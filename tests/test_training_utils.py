"""TrainingUtils: counters, storages, schedulers, full TrainingLoop runs."""
import math
import os

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from MLTools.Utilities import (
    AverageLossCounter,
    DelayedStorage,
    RecentLossCounter,
    TensorBatchStorage,
    TrainingLoop,
)
from MLTools.Utilities.TrainingUtils import CosineAnnealingWithWarmup
from MLTools.Utilities.Serialization import LoadDump


# ---------------------------------------------------------------- counters

def test_recent_loss_counter_window():
    c = RecentLossCounter(memory=3)
    for v in [1.0, 2.0, 3.0]:
        c.add(v)
    assert c.value() == pytest.approx(2.0)
    c.add(10.0)  # drops the 1.0
    assert c.value() == pytest.approx(5.0)
    assert len(c) == 3


def test_recent_loss_counter_empty():
    assert RecentLossCounter().value() == 0.0


def test_recent_loss_counter_many_evictions_stay_consistent():
    c = RecentLossCounter(memory=10)
    for i in range(1000):
        c.add(float(i % 7))
    assert c.value() == pytest.approx(sum((i % 7) for i in range(990, 1000)) / 10, abs=1e-6)


def test_average_loss_counter():
    c = AverageLossCounter()
    assert c.value() == 0.0
    for v in [1.0, 2.0, 6.0]:
        c.add(v)
    assert c.value() == pytest.approx(3.0)
    assert len(c) == 3


# ---------------------------------------------------------------- storages

def test_tensor_batch_storage_scalars_and_batches():
    s = TensorBatchStorage()
    s.add(torch.tensor(1.0))
    s.add(torch.tensor([2.0, 3.0]))
    out = s.value()
    assert torch.equal(out, torch.tensor([1.0, 2.0, 3.0]))
    s.clear()
    assert len(s) == 0
    assert s.value().numel() == 0


def test_tensor_batch_storage_2d_cat():
    s = TensorBatchStorage(cat_dim=0)
    s.add(torch.zeros(2, 4))
    s.add(torch.ones(3, 4))
    assert s.value().shape == (5, 4)


def test_delayed_storage_emits_every_k():
    d = DelayedStorage(itemize_every_k=3)
    assert d.add(torch.tensor(1.0)) is None
    assert d.add(torch.tensor(2.0)) is None
    out = d.add(torch.tensor(3.0))
    assert torch.equal(out, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(d.last_value(), out)
    assert len(d) == 0


def test_delayed_storage_flush_partial():
    d = DelayedStorage(itemize_every_k=10)
    d.add(torch.tensor(1.0))
    d.add(torch.tensor(2.0))
    out = d.flush()
    assert torch.equal(out, torch.tensor([1.0, 2.0]))
    assert d.flush() is None  # nothing left


# ---------------------------------------------------------------- scheduler

def test_cosine_annealing_with_warmup_shape():
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    sched = CosineAnnealingWithWarmup(opt, warmup_epochs=3, total_epochs=10, warmup_factor=0.1)
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.1)        # warmup start
    assert lrs[3] == pytest.approx(1.0)        # warmup done
    assert lrs[-1] < 0.1                       # annealed near zero
    assert all(b >= a - 1e-9 for a, b in zip(lrs[:3], lrs[1:4]))  # warmup monotonic


def test_cosine_annealing_validation():
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    with pytest.raises(ValueError):
        CosineAnnealingWithWarmup(opt, warmup_epochs=0, total_epochs=10)
    with pytest.raises(ValueError):
        CosineAnnealingWithWarmup(opt, warmup_epochs=5, total_epochs=5)


# ---------------------------------------------------------------- TrainingLoop

def make_regression_loaders(n=64, batch_size=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 1, generator=g)
    y = 2.0 * X + 1.0 + 0.01 * torch.randn(n, 1, generator=g)
    train = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    test = DataLoader(TensorDataset(X, y), batch_size=batch_size)
    return train, test


class QuietLoop(TrainingLoop):
    """Suppress prints for clean test output."""
    def print_epoch_results(self, *a, **k):
        pass
    def improved_val(self, *a, **k):
        pass
    def update_pbar(self, *a, **k):
        pass


class MetricLoop(QuietLoop):
    def compute_metrics(self, outputs, targets, valid_mode):
        if not valid_mode:
            return None
        return {"mse": float(((outputs - targets) ** 2).mean())}


def build_loop(cls=QuietLoop, epochs=3, tmp_path=None, log_every_k=2, **kw):
    train_loader, test_loader = make_regression_loaders()
    model = nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    paths = {}
    if tmp_path is not None:
        paths = {
            "checkpoint_path": str(tmp_path / "ckpt.pt"),
            "best_path": str(tmp_path / "best.pt"),
        }
    loop = cls(
        model=model,
        optimizer=opt,
        criterion=nn.MSELoss(),
        device="cpu",
        epochs=epochs,
        train_loop_constructor=lambda: iter(train_loader),
        test_loop_constructor=lambda: iter(test_loader),
        log_every_k=log_every_k,
        **paths,
        **kw,
    )
    return loop


def test_training_loop_learns():
    loop = build_loop(epochs=20)
    loop.run(resume=False)
    assert loop.history[-1]["test_loss"] < loop.history[0]["test_loss"]
    assert loop.history[-1]["test_loss"] < 0.5
    w = loop.model.weight.item()
    b = loop.model.bias.item()
    assert w == pytest.approx(2.0, abs=0.3)
    assert b == pytest.approx(1.0, abs=0.3)


def test_history_structure_and_test_metrics_key():
    """history['test_metrics'] used to store the *train* metrics blob."""
    loop = build_loop(cls=MetricLoop, epochs=2, metrics_train=False, metrics_test=True)
    loop.run(resume=False)
    assert len(loop.history) == 2
    entry = loop.history[-1]
    assert set(entry.keys()) == {"epoch", "train_loss", "test_loss", "train_metrics", "test_metrics"}
    train_m = LoadDump(entry["train_metrics"])
    test_m = LoadDump(entry["test_metrics"])
    assert train_m is None              # metrics_train=False
    assert isinstance(test_m, dict) and "mse" in test_m


def test_short_epoch_fewer_batches_than_k():
    """Loaders shorter than log_every_k used to yield empty histories and
    crash torch.cat([]). The flush fix must record every batch."""
    loop = build_loop(epochs=1, log_every_k=1000)
    loop.run(resume=False)
    assert len(loop.history) == 1
    assert loop.history[0]["train_loss"] > 0.0  # not the 0.0 of "no data seen"


def test_checkpoint_and_resume(tmp_path):
    loop = build_loop(epochs=2, tmp_path=tmp_path)
    loop.run(resume=False)
    assert os.path.isfile(tmp_path / "ckpt.pt")

    resumed = build_loop(epochs=4, tmp_path=tmp_path)
    resumed.run(resume=True)
    assert resumed.loaded
    assert len(resumed.history) == 4
    epochs_seen = [h["epoch"] for h in resumed.history]
    assert epochs_seen == [0, 1, 2, 3]


def test_best_checkpoint_written_on_first_epoch(tmp_path):
    """With a 1-epoch run, best_path must still be produced."""
    loop = build_loop(epochs=1, tmp_path=tmp_path)
    loop.run(resume=False)
    assert os.path.isfile(tmp_path / "best.pt")


def test_best_val_improves_and_saves(tmp_path):
    loop = build_loop(epochs=10, tmp_path=tmp_path)
    loop.run(resume=False)
    assert loop.best_val < math.inf
    assert loop.best_val <= loop.history[0]["test_loss"] + 1e-9
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert "model" in best and "epoch" in best


def test_early_stopping_via_post_epoch():
    class EarlyStop(QuietLoop):
        def post_epoch(self, history):
            return len(history) >= 2

    loop = build_loop(cls=EarlyStop, epochs=50)
    loop.run(resume=False)
    assert len(loop.history) == 2


def test_post_epoch_exception_does_not_kill_training(capsys):
    class Bad(QuietLoop):
        def post_epoch(self, history):
            raise RuntimeError("boom")

    loop = build_loop(cls=Bad, epochs=2)
    loop.run(resume=False)  # must not raise
    assert len(loop.history) == 2


def test_keep_outputs_false_runs():
    loop = build_loop(epochs=1, keep_outputs=False)
    loop.run(resume=False)
    assert len(loop.history) == 1


def test_improvement_check():
    check = TrainingLoop._improvement_check
    assert check(0.90, 1.00, minimal_change=0.005)        # clear improvement
    assert not check(0.999, 1.00, minimal_change=0.005)   # too small
    assert not check(1.10, 1.00, minimal_change=0.005)    # regression
    assert check(-1e-7, 0.0, minimal_change=0.005)        # eps floor (1e-8) at zero
    assert not check(-1e-9, 0.0, minimal_change=0.005)    # below the eps floor

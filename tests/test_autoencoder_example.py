"""End-to-end integration test: the autoencoder example pipeline.

Runs examples/autoencoder_tinyimagenet.py against a synthetic dataset written
in the exact TinyImageNetFeatures .bin format (low-rank features, so the
autoencoder genuinely has structure to learn). Verifies the full stack:
TinyImageNetFeatures -> StandardScaler -> TrainingLoop (+ scheduler,
checkpoints, metrics) -> summary artifacts.

The real-data version of this run (100k TinyImageNet ResNet50 features) is
the example script itself; see examples/autoencoder_tinyimagenet.py.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from MLTools.Utilities import SaveBin

pytestmark = pytest.mark.slow

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "examples", "autoencoder_tinyimagenet.py")


def write_features_bin(path, n, dim=64, rank=8, seed=0):
    """Synthetic blob in the TinyImageNetFeatures on-disk format.

    All splits share one low-rank basis (fixed seed) so train and val live in
    the same subspace; only the per-sample codes/noise differ per seed.
    """
    basis = torch.randn(rank, dim, generator=torch.Generator().manual_seed(1234))
    g = torch.Generator().manual_seed(seed)
    codes = torch.randn(n, rank, generator=g)
    feats = (codes @ basis + 0.05 * torch.randn(n, dim, generator=g) + 2.0).numpy()
    onehots = np.eye(200, dtype=np.float32)
    encoded = {
        f"img_{i:05d}.JPEG": (feats[i].astype(np.float32), onehots[i % 200])
        for i in range(n)
    }
    blob = {
        "wnid_to_id": {f"n{i:08d}": i for i in range(200)},
        "id_to_wnid": {i: f"n{i:08d}" for i in range(200)},
        "id_to_class": {i: f"class_{i}" for i in range(200)},
        "encoded": encoded,
    }
    SaveBin(str(path), blob)


@pytest.fixture
def feature_dir(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    write_features_bin(data / "train.bin", n=1200, seed=0)
    write_features_bin(data / "val.bin", n=200, seed=1)
    return data


def test_autoencoder_example_end_to_end(feature_dir, tmp_path):
    out = tmp_path / "run"
    proc = subprocess.run(
        [sys.executable, SCRIPT,
         "--data-dir", str(feature_dir), "--out-dir", str(out),
         "--epochs", "12", "--batch-size", "64",
         "--bottleneck", "16", "--num-workers", "0"],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=600,
    )
    assert proc.returncode == 0, f"script failed:\n{proc.stdout}\n{proc.stderr}"
    assert "OK: reconstruction loss decreased." in proc.stdout

    # artifacts
    assert (out / "ckpt.pt").is_file()
    assert (out / "best.pt").is_file()
    summary = json.loads((out / "summary.json").read_text())
    assert len(summary["epochs"]) == 12
    losses = [e["test_loss"] for e in summary["epochs"]]
    assert losses[-1] < losses[0]
    last_metrics = summary["epochs"][-1]["test_metrics"]
    assert last_metrics["r2"] > 0.3
    assert -1.0 <= last_metrics["cosine"] <= 1.0
    # loss chart rendered (matplotlib installed in the test env)
    assert (out / "loss_curve.png").is_file()


def test_autoencoder_example_resume(feature_dir, tmp_path):
    """Interrupt after 2 epochs, resume to 4 — history must be continuous."""
    out = tmp_path / "run"
    common = [sys.executable, SCRIPT,
              "--data-dir", str(feature_dir), "--out-dir", str(out),
              "--batch-size", "64", "--bottleneck", "16", "--num-workers", "0"]
    p1 = subprocess.run(common + ["--epochs", "2"],
                        capture_output=True, text=True, cwd=REPO_ROOT, timeout=600)
    assert p1.returncode == 0, p1.stderr
    p2 = subprocess.run(common + ["--epochs", "4", "--resume"],
                        capture_output=True, text=True, cwd=REPO_ROOT, timeout=600)
    assert p2.returncode == 0, p2.stderr
    summary = json.loads((out / "summary.json").read_text())
    assert [e["epoch"] for e in summary["epochs"]] == [0, 1, 2, 3]

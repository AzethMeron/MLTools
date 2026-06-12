#!/usr/bin/env python3
"""End-to-end integration exercise: feature autoencoder on TinyImageNet.

Trains an MLP autoencoder on the precomputed ResNet50 features of
TinyImageNet (MLTools.Dataset.TinyImageNetFeatures) using the library's own
TrainingLoop, StandardScaler, CosineAnnealingWithWarmup, Visualizer and
Cosmetics helpers. Serves both as a usage example and as a real-data smoke
test of the training stack.

Usage:
    python examples/autoencoder_tinyimagenet.py \
        --data-dir /tmp/tinf --out-dir /tmp/ae_run \
        --epochs 8 --batch-size 256 [--limit-train 20000]

Outputs (under --out-dir):
    ckpt.pt / best.pt   - TrainingLoop checkpoints
    loss_curve.png      - train/test loss chart (Visualizer)
    summary.json        - final losses + metrics per epoch
"""
import argparse
import json
import math
import os
import sys

import torch
import tqdm
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from MLTools.DataProcessing import StandardScaler
from MLTools.Dataset import TinyImageNetFeatures
from MLTools.Utilities import (
    CosineAnnealingWithWarmup,
    CountParameters,
    LoadDump,
    TrainingLoop,
)
from MLTools.Utilities.TrainingUtils import AverageLossCounter  # noqa: F401 (re-export check)


class FeatureAutoencoder(nn.Module):
    """2048 -> 512 -> bottleneck -> 512 -> 2048 MLP autoencoder."""

    def __init__(self, in_dim: int, bottleneck: int = 128, hidden: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderLoop(TrainingLoop):
    """TrainingLoop specialization: the reconstruction target is the input.

    The dataloader yields (features, onehot_label); labels are ignored and the
    standardized features serve as both input and target.
    """

    def __init__(self, *args, scaler: StandardScaler, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = scaler

    def _prepare(self, data):
        feats, _labels = data
        feats = feats.to(self.device, non_blocking=self.non_blocking).float()
        return self.scaler.transform(feats)

    def train_batch(self, data):
        x = self._prepare(data)
        self.optimizer.zero_grad(set_to_none=True)
        out = self.model(x)
        loss = self.criterion(out, x)
        loss.backward()
        self.optimizer.step()
        return loss, out, x

    def test_batch(self, data):
        with torch.inference_mode():
            x = self._prepare(data)
            out = self.model(x)
            loss = self.criterion(out, x)
            return loss, out, x

    def compute_metrics(self, outputs, targets, valid_mode):
        if not valid_mode or outputs is None:
            return None
        # R^2 of the reconstruction and mean cosine similarity, both on CPU
        ss_res = ((outputs - targets) ** 2).sum()
        ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum()
        r2 = 1.0 - float(ss_res / ss_tot)
        cos = float(nn.functional.cosine_similarity(outputs, targets, dim=1).mean())
        return {"r2": r2, "cosine": cos}

    def print_epoch_results(self, epoch, train_loss, train_metrics, test_loss, test_metrics):
        extra = ""
        if test_metrics:
            extra = f", r2={test_metrics['r2']:.4f}, cos={test_metrics['cosine']:.4f}"
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.5f}, test_loss={test_loss:.5f}{extra}")


def collect_features(dataset, limit=None):
    """Stack all feature vectors into one (N, D) float32 tensor."""
    n = len(dataset) if limit is None else min(limit, len(dataset))
    rows = [torch.as_tensor(dataset[i][0], dtype=torch.float32).reshape(-1) for i in range(n)]
    return torch.stack(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="/tmp/tinf")
    ap.add_argument("--out-dir", default="/tmp/ae_run")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--bottleneck", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--limit-train", type=int, default=None,
                    help="Optional cap on training samples (CPU-friendly runs)")
    ap.add_argument("--limit-val", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_bin = os.path.join(args.data_dir, "train.bin")
    val_bin = os.path.join(args.data_dir, "val.bin")
    if not (os.path.isfile(train_bin) and os.path.isfile(val_bin)):
        print("Feature files missing; downloading via TinyImageNetFeatures.download() ...")
        TinyImageNetFeatures.download(args.data_dir)

    train_ds = TinyImageNetFeatures(train_bin)
    val_ds = TinyImageNetFeatures(val_bin)
    print(f"Train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    if args.limit_train:
        train_ds = Subset(train_ds, range(min(args.limit_train, len(train_ds))))
    if args.limit_val:
        val_ds = Subset(val_ds, range(min(args.limit_val, len(val_ds))))

    # Fit the repo's StandardScaler on the training features
    base_train = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
    feats = collect_features(base_train, limit=args.limit_train)
    in_dim = feats.shape[1]
    print(f"Feature dim: {in_dim}; fitting StandardScaler on {feats.shape[0]} vectors")
    scaler = StandardScaler().fit(feats, batch_size=4096)
    scaler = scaler.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = FeatureAutoencoder(in_dim, bottleneck=args.bottleneck)
    print(f"Autoencoder parameters: {CountParameters(model, return_sci_notation=True)}"
          f" ({CountParameters(model)})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_epochs=1,
                                          total_epochs=args.epochs)

    loop = AutoencoderLoop(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        device=device,
        epochs=args.epochs,
        train_loop_constructor=lambda: tqdm.tqdm(train_loader, mininterval=1.0),
        test_loop_constructor=lambda: tqdm.tqdm(val_loader, mininterval=1.0),
        scheduler=scheduler,
        checkpoint_path=os.path.join(args.out_dir, "ckpt.pt"),
        best_path=os.path.join(args.out_dir, "best.pt"),
        log_every_k=10,
        metrics_test=True,
        scaler=scaler,
    )
    loop.run(resume=args.resume)

    # ---- report ----
    history = loop.history
    summary = {
        "epochs": [
            {
                "epoch": h["epoch"],
                "train_loss": h["train_loss"],
                "test_loss": h["test_loss"],
                "test_metrics": LoadDump(h["test_metrics"]),
            }
            for h in history
        ],
        "best_val": loop.best_val,
        "parameters": CountParameters(model),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    try:
        from MLTools.Utilities import Visualizer
        viz = Visualizer(figsize=(7, 4))
        fig, _ = viz.loss_chart(history)
        Visualizer.draw_to_pil(fig).save(os.path.join(args.out_dir, "loss_curve.png"))
        print(f"Loss chart saved to {os.path.join(args.out_dir, 'loss_curve.png')}")
    except ImportError:
        print("matplotlib not installed; skipping loss chart")

    first, last = history[0], history[-1]
    print(f"\ntrain loss {first['train_loss']:.5f} -> {last['train_loss']:.5f}")
    print(f"test  loss {first['test_loss']:.5f} -> {last['test_loss']:.5f}")
    assert last["test_loss"] < first["test_loss"], "autoencoder failed to learn"
    print("OK: reconstruction loss decreased.")


if __name__ == "__main__":
    main()

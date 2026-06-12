"""Randomized stress / fuzz tests ("blast it with data").

Property-based checks over many random configurations. Marked slow; run with
`pytest -m slow` or a full `pytest` to include them. All seeds are derived
from the parametrized trial index, so failures reproduce deterministically.
"""
import math
import random

import numpy as np
import pytest
import torch
from PIL import Image

from MLTools.DataProcessing import (
    IncrementalPCA,
    Letterbox,
    PCA,
    RandomJPEGCompression,
    RandomScale,
    StandardScaler,
)
from MLTools.Detections import Detection

pytestmark = pytest.mark.slow

TRIALS = 25


# ---------------------------------------------------------------- StandardScaler

@pytest.mark.parametrize("trial", range(TRIALS))
def test_scaler_fuzz(trial):
    g = torch.Generator().manual_seed(trial * 7919)
    N = int(torch.randint(1, 600, (1,), generator=g))
    D = int(torch.randint(1, 64, (1,), generator=g))
    offset = 10 ** float(torch.empty(1).uniform_(-2, 6, generator=g))
    spread = 10 ** float(torch.empty(1).uniform_(-2, 2, generator=g))
    X = torch.randn(N, D, generator=g) * spread + offset
    bs = int(torch.randint(1, max(N, 2), (1,), generator=g))

    scaler = StandardScaler().fit(X, batch_size=bs)
    Z = scaler.transform(X)

    assert torch.isfinite(Z).all(), "transform produced non-finite values"
    assert (scaler.var_ >= 0).all(), "negative variance"
    assert int(scaler.n_samples_seen_.item()) == N
    # round-trip within float32 tolerance scaled by data magnitude
    back = scaler.inverse_transform(Z)
    tol = 1e-3 * (abs(offset) + spread)
    assert torch.allclose(back, X, atol=tol), "inverse_transform round-trip failed"


# ---------------------------------------------------------------- PCA

@pytest.mark.parametrize("trial", range(TRIALS))
def test_pca_fuzz(trial):
    g = torch.Generator().manual_seed(trial * 104729)
    N = int(torch.randint(2, 300, (1,), generator=g))
    D = int(torch.randint(1, 32, (1,), generator=g))
    K = int(torch.randint(1, 40, (1,), generator=g))
    X = torch.randn(N, D, generator=g)
    X = X - X.mean(0, keepdim=True)
    bs = int(torch.randint(1, N + 1, (1,), generator=g))

    pca = PCA(n_components=K).fit(X, batch_size=bs)
    k = min(K, D)
    assert pca.components_.shape == (D, k)
    # orthonormal columns
    eye = torch.eye(k)
    assert torch.allclose(pca.components_.T @ pca.components_, eye, atol=1e-3)
    r = pca.explained_variance_ratio()
    assert torch.isfinite(r).all()
    assert (r >= -1e-6).all() and float(r.sum()) <= 1.0 + 1e-4
    Z = pca.transform(X, batch_size=max(1, N // 3))
    assert Z.shape == (N, k) and torch.isfinite(Z).all()


# ---------------------------------------------------------------- IncrementalPCA

@pytest.mark.parametrize("trial", range(TRIALS))
def test_incremental_pca_fuzz(trial):
    g = torch.Generator().manual_seed(trial * 1299709)
    N = int(torch.randint(2, 300, (1,), generator=g))
    D = int(torch.randint(2, 24, (1,), generator=g))
    K = int(torch.randint(1, D + 1, (1,), generator=g))
    offset = float(torch.empty(1).uniform_(-1e5, 1e5, generator=g))
    X = torch.randn(N, D, generator=g) * 3 + offset

    ipca = IncrementalPCA(n_components=K)
    splits = torch.split(X, max(1, N // int(torch.randint(1, 6, (1,), generator=g))))
    for chunk in splits:
        ipca.partial_fit(chunk)

    assert ipca.n_samples_seen_ == N
    assert torch.isfinite(ipca.explained_variance_).all()
    assert (ipca.explained_variance_ >= -1e-5).all()
    Z = ipca.transform(X)
    assert Z.shape == (N, K) and torch.isfinite(Z).all()
    # streaming result == one-shot result on the same data
    ref = IncrementalPCA(n_components=K).fit(X)
    assert torch.allclose(ipca.explained_variance_, ref.explained_variance_,
                          rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------- Letterbox

@pytest.mark.parametrize("trial", range(TRIALS))
def test_letterbox_fuzz(trial):
    rng = np.random.default_rng(trial)
    ow, oh = int(rng.integers(1, 400)), int(rng.integers(1, 400))
    tw, th = int(rng.integers(8, 256)), int(rng.integers(8, 256))
    fill = tuple(int(c) for c in rng.integers(0, 256, 3))
    img = Image.fromarray(rng.integers(0, 256, (oh, ow, 3), dtype=np.uint8))

    lb = Letterbox((tw, th), fill=fill)
    out = lb(img)
    assert out.size == (tw, th)

    # detections stay inside the canvas
    det = Detection.from_cxcywh(ow / 2, oh / 2, ow, oh, 0, 1.0)
    mapped = lb.transform_detections([det], (ow, oh))[0]
    x1, y1, x2, y2 = mapped.to_xyxy()
    assert -1.0 <= x1 and x2 <= tw + 1.0
    assert -1.0 <= y1 and y2 <= th + 1.0


# ---------------------------------------------------------------- pixel transforms

@pytest.mark.parametrize("trial", range(TRIALS))
def test_jpeg_fuzz(trial):
    rng = np.random.default_rng(trial + 999)
    w, h = int(rng.integers(1, 128)), int(rng.integers(1, 128))
    img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    t = RandomJPEGCompression(
        min_quality=int(rng.integers(1, 50)),
        max_quality=int(rng.integers(50, 96)),
        subsampling=int(rng.integers(0, 3)),
        seed=trial,
        p=float(rng.uniform(0.3, 1.0)),
    )
    out = t(img)
    assert out.size == img.size and out.mode == img.mode


@pytest.mark.parametrize("trial", range(TRIALS))
def test_random_scale_fuzz(trial):
    rng = np.random.default_rng(trial + 5555)
    random.seed(trial)
    c = int(rng.integers(1, 5))
    h, w = int(rng.integers(2, 200)), int(rng.integers(2, 200))
    lo = float(rng.uniform(0.05, 1.5))
    hi = lo + float(rng.uniform(0.0, 1.0))
    algo = ["nearest", "bilinear", "bicubic", "area"][trial % 4]
    t = RandomScale(min_scale=lo, max_scale=hi,
                    downscale_algorithm=algo, upscale_algorithm=algo)
    img = torch.rand(c, h, w)
    out = t(img)
    assert out.shape == (c, h, w)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------- Detection geometry

@pytest.mark.parametrize("trial", range(TRIALS))
def test_detection_geometry_fuzz(trial):
    rng = np.random.default_rng(trial + 31337)
    cx, cy = rng.uniform(-100, 100, 2)
    w, h = rng.uniform(0.1, 50, 2)
    rot = rng.uniform(-360, 360)
    d = Detection(cx, cy, w, h, rot, int(rng.integers(0, 10)), float(rng.uniform(0, 1)))

    # corners are consistent with the analytic envelope
    x1, y1, x2, y2 = d.to_xyxy()
    rad = math.radians(rot)
    half_w = (abs(w * math.cos(rad)) + abs(h * math.sin(rad))) / 2
    half_h = (abs(w * math.sin(rad)) + abs(h * math.cos(rad))) / 2
    assert x1 == pytest.approx(cx - half_w, abs=1e-6)
    assert x2 == pytest.approx(cx + half_w, abs=1e-6)
    assert y1 == pytest.approx(cy - half_h, abs=1e-6)
    assert y2 == pytest.approx(cy + half_h, abs=1e-6)

    # rotation around own center keeps the center fixed
    rotated = d.Rotate(rng.uniform(-180, 180), (cx, cy))
    assert rotated.cx == pytest.approx(cx, abs=1e-9)
    assert rotated.cy == pytest.approx(cy, abs=1e-9)

    # IOU with itself is 1 (degenerate thin boxes excluded by w,h >= 0.1)
    assert d.IOU(d) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("trial", range(10))
def test_nms_fuzz_never_increases_count_and_keeps_best(trial):
    rng = np.random.default_rng(trial)
    dets = []
    for _ in range(int(rng.integers(1, 60))):
        cx, cy = rng.uniform(0, 200, 2)
        w, h = rng.uniform(1, 60, 2)
        dets.append(Detection.from_cxcywh(cx, cy, w, h,
                                          int(rng.integers(0, 3)),
                                          float(rng.uniform(0, 1))))
    kept = Detection.NMS(dets, iou_threshold=0.5)
    assert len(kept) <= len(dets)
    # highest-confidence detection always survives class-aware NMS
    best = max(dets, key=lambda d: d.confidence)
    assert best in kept

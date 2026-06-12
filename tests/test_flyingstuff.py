"""FlyingStuff: flow-file IO, affine math, renderer flow correctness, loader."""
import math
import os
import struct

import numpy as np
import pytest
import torch
from PIL import Image

from MLTools.Dataset.FlyingStuff import (
    FlyingStuffDataset,
    Layer,
    WeightedPool,
    affine_matrix_about,
    apply_affine_to_points,
    background_layer,
    parse_pool_arg,
    pil_to_tensor_mask,
    random_object_layer,
    render_pair_and_flow,
    resize_keep_aspect,
    to_affine_grid_theta,
    write_flo,
)
from MLTools.Dataset.FlyingStuff import ForegroundSample

from conftest import make_pil, solid_pil


# ---------------------------------------------------------------- .flo IO

def test_write_flo_read_roundtrip(tmp_path):
    flow = np.random.default_rng(0).standard_normal((12, 16, 2)).astype(np.float32)
    path = str(tmp_path / "f.flo")
    write_flo(flow, path)
    with open(path, "rb") as f:
        magic = struct.unpack("f", f.read(4))[0]
        w = struct.unpack("i", f.read(4))[0]
        h = struct.unpack("i", f.read(4))[0]
        data = np.fromfile(f, np.float32).reshape(h, w, 2)
    assert magic == pytest.approx(202021.25)
    assert (w, h) == (16, 12)
    assert np.array_equal(data, flow)


def test_flyingstuff_dataset_loader(tmp_path):
    # synthesize a fake generated dataset on disk
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    H, W = 10, 14
    for bid in ("000001", "000002"):
        solid_pil(W, H, (10, 20, 30)).save(split_dir / f"{bid}_img1.png")
        solid_pil(W, H, (40, 50, 60)).save(split_dir / f"{bid}_img2.png")
        flow = np.full((H, W, 2), 1.5, dtype=np.float32)
        write_flo(flow, str(split_dir / f"{bid}_flow.flo"))
        Image.fromarray(np.full((H, W), 255, np.uint8)).save(split_dir / f"{bid}_valid.png")
    (tmp_path / "train.txt").write_text("000001\n000002\n")

    ds = FlyingStuffDataset(str(tmp_path), split="train")
    assert len(ds) == 2
    sample = ds[0]
    assert sample["img1"].shape == (3, H, W)
    assert sample["img2"].shape == (3, H, W)
    assert sample["flow"].shape == (2, H, W)
    assert sample["valid"].shape == (1, H, W)
    assert torch.allclose(sample["flow"], torch.full((2, H, W), 1.5))
    assert torch.allclose(sample["valid"], torch.ones(1, H, W))


def test_flo_reader_rejects_bad_magic(tmp_path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    bad = struct.pack("f", 1.0) + struct.pack("i", 2) + struct.pack("i", 2) + bytes(32)
    (split_dir / "x_flow.flo").write_bytes(bad)
    (tmp_path / "train.txt").write_text("x\n")
    ds = FlyingStuffDataset(str(tmp_path))
    with pytest.raises(ValueError):
        ds._read_flo(str(split_dir / "x_flow.flo"))


# ---------------------------------------------------------------- affine math

def test_affine_matrix_pure_translation():
    A = affine_matrix_about(5.0, -3.0, (1.0, 1.0), 0.0, (0.0, 0.0), pivot=(0, 0))
    pts = torch.tensor([[0.0, 0.0], [10.0, 20.0]])
    out = apply_affine_to_points(A, pts)
    assert torch.allclose(out, pts + torch.tensor([5.0, -3.0]))


def test_affine_matrix_rotation_about_pivot_keeps_pivot():
    A = affine_matrix_about(0.0, 0.0, (1.0, 1.0), 90.0, (0.0, 0.0), pivot=(7.0, 9.0))
    out = apply_affine_to_points(A, torch.tensor([[7.0, 9.0]]))
    assert torch.allclose(out, torch.tensor([[7.0, 9.0]]), atol=1e-5)


def test_affine_scale_about_pivot():
    A = affine_matrix_about(0.0, 0.0, (2.0, 2.0), 0.0, (0.0, 0.0), pivot=(10.0, 10.0))
    out = apply_affine_to_points(A, torch.tensor([[11.0, 10.0]]))
    assert torch.allclose(out, torch.tensor([[12.0, 10.0]]), atol=1e-5)


def test_to_affine_grid_theta_shapes_and_validation():
    A = torch.eye(3)
    theta = to_affine_grid_theta(A, (8, 6), (6, 8))
    assert theta.shape == (2, 3)
    theta2 = to_affine_grid_theta(A[:2, :], (8, 6), (6, 8))
    assert torch.allclose(theta, theta2)
    with pytest.raises(ValueError):
        to_affine_grid_theta(torch.eye(4), (8, 6), (6, 8))


# ---------------------------------------------------------------- helpers

def test_resize_keep_aspect():
    img = solid_pil(100, 50)
    out = resize_keep_aspect(img, 200)
    assert out.size == (200, 100)
    out2 = resize_keep_aspect(solid_pil(50, 100), 200)
    assert out2.size == (100, 200)


def test_pil_to_tensor_mask_binarizes():
    m = Image.fromarray(np.array([[0, 100, 200]], dtype=np.uint8), "L")
    t = pil_to_tensor_mask(m)
    assert t.shape == (1, 1, 3)
    assert t.tolist() == [[[0.0, 0.0, 1.0]]]


def test_parse_pool_arg():
    specs = parse_pool_arg("pets, voc_person:2.5 ,")
    assert [(s.name, s.weight) for s in specs] == [("pets", 1.0), ("voc_person", 2.5)]
    assert parse_pool_arg("") == []


def test_weighted_pool_sampling_distribution():
    import random
    random.seed(0)
    pool = WeightedPool(["a", "b"], [1.0, 9.0])
    counts = {"a": 0, "b": 0}
    for _ in range(2000):
        counts[pool.sample()] += 1
    assert counts["b"] > counts["a"] * 4


def test_layer_constructors():
    fg = ForegroundSample(make_pil(40, 30), Image.fromarray(np.full((30, 40), 255, np.uint8), "L"))
    layer = random_object_layer(fg, (64, 64))
    assert layer.tex.shape[0] == 3
    assert layer.alpha.shape[0] == 1
    assert layer.A1.shape == (3, 3) and layer.A2.shape == (3, 3)

    bg = background_layer(make_pil(40, 30), (64, 48))
    assert bg.tex.shape == (3, 64, 48) or bg.tex.shape == (3, 48, 64)
    assert bg.z == -1.0
    assert torch.allclose(bg.A1, torch.eye(3))


# ---------------------------------------------------------------- renderer

def translation_layer(H, W, tx, ty, seed=0):
    g = torch.Generator().manual_seed(seed)
    tex = torch.rand(3, H, W, generator=g)
    alpha = torch.ones(1, H, W)
    A1 = torch.eye(3)
    A2 = torch.eye(3)
    A2[0, 2] = tx
    A2[1, 2] = ty
    return Layer(tex=tex, alpha=alpha, A1=A1, A2=A2, z=-1.0, pad_mode="border")


def test_render_pure_translation_flow():
    """Background translated by (tx, ty): the flow field must equal it."""
    H, W = 32, 40
    tx, ty = 4.0, -2.0
    layer = translation_layer(H, W, tx, ty)
    I1, I2, flow, valid = render_pair_and_flow([layer], (H, W), torch.device("cpu"))
    assert I1.shape == (3, H, W) and I2.shape == (3, H, W)
    assert flow.shape == (2, H, W) and valid.shape == (H, W)
    assert torch.allclose(flow[0], torch.full((H, W), tx), atol=0.1)
    assert torch.allclose(flow[1], torch.full((H, W), ty), atol=0.1)
    assert valid.float().mean() > 0.5


def test_render_frame2_is_shifted_frame1():
    """Photometric check: I2 sampled at p+flow equals I1 at p (away from borders)."""
    H, W = 32, 32
    tx, ty = 5.0, 3.0
    layer = translation_layer(H, W, tx, ty, seed=1)
    I1, I2, flow, valid = render_pair_and_flow([layer], (H, W), torch.device("cpu"))
    # I2(x + tx, y + ty) == I1(x, y) for interior pixels
    interior1 = I1[:, 8:16, 8:16]
    interior2 = I2[:, 8 + int(ty):16 + int(ty), 8 + int(tx):16 + int(tx)]
    assert torch.allclose(interior1, interior2, atol=0.05)


def test_render_identity_motion_zero_flow():
    H, W = 16, 16
    layer = translation_layer(H, W, 0.0, 0.0)
    _, _, flow, valid = render_pair_and_flow([layer], (H, W), torch.device("cpu"))
    assert torch.allclose(flow, torch.zeros(2, H, W), atol=1e-3)
    assert valid.all()


def test_render_min_motion_enforced():
    H, W = 24, 24
    layer = translation_layer(H, W, 0.0, 0.0)
    layer.z = 0.5  # treat as object so bounds apply
    _, _, flow, valid = render_pair_and_flow(
        [layer], (H, W), torch.device("cpu"), min_motion_px=3.0
    )
    mag = flow.norm(dim=0)
    assert float(mag.median()) >= 2.5  # static layer got motion injected


def test_render_max_motion_enforced():
    H, W = 24, 24
    layer = translation_layer(H, W, 20.0, 0.0)
    layer.z = 0.5
    _, _, flow, valid = render_pair_and_flow(
        [layer], (H, W), torch.device("cpu"), max_motion_px=5.0,
        apply_to_background=True,
    )
    mag = flow.norm(dim=0)
    # 80th-percentile motion rescaled to <= max (allow tolerance)
    assert float(torch.quantile(mag.flatten(), 0.8)) <= 6.0


def test_render_object_over_background_occlusion():
    H, W = 32, 32
    bg = translation_layer(H, W, 0.0, 0.0)
    # small static object in the middle, on top
    tex = torch.ones(3, 8, 8) * 0.9
    alpha = torch.ones(1, 8, 8)
    A = torch.eye(3); A[0, 2] = 12; A[1, 2] = 12  # place at (12..19)
    obj = Layer(tex=tex, alpha=alpha, A1=A.clone(), A2=A.clone(), z=1.0, pad_mode="zeros")
    I1, I2, flow, valid = render_pair_and_flow([bg, obj], (H, W), torch.device("cpu"))
    # object pixels rendered on top
    assert torch.allclose(I1[:, 14, 14], torch.tensor([0.9, 0.9, 0.9]), atol=0.05)
    assert torch.allclose(flow[:, 14, 14], torch.zeros(2), atol=1e-3)

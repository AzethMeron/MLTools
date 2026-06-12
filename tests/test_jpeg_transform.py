"""RandomJPEGCompression: determinism, probability gate, mode handling."""
import numpy as np
import pytest
from PIL import Image

from MLTools.DataProcessing import RandomJPEGCompression

from conftest import make_pil


# ---------------------------------------------------------------- typical

def test_returns_same_mode_and_size():
    t = RandomJPEGCompression(seed=0)
    img = make_pil(64, 48)
    out = t(img)
    assert out.mode == "RGB" and out.size == (64, 48)


def test_low_quality_actually_degrades():
    t = RandomJPEGCompression(min_quality=1, max_quality=5, seed=0)
    img = make_pil(64, 64, seed=3)
    out = t(img)
    diff = np.abs(np.asarray(out).astype(int) - np.asarray(img).astype(int))
    assert diff.mean() > 1.0, "1-5 quality JPEG should visibly change the image"


def test_seed_determinism():
    img = make_pil(48, 48, seed=1)
    a = RandomJPEGCompression(seed=99)(img)
    b = RandomJPEGCompression(seed=99)(img)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_seeded_subsampling_choice_determinism():
    """Subsampling list used to be drawn from the *global* RNG, breaking
    seeded reproducibility. Both instances must pick identical params."""
    import random as global_random
    img = make_pil(48, 48, seed=2)
    t1 = RandomJPEGCompression(seed=7, subsampling=[0, 1, 2])
    t2 = RandomJPEGCompression(seed=7, subsampling=[0, 1, 2])
    global_random.seed(1)
    a = t1(img)
    global_random.seed(2)  # perturb global RNG between runs
    b = t2(img)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_probability_zero_returns_original():
    t = RandomJPEGCompression(p=0.0, seed=0)
    img = make_pil()
    assert t(img) is img


def test_probability_statistics():
    t = RandomJPEGCompression(p=0.5, seed=123, min_quality=1, max_quality=5)
    img = make_pil(16, 16, seed=4)
    applied = sum(1 for _ in range(200) if t(img) is not img)
    assert 60 < applied < 140  # ~100 expected


# ---------------------------------------------------------------- modes / edge

def test_grayscale_image():
    t = RandomJPEGCompression(seed=0)
    out = t(make_pil(32, 32, mode="L"))
    assert out.mode == "L"


def test_rgba_image_no_crash():
    """RGBA cannot be JPEG-encoded directly; used to raise OSError."""
    t = RandomJPEGCompression(seed=0)
    img = make_pil(32, 32).convert("RGBA")
    out = t(img)
    assert out.mode == "RGBA" and out.size == (32, 32)


def test_palette_image_no_crash():
    t = RandomJPEGCompression(seed=0)
    img = make_pil(32, 32).convert("P")
    out = t(img)
    assert out.mode == "P"


def test_tiny_image():
    out = RandomJPEGCompression(seed=0)(make_pil(1, 1))
    assert out.size == (1, 1)


# ---------------------------------------------------------------- errors / repr

def test_invalid_quality_range():
    with pytest.raises(ValueError):
        RandomJPEGCompression(min_quality=50, max_quality=25)
    with pytest.raises(ValueError):
        RandomJPEGCompression(min_quality=0)
    with pytest.raises(ValueError):
        RandomJPEGCompression(max_quality=100)
    with pytest.raises(ValueError):
        RandomJPEGCompression(p=1.5)


def test_rejects_non_pil():
    with pytest.raises(TypeError):
        RandomJPEGCompression(seed=0)(np.zeros((8, 8, 3), dtype=np.uint8))


def test_repr_is_well_formed():
    r = repr(RandomJPEGCompression(min_quality=10, max_quality=90, seed=5, p=0.7))
    assert "min_quality=10" in r and "seed=5" in r and "p=0.7" in r
    assert "bound method" not in r

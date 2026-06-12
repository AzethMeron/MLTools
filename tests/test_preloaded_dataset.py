"""PreloadedDataset: all storage modes, tensor/array inputs, transforms."""
import numpy as np
import pytest
import torch
from PIL import Image

from MLTools.Dataset import PreloadedDataset

from conftest import make_pil


class ListDataset:
    def __init__(self, items):
        self.items = items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]


def pil_base(n=4):
    return ListDataset([(make_pil(16, 12, seed=i), i) for i in range(n)])


# ---------------------------------------------------------------- storage modes

@pytest.mark.parametrize("mode", [None, "zlib", "PNG"])
def test_lossless_modes_roundtrip(mode):
    base = pil_base()
    ds = PreloadedDataset(base, compression=mode)
    assert len(ds) == 4
    for i in range(4):
        img, label = ds[i]
        assert isinstance(img, Image.Image)
        assert label == i
        assert np.array_equal(np.asarray(img), np.asarray(base[i][0]))


def test_jpeg_mode_lossy_but_close():
    # smooth gradient compresses well; random noise would not
    grad = np.linspace(0, 255, 16, dtype=np.uint8)
    arr = np.stack([np.tile(grad, (12, 1))] * 3, axis=-1)
    base = ListDataset([(Image.fromarray(arr), 0)])
    ds = PreloadedDataset(base, compression="JPG", jpg_quality=95)
    img, label = ds[0]
    diff = np.abs(np.asarray(img).astype(int) - arr.astype(int))
    assert diff.mean() < 10
    assert label == 0


def test_jpeg_quality_clamped():
    ds = PreloadedDataset(pil_base(1), compression="JPEG", jpg_quality=10_000)
    assert ds._jpg_quality == 95


def test_case_insensitive_modes():
    for mode in ("png", "jpeg", "Zlib", "NONE"):
        PreloadedDataset(pil_base(1), compression=mode)


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        PreloadedDataset(pil_base(1), compression="webp")


# ---------------------------------------------------------------- transform

def test_transform_applied():
    ds = PreloadedDataset(pil_base(), compression="PNG", transform=lambda im: im.size)
    out, label = ds[2]
    assert out == (16, 12)


# ---------------------------------------------------------------- input conversion

def test_tensor_chw_float_input():
    items = [(torch.rand(3, 8, 8), 0)]
    ds = PreloadedDataset(ListDataset(items), compression=None)
    img, _ = ds[0]
    assert isinstance(img, Image.Image) and img.size == (8, 8)


def test_tensor_chw_grayscale_input():
    """(1, H, W) tensors used to crash PIL with an (H, W, 1) array."""
    items = [(torch.rand(1, 8, 8), 0)]
    ds = PreloadedDataset(ListDataset(items), compression=None)
    img, _ = ds[0]
    assert img.size == (8, 8) and img.mode == "L"


def test_tensor_uint8_input():
    items = [(torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8), 1)]
    ds = PreloadedDataset(ListDataset(items), compression="PNG")
    img, label = ds[0]
    assert img.size == (8, 8) and label == 1


def test_numpy_float_input():
    items = [(np.random.rand(8, 8, 3).astype(np.float32), "lbl")]
    ds = PreloadedDataset(ListDataset(items), compression=None)
    img, label = ds[0]
    assert img.size == (8, 8) and label == "lbl"


def test_numpy_uint8_input():
    items = [(np.zeros((8, 8), dtype=np.uint8), 0)]
    ds = PreloadedDataset(ListDataset(items), compression="PNG")
    img, _ = ds[0]
    assert img.mode == "L"


# ---------------------------------------------------------------- misc

def test_empty_dataset():
    ds = PreloadedDataset(ListDataset([]), compression="PNG")
    assert len(ds) == 0


def test_labels_preserved_arbitrary_objects():
    labels = [np.eye(3)[1], {"a": 1}, (1, 2), None]
    items = [(make_pil(8, 8, seed=i), labels[i]) for i in range(4)]
    ds = PreloadedDataset(ListDataset(items), compression="PNG")
    assert np.array_equal(ds[0][1], labels[0])
    assert ds[1][1] == {"a": 1}
    assert ds[3][1] is None


def test_repr():
    assert "PNG" in repr(PreloadedDataset(pil_base(1), compression="PNG"))

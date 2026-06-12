"""MNIST / CIFAR10 / TinyImageNet loaders against synthetic on-disk fixtures.

No network access: the official file formats are reproduced byte-for-byte in
tmp directories, including adversarial corrupted variants.
"""
import gzip
import pickle
import struct

import numpy as np
import pytest
from PIL import Image

from MLTools.Dataset import CIFAR10, MNIST, TinyImageNet


# ================================================================ MNIST helpers

def write_idx_images(path, images):
    n, rows, cols = images.shape
    with open(path, "wb") as f:
        f.write(struct.pack(">IIII", 0x00000803, n, rows, cols))
        f.write(images.astype(np.uint8).tobytes())


def write_idx_labels(path, labels):
    with open(path, "wb") as f:
        f.write(struct.pack(">II", 0x00000801, len(labels)))
        f.write(labels.astype(np.uint8).tobytes())


@pytest.fixture
def mnist_dir(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "MNIST"
    root.mkdir()
    train_imgs = rng.integers(0, 256, (6, 28, 28), dtype=np.uint8)
    train_lbls = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
    test_imgs = rng.integers(0, 256, (3, 28, 28), dtype=np.uint8)
    test_lbls = np.array([7, 8, 9], dtype=np.uint8)
    write_idx_images(root / "train-images-idx3-ubyte", train_imgs)
    write_idx_labels(root / "train-labels-idx1-ubyte", train_lbls)
    write_idx_images(root / "t10k-images-idx3-ubyte", test_imgs)
    write_idx_labels(root / "t10k-labels-idx1-ubyte", test_lbls)
    return tmp_path, train_imgs, train_lbls


def test_mnist_train_loading(mnist_dir):
    root, imgs, lbls = mnist_dir
    ds = MNIST(root, train=True)
    assert len(ds) == 6
    img, onehot = ds[2]
    assert isinstance(img, Image.Image) and img.size == (28, 28)
    assert np.array_equal(np.asarray(img), imgs[2])
    assert onehot.shape == (10,) and onehot[2] == 1.0 and onehot.sum() == 1.0


def test_mnist_val_loading(mnist_dir):
    root, _, _ = mnist_dir
    ds = MNIST(root, train=False)
    assert len(ds) == 3
    _, onehot = ds[0]
    assert onehot[7] == 1.0


def test_mnist_transform(mnist_dir):
    root, _, _ = mnist_dir
    ds = MNIST(root, train=True, transform=lambda im: np.asarray(im).mean())
    val, _ = ds[0]
    assert isinstance(val, float) or isinstance(val, np.floating)


def test_mnist_gz_decompression(tmp_path):
    rng = np.random.default_rng(1)
    root = tmp_path / "MNIST"
    root.mkdir()
    imgs = rng.integers(0, 256, (2, 28, 28), dtype=np.uint8)
    lbls = np.array([1, 2], dtype=np.uint8)
    # write only .gz versions
    raw_img = struct.pack(">IIII", 0x00000803, 2, 28, 28) + imgs.tobytes()
    raw_lbl = struct.pack(">II", 0x00000801, 2) + lbls.tobytes()
    (root / "train-images-idx3-ubyte.gz").write_bytes(gzip.compress(raw_img))
    (root / "train-labels-idx1-ubyte.gz").write_bytes(gzip.compress(raw_lbl))
    ds = MNIST(tmp_path, train=True)
    assert len(ds) == 2
    assert (root / "train-images-idx3-ubyte").exists()  # decompressed on demand


def test_mnist_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        MNIST(tmp_path, train=True)


def test_mnist_corrupt_magic_rejected(tmp_path):
    """Adversarial: wrong magic number must raise, not load garbage."""
    root = tmp_path / "MNIST"
    root.mkdir()
    bad = struct.pack(">IIII", 0xDEADBEEF, 1, 28, 28) + bytes(28 * 28)
    (root / "train-images-idx3-ubyte").write_bytes(bad)
    write_idx_labels(root / "train-labels-idx1-ubyte", np.array([0], dtype=np.uint8))
    with pytest.raises(RuntimeError):
        MNIST(tmp_path, train=True)


def test_mnist_count_mismatch_rejected(tmp_path):
    root = tmp_path / "MNIST"
    root.mkdir()
    write_idx_images(root / "train-images-idx3-ubyte",
                     np.zeros((2, 28, 28), dtype=np.uint8))
    write_idx_labels(root / "train-labels-idx1-ubyte", np.zeros(5, dtype=np.uint8))
    with pytest.raises(RuntimeError):
        MNIST(tmp_path, train=True)


# ================================================================ CIFAR10

@pytest.fixture
def cifar_dir(tmp_path):
    rng = np.random.default_rng(2)
    root = tmp_path / "cifar-10-batches-py"
    root.mkdir()
    all_data = []
    for b in range(1, 6):
        data = rng.integers(0, 256, (4, 3072), dtype=np.uint8)
        labels = list(rng.integers(0, 10, 4))
        with open(root / f"data_batch_{b}", "wb") as f:
            pickle.dump({"data": data, "labels": labels}, f)
        all_data.append((data, labels))
    test_data = rng.integers(0, 256, (4, 3072), dtype=np.uint8)
    with open(root / "test_batch", "wb") as f:
        pickle.dump({"data": test_data, "labels": [1, 2, 3, 4]}, f)
    return tmp_path, all_data


def test_cifar_train_loading(cifar_dir):
    root, all_data = cifar_dir
    ds = CIFAR10(root, train=True)
    assert len(ds) == 20  # 5 batches x 4
    img, onehot = ds[0]
    assert img.size == (32, 32) and img.mode == "RGB"
    assert onehot.shape == (10,) and onehot.sum() == 1.0


def test_cifar_chw_to_hwc_layout(cifar_dir):
    """Verify R/G/B plane deinterleaving is correct."""
    root, all_data = cifar_dir
    ds = CIFAR10(root, train=True)
    data0, _ = all_data[0]
    expected = data0[0].reshape(3, 32, 32).transpose(1, 2, 0)
    assert np.array_equal(np.asarray(ds[0][0]), expected)


def test_cifar_test_split(cifar_dir):
    root, _ = cifar_dir
    ds = CIFAR10(root, train=False)
    assert len(ds) == 4
    _, onehot = ds[1]
    assert onehot[2] == 1.0


def test_cifar_accepts_batches_dir_directly(cifar_dir):
    root, _ = cifar_dir
    ds = CIFAR10(root / "cifar-10-batches-py", train=False)
    assert len(ds) == 4


def test_cifar_missing_batch(tmp_path):
    root = tmp_path / "cifar-10-batches-py"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        CIFAR10(tmp_path, train=True)


def test_cifar_corrupt_batch_without_labels(tmp_path):
    root = tmp_path / "cifar-10-batches-py"
    root.mkdir()
    for b in range(1, 6):
        with open(root / f"data_batch_{b}", "wb") as f:
            pickle.dump({"data": np.zeros((1, 3072), dtype=np.uint8)}, f)
    with pytest.raises(RuntimeError):
        CIFAR10(tmp_path, train=True)


# ================================================================ TinyImageNet

@pytest.fixture
def tiny_dir(tmp_path):
    rng = np.random.default_rng(3)
    root = tmp_path / "tiny-imagenet-200"
    wnids = [f"n{i:08d}" for i in range(200)]
    (root / "train").mkdir(parents=True)
    (root / "val" / "images").mkdir(parents=True)
    (root / "wnids.txt").write_text("\n".join(wnids))
    # Images only for the first 3 classes (others have no image dirs)
    for wnid in wnids[:3]:
        d = root / "train" / wnid / "images"
        d.mkdir(parents=True)
        for j in range(2):
            arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{wnid}_{j}.JPEG")
    # Validation split: 4 images mapped via val_annotations.txt
    lines = []
    for j in range(4):
        name = f"val_{j}.JPEG"
        arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(arr).save(root / "val" / "images" / name)
        lines.append(f"{name}\t{wnids[j % 3]}\t0\t0\t10\t10")
    (root / "val" / "val_annotations.txt").write_text("\n".join(lines))
    return tmp_path, wnids


def test_tiny_train_loading(tiny_dir):
    root, wnids = tiny_dir
    ds = TinyImageNet(root, train=True)
    assert len(ds) == 6  # 3 classes x 2 images
    img, onehot = ds[0]
    assert img.mode == "RGB" and img.size == (64, 64)
    assert onehot.shape == (200,) and onehot.sum() == 1.0
    assert onehot[0] == 1.0  # first class


def test_tiny_val_loading(tiny_dir):
    root, _ = tiny_dir
    ds = TinyImageNet(root, train=False)
    assert len(ds) == 4
    for i in range(4):
        _, onehot = ds[i]
        assert onehot.sum() == 1.0


def test_tiny_missing_structure(tmp_path):
    with pytest.raises(FileNotFoundError):
        TinyImageNet(tmp_path)


def test_tiny_wrong_class_count(tmp_path):
    root = tmp_path / "tiny-imagenet-200"
    (root / "train").mkdir(parents=True)
    (root / "val").mkdir()
    (root / "wnids.txt").write_text("n1\nn2\n")  # only 2 classes
    with pytest.raises(RuntimeError):
        TinyImageNet(tmp_path)


def test_tiny_transform(tiny_dir):
    root, _ = tiny_dir
    ds = TinyImageNet(root, train=True, transform=lambda im: im.size)
    size, _ = ds[0]
    assert size == (64, 64)

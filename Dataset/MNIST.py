import os
import struct
import gzip
import shutil
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import urllib.request


class MNIST(Dataset):
    """
    Minimal PyTorch-style MNIST dataset (preloads images & labels).

    API mirrors TinyImageNet:
      - __init__(root, train=True, transform=None)
      - __getitem__ -> (PIL.Image, one_hot_numpy)
      - download(dest_dir) -> Path

    Notes:
      - Images are preloaded as a contiguous (N, 28, 28) uint8 array.
      - Labels are preloaded as uint8; one-hot vectors are indexed from a precomputed eye().
    """
    NUM_CLASSES = 10

    def __init__(self, root: str | os.PathLike, train: bool = True,
                 transform: Optional[Callable] = None):
        root = Path(root)
        # Allow nested "MNIST" directory (common layout)
        if not any((root / f).exists() for f in [
            "train-images-idx3-ubyte", "train-images-idx3-ubyte.gz",
            "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"
        ]) and (root / "MNIST").exists():
            root = root / "MNIST"

        self.root = root
        self.split = "train" if train else "val"
        self.transform = transform

        img_base, lbl_base = (
            ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
            if train else
            ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
        )
        img_path = self._ensure_uncompressed(img_base)
        lbl_path = self._ensure_uncompressed(lbl_base)

        # Load entire dataset into memory (efficient & simple for MNIST size)
        self.images = self._load_idx_images(img_path)      # (N, 28, 28) uint8 contiguous
        self.targets = self._load_idx_labels(lbl_path)     # (N,) uint8 contiguous

        if self.images.shape[0] != self.targets.shape[0]:
            raise RuntimeError("Images and labels count mismatch.")

        # Precompute one-hot lookup (float32)
        self._one_hot_matrix = np.eye(self.NUM_CLASSES, dtype=np.float32)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> Tuple[Image.Image, np.ndarray]:
        # Convert a single 28x28 uint8 array to PIL grayscale image
        arr = self.images[index]  # view; no extra copy
        img = Image.fromarray(arr)

        if self.transform is not None:
            img = self.transform(img)

        target_idx = int(self.targets[index])
        label = self._one_hot_matrix[target_idx]  # view into precomputed eye()
        return img, label

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root={self.root!s}, split={self.split}, size={len(self)})"

    # ------------------------- Utilities -------------------------

    def _ensure_uncompressed(self, base_name: str) -> Path:
        """Return path to uncompressed IDX file; if only .gz exists, decompress it."""
        raw = self.root / base_name
        gz = self.root / (base_name + ".gz")

        if raw.exists():
            return raw
        if gz.exists():
            raw.parent.mkdir(parents=True, exist_ok=True)
            print(f"Decompressing {gz.name} ...")
            with gzip.open(gz, "rb") as fin, open(raw, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            return raw
        raise FileNotFoundError(
            f"Missing MNIST file(s) in {self.root}. Expected '{base_name}' or '{base_name}.gz'."
        )

    @staticmethod
    def _read_all_bytes(path: Path) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    @classmethod
    def _load_idx_images(cls, path: Path) -> np.ndarray:
        data = cls._read_all_bytes(path)
        # IDX header: >IIII (magic, num, rows, cols) for images
        magic, num, rows, cols = struct.unpack_from(">IIII", data, 0)
        if magic != 0x00000803 or rows != 28 or cols != 28:
            raise RuntimeError(f"Invalid MNIST images file header at {path}")
        offset = 16
        expected = num * rows * cols
        # Copy to detach from the file buffer and ensure C-contiguous
        arr = np.frombuffer(data, dtype=np.uint8, count=expected, offset=offset).copy()
        return arr.reshape(num, rows, cols)

    @classmethod
    def _load_idx_labels(cls, path: Path) -> np.ndarray:
        data = cls._read_all_bytes(path)
        # IDX header: >II (magic, num) for labels
        magic, num = struct.unpack_from(">II", data, 0)
        if magic != 0x00000801:
            raise RuntimeError(f"Invalid MNIST labels file header at {path}")
        offset = 8
        return np.frombuffer(data, dtype=np.uint8, count=num, offset=offset).copy()

    @staticmethod
    def download(dest_dir: str | os.PathLike,
                 url: str = "https://storage.googleapis.com/cvdf-datasets/mnist/") -> Path:
        """
        Download and decompress MNIST IDX files into dest_dir/MNIST.

        Returns:
            Path to the dataset directory containing uncompressed IDX files.
        """
        dest = Path(dest_dir)
        out_dir = dest / "MNIST"
        out_dir.mkdir(parents=True, exist_ok=True)

        files = [
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz",
        ]

        for fname in files:
            url = url.rstrip("/") + "/" + fname
            gz_path = out_dir / fname
            raw_path = out_dir / fname.replace(".gz", "")
            if not gz_path.exists() and not raw_path.exists():
                print(f"Downloading {fname} ...")
                urllib.request.urlretrieve(url, gz_path)
            if gz_path.exists() and not raw_path.exists():
                print(f"Decompressing {fname} ...")
                with gzip.open(gz_path, "rb") as f_in, open(raw_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        return out_dir

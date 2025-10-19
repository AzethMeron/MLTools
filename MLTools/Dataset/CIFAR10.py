import os
import pickle
import tarfile
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import urllib.request


class CIFAR10(Dataset):
    """
    Minimal PyTorch-style CIFAR-10 dataset (preloads images & labels).

    API mirrors TinyImageNet:
      - __init__(root, train=True, transform=None)
      - __getitem__ -> (PIL.Image, one_hot_numpy)
      - download(dest_dir) -> Path

    Notes:
      - Loads the official 'cifar-10-python' batches into memory.
      - Images stored as uint8 array (N, 32, 32, 3) in HWC layout.
      - One-hot labels are views into a precomputed identity matrix.
    """
    NUM_CLASSES = 10

    def __init__(self, root: str | os.PathLike, train: bool = True,
                 transform: Optional[Callable] = None):
        root = Path(root)

        # Accept either the directory containing 'cifar-10-batches-py' or that folder itself
        if (root / "cifar-10-batches-py").exists():
            data_dir = root / "cifar-10-batches-py"
        elif root.name == "cifar-10-batches-py":
            data_dir = root
        else:
            # common nested folder name used by download()
            if (root / "CIFAR10" / "cifar-10-batches-py").exists():
                data_dir = root / "CIFAR10" / "cifar-10-batches-py"
            else:
                data_dir = root

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Could not find CIFAR-10 at {root}. "
                f"Expected 'cifar-10-batches-py' directory."
            )

        self.root = data_dir
        self.split = "train" if train else "val"
        self.transform = transform

        # ---- Load batches ----
        if train:
            batch_files = [f"data_batch_{i}" for i in range(1, 5 + 1)]
        else:
            batch_files = ["test_batch"]

        # Pre-allocate for efficiency
        num_per_train_batch = 10000
        total = num_per_train_batch * len(batch_files)
        images = np.empty((total, 32, 32, 3), dtype=np.uint8)
        targets = np.empty((total,), dtype=np.uint8)

        cursor = 0
        for fname in batch_files:
            batch_path = self.root / fname
            if not batch_path.exists():
                raise FileNotFoundError(f"Missing CIFAR-10 batch file: {batch_path}")
            data, labels = self._load_batch(batch_path)
            n = data.shape[0]
            images[cursor:cursor + n] = data
            targets[cursor:cursor + n] = labels
            cursor += n

        if cursor != total:
            images = images[:cursor]
            targets = targets[:cursor]

        self.images = images   # (N, 32, 32, 3) uint8
        self.targets = targets # (N,) uint8

        # Precompute one-hot lookup
        self._one_hot_matrix = np.eye(self.NUM_CLASSES, dtype=np.float32)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> Tuple[Image.Image, np.ndarray]:
        # HWC uint8 -> PIL RGB (Pillow infers mode from 3-channel uint8)
        img = Image.fromarray(self.images[index])

        if self.transform is not None:
            img = self.transform(img)

        label_idx = int(self.targets[index])
        label = self._one_hot_matrix[label_idx]
        return img, label

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root={self.root!s}, split={self.split}, size={len(self)})"

    # ------------------------- Utilities -------------------------

    @staticmethod
    def _load_batch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single CIFAR-10 batch file.
        Returns:
            images: (N, 32, 32, 3) uint8
            labels: (N,) uint8
        """
        with open(path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")  # works with py2 pickles
        data = entry["data"]  # shape (N, 3072), R(1024),G(1024),B(1024)
        labels: List[int] = entry.get("labels") or entry.get("fine_labels")
        if labels is None:
            raise RuntimeError(f"Labels not found in {path}")

        data = np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32)
        # CHW -> HWC
        data = np.transpose(data, (0, 2, 3, 1)).copy(order="C")
        labels = np.asarray(labels, dtype=np.uint8)
        return data, labels

    @staticmethod
    def download(dest_dir: str | os.PathLike,
                 url: str = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz") -> Path:
        """
        Download and extract CIFAR-10 (python version) into dest_dir/CIFAR10/cifar-10-batches-py.

        Returns:
            Path to the extracted dataset directory (…/CIFAR10/cifar-10-batches-py).
        """
        dest = Path(dest_dir)
        out_root = dest / "CIFAR10"
        out_root.mkdir(parents=True, exist_ok=True)

        tar_path = out_root / "cifar-10-python.tar.gz"
        target_dir = out_root / "cifar-10-batches-py"

        # Skip if extracted already
        if target_dir.exists() and (target_dir / "data_batch_1").exists():
            return target_dir

        # Download if missing
        if not tar_path.exists():
            print(f"Downloading CIFAR-10 to {tar_path} ...")
            urllib.request.urlretrieve(url, tar_path)
            print("Download complete.")

        # Extract (creates 'cifar-10-batches-py' inside out_root)
        print(f"Extracting {tar_path} ...")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(path=out_root)
        print(f"Extracted to {target_dir}")

        return target_dir

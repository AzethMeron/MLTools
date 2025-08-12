import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import zipfile
import urllib.request

class TinyImageNet(Dataset):
    """
    Minimal PyTorch-style TinyImageNet dataset.

    Args:
        root: Path to Tiny ImageNet root directory (the one that contains 'train', 'val', ...).
        train: True for train split, False for val split.
        transform: Optional transform applied to PIL images.
    Notes:
        - __getitem__ returns (pil_image, labels) where labels is a numpy one-hot vector (200,).
        - Annotations are fully preloaded (paths + integer labels). Images are not preloaded.
    """
    NUM_CLASSES = 200

    def __init__(self, root: str | os.PathLike, train: bool = True, transform: Optional[Callable] = None):
        # Normalize root to the directory that actually contains train/ and val/
        root = Path(root)
        if not (root / "train").exists() and (root / "tiny-imagenet-200").exists():
            root = root / "tiny-imagenet-200"

        if not (root / "train").exists() or not (root / "val").exists():
            raise FileNotFoundError(
                f"Could not find Tiny ImageNet structure under {root}. "
                f"Expected subfolders 'train' and 'val'."
            )

        self.root = root
        self.split = "train" if train else "val"
        self.transform = transform

        # --- Load class list and map WNIDs -> indices (stable class ordering) ---
        wnids_path = self.root / "wnids.txt"
        if not wnids_path.exists():
            raise FileNotFoundError(f"Missing wnids.txt at {wnids_path}")
        self.wnids: List[str] = [line.strip() for line in wnids_path.read_text().splitlines() if line.strip()]
        if len(self.wnids) != self.NUM_CLASSES:
            # Not fatal, but helps catch broken downloads
            raise RuntimeError(f"Expected {self.NUM_CLASSES} classes, found {len(self.wnids)}")
        self.class_to_idx: Dict[str, int] = {wnid: i for i, wnid in enumerate(self.wnids)}

        # --- Preload all annotations: file paths + integer labels ---
        self.samples: List[str] = []
        targets: List[int] = []

        if train:
            # train/<wnid>/images/*.JPEG
            train_dir = self.root / "train"
            for wnid in self.wnids:
                img_dir = train_dir / wnid / "images"
                # guard: some distributions may include non-JPEGs; filter tight
                for p in sorted(img_dir.glob("*.JPEG")):
                    self.samples.append(str(p))
                    targets.append(self.class_to_idx[wnid])
        else:
            # val/images/*.JPEG with mapping in val_annotations.txt
            val_dir = self.root / "val"
            anno_path = val_dir / "val_annotations.txt"
            if not anno_path.exists():
                raise FileNotFoundError(f"Missing val annotations at {anno_path}")

            img_to_wnid: Dict[str, str] = {}
            # Each line: <filename>\t<wnid>\t<x>\t<y>\t<w>\t<h>
            for line in anno_path.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                fname, wnid = parts[0], parts[1]
                img_to_wnid[fname] = wnid

            images_dir = val_dir / "images"
            for p in sorted(images_dir.glob("*.JPEG")):
                fname = p.name
                wnid = img_to_wnid.get(fname, None)
                if wnid is None:
                    # Skip if not annotated (should not happen)
                    continue
                self.samples.append(str(p))
                targets.append(self.class_to_idx[wnid])

        # Efficient numeric storage for targets
        self.targets = np.asarray(targets, dtype=np.int16)

        # Precompute an identity matrix once for fast one-hot lookup (float32)
        self._one_hot_matrix = np.eye(self.NUM_CLASSES, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, np.ndarray]:
        path = self.samples[index]
        target_idx: int = int(self.targets[index])

        # Open as RGB PIL image
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        # One-hot (view into the identity; zero-copy for speed/memory)
        label = self._one_hot_matrix[target_idx]
        return img, label

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(root={self.root!s}, split={self.split}, size={len(self)})"

    # ------------------------- Utilities -------------------------

    @staticmethod
    def download(dest_dir: str | os.PathLike, url: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip") -> Path:
        """
        Download and extract Tiny ImageNet (200 classes) into dest_dir.

        Returns:
            Path to the extracted dataset root directory (…/tiny-imagenet-200).
        """
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        zip_path = dest / "tiny-imagenet-200.zip"
        root_dir = dest / "tiny-imagenet-200"

        # Skip if extracted already
        if root_dir.exists() and (root_dir / "train").exists() and (root_dir / "val").exists():
            return root_dir

        # Download if missing
        if not zip_path.exists():
            print(f"Downloading Tiny ImageNet to {zip_path} ...")
            urllib.request.urlretrieve(url, zip_path)
            print("Download complete.")

        # Extract
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        print(f"Extracted to {root_dir}")

        return root_dir

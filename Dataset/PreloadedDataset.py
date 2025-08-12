from __future__ import annotations

import io
import pickle
import zlib
from typing import Optional, Callable, Tuple, Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

try:
    import torch
except ImportError:
    torch = None


class PreloadedDataset(Dataset):
    """
    A dataset that eagerly loads all (image, label) pairs from another PyTorch Dataset
    and stores images in one of several formats for speed/memory tradeoffs.

    Storage modes (via `compression`):
      - None:  keep PIL.Image objects directly (fastest access, largest memory).
      - 'zlib': pickle the PIL image and zlib-compress the pickle (compact, CPU to decode).
      - 'PNG':  store PNG-encoded bytes in memory (lossless, smaller than raw for many images).
      - 'JPG'/'JPEG': store JPEG-encoded bytes in memory (lossy, smallest; uses `jpg_quality`).

    Notes:
      - Labels are copied as-is (no conversion).
      - __getitem__ returns (PIL.Image, label_as_is). If you pass a `transform`, it is applied to the PIL image.
      - This class does not keep a reference to the input dataset.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        compression: Optional[str] = "PNG",
        transform: Optional[Callable[[Image.Image], Any]] = None,
        jpg_quality: int = 90,
        png_compression: int = 6,
    ):
        super().__init__()
        mode = None if compression is None else str(compression).upper()
        if mode in ("JPG", "JPEG"):
            mode = "JPEG"
        elif mode == "PNG":
            pass
        elif mode == "ZLIB":
            pass
        elif (mode is None) or (mode == "NONE"):
            pass
        else:
            raise ValueError("compression must be one of: None, 'zlib', 'PNG', 'JPG'/'JPEG'")

        if mode == "JPEG":
            # Clamp to a sensible JPEG quality range
            jpg_quality = int(max(1, min(95, jpg_quality)))

        self._mode = mode
        self._transform = transform
        self._jpg_quality = jpg_quality
        self._png_compression = png_compression

        self._images = []
        self._labels = []

        n = len(base_dataset)

        for i in range(n):
            img, label = base_dataset[i]
            pil_img = self._ensure_pil(img)

            if self._mode is None:
                # Store a copy so we don't keep underlying references
                self._images.append(pil_img.copy())
            elif self._mode == "ZLIB":
                payload = pickle.dumps(pil_img, protocol=pickle.HIGHEST_PROTOCOL)
                self._images.append(zlib.compress(payload))
            elif self._mode == "PNG":
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG", compress_level = self._png_compression)
                self._images.append(buf.getvalue())
            elif self._mode == "JPEG":
                buf = io.BytesIO()
                # JPEG requires 8-bit/channel; force RGB for consistency
                pil_img.convert("RGB").save(buf, format="JPEG", quality=self._jpg_quality)
                self._images.append(buf.getvalue())

            # Labels: copy as-is (shallow copy for common types)
            self._labels.append(label)

        # Do not keep any reference to the original dataset object
        del base_dataset

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> Tuple[Image.Image, Any]:
        stored = self._images[index]

        if self._mode is None:
            img = stored
        elif self._mode == "ZLIB":
            img = pickle.loads(zlib.decompress(stored))
        else:
            # PNG / JPEG bytes
            with Image.open(io.BytesIO(stored)) as im:
                # Load fully to detach from buffer context
                img = im.copy()

        if self._transform is not None:
            img = self._transform(img)

        return img, self._labels[index]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={len(self)}, compression={self._mode or 'None'})"

    # -------------------- Helpers --------------------

    @staticmethod
    def _ensure_pil(x) -> Image.Image:
        if isinstance(x, Image.Image):
            return x
        if torch is not None and hasattr(torch, "Tensor") and isinstance(x, torch.Tensor):
            # Handle CHW or HWC, normalize to uint8 if needed
            arr = x.detach().cpu()
            if arr.dtype in (getattr(torch, "float32"), getattr(torch, "float16"), getattr(torch, "float64")):
                arr = arr.clamp(0, 1).mul(255).round().to(torch.uint8)
            elif arr.dtype not in (getattr(torch, "uint8"),):
                arr = arr.to(torch.uint8)

            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):  # CHW
                arr = arr.permute(1, 2, 0).contiguous()
            np_arr = arr.numpy()
            return Image.fromarray(np_arr)
        if isinstance(x, np.ndarray):
            # Assume HWC or HW
            if x.dtype != np.uint8:
                x = np.clip(x, 0, 1) * 255 if x.dtype.kind == "f" else x
                x = x.astype(np.uint8)
            return Image.fromarray(x)
        # Fallback: try PIL conversion
        return Image.fromarray(np.array(x))


import torch
import random
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

class RandomScale:
    def __init__(
        self,
        min_scale=0.25,
        max_scale=0.75,
        downscale_algorithm="bilinear",
        upscale_algorithm="bilinear",
        p=1.0,
    ):
        if min_scale <= 0:
            raise ValueError(f"min_scale must be > 0, got {min_scale}")
        if max_scale < min_scale:
            raise ValueError(f"max_scale ({max_scale}) must be >= min_scale ({min_scale})")
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p must be in [0, 1], got {p}")
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

        self.down_interp = self._to_interp(downscale_algorithm)
        self.up_interp = self._to_interp(upscale_algorithm)

    def _to_interp(self, name):
        modes = {
            "nearest": InterpolationMode.NEAREST,
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
            "area": InterpolationMode.BOX,
        }
        if name not in modes:
            raise ValueError(f"Unknown interpolation '{name}'. Available: {sorted(modes.keys())}")
        return modes[name]

    @staticmethod
    def _resize(img: torch.Tensor, size, interp):
        # torchvision's tensor resize cannot do BOX ("area"); route that case
        # through torch.nn.functional.interpolate(mode="area") instead.
        if interp == InterpolationMode.BOX:
            import torch.nn.functional as TF
            batched = img.ndim == 3
            x = img.unsqueeze(0) if batched else img
            x = TF.interpolate(x.float(), size=size, mode="area").to(img.dtype)
            return x.squeeze(0) if batched else x
        return F.resize(img, size, interpolation=interp)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.p < 1.0 and random.random() > self.p: return img

        _, h, w = img.shape

        scale = random.uniform(self.min_scale, self.max_scale)

        scaled_h = max(1, int(round(h * scale)))
        scaled_w = max(1, int(round(w * scale)))

        interp1 = self.down_interp if scale < 1.0 else self.up_interp
        interp2 = self.down_interp if scale > 1.0 else self.up_interp

        # scale down / up
        img = self._resize(img, (scaled_h, scaled_w), interp1)
        # scale back to original size
        img = self._resize(img, (h, w), interp2)

        return img

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"min_scale={self.min_scale}, "
            f"max_scale={self.max_scale}, "
            f"p={self.p})"
        )

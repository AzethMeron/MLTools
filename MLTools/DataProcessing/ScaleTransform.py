
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
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

        self.down_interp = self._to_interp(downscale_algorithm)
        self.up_interp = self._to_interp(upscale_algorithm)

    def _to_interp(self, name):
        return {
            "nearest": InterpolationMode.NEAREST,
            "bilinear": InterpolationMode.BILINEAR,
            "bicubic": InterpolationMode.BICUBIC,
            "area": InterpolationMode.BOX,
        }[name]

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.p < 1.0 and random.random() > self.p: return img

        _, h, w = img.shape

        scale = random.uniform(self.min_scale, self.max_scale)

        scaled_h = max(1, int(round(h * scale)))
        scaled_w = max(1, int(round(w * scale)))

        interp1 = self.down_interp if scale < 1.0 else self.up_interp
        interp2 = self.down_interp if scale > 1.0 else self.up_interp

        # scale down / up
        img = F.resize(img, (scaled_h, scaled_w), interpolation=interp1)
        # scale back to original size
        img = F.resize(img, (h, w), interpolation=interp2)

        return img

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"min_scale={self.min_scale}, "
            f"max_scale={self.max_scale}, "
            f"p={self.p})"
        )

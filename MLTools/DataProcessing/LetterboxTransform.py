from typing import List, Tuple
from PIL import Image

import torch
import torch.nn.functional as F

# ----------------------------------------------------
# Torch-backed Letterbox transform (PIL -> PIL)
# ----------------------------------------------------
class Letterbox:
    """
    Letterbox resize using Torch ops (resize + pad), returning PIL.Image.

    Args:
        image_size: (W, H) target canvas size
        fill: RGB padding color (default 114 like YOLO)
        device: torch device for internal ops
    """
    def __init__(self,
                 image_size: Tuple[int, int],
                 fill: Tuple[int, int, int] = (114, 114, 114),
                 device: str = "cpu"):
        self.tw, self.th = int(image_size[0]), int(image_size[1])
        self.fill = fill
        self.device = torch.device(device)

    @torch.no_grad()
    def __call__(self, img_pil: Image.Image) -> Image.Image:
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")

        ow, oh = img_pil.size  # (W, H)

        # Compute resize ratio
        r = min(self.th / oh, self.tw / ow)
        nw, nh = int(round(ow * r)), int(round(oh * r))

        # Convert PIL -> torch tensor [1,3,H,W]
        img = torch.frombuffer(bytearray(img_pil.tobytes()), dtype=torch.uint8)
        img = img.view(oh, ow, 3).permute(2, 0, 1).unsqueeze(0).to(self.device)  # [1,3,H,W]
        img = img.float()  # interpolate needs float

        # Resize with bilinear
        if (oh, ow) != (nh, nw):
            img = F.interpolate(img, size=(nh, nw), mode="bilinear", align_corners=False)

        # Compute symmetric padding
        pad_w = self.tw - nw
        pad_h = self.th - nh
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        # Pad with constant fill
        img = F.pad(img, (left, right, top, bottom), value=float(self.fill[0]))
        if len(self.fill) == 3:
            # Different R,G,B fill
            for c in range(3):
                if self.fill[c] != self.fill[0]:
                    img[:, c:c+1] = F.pad(
                        img[:, c:c+1],
                        (left, right, top, bottom),
                        value=float(self.fill[c])
                    )

        # Back to uint8 PIL.Image
        img = img.clamp(0, 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(img)

    def __repr__(self):
        return f"{self.__class__.__name__}(size=({self.tw},{self.th}), fill={self.fill})"

    def transform_image(self, pil_img):
      return self(pil_img)

    def transform_detections(self,
                            detections,
                            orig_size: Tuple[int, int]):
        """
        Adjust detections (CXCYWH) for letterboxed resize.
        - orig_size: (W,H) before resize
        - new_size: (W,H) canvas after letterbox
        """
        from MLTools.Detections import Detection
        ow, oh = orig_size
        tw, th = self.tw, self.th

        r = min(th / oh, tw / ow)
        nw, nh = int(round(ow * r)), int(round(oh * r))

        pad_w = tw - nw
        pad_h = th - nh
        left = pad_w // 2
        top = pad_h // 2

        out = []
        for d in detections:
            out.append(Detection(
                cx=d.cx * r + left,
                cy=d.cy * r + top,
                w=d.w * r,
                h=d.h * r,
                rotation=d.rotation,
                class_id=d.class_id,
                confidence=d.confidence
            ))
        return out
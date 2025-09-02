
#!/usr/bin/env python3
"""
FlyingStuff: Modular Flying-Chairs–style dataset generator (PyTorch)
===================================================================

Generate FlowNet-style training pairs (img1, img2, dense flow, valid mask,
metadata) while being **dataset-agnostic**. Plug in any number of foreground
and background providers via tiny interfaces:

- `ForegroundDataset` → yields `(RGB image, binary alpha mask)` cutouts
- `BackgroundDataset` → yields background `RGB image`

Included adapters:
- Foregrounds: `pets` (Oxford-IIIT Pet trimaps), `voc_person` (Pascal VOC),
  `coco_person` (COCO persons, needs `pycocotools`), `clouds` (synthetic).
- Backgrounds: `dtd` (Describable Textures Dataset).

You can **mix multiple foreground/background datasets at once** with optional
weights. The renderer composes N moving foreground layers on 1 background and
computes dense optical flow with occlusions.

Output structure
----------------
<out_root>/
  train/ | val/
    000001_img1.png
    000001_img2.png
    000001_flow.flo
    000001_valid.png
    000001_meta.json
  train.txt
  val.txt
  _sources/   # cached raw datasets

Examples
--------
# Pets + VOC people over DTD backgrounds
python flying_stuff.py \
  --out-root ./FlyingStuff \
  --fg pets,voc_person \
  --bg dtd \
  --num-train 20000 --num-val 2000 \
  --height 384 --width 512 --max-objects 3

# Add COCO persons (requires `pip install pycocotools`) and synthetic clouds
python flying_stuff.py \
  --out-root ./FlyingStuffPlus \
  --fg pets,voc_person,coco_person,clouds \
  --bg dtd \
  --num-train 30000 --num-val 3000 --max-objects 4

"""
from __future__ import annotations
import json
import math
import os
import random
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T
import tqdm

# ================================================================
# Core data contracts
# ================================================================

@dataclass
class ForegroundSample:
    """A single foreground cutout.
    - image: RGB PIL.Image
    - mask:  PIL.Image in mode 'L' (0 background, 255 foreground)
    """
    image: Image.Image
    mask: Image.Image


class ForegroundDataset(ABC):
    """Interface any foreground provider must implement."""

    @abstractmethod
    def get_random(self) -> ForegroundSample:
        """Return a random foreground sample (image+binary mask)."""
        raise NotImplementedError

    def __len__(self) -> int:  # optional hint for logging
        return 0


class BackgroundDataset(ABC):
    """Interface any background provider must implement."""

    @abstractmethod
    def get_random(self) -> Image.Image:
        """Return a random RGB background image (PIL.Image)."""
        raise NotImplementedError

    def __len__(self) -> int:
        return 0


# ================================================================
# Registry helpers (name → constructor)
# ================================================================

FG_REGISTRY: Dict[str, callable] = {}
BG_REGISTRY: Dict[str, callable] = {}


def register_fg(name: str):
    def deco(ctor):
        FG_REGISTRY[name] = ctor
        return ctor
    return deco


def register_bg(name: str):
    def deco(ctor):
        BG_REGISTRY[name] = ctor
        return ctor
    return deco


# ================================================================
# Built-in dataset adapters
# ================================================================

@register_fg('pets')
class PetsForeground(ForegroundDataset):
    """Oxford-IIIT Pets with trimaps → binary pet mask."""
    def __init__(self, root: str, split: str = 'trainval'):
        self.ds = torchvision.datasets.OxfordIIITPet(
            root=root, split=split, target_types=("segmentation",), download=True
        )

    def __len__(self):
        return len(self.ds)

    def get_random(self) -> ForegroundSample:
        idx = random.randrange(len(self.ds))
        img, trimap = self.ds[idx]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        m = np.array(trimap, dtype=np.uint8)
        # Oxford trimap: 1=pet, 2=background, 3=outline (treat outline as pet)
        binmask = np.where(m == 2, 0, 255).astype(np.uint8)
        return ForegroundSample(img, Image.fromarray(binmask, mode='L'))


@register_fg('voc_person')
class VOCPersonForeground(ForegroundDataset):
    """Pascal VOC 2012 segmentation — class 15 is 'person'."""
    def __init__(self, root: str, image_set: str = 'train'):
        self.ds = torchvision.datasets.VOCSegmentation(root=root, year='2012', image_set=image_set, download=True)
        self.person_id = 15

    def __len__(self):
        return len(self.ds)

    def get_random(self) -> ForegroundSample:
        for _ in range(25):
            idx = random.randrange(len(self.ds))
            img, target = self.ds[idx]
            m = (np.array(target, dtype=np.uint8) == self.person_id).astype(np.uint8) * 255
            if m.sum() > 0:
                return ForegroundSample(img.convert('RGB'), Image.fromarray(m, 'L'))
        # Fallback (rare)
        img, target = self.ds[idx]
        m = (np.array(target, dtype=np.uint8) == self.person_id).astype(np.uint8) * 255
        return ForegroundSample(img.convert('RGB'), Image.fromarray(m, 'L'))


@register_fg('coco_person')
class COCOPersonForeground(ForegroundDataset):
    """COCO persons; requires `pycocotools`. Large & varied."""
    def __init__(self, root: str, split: str = 'train'):
        try:
            from pycocotools.coco import COCO  # noqa: F401
        except Exception as e:
            raise RuntimeError("coco_person needs pycocotools. Install with `pip install pycocotools`.\n" + str(e))
        year = '2017'
        if split not in ('train', 'val'):
            split = 'train'
        img_dir = os.path.join(root, 'coco', f'{split}{year}')
        ann_file = os.path.join(root, 'coco', 'annotations', f'instances_{split}{year}.json')
        self.ds = torchvision.datasets.CocoDetection(img_dir, ann_file, download=True)
        # cache person ids
        cats = self.ds.coco.loadCats(self.ds.coco.getCatIds())
        self.person_ids = [c['id'] for c in cats if c['name'] == 'person'] or [1]

    def __len__(self):
        return len(self.ds)

    def get_random(self) -> ForegroundSample:
        from pycocotools import mask as maskUtils
        for _ in range(50):
            idx = random.randrange(len(self.ds))
            img, anns = self.ds[idx]
            w, h = img.size
            rles = []
            for a in anns:
                if a.get('category_id') not in self.person_ids:
                    continue
                seg = a.get('segmentation')
                if seg is None:
                    continue
                if isinstance(seg, list):
                    rles.extend(maskUtils.frPyObjects(seg, h, w))
                elif isinstance(seg, dict):
                    rles.append(seg)
            if not rles:
                continue
            m = maskUtils.merge(rles)
            m = maskUtils.decode(m)
            if m.ndim == 3:
                m = np.any(m, axis=2)
            binmask = (m.astype(np.uint8) * 255)
            if binmask.sum() == 0:
                continue
            return ForegroundSample(img.convert('RGB'), Image.fromarray(binmask, 'L'))
        # fallback empty
        img, _ = self.ds[idx]
        return ForegroundSample(img.convert('RGB'), Image.fromarray(np.zeros((img.height, img.width), np.uint8), 'L'))


@register_fg('clouds')
class SyntheticCloudsForeground(ForegroundDataset):
    """Simple synthetic cloud billows (white blobs with soft alpha)."""
    def __init__(self, root: str, size_range: Tuple[int, int] = (160, 480)):
        self.size_range = size_range

    def _make_blob(self, W: int, H: int) -> ForegroundSample:
        # random white texture
        img = Image.new('RGB', (W, H), (255, 255, 255))
        # random noise mask → blur → threshold
        noise = (np.random.rand(H, W) * 255).astype(np.uint8)
        mask = Image.fromarray(noise, 'L').filter(ImageFilter.GaussianBlur(radius=random.uniform(6, 18)))
        mask_np = np.array(mask)
        mask_np = (mask_np > 96).astype(np.uint8) * 255
        return ForegroundSample(img, Image.fromarray(mask_np, 'L'))

    def get_random(self) -> ForegroundSample:
        long = random.randint(*self.size_range)
        if random.random() < 0.5:
            W = long; H = int(long * random.uniform(0.6, 1.2))
        else:
            H = long; W = int(long * random.uniform(0.6, 1.2))
        return self._make_blob(W, H)


@register_bg('dtd')
class DTDBackground(BackgroundDataset):
    """Describable Textures as backgrounds."""
    def __init__(self, root: str, split: str = 'train'):
        self.ds = torchvision.datasets.DTD(root=root, split=split, download=True)

    def __len__(self):
        return len(self.ds)

    def get_random(self) -> Image.Image:
        img, _ = self.ds[random.randrange(len(self.ds))]
        return img.convert('RGB')


# ================================================================
# Rendering utilities
# ================================================================

def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_flo(flow: np.ndarray, path: str):
    assert flow.ndim == 3 and flow.shape[2] == 2
    h, w, _ = flow.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('f', 202021.25))  # magic
        f.write(struct.pack('i', w)); f.write(struct.pack('i', h))
        flow.astype(np.float32).tofile(f)


def save_png(t: torch.Tensor, path: str):
    t = t.clamp(0, 1)
    if t.dim() == 3 and t.size(0) == 3:
        img = T.ToPILImage()(t.cpu())
    elif t.dim() == 2:
        img = T.ToPILImage()(t.cpu())
    else:
        raise ValueError("Expected CHW (3,H,W) or (H,W)")
    img.save(path, format='PNG')


def pil_to_tensor_rgb(img: Image.Image) -> torch.Tensor:
    return T.ToTensor()(img)


def pil_to_tensor_mask(mask: Image.Image) -> torch.Tensor:
    m = torch.from_numpy(np.array(mask, dtype=np.uint8))
    m = (m > 127).to(torch.float32)
    return m.unsqueeze(0)


def resize_keep_aspect(img: Image.Image, target_long: int) -> Image.Image:
    w, h = img.size
    if w >= h:
        new_w = target_long
        new_h = int(round(h * target_long / max(1, w)))
    else:
        new_h = target_long
        new_w = int(round(w * target_long / max(1, h)))
    return img.resize((new_w, new_h), resample=Image.BICUBIC)


def affine_matrix_about(tx: float, ty: float, scale: Tuple[float, float], rot_deg: float, shear_deg: Tuple[float, float], pivot: Tuple[float, float]) -> torch.Tensor:
    """Create a 3x3 affine matrix (local→world) that applies scale/shear/rotation
    about a given pivot (px, py) in **local pixel coordinates**, then translates
    by (tx, ty) in **world pixels**.
    """
    sx, sy = scale
    rot = math.radians(rot_deg)
    shx, shy = map(math.radians, shear_deg)
    px, py = pivot
    S = torch.tensor([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=torch.float32)
    Sh = torch.tensor([[1, math.tan(shx), 0], [math.tan(shy), 1, 0], [0, 0, 1]], dtype=torch.float32)
    R = torch.tensor([[math.cos(rot), -math.sin(rot), 0], [math.sin(rot), math.cos(rot), 0], [0, 0, 1]], dtype=torch.float32)
    T_p = torch.tensor([[1, 0, -px], [0, 1, -py], [0, 0, 1]], dtype=torch.float32)
    T_p_inv = torch.tensor([[1, 0, px], [0, 1, py], [0, 0, 1]], dtype=torch.float32)
    Tm = torch.tensor([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=torch.float32)
    # Translate to pivot, apply S/Shear/R, translate back, then global T
    return Tm @ (T_p_inv @ (R @ (Sh @ (S @ T_p))))


def to_affine_grid_theta(A_world_from_local: torch.Tensor, src_wh: Tuple[int, int], dst_hw: Tuple[int, int]) -> torch.Tensor:
    """Return theta for grid_sample so that sampling reads from *source* given a forward affine A.

    Accepts A as (2x3) or (3x3). Builds the 3x3 homogeneous transform, inverts it,
    and composes with pixel<->normalized coordinate transforms:
        theta = N_src^{-1} * A^{-1} * N_dst  (take top 2x3)
    """
    src_w, src_h = src_wh
    dst_h, dst_w = dst_hw
    device = A_world_from_local.device
    dtype = A_world_from_local.dtype

    def N_pix_to_norm(w, h):
        return torch.tensor([[2.0/(w-1), 0, -1], [0, 2.0/(h-1), -1], [0, 0, 1]], dtype=dtype, device=device)

    def N_norm_to_pix(w, h):
        return torch.tensor([[(w-1)/2.0, 0, (w-1)/2.0], [0, (h-1)/2.0, (h-1)/2.0], [0, 0, 1]], dtype=dtype, device=device)

    A = A_world_from_local
    if A.shape == (2, 3):
        A3 = torch.eye(3, dtype=dtype, device=device)
        A3[:2, :3] = A
    elif A.shape == (3, 3):
        A3 = A
    else:
        raise ValueError(f"A must be (2,3) or (3,3), got {tuple(A.shape)}")

    Ainv = torch.linalg.inv(A3)
    Theta = N_pix_to_norm(src_w, src_h) @ Ainv @ N_norm_to_pix(dst_w, dst_h)
    return Theta[:2, :3]


def apply_affine_to_points(A: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    if A.shape == (2,3):
        A = torch.cat([A, torch.tensor([[0,0,1]], dtype=A.dtype, device=A.device)], dim=0)
    ones = torch.ones((xy.shape[0], 1), dtype=xy.dtype, device=xy.device)
    out = (A @ torch.cat([xy, ones], dim=1).T).T
    return out[:, :2]


# ================================================================
# Layers & rendering
# ================================================================

@dataclass
class Layer:
    tex: torch.Tensor   # (3,H,W)
    alpha: torch.Tensor # (1,H,W)
    A1: torch.Tensor    # (3,3)
    A2: torch.Tensor    # (3,3)
    z: float
    pad_mode: str = 'zeros'  # 'zeros' | 'border' | 'reflection'


def random_object_layer(fg: ForegroundSample, canvas_hw: Tuple[int,int]) -> Layer:
    Hc, Wc = canvas_hw
    img, msk = fg.image, fg.mask
    long_side = max(Wc, Hc)
    scale_long = random.uniform(0.45, 0.9) * long_side
    img = resize_keep_aspect(img, int(scale_long))
    msk = resize_keep_aspect(msk, int(scale_long))

    tex = pil_to_tensor_rgb(img)
    alpha = pil_to_tensor_mask(msk)

    Hl, Wl = tex.shape[1:]
    px, py = ( (Wl - 1) / 2.0, (Hl - 1) / 2.0 )

    def sample_affine():
        tx = random.uniform(-0.25, 0.25) * Wc
        ty = random.uniform(-0.30, 0.30) * Hc
        scl = (random.uniform(0.85, 1.15), random.uniform(0.85, 1.15))
        rot = random.uniform(-35, 35)
        shear = (random.uniform(-12, 12), random.uniform(-12, 12))
        return affine_matrix_about(tx, ty, scl, rot, shear, pivot=(px, py))

    A1 = sample_affine()
    A2 = sample_affine()
    z = random.random()
    return Layer(tex=tex, alpha=alpha, A1=A1, A2=A2, z=z, pad_mode='zeros')

    A1 = sample_affine()
    A2 = sample_affine()
    z = random.random()
    return Layer(tex=tex, alpha=alpha, A1=A1, A2=A2, z=z, pad_mode='zeros')


def background_layer(bg: Image.Image, canvas_hw: Tuple[int,int]) -> Layer:
    Hc, Wc = canvas_hw
    # Identity at frame1 to guarantee full coverage; gentle motion at frame2.
    bg = bg.resize((Wc, Hc), resample=Image.BICUBIC)
    tex = pil_to_tensor_rgb(bg)
    alpha = torch.ones(1, Hc, Wc)

    cx, cy = ((Wc - 1) / 2.0, (Hc - 1) / 2.0)

    def sample_small_motion():
        tx = random.uniform(-0.03, 0.03) * Wc
        ty = random.uniform(-0.03, 0.03) * Hc
        scl = (random.uniform(0.985, 1.015), random.uniform(0.985, 1.015))
        rot = random.uniform(-3, 3)
        shear = (random.uniform(-2, 2), random.uniform(-2, 2))
        return affine_matrix_about(tx, ty, scl, rot, shear, pivot=(cx, cy))

    A1 = torch.eye(3, dtype=torch.float32)
    A2 = sample_small_motion()
    return Layer(tex=tex, alpha=alpha, A1=A1, A2=A2, z=-1.0, pad_mode='border')

    A1 = torch.eye(3, dtype=torch.float32)
    A2 = sample_small_motion()
    return Layer(tex=tex, alpha=alpha, A1=A1, A2=A2, z=-1.0, pad_mode='border')


import torch
import torch.nn.functional as F

@torch.no_grad()
def render_pair_and_flow(
    layers: List[Layer],
    canvas_hw: Tuple[int,int],
    device: torch.device,
    *,
    min_motion_px: float | None = None,
    max_motion_px: float | None = None,
    motion_percentile: float = 80.0,
    apply_to_background: bool = False,
):
    """
    Render (img1, img2) and forward flow with optional motion bounds.
    Motion bounds are enforced PER LAYER by rescaling the relative affine A_delta = A2 @ inv(A1)
    so that the chosen percentile of |A2·uv - A1·uv| on visible pixels falls within [min,max].

    Args:
        min_motion_px: if set, ensure motion >= this (approx.) for each object
        max_motion_px: if set, ensure motion <= this (approx.) for each object
        motion_percentile: which percentile of per-pixel motion to enforce (e.g., 80)
        apply_to_background: also enforce bounds for the background layer
    """
    Hc, Wc = canvas_hw
    layers = sorted(layers, key=lambda L: L.z)

    I1 = torch.zeros(3, Hc, Wc, device=device)
    I2 = torch.zeros(3, Hc, Wc, device=device)
    vis1 = -torch.ones(Hc, Wc, dtype=torch.long, device=device)
    vis2 = -torch.ones(Hc, Wc, dtype=torch.long, device=device)
    local_uv1 = torch.full((2, Hc, Wc), float('nan'), device=device)

    yy, xx = torch.meshgrid(torch.arange(Hc, device=device),
                            torch.arange(Wc, device=device), indexing='ij')
    canvas_xy = torch.stack([xx, yy], dim=0).float()

    def _enforce_bounds_on_layer_A2(A1_3x3: torch.Tensor,
                                    A2_3x3: torch.Tensor,
                                    uv_vis: torch.Tensor,  # (N,2) in layer-local pixels
                                   ):
        """Rescale relative affine so chosen motion percentile lies in [min,max]."""
        needs_bounds = (min_motion_px is not None) or (max_motion_px is not None)
        if (not needs_bounds) or (uv_vis.numel() == 0):
            return A2_3x3

        # Compute current per-pixel displacement magnitudes
        xy1 = apply_affine_to_points(A1_3x3[:2, :], uv_vis)    # (N,2)
        xy2 = apply_affine_to_points(A2_3x3[:2, :], uv_vis)    # (N,2)
        disp = xy2 - xy1                                       # (N,2)
        mag = torch.linalg.norm(disp, dim=1)                   # (N,)

        # Percentile statistic
        p = float(motion_percentile) / 100.0
        stat = torch.quantile(mag, torch.tensor(p, device=mag.device)) if mag.numel() > 0 else torch.tensor(0.0, device=mag.device)
        stat = float(stat.item())
        if stat <= 0.0:
            # If absolutely no motion but min>0 requested, inject a tiny translation
            if (min_motion_px is not None) and (min_motion_px > 0):
                A_delta = A2_3x3 @ torch.linalg.inv(A1_3x3)
                # add a small translation in a random direction
                angle = torch.rand(1).item() * 2*math.pi
                tx = min_motion_px * math.cos(angle)
                ty = min_motion_px * math.sin(angle)
                A_delta_adj = A_delta.clone()
                A_delta_adj[0,2] += tx
                A_delta_adj[1,2] += ty
                return A_delta_adj @ A1_3x3
            return A2_3x3

        # Compute scale factor alpha to bring percentile into [min,max]
        alpha = 1.0
        if (min_motion_px is not None) and (stat < min_motion_px):
            alpha = max(alpha, (min_motion_px / max(stat, 1e-6)))
        if (max_motion_px is not None) and (stat > max_motion_px):
            alpha = min(alpha, (max_motion_px / max(stat, 1e-6)))
        if abs(alpha - 1.0) < 1e-6:
            return A2_3x3

        # Rescale relative affine: A_delta = A2 @ inv(A1)
        A_delta = A2_3x3 @ torch.linalg.inv(A1_3x3)
        M = A_delta[:2, :2]
        t = A_delta[:2, 2]

        I2 = torch.eye(2, dtype=M.dtype, device=M.device)
        M_adj = I2 + alpha * (M - I2)   # linear blend of linear part
        t_adj = alpha * t               # scale translation

        A_delta_adj = torch.eye(3, dtype=M.dtype, device=M.device)
        A_delta_adj[:2, :2] = M_adj
        A_delta_adj[:2, 2] = t_adj

        return A_delta_adj @ A1_3x3

    for lid, L in enumerate(layers):
        tex = L.tex.to(device)
        alpha = L.alpha.to(device)
        Hl, Wl = tex.shape[1:]

        theta1 = to_affine_grid_theta(L.A1, (Wl, Hl), (Hc, Wc)).to(device)
        grid1 = F.affine_grid(theta1.unsqueeze(0), size=(1, 3, Hc, Wc), align_corners=True).to(device)
        samp_tex1 = F.grid_sample(tex.unsqueeze(0), grid1, mode='bilinear',
                                  padding_mode=L.pad_mode, align_corners=True)[0]
        samp_a1 = F.grid_sample(alpha.unsqueeze(0), grid1, mode='bilinear',
                                padding_mode=L.pad_mode, align_corners=True)[0]
        m1 = (samp_a1[0] > 0.5)
        I1[:, m1] = samp_tex1[:, m1]
        vis1[m1] = lid

        # Layer-local coordinates at frame1
        u = torch.linspace(0, Wl - 1, Wl, device=device).view(1, 1, Wl).expand(1, Hl, Wl)
        v = torch.linspace(0, Hl - 1, Hl, device=device).view(1, Hl, 1).expand(1, Hl, Wl)
        uv = torch.cat([u, v], dim=0)
        samp_uv1 = F.grid_sample(uv.unsqueeze(0), grid1, mode='bilinear',
                                 padding_mode=L.pad_mode, align_corners=True)[0]
        # store local coords for visible pixels (topmost overwrite already in effect)
        for c in range(2):
            tmp = local_uv1[c]; tmp[m1] = samp_uv1[c][m1]; local_uv1[c] = tmp

        # --- Enforce motion bounds on this layer (skip background unless requested) ---
        is_bg = (L.z <= -0.5)  # background we set to z=-1.0 earlier
        if (apply_to_background or not is_bg) and ((min_motion_px is not None) or (max_motion_px is not None)):
            # visible uv points only for this layer
            uv_vis = samp_uv1[:, m1].T  # (N,2) in layer-local pixels
            # Make sure A1/A2 are 3x3 on device
            A1_3 = L.A1.to(device)
            if A1_3.shape == (2,3):
                A1t = torch.eye(3, dtype=A1_3.dtype, device=device); A1t[:2,:3] = A1_3; A1_3 = A1t
            A2_3 = L.A2.to(device)
            if A2_3.shape == (2,3):
                A2t = torch.eye(3, dtype=A2_3.dtype, device=device); A2t[:2,:3] = A2_3; A2_3 = A2t

            A2_3_new = _enforce_bounds_on_layer_A2(A1_3, A2_3, uv_vis)
            # write back (keep 3x3)
            L.A2 = A2_3_new.detach().cpu()

        # Now sample frame2 with (possibly) adjusted A2
        theta2 = to_affine_grid_theta(L.A2, (Wl, Hl), (Hc, Wc)).to(device)
        grid2 = F.affine_grid(theta2.unsqueeze(0), size=(1, 3, Hc, Wc), align_corners=True).to(device)
        samp_tex2 = F.grid_sample(tex.unsqueeze(0), grid2, mode='bilinear',
                                  padding_mode=L.pad_mode, align_corners=True)[0]
        samp_a2 = F.grid_sample(alpha.unsqueeze(0), grid2, mode='bilinear',
                                padding_mode=L.pad_mode, align_corners=True)[0]
        m2 = (samp_a2[0] > 0.5)
        I2[:, m2] = samp_tex2[:, m2]
        vis2[m2] = lid

    # ---- Compute forward flow + valid (same as before, with adjusted A2) ----
    flow = torch.zeros(2, Hc, Wc, device=device)
    valid = torch.zeros(Hc, Wc, dtype=torch.bool, device=device)

    for lid, L in enumerate(layers):
        mask_l = (vis1 == lid)
        if not mask_l.any():
            continue
        uv = local_uv1[:, mask_l].T
        xy2 = apply_affine_to_points(L.A2.to(device)[:2, :], uv)
        xy1_calc = apply_affine_to_points(L.A1.to(device)[:2, :], uv)
        disp = (xy2 - xy1_calc).T
        flow[:, mask_l] = disp

        x2 = xy2[:, 0]; y2 = xy2[:, 1]
        inside = (x2 >= 0) & (x2 <= (Wc - 1)) & (y2 >= 0) & (y2 <= (Hc - 1))
        xi = x2.round().clamp(0, Wc - 1).long()
        yi = y2.round().clamp(0, Hc - 1).long()
        same_layer = (vis2[yi, xi] == lid)
        valid[mask_l] = (inside & same_layer)

    return I1.cpu(), I2.cpu(), flow.cpu(), valid.cpu()



# ================================================================
# Sampling pools and synthesis driver
# ================================================================

@dataclass
class PoolSpec:
    name: str
    weight: float


def parse_pool_arg(arg: str) -> List[PoolSpec]:
    specs: List[PoolSpec] = []
    if not arg:
        return specs
    for token in arg.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' in token:
            n, w = token.split(':', 1)
            specs.append(PoolSpec(n.strip(), float(w)))
        else:
            specs.append(PoolSpec(token, 1.0))
    return specs


class WeightedPool:
    def __init__(self, items: Sequence, weights: Sequence[float]):
        assert len(items) == len(weights) and len(items) > 0
        self.items = list(items)
        self.weights = list(weights)
        self.total = float(sum(weights))
        self.cum = []
        s = 0.0
        for w in weights:
            s += float(w)
            self.cum.append(s)

    def sample(self):
        r = random.random() * self.total
        for item, c in zip(self.items, self.cum):
            if r <= c:
                return item
        return self.items[-1]


def build_fg_pool(specs: List[PoolSpec], root: str) -> WeightedPool:
    if not specs:
        raise ValueError("No foreground datasets specified. Provide --fg ...")
    providers = []
    weights = []
    for s in specs:
        if s.name not in FG_REGISTRY:
            raise KeyError(f"Unknown foreground '{s.name}'. Available: {sorted(FG_REGISTRY.keys())}")
        ctor = FG_REGISTRY[s.name]
        if s.name == 'coco_person':
            ds = ctor(root=root, split='train')
        elif s.name == 'voc_person':
            ds = ctor(root=root, image_set='train')
        elif s.name == 'pets':
            ds = ctor(root=root, split='trainval')
        else:
            ds = ctor(root=root)
        providers.append(ds)
        weights.append(s.weight)
    return WeightedPool(providers, weights)


def build_bg_pool(specs: List[PoolSpec], root: str) -> WeightedPool:
    if not specs:
        raise ValueError("No background datasets specified. Provide --bg ...")
    providers = []
    weights = []
    for s in specs:
        if s.name not in BG_REGISTRY:
            raise KeyError(f"Unknown background '{s.name}'. Available: {sorted(BG_REGISTRY.keys())}")
        ctor = BG_REGISTRY[s.name]
        ds = ctor(root=root)
        providers.append(ds)
        weights.append(s.weight)
    return WeightedPool(providers, weights)


def synthesize_dataset(
    out_root: str,
    num_train: int,
    num_val: int,
    H: int,
    W: int,
    max_objects: int,
    bg_pool: WeightedPool,
    fg_pool: WeightedPool,
    seed: int = 1234,
    preview: int = 0,
    min_motion_px: float | None = None,
    max_motion_px: float | None = None
):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dir = os.path.join(out_root, 'train'); ensure_dir(train_dir)
    val_dir = os.path.join(out_root, 'val'); ensure_dir(val_dir)

    def one_example(idx: int, split_dir: str):
        bg_ds = bg_pool.sample()
        bg_img = bg_ds.get_random()
        layers: List[Layer] = [background_layer(bg_img, (H, W))]
        n_obj = random.randint(1, max_objects)
        for _ in range(n_obj):
            fg_ds = fg_pool.sample()
            fg = fg_ds.get_random()
            layers.append(random_object_layer(fg, (H, W)))

        I1, I2, flow, valid = render_pair_and_flow(layers, (H, W), device, max_motion_px = max_motion_px, min_motion_px = min_motion_px)

        base = f"{idx:06d}"
        save_png(I1, os.path.join(split_dir, f"{base}_img1.png"))
        save_png(I2, os.path.join(split_dir, f"{base}_img2.png"))
        write_flo(flow.permute(1, 2, 0).numpy(), os.path.join(split_dir, f"{base}_flow.flo"))
        save_png(valid.float(), os.path.join(split_dir, f"{base}_valid.png"))

        meta = {
            'id': base,
            'height': H,
            'width': W,
            'num_layers': len(layers),
            'layers': [
                {
                    'z': float(L.z),
                    'A1': L.A1.numpy().tolist(),
                    'A2': L.A2.numpy().tolist(),
                    'tex_shape': list(L.tex.shape),
                } for L in layers
            ],
        }
        with open(os.path.join(split_dir, f"{base}_meta.json"), 'w') as f:
            json.dump(meta, f, indent=2)
        return base

    train_ids: List[str] = []
    val_ids: List[str] = []
    total = num_train + num_val
    for i in tqdm.tqdm([ _x for _x in range(1, total + 1)]):
        split_dir = train_dir if i <= num_train else val_dir
        base = one_example(i, split_dir)
        (train_ids if i <= num_train else val_ids).append(base)

    with open(os.path.join(out_root, 'train.txt'), 'w') as f:
        f.write('\n'.join(train_ids))
    with open(os.path.join(out_root, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_ids))

    print(f"Done. Wrote {len(train_ids)} train and {len(val_ids)} val samples to {out_root}")

    # =====================
    # Fast, masked previews
    # =====================
    if preview > 0:
        def flow_to_rgb(flow: torch.Tensor, valid: Optional[torch.Tensor] = None, invert_v: bool = True) -> torch.Tensor:
            # flow: (2,H,W) -> (3,H,W) RGB [0,1]
            u, v = flow[0], flow[1]
            if invert_v:
                v = -v  # display convention: up is positive
            ang = torch.atan2(v, u)                     # [-pi, pi]
            h = (ang + math.pi) / (2 * math.pi)         # [0,1)
            mag = torch.sqrt(u*u + v*v)
            mag = mag / (mag.max() + 1e-8)
            s = torch.ones_like(h)
            vval = mag
            c = vval * s
            h6 = h * 6.0
            k = torch.floor(h6).to(torch.int64) % 6
            f = h6 - torch.floor(h6)
            p = vval - c
            x = c * (1.0 - torch.abs(2.0*f - 1.0))
            zero = torch.zeros_like(h)
            r = torch.zeros_like(h); g = torch.zeros_like(h); b = torch.zeros_like(h)
            r = torch.where(k == 0, c, r); g = torch.where(k == 0, x, g); b = torch.where(k == 0, zero, b)
            r = torch.where(k == 1, x, r); g = torch.where(k == 1, c, g)
            g = torch.where(k == 2, c, g); b = torch.where(k == 2, x, b)
            r = torch.where(k == 3, zero, r); g = torch.where(k == 3, x, g); b = torch.where(k == 3, c, b)
            r = torch.where(k == 4, x, r); b = torch.where(k == 4, c, b)
            r = torch.where(k == 5, c, r); g = torch.where(k == 5, zero, g)
            r = (r + p).clamp(0,1); g = (g + p).clamp(0,1); b = (b + p).clamp(0,1)
            rgb = torch.stack([r, g, b], dim=0)
            if valid is not None:
                mask = (valid > 0.5).to(rgb.dtype)
                rgb = rgb * mask.unsqueeze(0)
            return rgb

        k = min(preview, len(train_ids))
        for base in train_ids[:k]:
            img1 = Image.open(os.path.join(train_dir, f"{base}_img1.png"))
            img2 = Image.open(os.path.join(train_dir, f"{base}_img2.png"))
            with open(os.path.join(train_dir, f"{base}_flow.flo"), 'rb') as f:
                _ = struct.unpack('f', f.read(4))
                w = struct.unpack('i', f.read(4))[0]
                h = struct.unpack('i', f.read(4))[0]
                data = np.fromfile(f, np.float32, count=2*w*h).reshape(h, w, 2)
            flow = torch.from_numpy(data).permute(2, 0, 1).float()
            valid_img = Image.open(os.path.join(train_dir, f"{base}_valid.png"))
            valid = torch.from_numpy((np.array(valid_img, dtype=np.uint8) > 127).astype(np.float32))
            rgb = flow_to_rgb(flow, valid=valid)
            canvas_img = Image.new('RGB', (img1.width * 3, img1.height))
            canvas_img.paste(img1, (0, 0))
            canvas_img.paste(img2, (img1.width, 0))
            canvas_img.paste(T.ToPILImage()(rgb.clamp(0, 1)), (img1.width * 2, 0))
            canvas_img.save(os.path.join(train_dir, f"{base}_preview.png"))


# ================================================================
# PyTorch loader for generated data
# ================================================================

class FlyingStuffDataset(torch.utils.data.Dataset):
    """FlowNet-style loader for generated pairs.
    Returns dict: img1 (3xHxW), img2 (3xHxW), flow (2xHxW), valid (1xHxW).
    """
    def __init__(self, root: str, split: str = 'train'):
        self.root = root
        self.split = split
        self.dir = os.path.join(root, split)
        with open(os.path.join(root, f"{split}.txt"), 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.ids)

    def _read_flo(self, path: str) -> torch.Tensor:
        with open(path, 'rb') as f:
            magic = struct.unpack('f', f.read(4))[0]
            if abs(magic - 202021.25) > 1e-3:
                raise ValueError('Invalid .flo file')
            w = struct.unpack('i', f.read(4))[0]
            h = struct.unpack('i', f.read(4))[0]
            data = np.fromfile(f, np.float32, count=2*w*h).reshape(h,w,2)
        return torch.from_numpy(data).permute(2,0,1).float()

    def __getitem__(self, i: int):
        bid = self.ids[i]
        img1 = Image.open(os.path.join(self.dir, f"{bid}_img1.png")).convert('RGB')
        img2 = Image.open(os.path.join(self.dir, f"{bid}_img2.png")).convert('RGB')
        valid = Image.open(os.path.join(self.dir, f"{bid}_valid.png"))
        flow = self._read_flo(os.path.join(self.dir, f"{bid}_flow.flo"))
        return {
            'img1': self.to_tensor(img1),
            'img2': self.to_tensor(img2),
            'flow': flow,
            'valid': torch.from_numpy(np.array(valid, dtype=np.uint8)).float().unsqueeze(0) / 255.0,
        }


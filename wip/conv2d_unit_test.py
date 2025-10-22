import math
import random
import torch
import torch.nn.functional as F
from torch import nn

import torch
import torch.nn.functional as F
from torch import nn

from conv2d_with_features import FastConv2d

torch.manual_seed(0)
random.seed(0)

def make_pair(in_ch, out_ch, k, stride, pad, dil, groups):
    """Create FastConv2d and nn.Conv2d with identical params and copied weights/bias."""
    m_fast = FastConv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=k,
        stride=stride,
        padding=pad,
        dilation=dil,
        groups=groups,
        bias=True,
    )
    m_ref = nn.Conv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=k,
        stride=stride,
        padding=pad,
        dilation=dil,
        groups=groups,
        bias=True,
    )
    # copy weights/bias for exact numerical equality
    with torch.no_grad():
        m_ref.weight.copy_(m_fast.weight)
        if m_fast.bias is not None:
            m_ref.bias.copy_(m_fast.bias)
    return m_fast, m_ref

def run_case(B, H, W, in_ch, out_ch, k, stride, pad, dil, groups, atol=0, rtol=0):
    x = torch.randn(B, in_ch, H, W)  # CPU

    m_fast, m_ref = make_pair(in_ch, out_ch, k, stride, pad, dil, groups)

    # --- forward & equality ---
    out_fast, shared = m_fast(x, include_features=True)
    out_ref = m_ref(x)
    assert torch.equal(out_fast, out_ref), "Conv outputs differ (should be bitwise equal with same weights)."

    # --- shared feats shape checks ---
    kH, kW = (k if isinstance(k, tuple) else (k, k))
    H_out, W_out = out_fast.shape[-2:]
    assert shared.shape == (B, H_out, W_out, in_ch * kH * kW), \
        f"shared_feats shape mismatch: got {shared.shape}, expected {(B, H_out, W_out, in_ch * kH * kW)}"

    # --- unfold alignment already checked in forward via assert L == H_out*W_out ---

    # --- grouped slice correctness + per-class reconstruction ---
    G = groups
    Cg_in = in_ch // G
    Co_per_g = out_ch // G
    Cg_vec = Cg_in * kH * kW

    # pick a few classes to test thoroughly
    test_classes = {0, out_ch - 1, out_ch // 2}
    test_classes = [c for c in test_classes if 0 <= c < out_ch]

    shared_dev = shared  # already on CPU
    for class_id in test_classes:
        feats_g = m_fast.features_for_class_from_shared(shared_dev, class_id)  # [B, H_out, W_out, Cg_vec]
        assert feats_g.shape[-1] == Cg_vec

        # figure out which input channels this class connects to (group mapping)
        g = class_id // Co_per_g
        in_start = g * Cg_in
        in_end = (g + 1) * Cg_in

        # get flattened weight vector for this class: [Cg_in * kH * kW]
        # weight layout: [Co, C_in/G, kH, kW]
        w = m_fast.weight[class_id, :, :, :].reshape(-1)  # [Cg_vec]
        b = m_fast.bias[class_id].item() if m_fast.bias is not None else 0.0

        # reconstruct conv output channel via dot(feats_g, w) + b
        # reshape feats_g to [B*H_out*W_out, Cg_vec]
        BHW = B * H_out * W_out
        feats_flat = feats_g.reshape(BHW, Cg_vec)
        rec_flat = feats_flat @ w  # [BHW]
        rec = rec_flat.reshape(B, H_out, W_out) + b  # [B, H_out, W_out]

        # compare to actual conv output channel
        target = out_fast[:, class_id, :, :]  # [B, H_out, W_out]
        if not torch.allclose(rec, target, atol=1e-6, rtol=1e-6):
            max_abs = (rec - target).abs().max().item()
            raise AssertionError(f"Per-class reconstruction mismatch for class {class_id}. max|diff|={max_abs}")

def test_fastconv2d_cpu():
    # A set of configurations; each tuple: (B, H, W, in_ch, out_ch, k, stride, pad, dil, groups)
    configs = [
        # simple
        (2, 16, 20, 4, 6, 3, 1, 1, 1, 1),
        # stride > 1
        (1, 31, 29, 8, 8, 3, 2, 1, 1, 1),
        # padding same-ish
        (1, 32, 32, 3, 7, 5, 1, 2, 1, 1),
        # dilation
        (1, 35, 35, 3, 5, 3, 1, 0, 2, 1),
        # groups=2
        (2, 28, 30, 8, 10, 3, 2, 1, 1, 2),  # in_ch % 2 == 0, out_ch % 2 == 0
        # depthwise (groups == in_ch == out_ch)
        (1, 27, 25, 12, 12, 3, 1, 1, 1, 12),
        # rectangular kernel
        (1, 20, 18, 6, 9, (3, 2), (2, 1), (1, 0), (1, 1), 3),
    ]

    for cfg in configs:
        run_case(*cfg)

if __name__ == "__main__":
    # Run as a simple script (no pytest needed)
    test_fastconv2d_cpu()
    print("All FastConv2d CPU tests passed ✅")

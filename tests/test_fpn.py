"""FPN variants: shapes, fuse modes, extra levels, validation."""
import pytest
import torch

from MLTools.Network import ClassicFPN, CspFPN, GenericFPN


def features(channels=(8, 16, 32), base=32):
    """Pyramid of feature maps, shallow->deep, halving resolution."""
    feats = []
    size = base
    for c in channels:
        feats.append(torch.randn(2, c, size, size))
        size //= 2
    return feats


# ---------------------------------------------------------------- GenericFPN

@pytest.mark.parametrize("fuse", ["sum", "concat", "csp"])
def test_fuse_types_preserve_resolution(fuse):
    fpn = GenericFPN([8, 16, 32], fuse_type=fuse)
    outs = fpn(features())
    assert len(outs) == 3
    for out, ref, ch in zip(outs, features(), [8, 16, 32]):
        assert out.shape == (2, ch, ref.shape[2], ref.shape[3])


def test_unified_out_channels():
    fpn = GenericFPN([8, 16, 32], out_channels=24, fuse_type="sum")
    outs = fpn(features())
    assert all(o.shape[1] == 24 for o in outs)


def test_per_level_out_channels():
    fpn = GenericFPN([8, 16, 32], out_channels=[4, 8, 12], fuse_type="concat")
    outs = fpn(features())
    assert [o.shape[1] for o in outs] == [4, 8, 12]


def test_sum_fuse_with_differing_widths_projects_top():
    fpn = GenericFPN([8, 16, 32], out_channels=[4, 8, 12], fuse_type="sum")
    outs = fpn(features())
    assert [o.shape[1] for o in outs] == [4, 8, 12]


def test_smoothing():
    fpn = GenericFPN([8, 16], smooth="3x3", fuse_type="sum", out_channels=8)
    outs = fpn(features(channels=(8, 16)))
    assert len(outs) == 2


def test_extra_levels():
    fpn = GenericFPN([8, 16, 32], fuse_type="sum", extra_levels=2)
    outs = fpn(features())
    assert len(outs) == 5
    # each extra level halves spatial resolution of the previous
    assert outs[3].shape[-1] == outs[2].shape[-1] // 2
    assert outs[4].shape[-1] == outs[3].shape[-1] // 2
    assert outs[3].shape[1] == 32 and outs[4].shape[1] == 32


def test_extra_levels_custom_channels():
    fpn = GenericFPN([8, 16], fuse_type="sum", extra_levels=2, extra_conv_channels=[64, 128])
    outs = fpn(features(channels=(8, 16)))
    assert outs[2].shape[1] == 64 and outs[3].shape[1] == 128


def test_single_level_passthrough():
    fpn = GenericFPN([8], fuse_type="sum")
    outs = fpn([torch.randn(2, 8, 16, 16)])
    assert len(outs) == 1 and outs[0].shape == (2, 8, 16, 16)


def test_gradients_flow():
    fpn = GenericFPN([8, 16], fuse_type="csp")
    feats = [f.requires_grad_(True) for f in features(channels=(8, 16))]
    outs = fpn(feats)
    sum(o.sum() for o in outs).backward()
    for f in feats:
        assert f.grad is not None and torch.isfinite(f.grad).all()


def test_nearest_upsample_mode():
    fpn = GenericFPN([8, 16], fuse_type="sum", upsample_mode="nearest", align_corners=None)
    assert len(fpn(features(channels=(8, 16)))) == 2


def test_odd_spatial_sizes():
    """Top-down upsample must match arbitrary (non-power-of-two) sizes."""
    fpn = GenericFPN([8, 16], fuse_type="concat")
    feats = [torch.randn(1, 8, 17, 23), torch.randn(1, 16, 9, 11)]
    outs = fpn(feats)
    assert outs[0].shape[-2:] == (17, 23)


# ---------------------------------------------------------------- validation

def test_invalid_fuse_type():
    with pytest.raises(ValueError):
        GenericFPN([8, 16], fuse_type="magic")


def test_wrong_number_of_inputs():
    fpn = GenericFPN([8, 16], fuse_type="sum")
    with pytest.raises(ValueError):
        fpn([torch.randn(1, 8, 8, 8)])


def test_out_channels_length_mismatch():
    with pytest.raises(ValueError):
        GenericFPN([8, 16, 32], out_channels=[8, 16])


# ---------------------------------------------------------------- subclasses

def test_classic_fpn():
    fpn = ClassicFPN([8, 16, 32], out_channels=16)
    outs = fpn(features())
    assert all(o.shape[1] == 16 for o in outs)


def test_classic_fpn_extra_levels():
    fpn = ClassicFPN([8, 16], out_channels=8, extra_levels=1)
    assert len(fpn(features(channels=(8, 16)))) == 3


def test_csp_fpn_keeps_per_level_widths():
    fpn = CspFPN([8, 16, 32])
    outs = fpn(features())
    assert [o.shape[1] for o in outs] == [8, 16, 32]

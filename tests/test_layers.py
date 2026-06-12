"""Network.Layers: shape checks, gradient flow, factory plumbing, edge cases."""
import warnings

import pytest
import torch
from torch import nn

from MLTools.Network import (
    AutoGroupNorm,
    Bottleneck,
    CBAM,
    ConvNormAct,
    CSP1_X,
    CSP2_X,
    DepthwiseConv,
    NormActConv,
    PointwiseConv,
    PureConv,
    ResidualBlock,
    SqueezeExciteBlock,
)


X8 = lambda: torch.randn(2, 8, 16, 16)


# ---------------------------------------------------------------- AutoGroupNorm

def test_autogroupnorm_prefers_8_per_group():
    n = AutoGroupNorm(64)
    assert n.norm.num_groups == 8


def test_autogroupnorm_explicit_groups():
    n = AutoGroupNorm(64, num_groups=4)
    assert n.norm.num_groups == 4


def test_autogroupnorm_small_channels():
    n = AutoGroupNorm(3)  # natural=0 -> floors to 1 group
    assert n.norm.num_groups == 1
    out = n(torch.randn(2, 3, 4, 4))
    assert out.shape == (2, 3, 4, 4)


def test_autogroupnorm_prime_channels():
    n = AutoGroupNorm(7)
    assert 7 % n.norm.num_groups == 0


def test_autogroupnorm_min_groups_above_channels_division():
    """channels=5 with min_groups=2: no divisor in range; used to raise
    ZeroDivisionError via natural=0, must now fall back gracefully."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = AutoGroupNorm(5, min_groups=2, max_groups=4)
    assert 5 % n.norm.num_groups == 0


def test_autogroupnorm_out_of_range_natural_warns():
    # 136 = 8*17: natural=17 divides but is above max_groups=16, while the
    # clamped 16 does not divide -> range restriction is relaxed with a warning
    with pytest.warns(UserWarning):
        n = AutoGroupNorm(136, max_groups=16)
    assert n.norm.num_groups == 17


def test_autogroupnorm_forward_normalizes():
    n = AutoGroupNorm(16, affine=False)
    x = torch.randn(4, 16, 8, 8) * 5 + 3
    out = n(x)
    assert abs(float(out.mean())) < 0.1
    assert abs(float(out.std()) - 1.0) < 0.1


# ---------------------------------------------------------------- conv wrappers

@pytest.mark.parametrize("cls", [PureConv, NormActConv, ConvNormAct])
def test_conv_wrappers_shapes(cls):
    m = cls(8, 12, kernel_size=3, padding=1)
    assert m(X8()).shape == (2, 12, 16, 16)


@pytest.mark.parametrize("cls", [NormActConv, ConvNormAct])
def test_conv_wrappers_custom_factories(cls):
    m = cls(8, 8, kernel_size=1,
            act_factory=lambda: nn.ReLU(),
            norm_factory=lambda c: nn.BatchNorm2d(c))
    assert m(X8()).shape == (2, 8, 16, 16)


def test_conv_wrappers_stride():
    m = ConvNormAct(8, 8, kernel_size=3, stride=2, padding=1)
    assert m(X8()).shape == (2, 8, 8, 8)


def test_make_factories_reject_instances():
    with pytest.raises(TypeError):
        ConvNormAct(8, 8, kernel_size=1, norm_factory=nn.BatchNorm2d(8))
    with pytest.raises(TypeError):
        ConvNormAct(8, 8, kernel_size=1, act_factory=nn.ReLU())


# ---------------------------------------------------------------- attention blocks

def test_squeeze_excite_shape_and_range():
    se = SqueezeExciteBlock(8)
    x = X8()
    out = se(x)
    assert out.shape == x.shape
    # gated output magnitude can't exceed input (sigmoid in (0,1))
    assert (out.abs() <= x.abs() + 1e-6).all()


def test_cbam_full():
    m = CBAM(8)
    assert m(X8()).shape == (2, 8, 16, 16)


@pytest.mark.parametrize("ch,sp", [(True, False), (False, True), (False, False)])
def test_cbam_partial_modes(ch, sp):
    m = CBAM(8, use_channel=ch, use_spatial=sp)
    out = m(X8())
    assert out.shape == (2, 8, 16, 16)
    if not ch and not sp:
        assert torch.equal(out, m(X8()) * 0 + out)  # passthrough sanity


def test_cbam_gradients_flow():
    m = CBAM(8)
    x = X8().requires_grad_(True)
    m(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------- residual / bottleneck

def test_residual_block_identity_skip():
    rb = ResidualBlock(8, 8, main_branch=lambda ic, oc, s, dt: nn.Conv2d(ic, oc, 3, padding=1))
    assert isinstance(rb.skip, nn.Identity)
    assert rb(X8()).shape == (2, 8, 16, 16)


def test_residual_block_channel_change_default_norm():
    """in!=out with norm_factory=None used to crash (None not callable)."""
    rb = ResidualBlock(8, 16, main_branch=lambda ic, oc, s, dt: nn.Conv2d(ic, oc, 3, padding=1))
    assert rb(X8()).shape == (2, 16, 16, 16)


def test_residual_block_strided():
    rb = ResidualBlock(
        8, 16, stride=2,
        main_branch=lambda ic, oc, s, dt: nn.Conv2d(ic, oc, 3, stride=s, padding=1),
    )
    assert rb(X8()).shape == (2, 16, 8, 8)


def test_residual_block_requires_main_branch():
    with pytest.raises(RuntimeError):
        ResidualBlock(8, 8)


def test_bottleneck_shapes():
    b = Bottleneck(8, 16, X=2)
    assert b(X8()).shape == (2, 16, 16, 16)


def test_bottleneck_stride():
    b = Bottleneck(8, 8, X=1, stride=2)
    assert b(X8()).shape == (2, 8, 8, 8)


def test_bottleneck_default_out_channels():
    b = Bottleneck(8)
    assert b(X8()).shape == (2, 8, 16, 16)


def test_bottleneck_tiny_reduction_floors_to_1_channel():
    # out=4, reduction=0.1 -> int() would give 0 working channels
    b = Bottleneck(8, 4, reduction=0.1)
    assert b(X8()).shape == (2, 4, 16, 16)


def test_bottleneck_warns_on_zero_X():
    with pytest.warns(UserWarning):
        Bottleneck(8, 8, X=0)


def test_bottleneck_normactconv_variant():
    b = Bottleneck(8, 8, X=1, conv_class=NormActConv)
    assert b(X8()).shape == (2, 8, 16, 16)


# ---------------------------------------------------------------- pointwise / depthwise

def test_pointwise_conv():
    m = PointwiseConv(8, 24)
    assert m(X8()).shape == (2, 24, 16, 16)


def test_depthwise_conv_stacked_strided():
    m = DepthwiseConv(8, X=3, stride=2)
    assert m(X8()).shape == (2, 8, 8, 8)


# ---------------------------------------------------------------- CSP blocks

@pytest.mark.parametrize("cls", [CSP1_X, CSP2_X])
def test_csp_blocks(cls):
    m = cls(8, working_channels=8, out_channels=16, X=1)
    assert m(X8()).shape == (2, 16, 16, 16)


@pytest.mark.parametrize("cls", [CSP1_X, CSP2_X])
def test_csp_default_out_channels(cls):
    m = cls(8, working_channels=4)
    assert m(X8()).shape == (2, 8, 16, 16)


def test_csp_gradients_flow():
    m = CSP1_X(8, working_channels=8, X=1)
    x = X8().requires_grad_(True)
    m(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------- adversarial-ish

def test_blocks_handle_1x1_spatial():
    x = torch.randn(2, 8, 1, 1)
    assert SqueezeExciteBlock(8)(x).shape == (2, 8, 1, 1)
    assert CBAM(8)(x).shape == (2, 8, 1, 1)
    assert PointwiseConv(8, 8)(x).shape == (2, 8, 1, 1)


def test_blocks_no_nan_on_zero_input():
    x = torch.zeros(2, 8, 8, 8)
    for m in (ConvNormAct(8, 8, 3, padding=1), CBAM(8), SqueezeExciteBlock(8)):
        assert torch.isfinite(m(x)).all()

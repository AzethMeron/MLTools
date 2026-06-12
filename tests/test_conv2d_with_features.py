"""Conv2dWithFeatures: output parity with nn.Conv2d and feature correctness.

The strongest possible check: the conv output must be exactly reconstructible
as <shared receptive-field features, kernel weights> + bias.
"""
import pytest
import torch
from torch import nn

from MLTools.Network import Conv2dWithFeatures


def test_output_identical_to_nn_conv2d():
    conv = Conv2dWithFeatures(4, 6, kernel_size=3, padding=1, stride=2, bias=True)
    ref = nn.Conv2d(4, 6, kernel_size=3, padding=1, stride=2, bias=True)
    ref.load_state_dict(conv.state_dict())
    x = torch.randn(2, 4, 11, 13)
    assert torch.allclose(conv(x), ref(x), atol=1e-6)


def test_without_features_returns_tensor_only():
    conv = Conv2dWithFeatures(3, 5, kernel_size=3)
    out = conv(torch.randn(1, 3, 8, 8))
    assert isinstance(out, torch.Tensor)


def test_features_shape_and_detachment():
    conv = Conv2dWithFeatures(3, 5, kernel_size=3, padding=1)
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    out, feats = conv(x, include_features=True)
    assert out.shape == (2, 5, 8, 8)
    assert feats.shape == (2, 8, 8, 3 * 3 * 3)
    assert not feats.requires_grad
    assert feats.device.type == "cpu"
    assert out.requires_grad  # the real conv keeps gradients


@pytest.mark.parametrize("groups", [1, 2])
def test_output_reconstructible_from_features(groups):
    """out[b, c, i, j] == feats_for_class(c)[b, i, j] . weight[c].flatten() + bias[c]"""
    conv = Conv2dWithFeatures(4, 6, kernel_size=3, padding=1, groups=groups, bias=True)
    x = torch.randn(2, 4, 7, 9)
    out, feats = conv(x, include_features=True)
    for class_id in range(6):
        fg = conv.features_for_class_from_shared(feats, class_id)
        w = conv.weight[class_id].reshape(-1)  # (C_in/G * kH * kW)
        recon = fg @ w + conv.bias[class_id]
        assert torch.allclose(recon, out[:, class_id].cpu(), atol=1e-4), \
            f"class {class_id} (groups={groups}) reconstruction mismatch"


def test_grouped_feature_slice_shape():
    conv = Conv2dWithFeatures(8, 8, kernel_size=3, padding=1, groups=4)
    x = torch.randn(1, 8, 6, 6)
    _, feats = conv(x, include_features=True)
    fg = conv.features_for_class_from_shared(feats, class_id=5)
    assert fg.shape == (1, 6, 6, 2 * 3 * 3)  # C_in/G = 2


def test_stride_and_dilation():
    conv = Conv2dWithFeatures(3, 4, kernel_size=3, stride=2, dilation=2, padding=2)
    x = torch.randn(1, 3, 16, 16)
    out, feats = conv(x, include_features=True)
    assert feats.shape[:3] == (1, out.shape[2], out.shape[3])


def test_class_id_out_of_range():
    conv = Conv2dWithFeatures(3, 4, kernel_size=1)
    _, feats = conv(torch.randn(1, 3, 4, 4), include_features=True)
    with pytest.raises(AssertionError):
        conv.features_for_class_from_shared(feats, class_id=4)
    with pytest.raises(AssertionError):
        conv.features_for_class_from_shared(feats, class_id=-1)


def test_string_padding_clear_error():
    conv = Conv2dWithFeatures(3, 4, kernel_size=3, padding="same")
    x = torch.randn(1, 3, 8, 8)
    conv(x)  # plain forward is fine
    with pytest.raises(NotImplementedError):
        conv(x, include_features=True)

"""RandomScale: shape preservation, probability gate, validation."""
import random

import pytest
import torch

from MLTools.DataProcessing import RandomScale


def test_shape_preserved():
    t = RandomScale(min_scale=0.25, max_scale=0.75)
    img = torch.rand(3, 33, 47)
    out = t(img)
    assert out.shape == img.shape


def test_degradation_changes_pixels():
    t = RandomScale(min_scale=0.1, max_scale=0.2)
    img = torch.rand(3, 64, 64)
    out = t(img)
    assert not torch.allclose(out, img)


def test_p_zero_identity():
    t = RandomScale(p=0.0)
    img = torch.rand(3, 16, 16)
    assert t(img) is img


def test_upscale_range_supported():
    t = RandomScale(min_scale=1.5, max_scale=2.0)
    img = torch.rand(3, 20, 20)
    assert t(img).shape == img.shape


def test_fixed_scale_when_min_equals_max():
    t = RandomScale(min_scale=0.5, max_scale=0.5)
    img = torch.rand(3, 32, 32)
    assert t(img).shape == img.shape


def test_deterministic_under_global_seed():
    img = torch.rand(3, 40, 40)
    t = RandomScale(min_scale=0.2, max_scale=0.9)
    random.seed(7); a = t(img.clone())
    random.seed(7); b = t(img.clone())
    assert torch.equal(a, b)


@pytest.mark.parametrize("algo", ["nearest", "bilinear", "bicubic", "area"])
def test_all_interpolations(algo):
    t = RandomScale(downscale_algorithm=algo, upscale_algorithm=algo)
    assert t(torch.rand(3, 24, 24)).shape == (3, 24, 24)


def test_tiny_image_scale_floors_to_1px():
    t = RandomScale(min_scale=0.01, max_scale=0.02)
    out = t(torch.rand(3, 4, 4))
    assert out.shape == (3, 4, 4)
    assert torch.isfinite(out).all()


def test_invalid_params():
    with pytest.raises(ValueError):
        RandomScale(min_scale=0.0)
    with pytest.raises(ValueError):
        RandomScale(min_scale=0.8, max_scale=0.2)
    with pytest.raises(ValueError):
        RandomScale(p=2.0)
    with pytest.raises(ValueError):
        RandomScale(downscale_algorithm="nope")


def test_repr():
    r = repr(RandomScale(min_scale=0.3, max_scale=0.6, p=0.5))
    assert "min_scale=0.3" in r and "p=0.5" in r

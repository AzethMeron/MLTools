"""StandardScaler: typical, edge, randomized and adversarial coverage."""
import pytest
import torch

from MLTools.DataProcessing import StandardScaler


def reference_stats(X):
    """Float64 ground truth: population mean/variance."""
    X64 = X.to(torch.float64)
    return X64.mean(dim=0), X64.var(dim=0, unbiased=False)


# ---------------------------------------------------------------- typical

def test_fit_transform_zero_mean_unit_var():
    X = torch.randn(500, 8) * 3.0 + 5.0
    scaler = StandardScaler().fit(X, batch_size=64)
    Z = scaler.transform(X)
    assert Z.shape == X.shape
    assert torch.allclose(Z.mean(dim=0), torch.zeros(8), atol=1e-4)
    assert torch.allclose(Z.std(dim=0, unbiased=False), torch.ones(8), atol=1e-4)


def test_matches_float64_reference():
    X = torch.randn(321, 5) * torch.tensor([1.0, 10.0, 0.1, 100.0, 5.0]) + 7.0
    scaler = StandardScaler().fit(X, batch_size=50)
    mean_ref, var_ref = reference_stats(X)
    assert torch.allclose(scaler.mean_.double(), mean_ref, atol=1e-4)
    assert torch.allclose(scaler.var_.double(), var_ref, rtol=1e-4, atol=1e-5)


def test_inverse_transform_roundtrip():
    X = torch.randn(100, 4) * 2.5 - 3.0
    scaler = StandardScaler().fit(X)
    assert torch.allclose(scaler.inverse_transform(scaler.transform(X)), X, atol=1e-4)


def test_partial_fit_equals_fit():
    X = torch.randn(200, 6)
    s_full = StandardScaler().fit(X, batch_size=200)
    s_inc = StandardScaler()
    for chunk in torch.split(X, 17):
        s_inc.partial_fit(chunk)
    assert torch.allclose(s_full.mean_, s_inc.mean_, atol=1e-5)
    assert torch.allclose(s_full.var_, s_inc.var_, rtol=1e-4, atol=1e-6)


def test_with_mean_with_std_flags():
    X = torch.randn(64, 3) * 4 + 9
    no_mean = StandardScaler(with_mean=False).fit(X)
    no_std = StandardScaler(with_std=False).fit(X)
    Zm = no_mean.transform(X)
    Zs = no_std.transform(X)
    # no centering: mean preserved up to scale
    assert torch.allclose(Zm.mean(0), X.mean(0) / no_mean.scale_, atol=1e-4)
    # no scaling: variance preserved
    assert torch.allclose(Zs.var(0, unbiased=False), X.var(0, unbiased=False), rtol=1e-4)


def test_chunked_transform_matches_full():
    X = torch.randn(101, 7)
    scaler = StandardScaler().fit(X)
    assert torch.allclose(scaler.transform(X), scaler.transform(X, batch_size=13), atol=1e-6)
    assert torch.allclose(
        scaler.inverse_transform(X), scaler.inverse_transform(X, batch_size=13), atol=1e-6
    )


# ---------------------------------------------------------------- edge cases

def test_single_sample_fit():
    X = torch.tensor([[1.0, 2.0, 3.0]])
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    # zero variance -> output is 0 (mean removed, scale = sqrt(eps))
    assert torch.allclose(Z, torch.zeros_like(Z), atol=1e-4)


def test_constant_feature_no_nan():
    X = torch.ones(50, 3) * 7.0
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    assert torch.isfinite(Z).all()
    assert torch.allclose(Z, torch.zeros_like(Z), atol=1e-3)


def test_empty_batch_partial_fit_is_noop():
    scaler = StandardScaler()
    scaler.partial_fit(torch.empty(0, 4))
    assert not scaler.fitted_


def test_refit_resets_statistics():
    s = StandardScaler()
    s.fit(torch.randn(100, 3) + 100.0)
    X2 = torch.randn(100, 3)
    s.fit(X2)
    mean_ref, _ = reference_stats(X2)
    assert torch.allclose(s.mean_.double(), mean_ref, atol=1e-4)
    assert int(s.n_samples_seen_.item()) == 100


# ---------------------------------------------------------------- error handling

def test_rejects_wrong_dims():
    with pytest.raises(ValueError):
        StandardScaler().fit(torch.randn(10))
    with pytest.raises(ValueError):
        StandardScaler().partial_fit(torch.randn(2, 3, 4))


def test_rejects_unfitted_transform():
    with pytest.raises(RuntimeError):
        StandardScaler().transform(torch.randn(4, 4))
    with pytest.raises(RuntimeError):
        StandardScaler().inverse_transform(torch.randn(4, 4))


def test_rejects_feature_mismatch():
    scaler = StandardScaler().fit(torch.randn(10, 4))
    with pytest.raises(ValueError):
        scaler.transform(torch.randn(10, 5))
    with pytest.raises(ValueError):
        scaler.partial_fit(torch.randn(10, 5))


def test_rejects_empty_fit_and_bad_batch_size():
    with pytest.raises(ValueError):
        StandardScaler().fit(torch.empty(0, 4))
    with pytest.raises(ValueError):
        StandardScaler().fit(torch.randn(10, 4), batch_size=0)


# ---------------------------------------------------------------- randomized

@pytest.mark.parametrize("trial", range(5))
def test_random_shapes_and_scales(trial):
    g = torch.Generator().manual_seed(trial)
    N = int(torch.randint(2, 400, (1,), generator=g))
    D = int(torch.randint(1, 30, (1,), generator=g))
    scale = 10 ** float(torch.empty(1).uniform_(-3, 3, generator=g))
    X = torch.randn(N, D, generator=g) * scale
    bs = int(torch.randint(1, N + 1, (1,), generator=g))
    scaler = StandardScaler().fit(X, batch_size=bs)
    mean_ref, var_ref = reference_stats(X)
    assert torch.allclose(scaler.mean_.double(), mean_ref, rtol=1e-3, atol=1e-3 * scale)
    assert torch.allclose(scaler.var_.double(), var_ref, rtol=1e-3, atol=1e-5 * scale * scale)


# ---------------------------------------------------------------- adversarial

def test_large_mean_small_std_numerical_stability():
    """The classic catastrophic-cancellation killer: mean >> std in float32.

    The naive sum-of-squares identity loses all precision here; the
    two-pass batch statistic must survive it.
    """
    torch.manual_seed(0)
    X = torch.randn(2000, 4) + 1e6  # std=1 around a huge mean
    scaler = StandardScaler().fit(X, batch_size=128)
    # Population std must be ~1, not 0 / NaN / wildly off
    assert torch.isfinite(scaler.scale_).all()
    assert torch.allclose(scaler.scale_, torch.ones(4), rtol=0.05)


def test_variance_never_negative():
    # Repeated identical rows + huge offset try to drive m2 below zero
    X = torch.full((1000, 3), 12345678.0)
    scaler = StandardScaler().fit(X, batch_size=7)
    assert (scaler.var_ >= 0).all()
    assert torch.isfinite(scaler.scale_).all()


def test_extreme_value_range():
    X = torch.tensor([[1e-30, 1e30], [2e-30, 2e30], [3e-30, 3e30]])
    scaler = StandardScaler(dtype=torch.float64).fit(X)
    Z = scaler.transform(X)
    assert torch.isfinite(Z).all()


def test_dtype_float64_pipeline():
    X = torch.randn(50, 3, dtype=torch.float64)
    scaler = StandardScaler(dtype=torch.float64).fit(X)
    assert scaler.mean_.dtype == torch.float64
    assert scaler.transform(X).dtype == torch.float64

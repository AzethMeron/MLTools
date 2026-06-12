"""IncrementalPCA: exact-moment streaming PCA vs dense references."""
import pytest
import torch

from MLTools.DataProcessing import IncrementalPCA


def data(N=300, D=8, seed=0, offset=0.0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(N, D, generator=g) * torch.linspace(0.5, 3.0, D) + offset


def population_cov(X):
    X64 = X.to(torch.float64)
    mean = X64.mean(0)
    Xc = X64 - mean
    return (Xc.T @ Xc) / X.shape[0]


# ---------------------------------------------------------------- typical

def test_fit_recovers_eigenstructure():
    X = data()
    ipca = IncrementalPCA(n_components=8).fit(X)
    cov = population_cov(X)
    evals_ref = torch.linalg.eigvalsh(cov).flip(0)
    assert torch.allclose(ipca.explained_variance_.double(), evals_ref, rtol=1e-3, atol=1e-5)
    # components are rows, orthonormal
    C = ipca.components_
    assert C.shape == (8, 8)
    assert torch.allclose(C @ C.T, torch.eye(8), atol=1e-4)


def test_partial_fit_chunks_equal_full_fit():
    X = data(N=257, seed=1)
    full = IncrementalPCA(n_components=4).fit(X)
    inc = IncrementalPCA(n_components=4)
    for chunk in torch.split(X, 31):
        inc.partial_fit(chunk)
    assert torch.allclose(full.explained_variance_, inc.explained_variance_, rtol=1e-4)
    assert torch.allclose(full.mean_, inc.mean_, atol=1e-5)
    dots = (full.components_ * inc.components_).sum(dim=1).abs()
    assert torch.allclose(dots, torch.ones(4), atol=1e-3)


def test_transform_centers_input():
    X = data(N=100, D=5, seed=2, offset=10.0)
    ipca = IncrementalPCA(n_components=3).fit(X)
    Z = ipca.transform(X)
    assert Z.shape == (100, 3)
    Z_ref = (X - ipca.mean_) @ ipca.components_.T
    assert torch.allclose(Z, Z_ref, atol=1e-4)
    # projections of centered data have ~zero mean
    assert torch.allclose(Z.mean(0), torch.zeros(3), atol=1e-3)


def test_full_rank_inverse_transform_roundtrip():
    X = data(N=80, D=6, seed=3, offset=-4.0)
    ipca = IncrementalPCA(n_components=6).fit(X)
    X_rec = ipca.inverse_transform(ipca.transform(X))
    assert torch.allclose(X_rec, X, atol=1e-3)


def test_whitening_unit_variance():
    X = data(N=2000, D=6, seed=4)
    ipca = IncrementalPCA(n_components=6, whiten=True).fit(X)
    Z = ipca.transform(X)
    # whitened projections: population variance ~ 1 in every direction
    assert torch.allclose(Z.var(0, unbiased=False), torch.ones(6), rtol=0.05)
    # and whiten round-trips through inverse_transform
    X_rec = ipca.inverse_transform(Z)
    assert torch.allclose(X_rec, X, atol=1e-2)


def test_fit_transform():
    X = data(N=64, D=4, seed=5)
    ipca = IncrementalPCA(n_components=2)
    Z = ipca.fit_transform(X)
    assert torch.allclose(Z, ipca.transform(X), atol=1e-5)


def test_explained_variance_ratio_sums_to_one_full_rank():
    X = data(N=128, D=5, seed=6)
    ipca = IncrementalPCA().fit(X)  # keep all
    assert ipca.n_components_ == 5
    assert abs(float(ipca.explained_variance_ratio_.sum()) - 1.0) < 1e-4


def test_singular_values_match_sklearn_convention():
    X = data(N=100, D=4, seed=7)
    ipca = IncrementalPCA(n_components=4).fit(X)
    expected = torch.sqrt(ipca.explained_variance_.clamp(min=0) * (100 - 1))
    assert torch.allclose(ipca.singular_values_, expected, atol=1e-4)


# ---------------------------------------------------------------- the refit bug

def test_fit_twice_does_not_accumulate():
    """fit() must reset accumulators (sklearn parity); double-fit used to
    double n_samples_seen_ and blend stale moments in."""
    X1 = data(N=100, D=4, seed=8, offset=50.0)
    X2 = data(N=100, D=4, seed=9)
    ipca = IncrementalPCA(n_components=4)
    ipca.fit(X1)
    ipca.fit(X2)
    fresh = IncrementalPCA(n_components=4).fit(X2)
    assert ipca.n_samples_seen_ == 100
    assert torch.allclose(ipca.mean_, fresh.mean_, atol=1e-5)
    assert torch.allclose(ipca.explained_variance_, fresh.explained_variance_, rtol=1e-5)


def test_batch_size_streaming_fit():
    X = data(N=205, D=6, seed=10)
    a = IncrementalPCA(n_components=3, batch_size=32).fit(X)
    b = IncrementalPCA(n_components=3).fit(X)
    assert torch.allclose(a.explained_variance_, b.explained_variance_, rtol=1e-4)


# ---------------------------------------------------------------- covariance / precision

def test_get_covariance_full_rank_matches_population_cov():
    X = data(N=400, D=5, seed=11)
    ipca = IncrementalPCA(n_components=5).fit(X)
    C = ipca.get_covariance()
    assert torch.allclose(C.double(), population_cov(X), rtol=1e-3, atol=1e-4)


def test_get_precision_inverts_covariance():
    X = data(N=400, D=5, seed=12)
    ipca = IncrementalPCA(n_components=5).fit(X)
    C = ipca.get_covariance()
    P = ipca.get_precision()
    assert torch.allclose(C @ P, torch.eye(5), atol=1e-3)


def test_get_covariance_reduced_rank_diag_reasonable():
    X = data(N=500, D=8, seed=13)
    ipca = IncrementalPCA(n_components=3).fit(X)
    C = ipca.get_covariance()
    assert C.shape == (8, 8)
    assert torch.allclose(C, C.T, atol=1e-5)  # symmetric
    # eigenvalues of reconstruction are >= 0 (PSD-ish up to noise floor)
    evals = torch.linalg.eigvalsh(C)
    assert float(evals.min()) > -1e-4


# ---------------------------------------------------------------- edge cases

def test_single_sample_partial_fit():
    ipca = IncrementalPCA()
    ipca.partial_fit(torch.tensor([[1.0, 2.0, 3.0]]))
    assert ipca.n_samples_seen_ == 1
    assert torch.allclose(ipca.mean_, torch.tensor([1.0, 2.0, 3.0]))
    assert torch.isfinite(ipca.explained_variance_).all()


def test_reset():
    ipca = IncrementalPCA(n_components=2).fit(data(N=50, D=4))
    ipca.reset()
    assert ipca.components_ is None and ipca.n_samples_seen_ == 0
    with pytest.raises(RuntimeError):
        ipca.transform(torch.randn(3, 4))


def test_feature_dim_change_rejected():
    ipca = IncrementalPCA()
    ipca.partial_fit(torch.randn(10, 4))
    with pytest.raises(ValueError):
        ipca.partial_fit(torch.randn(10, 5))


def test_invalid_inputs():
    ipca = IncrementalPCA()
    with pytest.raises(TypeError):
        ipca.partial_fit([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        ipca.partial_fit(torch.randn(8))
    with pytest.raises(ValueError):
        IncrementalPCA(n_components=99).partial_fit(torch.randn(10, 4))
    fitted = IncrementalPCA(n_components=2).fit(torch.randn(10, 4))
    with pytest.raises(ValueError):
        fitted.inverse_transform(torch.randn(5, 3))
    with pytest.raises(RuntimeError):
        IncrementalPCA().inverse_transform(torch.randn(5, 3))


# ---------------------------------------------------------------- adversarial

def test_huge_offset_stability():
    """float64 accumulators must survive mean >> std."""
    X = torch.randn(1000, 4) + 1e7
    ipca = IncrementalPCA(n_components=4).fit(X)
    assert torch.isfinite(ipca.explained_variance_).all()
    assert torch.allclose(ipca.var_, X.var(0, unbiased=True), rtol=0.05)


def test_constant_data():
    X = torch.full((100, 3), 5.0)
    ipca = IncrementalPCA(n_components=3).fit(X)
    assert torch.isfinite(ipca.explained_variance_).all()
    assert torch.allclose(ipca.explained_variance_, torch.zeros(3), atol=1e-5)


def test_input_dtype_and_device_preserved():
    X = data(N=40, D=4, seed=14).to(torch.float64)
    ipca = IncrementalPCA(n_components=2).fit(X)
    Z = ipca.transform(X)
    assert Z.dtype == torch.float64
    assert Z.device == X.device

"""PCA (batched, no internal centering): correctness vs dense reference."""
import pytest
import torch

from MLTools.DataProcessing import PCA


def centered(N=300, D=10, seed=0, scale=None):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, D, generator=g)
    if scale is not None:
        X = X * scale
    return X - X.mean(dim=0, keepdim=True)


# ---------------------------------------------------------------- typical

def test_components_orthonormal():
    X = centered()
    pca = PCA(n_components=5).fit(X)
    C = pca.components_  # (D, K)
    assert C.shape == (10, 5)
    assert torch.allclose(C.T @ C, torch.eye(5), atol=1e-4)


def test_matches_direct_eigendecomposition():
    X = centered(N=400, D=8, seed=1)
    pca = PCA(n_components=8).fit(X, batch_size=37)
    cov = X.T @ X / (X.shape[0] - 1)
    evals_ref = torch.linalg.eigvalsh(cov).flip(0)  # descending
    assert torch.allclose(pca.explained_variance_values(), evals_ref, rtol=1e-3, atol=1e-5)


def test_transform_projects_onto_components():
    X = centered(N=100, D=6, seed=2)
    pca = PCA(n_components=3).fit(X)
    Z = pca.transform(X)
    assert Z.shape == (100, 3)
    assert torch.allclose(Z, X @ pca.components_, atol=1e-5)


def test_batched_fit_equals_full_fit():
    X = centered(N=257, D=12, seed=3)
    p1 = PCA(n_components=4).fit(X, batch_size=257)
    p2 = PCA(n_components=4).fit(X, batch_size=10)
    assert torch.allclose(p1.explained_variance_values(), p2.explained_variance_values(), rtol=1e-4)
    # eigenvectors may differ in sign only
    dots = (p1.components_ * p2.components_).sum(dim=0).abs()
    assert torch.allclose(dots, torch.ones(4), atol=1e-3)


def test_batched_transform_equals_full():
    X = centered(N=101, D=7, seed=4)
    pca = PCA(n_components=5).fit(X)
    assert torch.allclose(pca.transform(X), pca.transform(X, batch_size=13), atol=1e-5)


def test_explained_variance_ratio_properties():
    X = centered(N=200, D=9, seed=5)
    pca = PCA(n_components=4).fit(X)
    r = pca.explained_variance_ratio()
    assert r.shape == (4,)
    assert (r >= 0).all() and (r <= 1).all()
    assert float(r.sum()) <= 1.0 + 1e-5
    # ratios are sorted descending (eigenvalues sorted)
    assert (r[:-1] >= r[1:] - 1e-7).all()


def test_dominant_direction_recovered():
    # Variance concentrated along a known direction must be the top component
    g = torch.Generator().manual_seed(6)
    direction = torch.zeros(5); direction[2] = 1.0
    X = torch.randn(500, 1, generator=g) * 50.0 * direction + torch.randn(500, 5, generator=g) * 0.1
    X = X - X.mean(0, keepdim=True)
    pca = PCA(n_components=1).fit(X)
    top = pca.components_[:, 0].abs()
    assert top.argmax() == 2
    assert float(pca.explained_variance_ratio()[0]) > 0.99


# ---------------------------------------------------------------- edge cases

def test_n_components_capped_at_D():
    X = centered(N=50, D=3)
    pca = PCA(n_components=10).fit(X)
    assert pca.components_.shape == (3, 3)


def test_minimum_two_samples():
    X = centered(N=2, D=4)
    PCA(n_components=2).fit(X)  # should not raise
    with pytest.raises(ValueError):
        PCA(n_components=2).fit(X[:1])


def test_zero_variance_data():
    X = torch.zeros(20, 4)
    pca = PCA(n_components=2).fit(X)
    r = pca.explained_variance_ratio()
    assert torch.isfinite(r).all()
    assert torch.allclose(r, torch.zeros(2))


# ---------------------------------------------------------------- error handling

def test_invalid_constructor_and_inputs():
    with pytest.raises(ValueError):
        PCA(n_components=0)
    with pytest.raises(ValueError):
        PCA(n_components=-3)
    with pytest.raises(ValueError):
        PCA(n_components=2).fit(torch.randn(10))
    with pytest.raises(ValueError):
        PCA(n_components=2).fit(torch.randn(10, 3), batch_size=0)


def test_unfitted_and_mismatched():
    pca = PCA(n_components=2)
    with pytest.raises(RuntimeError):
        pca.transform(torch.randn(5, 4))
    with pytest.raises(RuntimeError):
        pca.explained_variance_ratio()
    pca.fit(centered(N=20, D=4))
    with pytest.raises(ValueError):
        pca.transform(torch.randn(5, 6))
    with pytest.raises(ValueError):
        pca.transform(torch.randn(5, 4), batch_size=-1)


# ---------------------------------------------------------------- persistence

def test_state_dict_includes_ratio_buffer():
    """explained_ratio_ must live in state_dict so save/.to(device) keep it."""
    X = centered(N=60, D=5)
    pca = PCA(n_components=3).fit(X)
    sd = pca.state_dict()
    for key in ("components_", "explained_variance_", "explained_ratio_", "total_variance_"):
        assert key in sd, f"buffer {key} missing from state_dict"
    # round-trip into another fitted instance
    other = PCA(n_components=3).fit(centered(N=60, D=5, seed=9))
    other.load_state_dict(sd)
    assert torch.allclose(other.explained_ratio_, pca.explained_ratio_)
    assert torch.allclose(other.components_, pca.components_)


# ---------------------------------------------------------------- randomized / adversarial

@pytest.mark.parametrize("seed", range(4))
def test_reconstruction_error_bounded_by_discarded_variance(seed):
    X = centered(N=150, D=12, seed=seed)
    pca = PCA(n_components=6).fit(X)
    Z = pca.transform(X)
    X_rec = Z @ pca.components_.T
    rec_err = ((X - X_rec) ** 2).sum() / (X.shape[0] - 1)
    discarded = pca.total_variance_ - pca.explained_variance_values().sum()
    assert torch.allclose(rec_err, discarded, rtol=1e-2, atol=1e-4)


def test_wildly_anisotropic_scales():
    scale = torch.tensor([1e-4, 1.0, 1e4, 1e-2, 1e2])
    X = centered(N=300, D=5, seed=7, scale=scale)
    pca = PCA(n_components=5, dtype=torch.float64).fit(X.double())
    r = pca.explained_variance_ratio()
    assert torch.isfinite(r).all()
    assert float(r[0]) > 0.99  # the 1e4 axis dominates

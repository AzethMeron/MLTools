import torch
import torch.nn as nn
from typing import Optional

class PCA(nn.Module):
    """
    PCA in PyTorch with batched fitting, **no internal centering**.

    - Assumes inputs are already centered/standardized externally.
    - Device-agnostic: move the module or the input tensor to CUDA to use the GPU.
    - fit: streams batches to build X^T X (Gram) → covariance → eigendecomposition.
    - transform: single big batch to maximize GPU throughput.

    Args:
        n_components (int): number of principal components to keep.
        dtype (torch.dtype): dtype for internal math and stored params (default torch.float32).
    """

    def __init__(self, n_components: int, dtype: torch.dtype = torch.float32):
        super(PCA, self).__init__()
        if n_components <= 0:
            raise ValueError("n_components must be a positive integer.")
        self.n_components = int(n_components)
        self.dtype = dtype

        # Learned parameters (moved by .to(device))
        self.register_buffer("components_", None)                 # (D, K)
        self.register_buffer("explained_variance_", None)         # (K,)  -- RATIOS (0..1)
        self.register_buffer("explained_variance_values_", None)  # (K,)  -- raw eigenvalues (variances)
        self.register_buffer("total_variance_", None)             # ()    -- sum of all eigenvalues
        self.fitted_ = False

    @torch.no_grad()
    def fit(self, X: torch.Tensor, batch_size: int = 64) -> "PCA":
        """
        Fit PCA on **pre-centered** data X using batched processing.

        Args:
            X: (N, D) tensor on CPU or CUDA. Will be viewed as self.dtype.

        Returns:
            self
        """
        if X.dim() != 2:
            raise ValueError("X must be 2D (N, D).")
        N, D = X.shape
        if N < 2:
            raise ValueError("Need at least 2 samples to compute covariance.")

        device = X.device
        X = X.to(dtype=self.dtype, copy=False)

        # Accumulate Gram matrix G = X^T X in batches (assumes X is centered already)
        gram = torch.zeros((D, D), device=device, dtype=self.dtype)
        n_total = 0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = X[start:end]  # (B, D)
            gram += batch.t().matmul(batch)  # (D, D)
            n_total += batch.size(0)

        # Sample covariance (valid if X is mean-centered)
        cov = gram / float(n_total - 1)

        # Eigendecomposition (symmetric)
        evals, evecs = torch.linalg.eigh(cov)  # ascending
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx].clamp_min(0)        # numerical safety
        evecs = evecs[:, idx]

        k = min(self.n_components, D)
        self.components_ = evecs[:, :k]                 # (D, k)
        self.explained_variance_ = evals[:k]     # raw variances for kept comps
        self.total_variance_ = evals.sum()              # total variance across all features

        if self.total_variance_.item() > 0:
            self.explained_ratio_ = self.explained_variance_ / self.total_variance_
        else:
            self.explained_ratio_ = torch.zeros_like(self.explained_variance_)

        self.fitted_ = True
        return self

    @torch.no_grad()
    def transform(self, X: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Project **pre-centered** X onto learned components (optionally in chunks).

        Args:
            X: (N, D) tensor (already centered the same way as data used in fit).
            batch_size: if provided, processes X in chunks of this size.

        Returns:
            (N, K) tensor on the same device as X.
        """
        if not self.fitted_:
            raise RuntimeError("PCA must be fit before calling transform().")
        if X.dim() != 2:
            raise ValueError("X must be 2D (N, D).")
        if X.shape[1] != self.components_.shape[0]:
            raise ValueError(
                f"X has D={X.shape[1]} features, but PCA was fit with D={self.components_.shape[0]}."
            )
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None.")

        # Ensure dtype matches PCA dtype; keep device as-is
        X = X.to(dtype=self.dtype, copy=False)
        N = X.shape[0]
        K = self.components_.shape[1]

        # Fast path: single big matmul
        comps = self.components_.to(device=X.device, dtype=X.dtype)
        if batch_size is None or batch_size >= N:
            return X.matmul(comps)

        # Batched path
        out = X.new_empty((N, K))  # same device/dtype as X
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            out[start:end] = X[start:end].matmul(comps)
        return out

    def explained_variance_ratio(self) -> torch.Tensor:
        """
        Ratio of variance explained by each selected component (length K), in [0, 1].
        Sum of this vector is the fraction of variance captured by the kept components.
        Multiply by 100 if you want percentage.
        """
        if not self.fitted_:
            raise RuntimeError("PCA must be fit before calling explained_variance().")
        return self.explained_ratio_

    def explained_variance_values(self) -> torch.Tensor:
        """
        Raw variances (eigenvalues) for each selected component (length K).
        """
        if not self.fitted_:
            raise RuntimeError("PCA must be fit before calling explained_variance_values().")
        return self.explained_variance_

import torch
import torch.nn as nn
from typing import Optional

class StandardScaler(nn.Module):
    """
    Standardize features by removing the mean and scaling to unit variance.

    - Works on CPU/CUDA (follows the device of your tensors/module).
    - Numerically-stable streaming stats via Welford/Chan.
    - `fit` uses batched passes; `partial_fit` updates stats incrementally.
    - `transform` defaults to a single big batch (fast on GPU); can also chunk.
    - Population variance (ddof=0), matching sklearn's StandardScaler default.

    Args:
        dtype (torch.dtype): dtype for internal computation and stored stats.
        with_mean (bool): center data.
        with_std (bool): scale by std (population).
        eps (float): numerical floor to avoid division by zero.
    """

    def __init__(
        self,
        dtype: torch.dtype = torch.float32,
        with_mean: bool = True,
        with_std: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.dtype = dtype
        self.with_mean = bool(with_mean)
        self.with_std = bool(with_std)
        self.eps = float(eps)

        # Running statistics (true stats; flags are applied only at transform time)
        self.register_buffer("mean_", None)   # (D,)
        self.register_buffer("m2_", None)     # (D,) sum of squared deviations
        self.register_buffer("var_", None)    # (D,) population variance = m2 / n
        self.register_buffer("scale_", None)  # (D,) std = sqrt(var + eps)
        self.register_buffer("n_samples_seen_", torch.tensor(0, dtype=torch.long))

        self.fitted_ = False

    def _ensure_initialized(self, D: int, device: torch.device):
        """Allocate buffers on-demand with correct shape/device."""
        if self.mean_ is None:
            self.mean_ = torch.zeros(D, device=device, dtype=self.dtype)
            self.m2_   = torch.zeros(D, device=device, dtype=self.dtype)
            self.var_  = torch.zeros(D, device=device, dtype=self.dtype)
            self.scale_= torch.ones (D, device=device, dtype=self.dtype)
            self.n_samples_seen_ = torch.tensor(0, device=device, dtype=torch.long)

    @torch.no_grad()
    def partial_fit(self, X: torch.Tensor) -> "StandardScaler":
        """
        Incrementally update scaler statistics with a mini-batch.

        Args:
            X: (B, D) tensor.

        Returns:
            self
        """
        if X.dim() != 2:
            raise ValueError("X must be 2D (B, D).")
        B, D = X.shape
        if B < 1:
            return self  # nothing to do

        device = X.device
        X = X.to(dtype=self.dtype, copy=False)
        self._ensure_initialized(D, device)

        # Per-batch stats
        batch_mean = X.mean(dim=0)  # (D,)
        # Sum of squared deviations within the batch: Σ (x - mean)^2
        # Use the identity: sum((x - mean)^2) = sum(x^2) - B * mean^2
        sumsq = (X * X).sum(dim=0)
        batch_m2 = sumsq - batch_mean * batch_mean * float(B)

        n_old = int(self.n_samples_seen_.item())
        if n_old == 0:
            # First batch initializes the stats
            self.mean_.copy_(batch_mean)
            self.m2_.copy_(batch_m2)
            self.n_samples_seen_.fill_(B)
        else:
            # Merge old and batch stats (Chan's parallel algorithm)
            n_new = n_old + B
            delta = batch_mean - self.mean_            # (D,)
            self.mean_ += delta * (float(B) / float(n_new))
            self.m2_ += batch_m2 + delta * delta * (float(n_old) * float(B) / float(n_new))
            self.n_samples_seen_.fill_(n_new)

        # Update derived quantities
        n = max(int(self.n_samples_seen_.item()), 1)
        self.var_.copy_(torch.clamp(self.m2_ / float(n), min=0.0))
        self.scale_.copy_(torch.sqrt(self.var_ + self.eps))

        self.fitted_ = True
        return self

    @torch.no_grad()
    def fit(self, X: torch.Tensor, batch_size: int = 64) -> "StandardScaler":
        """
        Fit scaler on X using batched streaming stats (calls partial_fit on chunks).

        Args:
            X: (N, D) tensor on any device.

        Returns:
            self
        """
        if X.dim() != 2:
            raise ValueError("X must be 2D (N, D).")
        N, D = X.shape
        if N < 1:
            raise ValueError("Need at least 1 sample to fit StandardScaler.")
        device = X.device
        X = X.to(dtype=self.dtype, copy=False)

        # Reset stats
        self.mean_ = None
        self.m2_ = None
        self.var_ = None
        self.scale_ = None
        self.n_samples_seen_ = torch.tensor(0, device=device, dtype=torch.long)
        self.fitted_ = False

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            self.partial_fit(X[start:end])

        return self

    @torch.no_grad()
    def transform(self, X: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Standardize X with learned stats.

        Args:
            X: (N, D) tensor.
            batch_size: if None (default), transform in one go; otherwise stream.

        Returns:
            (N, D) standardized tensor on same device as X.
        """
        if not self.fitted_:
            raise RuntimeError("StandardScaler must be fit before calling transform().")
        if X.dim() != 2:
            raise ValueError("X must be 2D (N, D).")
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"X has D={X.shape[1]} features, but scaler was fit with D={self.mean_.shape[0]}.")

        X = X.to(dtype=self.dtype, copy=False)
        device = X.device

        mean = self.mean_.to(device=device, dtype=self.dtype) if self.with_mean else torch.zeros_like(self.mean_, device=device, dtype=self.dtype)
        scale = self.scale_.to(device=device, dtype=self.dtype) if self.with_std else torch.ones_like(self.scale_, device=device, dtype=self.dtype)

        if batch_size is None:
            return (X - mean) / scale

        outs = []
        N = X.shape[0]
        for start in range(0, N, int(batch_size)):
            end = min(start + int(batch_size), N)
            chunk = (X[start:end] - mean) / scale
            outs.append(chunk)
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def inverse_transform(self, X: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Revert standardization: X * scale + mean.

        Args:
            X: (N, D) tensor in standardized space.
            batch_size: optional chunk size.

        Returns:
            (N, D) tensor in original feature scale.
        """
        if not self.fitted_:
            raise RuntimeError("StandardScaler must be fit before calling inverse_transform().")
        if X.dim() != 2:
            raise ValueError("X must be 2D (N, D).")
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"X has D={X.shape[1]} features, but scaler was fit with D={self.mean_.shape[0]}.")

        X = X.to(dtype=self.dtype, copy=False)
        device = X.device

        mean = self.mean_.to(device=device, dtype=self.dtype) if self.with_mean else torch.zeros_like(self.mean_, device=device, dtype=self.dtype)
        scale = self.scale_.to(device=device, dtype=self.dtype) if self.with_std else torch.ones_like(self.scale_, device=device, dtype=self.dtype)

        if batch_size is None:
            return X * scale + mean

        outs = []
        N = X.shape[0]
        for start in range(0, N, int(batch_size)):
            end = min(start + int(batch_size), N)
            outs.append(X[start:end] * scale + mean)
        return torch.cat(outs, dim=0)

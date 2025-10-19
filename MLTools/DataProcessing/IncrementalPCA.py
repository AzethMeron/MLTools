
import torch


class IncrementalPCA:
    """
    Pure-PyTorch Incremental PCA with tensor I/O, dtype & device support.

    API parity with sklearn.decomposition.IncrementalPCA (most-used parts):
      - __init__(n_components=None, whiten=False, batch_size=None)
      - fit(X), partial_fit(X), fit_transform(X), transform(X), inverse_transform(Z)
      - get_covariance(), get_precision()
    Attributes (after fit/partial_fit):
      - components_ : (n_components, n_features)
      - explained_variance_ : (n_components,)
      - explained_variance_ratio_ : (n_components,)
      - singular_values_ : (n_components,)
      - mean_ : (n_features,)
      - var_ : (n_features,)  # feature variances
      - noise_variance_ : scalar (float)
      - n_components_ : int
      - n_features_ : int
      - n_samples_seen_ : int

    Implementation note:
      We maintain exact first/second moments (S1 = sum x, S2 = sum x^T x).
      After each partial_fit, we recompute eigenpairs of the current covariance.
      This is O(F^3) per update (F = n_features), but extremely robust and simple.
      For very large F (e.g., > 2k), consider reducing dimensionality first.

    Parameters
    ----------
    n_components : int | None
      If None, keep all components.
    whiten : bool
      If True, `transform` divides by sqrt(explained_variance_ + eps).
    batch_size : int | None
      For `fit(X)` only; if set and X is a 2D tensor, we chunk rows by this size.
    dtype : torch.dtype
      Internal dtype for model parameters (default: torch.float32).
    device : torch.device | str | None
      Device to host model parameters/accumulators. If None, inferred from first input.
    acc_dtype : torch.dtype
      Accumulator precision for S1/S2 (default: torch.float64 for stability).
    eps : float
      Numerical epsilon for whitening.
    """

    def __init__(self,
                 n_components: int | None = None,
                 whiten: bool = False,
                 batch_size: int | None = None,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device | str | None = None,
                 acc_dtype: torch.dtype = torch.float64,
                 eps: float = 1e-9):
        self.n_components = n_components
        self.whiten = bool(whiten)
        self.batch_size = batch_size
        self._dtype = dtype
        self._device = torch.device(device) if device is not None else None
        self._acc_dtype = acc_dtype
        self.eps = float(eps)

        # Lazy-initialized on first partial_fit:
        self.n_features_ = None
        self.n_samples_seen_ = 0

        # Accumulators (set after first batch)
        self._S1 = None   # (F,) sum(x)
        self._S2 = None   # (F,F) sum(x^T x)

        # Model attributes (populated after eigenupdate)
        self.n_components_ = None
        self.components_ = None                 # (K, F)
        self.explained_variance_ = None         # (K,)
        self.explained_variance_ratio_ = None   # (K,)
        self.singular_values_ = None            # (K,)
        self.mean_ = None                       # (F,)
        self.var_ = None                        # (F,)
        self.noise_variance_ = None             # scalar

    # ------------------------- utilities -------------------------

    def _infer_device_dtype(self, X: torch.Tensor):
        if self._device is None:
            self._device = X.device
        # ensure dtype for params; we allow different input dtype but cast as needed

    def _ensure_accumulators(self, n_features: int):
        dev = self._device
        acc = self._acc_dtype
        if self._S1 is None:
            self._S1 = torch.zeros(n_features, device=dev, dtype=acc)
            self._S2 = torch.zeros(n_features, n_features, device=dev, dtype=acc)

    @staticmethod
    def _check_2d(X: torch.Tensor) -> torch.Tensor:
        if not isinstance(X, torch.Tensor):
            raise TypeError("X must be a torch.Tensor")
        if X.dim() != 2:
            raise ValueError(f"X must be 2D (n_samples, n_features), got shape {tuple(X.shape)}")
        return X

    def _covariance_from_moments(self):
        """Return (mean, Cov, total_var) with Cov as *population* covariance (divide by N)."""
        N = self.n_samples_seen_
        if N <= 0:
            raise RuntimeError("No samples seen; call partial_fit first.")
        mean = self._S1 / N                                  # (F,)
        E_xx = self._S2 / N                                  # (F,F)
        Cov = E_xx - torch.outer(mean, mean)                 # (F,F)
        total_var = torch.trace(Cov)                         # scalar
        return mean, Cov, total_var

    def _eigendecompose_update(self):
        """Update components_ etc from current accumulators."""
        with torch.no_grad():
            mean, Cov, total_var = self._covariance_from_moments()

            # eigh on (F,F), returns ascending
            evals, evecs = torch.linalg.eigh(Cov)

            # sort descending
            idx = torch.argsort(evals, descending=True)
            evals = evals[idx]
            evecs = evecs[:, idx]

            F = evals.numel()
            if self.n_components is None:
                K = F
            else:
                K = int(self.n_components)
                if not (1 <= K <= F):
                    raise ValueError(f"n_components must be in [1, {F}] or None, got {K}")

            # Top-K
            evals_K = evals[:K].to(self._dtype)
            comps_K = evecs[:, :K].T.to(self._dtype)  # (K, F)

            # Attributes
            self.n_components_ = K
            self.components_ = comps_K.to(self._device, dtype=self._dtype)           # (K,F)
            self.explained_variance_ = evals_K.to(self._device, dtype=self._dtype)   # (K,)

            denom = (total_var.to(evals_K) + torch.finfo(evals_K.dtype).eps)
            self.explained_variance_ratio_ = (evals_K / denom).to(self._device)

            N = self.n_samples_seen_
            # Align with sklearn: singular_values_ = sqrt((n_samples - 1) * explained_variance_)
            svals = torch.sqrt(torch.clamp(self.explained_variance_, min=0) * max(N - 1, 1))
            self.singular_values_ = svals.to(self._device, dtype=self._dtype)

            # Feature stats
            self.mean_ = mean.to(self._device, dtype=self._dtype)
            # Unbiased per-feature variance estimate (≈ sklearn’s PCA.var_)
            if N > 1:
                var_unbiased = torch.diag(Cov) * (N / (N - 1))
            else:
                var_unbiased = torch.diag(Cov)
            self.var_ = var_unbiased.to(self._device, dtype=self._dtype)

            # Residual/noise variance (average of discarded eigenvalues)
            if K < F:
                resid = evals[K:].mean()
                self.noise_variance_ = resid.to(self._device, dtype=self._dtype)
            else:
                self.noise_variance_ = torch.zeros((), device=self._device, dtype=self._dtype)

    # ------------------------- core API -------------------------

    @torch.no_grad()
    def partial_fit(self, X: torch.Tensor):
        """
        Incrementally update the model with a batch X: (n_samples, n_features).
        After each call, components_ and related attributes are refreshed.
        """
        X = self._check_2d(X)
        self._infer_device_dtype(X)

        # Initialize feature count & accumulators
        n_features = X.shape[1]
        if self.n_features_ is None:
            self.n_features_ = int(n_features)
            self._ensure_accumulators(self.n_features_)
        elif self.n_features_ != int(n_features):
            raise ValueError(f"Feature dimension changed: was {self.n_features_}, got {n_features}")

        # Move to model device and accumulate in acc dtype
        Xd = X.to(self._device, dtype=self._acc_dtype, non_blocking=True)

        # Update moments
        self._S1 += Xd.sum(dim=0)               # (F,)
        self._S2 += Xd.T @ Xd                   # (F,F)
        self.n_samples_seen_ += X.shape[0]

        # Update eigen model
        self._eigendecompose_update()
        return self

    @torch.no_grad()
    def fit(self, X: torch.Tensor):
        """
        Fit from a 2D tensor. If batch_size is set, stream rows in chunks.
        """
        X = self._check_2d(X)
        if self.batch_size is None:
            return self.partial_fit(X)
        else:
            bs = int(self.batch_size)
            for i in range(0, X.shape[0], bs):
                self.partial_fit(X[i:i+bs])
            return self

    @torch.no_grad()
    def fit_transform(self, X: torch.Tensor) -> torch.Tensor:
        """Fit the model and return the transformed data."""
        self.fit(X)
        return self.transform(X)

    @torch.no_grad()
    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Project X onto the principal component space. Returns a tensor on X.device/dtype."""
        if self.components_ is None:
            raise RuntimeError("Model is not fitted yet. Call fit/partial_fit first.")
        X = self._check_2d(X)
        dev_in, dt_in = X.device, X.dtype

        # center on model device
        Xc = (X.to(self._device, dtype=self._dtype, non_blocking=True) -
              self.mean_.to(self._device))
        Z = Xc @ self.components_.T  # (N,K)

        if self.whiten:
            Z = Z / torch.sqrt(self.explained_variance_ + self.eps)

        return Z.to(dev_in, dtype=dt_in)

    @torch.no_grad()
    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from principal components. Returns on Z.device/dtype."""
        if self.components_ is None:
            raise RuntimeError("Model is not fitted yet.")
        if Z.dim() != 2 or Z.shape[1] != self.n_components_:
            raise ValueError(f"Z must have shape (n_samples, n_components={self.n_components_})")

        dev_in, dt_in = Z.device, Z.dtype
        Zd = Z.to(self._device, dtype=self._dtype, non_blocking=True)

        if self.whiten:
            Zd = Zd * torch.sqrt(self.explained_variance_ + self.eps)

        Xr = Zd @ self.components_ + self.mean_
        return Xr.to(dev_in, dtype=dt_in)

    # ------------------------- covariance / precision -------------------------

    @torch.no_grad()
    def get_covariance(self) -> torch.Tensor:
        """
        Reconstruct the feature covariance:
            C ≈ components_.T diag(explained_variance_) components_ + noise_variance_ * I
        """
        if self.components_ is None:
            raise RuntimeError("Model is not fitted yet.")
        K, F = self.components_.shape
        C = (self.components_.T * self.explained_variance_) @ self.components_
        if self.noise_variance_ is not None and self.noise_variance_.numel() == 1:
            C = C + self.noise_variance_ * torch.eye(F, device=C.device, dtype=C.dtype)
        return C

    @torch.no_grad()
    def get_precision(self) -> torch.Tensor:
        """Return the inverse covariance (precision). Uses pinv for stability."""
        C = self.get_covariance()
        return torch.linalg.pinv(C)

    # ------------------------- utilities / reset -------------------------

    @torch.no_grad()
    def reset(self):
        """Reset the estimator to its initial, unfitted state."""
        self.n_features_ = None
        self.n_samples_seen_ = 0
        self._S1 = None
        self._S2 = None
        self.n_components_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.var_ = None
        self.noise_variance_ = None

    def __repr__(self):
        name = self.__class__.__name__
        return (f"{name}(n_components={self.n_components}, whiten={self.whiten}, "
                f"batch_size={self.batch_size}, dtype={self._dtype}, device={self._device}, "
                f"acc_dtype={self._acc_dtype})")
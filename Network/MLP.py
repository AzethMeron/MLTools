import math
from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm.auto import tqdm
import warnings


class MLP(nn.Module):
    """
    MLP with sklearn-like .fit() using tqdm progress bars, plus .predict() and .predict_proba().

    All tensors created by the model use the dtype specified in the constructor (default torch.float32).

    Problem types:
      - 'binary'     : y shape (N,) in {0,1} or (N,1) float in {0,1}
                       -> sigmoid, BCEWithLogits by default
      - 'multiclass' : y is class indices (N,) or one-hot (N, C>=2, rows sum to 1)
                       -> softmax, CrossEntropy by default
      - 'multilabel' : y multi-hot (N, C), rows need not sum to 1
                       -> sigmoid, BCEWithLogits by default

    Notes:
      - If you call inference before .fit(), you MUST have provided out_features in the ctor,
        otherwise the output head doesn't exist and inference will raise a friendly error.
      - Safe default mapping from logits->probs:
          * multiclass -> softmax
          * otherwise  -> sigmoid (binary & multilabel safe)
      - ReduceLROnPlateau: when constructed via class/factory, mode defaults to 'max' for F1/AUC.
        When passing a prebuilt instance, a warning is emitted if mode looks mismatched.
      - Validation can be batched to avoid OOM (val_batch_size).
      - Optional learning of multiclass thresholds (off by default; argmax is standard).
      - Supports (N,) index labels for multiclass training and metrics.
      - Optional pos_weight for BCEWithLogits to handle imbalance; also honored for
        F.binary_cross_entropy_with_logits via a wrapper.
      - Inference uses torch.inference_mode().
      - Tracks training history per epoch.

    Implemented robustness:
      - predict_logits/predict_proba/_predict_proba_batched toggle eval() and restore training mode.
      - y_val one-hot now moved to device; binary y_val handled as (N,1) float.
      - CrossEntropy training path accepts one-hot by converting to indices.

    Tips:
      - When use_batchnorm=True, very small batch sizes can destabilize BN statistics during training.
        Consider larger batches or disabling BN in such cases.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: List[int],
        activation: nn.Module = nn.ReLU(),
        out_features: Optional[int] = None,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        problem_type: Optional[str] = None,  # "binary" | "multiclass" | "multilabel"
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_features = list(hidden_features)
        self.activation_template = activation
        self.out_features = out_features
        self.use_batchnorm = bool(use_batchnorm)
        self.dropout = float(dropout)
        self.model_dtype = dtype  # preferred dtype for all model tensors

        layers: List[nn.Module] = []
        last = self.in_features
        for h in self.hidden_features:
            layers.append(nn.Linear(last, h, dtype=dtype))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(h, dtype=dtype))
            layers.append(deepcopy(self.activation_template))
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            last = h
        self.backbone = nn.Sequential(*layers)

        # Head can be deferred until .fit()
        self.head: Optional[nn.Linear] = (
            nn.Linear(last, self.out_features, dtype=dtype) if self.out_features is not None else None
        )

        # Learned/declared after .fit() or ctor
        self.problem_type: Optional[str] = problem_type
        self.num_classes: Optional[int] = self.out_features

        # Persist thresholds as a buffer so they save/move with state_dict / .to(...)
        self.register_buffer("best_thresholds", torch.tensor([], dtype=self.model_dtype), persistent=True)

        self._fitted: bool = False

        # Optional training history
        self.history: List[Dict[str, float]] = []

    def _ensure_head(self, out_features: int):
        if self.head is None:
            last = self.hidden_features[-1] if self.hidden_features else self.in_features
            self.head = nn.Linear(last, out_features, dtype=self.model_dtype)  # registers module
            self.out_features = out_features
            self.num_classes = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x.to(dtype=self.model_dtype))
        if self.head is None:
            raise RuntimeError(
                "Output head not initialized. Pass out_features in the constructor or call .fit(...) to infer it."
            )
        return self.head(x)  # logits

    @torch.inference_mode()
    def predict_logits(self, X: torch.Tensor) -> torch.Tensor:
        """Return raw logits on the model's device; uses eval() during the pass and restores mode."""
        if self.head is None:
            raise RuntimeError(
                "Output head not initialized. Pass out_features in the constructor or call .fit(...) to infer it."
            )
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            return self.forward(X.to(device=device, dtype=self.model_dtype))
        finally:
            if was_training:
                self.train()

    @torch.inference_mode()
    def predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """
        Inference that returns probabilities of shape (N, C).
        Automatically moves X to the model's device.

        Safety: If problem_type is unknown (e.g., out_features set but not fit yet),
        - If problem_type == 'multiclass': softmax
        - Else: sigmoid (covers binary and multilabel safely)

        Requires a defined output head: either constructed via ctor (out_features) or learned via .fit().
        """
        if self.head is None:
            raise RuntimeError(
                "Output head not initialized. Pass out_features in the constructor or call .fit(...) to infer it."
            )
        device = next(self.parameters()).device
        X = X.to(device=device, dtype=self.model_dtype)

        was_training = self.training
        self.eval()
        try:
            logits = self.forward(X)
            probs = self._logits_to_probs(logits)
            return probs
        finally:
            if was_training:
                self.train()

    @torch.inference_mode()
    def predict(
        self,
        X: torch.Tensor,
        force_single_label: bool = False,
    ) -> torch.Tensor:
        """
        Inference with automatic per-class thresholding.
        Returns (N, C) predictions as 0/1 floats by default.
        If force_single_label=True and problem_type == 'multiclass', returns strict one-hot via argmax.
        """
        probs = self.predict_proba(X)

        # Thresholds (buffer may be empty or mismatched before/after fitting)
        C = probs.size(1)
        if self.best_thresholds.numel() != C:
            thr = torch.full((C,), 0.5, device=probs.device, dtype=self.model_dtype)
        else:
            thr = self.best_thresholds.to(probs.device, dtype=self.model_dtype)

        if self.problem_type == "multiclass" and (force_single_label or not self._learn_multiclass_thresholds_default()):
            preds = self._argmax_one_hot(probs)
        else:
            preds = (probs >= thr.unsqueeze(0)).to(self.model_dtype)
            if self.problem_type == "multiclass":
                # Ensure at least one class chosen for multiclass
                none_selected = preds.sum(dim=1) == 0
                if none_selected.any():
                    preds[none_selected] = self._argmax_one_hot(probs[none_selected])

        return preds

    # Backward compatibility alias
    transform = predict

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_val: Optional[torch.Tensor] = None,
        X_val: Optional[torch.Tensor] = None,
        batch_size: int = 64,
        val_batch_size: Optional[int] = None,
        optimizer: Union[torch.optim.Optimizer, Callable] = torch.optim.Adam,
        epochs: int = 100,
        criterion: Union[nn.Module, Callable] = nn.CrossEntropyLoss,
        metric: str = "f1",                       # 'f1' or 'auc'
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        nesterov: bool = False,
        shuffle: bool = True,
        patience: Optional[int] = None,
        grad_clip: Optional[float] = None,
        scheduler: Optional[Union[torch.optim.lr_scheduler._LRScheduler, Callable]] = None,
        scheduler_kwargs: Optional[dict] = None,
        scheduler_step: str = "epoch",  # 'epoch' | 'batch' (ReduceLROnPlateau handled automatically)
        verbose: bool = True,
        seed: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        # New knobs:
        learn_multiclass_thresholds: bool = False,   # by default use argmax for multiclass
        pos_weight: Optional[torch.Tensor] = None,   # for BCEWithLogitsLoss (multilabel/binary)
    ) -> "MLP":
        """
        Train with tqdm progress bars (disable with verbose=False).

        Notes/fixes:
        - Validation runs in eval() mode (Dropout off, BatchNorm running stats).
        - DataLoader uses CPU tensors; batches are moved to the model device.
        - best_thresholds is a buffer and best weights are restored at the end.
        - Supports ReduceLROnPlateau (step(metric)) and optional per-batch schedulers via scheduler_step='batch'.
        - Accepts (N,) index labels for multiclass and (N,) {0,1} for binary.
        - Optional validation batching via val_batch_size.
        - Optional pos_weight for BCEWithLogitsLoss (and functional BCEWithLogits) to handle class imbalance.
        """
        if seed is not None:
            torch.manual_seed(int(seed))

        # ---- Label normalization & head/problem inference ----
        def _is_binary_vector(vec: torch.Tensor) -> bool:
            # Accept floats/ints; allow only values in {0,1}
            if vec.numel() == 0:
                return False
            u = torch.unique(vec.detach().to(torch.int64))
            return (u.numel() <= 2) and torch.all((u == 0) | (u == 1))

        if y.dim() == 1:
            # Decide between binary vs multiclass indices
            want_binary = (
                (self.problem_type == "binary") or
                (self.out_features == 1) or
                _is_binary_vector(y)
            )
            if want_binary:
                # Binary: BCE with single logit head
                self._ensure_head(1)
                y_is_indices = False
                y_oh = y.to(torch.float32).view(-1, 1).to(dtype=self.model_dtype)
                inferred_problem = "binary"
            else:
                # Multiclass indices
                y_is_indices = True
                if self.out_features is None:
                    inferred_classes = int(y.max().item()) + 1
                    self._ensure_head(inferred_classes)
                else:
                    self._ensure_head(self.out_features)
                C = int(self.out_features)
                y_oh = F.one_hot(y.to(torch.int64), num_classes=C).to(dtype=self.model_dtype)
                inferred_problem = "multiclass"
        elif y.dim() == 2:
            y_is_indices = False
            N, C = y.shape
            self._ensure_head(C)
            y_oh = y.to(torch.float32).to(dtype=self.model_dtype)
            inferred_problem = self._infer_problem_type(y_oh)
        else:
            raise AssertionError("y must be (N,) class indices/(binary {0,1}) or (N, C) one-hot / multi-hot")

        # Problem-type consistency check (fix #1)
        if self.problem_type is None:
            self.problem_type = inferred_problem
        elif self.problem_type != inferred_problem:
            warnings.warn(
                f"Provided problem_type='{self.problem_type}' but data implies '{inferred_problem}'. "
                "Proceeding with provided problem_type."
            )

        # Choose device (prefer GPU if any input is already on CUDA)
        if X.is_cuda:
            device = X.device
        elif y.is_cuda:
            device = y.device
        else:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        # ---- Validation tensors ----
        if X_val is not None and y_val is None:
            raise ValueError("Pass both X_val and y_val, or neither.")
        if X_val is None and y_val is not None:
            raise ValueError("Pass both X_val and y_val, or neither.")

        if X_val is not None:
            # Early shape check for 2D y_val vs model head (fix #2)
            if y_val.dim() == 2 and self.out_features is not None and y_val.shape[1] != self.out_features:
                raise ValueError(f"y_val has C={y_val.shape[1]} but model out_features={self.out_features}.")

            X_val = X_val.to(device=device, dtype=self.model_dtype)
            if y_val.dim() == 1:
                # If binary (by config or head), treat as (N,1) float; else one-hot indices.
                if (self.problem_type == "binary") or (self.out_features == 1) or _is_binary_vector(y_val):
                    y_val_oh = y_val.to(torch.float32).view(-1, 1).to(device=device, dtype=self.model_dtype)
                else:
                    y_val_oh = F.one_hot(
                        y_val.to(torch.int64), num_classes=self.out_features
                    ).to(dtype=self.model_dtype).to(device)
            else:
                y_val_oh = y_val.to(torch.float32).to(device=device, dtype=self.model_dtype)
        else:
            y_val_oh = None

        # num_classes always reflect head
        self.num_classes = self.out_features

        # Move the model to device BEFORE creating optimizer/scheduler
        self.to(device=device, dtype=self.model_dtype)

        # ---- Datasets & loaders (always CPU for DataLoader friendliness) ----
        X_cpu = X.detach().cpu()
        # Keep CE target as indices for training when appropriate; if one-hot/binary, keep as is and convert per-batch if CE.
        if y_is_indices:
            y_train_for_loss = y.detach().cpu()
        else:
            y_train_for_loss = y_oh.detach().cpu()

        train_ds = TensorDataset(X_cpu, y_train_for_loss)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        # ---- Loss selection ----
        loss_fn, loss_is_ce = self._prepare_loss(criterion)

        # If not multiclass, prefer BCEWithLogits (with optional pos_weight)
        if self.problem_type in ("binary", "multilabel"):
            if loss_is_ce:
                # override CE with BCEWithLogits
                if pos_weight is not None:
                    pos_weight = pos_weight.to(device=device, dtype=self.model_dtype)
                    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                else:
                    loss_fn = nn.BCEWithLogitsLoss()
                loss_is_ce = False
            else:
                # If functional BCEWithLogits was passed, honor pos_weight by wrapping
                if criterion is F.binary_cross_entropy_with_logits and pos_weight is not None:
                    pw = pos_weight.to(device=device, dtype=self.model_dtype)

                    def _bce_logits_with_pw(logits, target):
                        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)

                    loss_fn = _bce_logits_with_pw

        # ---- Optimizer ----
        if isinstance(optimizer, torch.optim.Optimizer):
            opt = optimizer
        else:
            opt_cls = optimizer
            if opt_cls is torch.optim.SGD:
                opt = opt_cls(
                    self.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov
                )
            else:
                opt = opt_cls(self.parameters(), lr=lr, weight_decay=weight_decay)

        # ---- Scheduler ----
        self.history.clear()
        metric_l = metric.lower()

        # Validate scheduler_step (fix #6)
        if scheduler_step not in {"epoch", "batch"}:
            raise ValueError("scheduler_step must be 'epoch' or 'batch'")

        if scheduler is None:
            sched = None
        elif isinstance(scheduler, (torch.optim.lr_scheduler._LRScheduler,
                                    torch.optim.lr_scheduler.ReduceLROnPlateau)):
            sched = scheduler
            # Hard-fail if instance optimizer mismatches (fix #5)
            opt_attr = getattr(sched, "optimizer", None)
            if opt_attr is not None and opt_attr is not opt:
                raise ValueError("Scheduler instance is bound to a different optimizer.")
        else:
            kwargs = {} if scheduler_kwargs is None else dict(scheduler_kwargs)
            # If user asked for ReduceLROnPlateau via factory, default to mode="max" for F1/AUC
            from torch.optim.lr_scheduler import ReduceLROnPlateau as _RLP
            if scheduler is _RLP:
                kwargs.setdefault("mode", "max" if metric_l in {"f1", "auc"} else "min")
            sched = scheduler(opt, **kwargs)

        # If user passed a ReduceLROnPlateau INSTANCE, warn on mode mismatch for maximized metrics
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            mode = getattr(sched, "mode", None)
            if metric_l in {"f1", "auc"} and mode not in {None, "max"}:
                warnings.warn(
                    "ReduceLROnPlateau is tracking a maximized metric (F1/AUC) but mode!='max'. "
                    "Consider setting mode='max'."
                )

        # Internal flag for whether to learn multiclass thresholds
        self._learn_multiclass_thresholds_flag = bool(learn_multiclass_thresholds)

        best_val_metric = -math.inf
        epochs_no_improve = 0
        best_state: Optional[dict] = None  # state_dict snapshot of best epoch

        epoch_iter = tqdm(
            range(1, epochs + 1),
            disable=not verbose,
            desc="Training",
            dynamic_ncols=True,
            leave=True,
        )

        N = X.shape[0]
        vbs = val_batch_size or batch_size

        for epoch in epoch_iter:
            self.train()
            running_loss = 0.0

            batch_iter = tqdm(
                train_loader,
                disable=not verbose,
                desc=f"Epoch {epoch}/{epochs}",
                dynamic_ncols=True,
                leave=False,
            )

            for xb_cpu, yb_cpu in batch_iter:
                xb = xb_cpu.to(device=device, dtype=self.model_dtype, non_blocking=pin_memory)

                opt.zero_grad(set_to_none=True)
                logits = self.forward(xb)

                if loss_is_ce:
                    # Accept either indices (1D) or one-hot (2D) and convert the latter to indices.
                    if yb_cpu.dim() == 2:
                        target = yb_cpu.argmax(dim=1).to(device=device, non_blocking=pin_memory).to(torch.int64)
                    else:
                        target = yb_cpu.to(device=device, non_blocking=pin_memory).to(torch.int64)
                    loss = loss_fn(logits, target)
                else:
                    # BCE path expects multi-hot/one-hot float (binary/multilabel)
                    target = yb_cpu.to(device=device, non_blocking=pin_memory).to(dtype=self.model_dtype)
                    # Warn if labels fall outside [0,1] (fix #3)
                    if not torch.all((target >= 0) & (target <= 1)):
                        warnings.warn("BCE path received targets outside [0,1]. Check your labels.")
                    loss = loss_fn(logits, target)

                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
                opt.step()

                # Optional per-batch scheduler stepping
                if sched is not None and scheduler_step == "batch" and not isinstance(
                    sched, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    sched.step()

                running_loss += float(loss.detach()) * xb.size(0)

            # Epoch-end scheduler step (non-plateau)
            if sched is not None and scheduler_step == "epoch" and not isinstance(
                sched, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                sched.step()

            # ---- Evaluation (strict eval mode) ----
            was_training = self.training
            self.eval()

            with torch.inference_mode():
                train_loss = running_loss / N

                # Decide eval tensors
                eval_X = X_val if X_val is not None else X.to(device=device, dtype=self.model_dtype)
                eval_y_oh = y_val_oh if y_val_oh is not None else y_oh.to(device=device, dtype=self.model_dtype)

                # Batched probabilities for memory safety
                probs_eval = self._predict_proba_batched(eval_X, batch_size=vbs, device=device)

                # Metric & thresholds
                if metric_l == "auc":
                    mval = self._macro_auc(probs_eval, eval_y_oh)
                    thr = self._best_thresholds_by_youden(probs_eval, eval_y_oh)
                else:
                    if self.problem_type == "multiclass" and not self._learn_multiclass_thresholds_flag:
                        preds_eval = self._argmax_one_hot(probs_eval)
                        mval = self._macro_f1(preds_eval, eval_y_oh)
                        # Thresholds are unused in this mode; keep a 0.5 default buffer
                        thr = torch.full((probs_eval.size(1),), 0.5, device=probs_eval.device, dtype=self.model_dtype)
                    else:
                        thr = self._best_thresholds_by_f1(probs_eval, eval_y_oh)
                        preds_eval = (probs_eval >= thr.unsqueeze(0)).to(eval_y_oh.dtype)
                        if self.problem_type == "multiclass":
                            none_sel = preds_eval.sum(1) == 0
                            if none_sel.any():
                                preds_eval[none_sel] = self._argmax_one_hot(probs_eval[none_sel])
                        mval = self._macro_f1(preds_eval, eval_y_oh)

                # ReduceLROnPlateau: step with metric
                if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    sched.step(float(mval))

                # Keep best thresholds, metric, and full model weights
                improved = mval > best_val_metric + 1e-12
                if improved:
                    best_val_metric = float(mval)

                    # Update buffer (clone to detach from graph)
                    self.best_thresholds = thr.detach().clone()

                    # Snapshot model state (includes buffers)
                    best_state = deepcopy(self.state_dict())

                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Log history (keep original casing behavior)
                self.history.append({"epoch": epoch, "loss": float(train_loss), metric.upper(): float(mval)})

            if was_training:
                self.train()

            if verbose:
                epoch_iter.set_postfix(loss=f"{train_loss:.4f}", **{metric.upper(): f"{float(mval):.4f}"})

            if patience is not None and epochs_no_improve >= patience:
                if verbose:
                    tqdm.write("Early stopping.")
                break

        # Restore best weights (and thresholds buffer) if we have them
        if best_state is not None:
            self.load_state_dict(best_state)

        # Remember how multiclass thresholds were handled for predict()
        self._learn_multiclass_thresholds_flag = bool(learn_multiclass_thresholds)

        self._fitted = True
        return self

    # ----- Internals -----

    def _learn_multiclass_thresholds_default(self) -> bool:
        # Helper for predict(): if problem_type is multiclass, should we use learned thresholds?
        return getattr(self, "_learn_multiclass_thresholds_flag", False)

    @staticmethod
    def _infer_problem_type(y: torch.Tensor) -> str:
        # y is (N, C) float (one-hot or multi-hot)
        C = y.size(1)
        if C == 1:
            return "binary"
        row_sums = y.sum(dim=1)
        ones = torch.ones_like(row_sums, dtype=row_sums.dtype, device=row_sums.device)
        if torch.all(torch.isclose(row_sums, ones, atol=1e-4, rtol=0.0)):
            return "multiclass"
        return "multilabel"

    def _logits_to_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Safer default when problem_type is None (e.g., out_features set but .fit() not called):
        - If problem_type == 'multiclass': softmax
        - Else: sigmoid (covers binary and multilabel safely)
        """
        ptype = self.problem_type
        if ptype == "multiclass":
            return torch.softmax(logits, dim=1)
        else:
            return torch.sigmoid(logits)

    @staticmethod
    def _argmax_one_hot(probs: torch.Tensor) -> torch.Tensor:
        idx = probs.argmax(dim=1)
        out = torch.zeros_like(probs)
        out.scatter_(1, idx.unsqueeze(1), 1.0)
        return out  # float 0/1

    @staticmethod
    def _safe_div(numer: torch.Tensor, denom: torch.Tensor) -> torch.Tensor:
        return numer / torch.clamp(denom, min=1e-12)

    @classmethod
    def _macro_f1(cls, preds01: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Guard: warn if non-binary inputs
        if torch.any((preds01 < 0) | (preds01 > 1)) or torch.any((y < 0) | (y > 1)):
            warnings.warn("macro_f1 received non-binary inputs; thresholding preds at 0.5 and y at >0.5.")

        # Work in boolean space to avoid float equality pitfalls
        preds_b = preds01 > 0.5
        y_b = y > 0.5

        tp = (preds_b & y_b).sum(dim=0).to(dtype=preds01.dtype)
        fp = (preds_b & ~y_b).sum(dim=0).to(dtype=preds01.dtype)
        fn = (~preds_b & y_b).sum(dim=0).to(dtype=preds01.dtype)
        prec = cls._safe_div(tp, tp + fp)
        rec = cls._safe_div(tp, tp + fn)
        f1 = cls._safe_div(2 * prec * rec, prec + rec).nan_to_num(0.0)
        return f1.mean()

    @classmethod
    def _auc_binary(
        cls,
        scores: torch.Tensor,
        targets01: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes AUC and Youden's J over thresholds implied by sorting scores descending.
        Returns:
            auc: scalar tensor (same dtype as scores)
            youdenJ: tensor aligned with scores sorted descending
            s_sorted: the scores sorted descending (aligns with youdenJ)
        """
        order = torch.argsort(scores, descending=True)
        s_sorted = scores[order]
        y_sorted = targets01[order].to(torch.int64)

        P = y_sorted.sum().to(dtype=scores.dtype)
        N = (y_sorted.numel() - y_sorted.sum()).to(dtype=scores.dtype)
        if P == 0 or N == 0:
            # Degenerate: no positives or no negatives
            auc = torch.tensor(0.5, dtype=scores.dtype, device=scores.device)
            youdenJ = torch.zeros_like(s_sorted, dtype=scores.dtype)
            return auc, youdenJ, s_sorted

        tp = torch.cumsum(y_sorted, dim=0).to(dtype=scores.dtype)
        fp = torch.cumsum(1 - y_sorted, dim=0).to(dtype=scores.dtype)

        tpr = tp / P
        fpr = fp / N

        tpr_curve = torch.cat([torch.tensor([0.0], device=scores.device, dtype=scores.dtype), tpr])
        fpr_curve = torch.cat([torch.tensor([0.0], device=scores.device, dtype=scores.dtype), fpr])
        auc = torch.trapz(tpr_curve, fpr_curve)

        youdenJ = tpr - fpr  # aligned with s_sorted (descending)
        return auc, youdenJ, s_sorted

    @classmethod
    def _macro_auc(cls, probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C = y.shape[1]
        aucs = []
        for c in range(C):
            s = probs[:, c]
            t = y[:, c]
            auc_c, _, _ = cls._auc_binary(s, t)
            aucs.append(auc_c)
        return torch.stack(aucs).mean()

    @classmethod
    def _best_thresholds_by_f1(cls, probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C = y.shape[1]
        thresholds = torch.empty(C, device=probs.device, dtype=probs.dtype)
        for c in range(C):
            p = probs[:, c]
            t = y[:, c].to(torch.int64)
            P = t.sum().to(dtype=probs.dtype)

            # Degenerate cases
            if P == 0:
                thresholds[c] = torch.tensor(1.0, device=probs.device, dtype=probs.dtype)
                continue
            if (t.numel() - int(P.item())) == 0:
                thresholds[c] = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
                continue

            order = torch.argsort(p, descending=True)
            p_sorted = p[order]
            t_sorted = t[order]

            cum_tp = torch.cumsum(t_sorted, dim=0).to(dtype=probs.dtype)
            n_samples = p.shape[0]  # fix #4
            idx = torch.arange(1, n_samples + 1, device=probs.device, dtype=probs.dtype)
            cum_fp = idx - cum_tp

            prec = cls._safe_div(cum_tp, cum_tp + cum_fp)
            rec = cls._safe_div(cum_tp, torch.clamp(P, min=probs.new_tensor(1.0)))
            f1 = cls._safe_div(2 * prec * rec, prec + rec).nan_to_num(0.0)

            best_i = int(torch.argmax(f1).item())
            thresholds[c] = p_sorted[best_i]
        return thresholds

    @classmethod
    def _best_thresholds_by_youden(cls, probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C = y.shape[1]
        thresholds = torch.empty(C, device=probs.device, dtype=probs.dtype)
        for c in range(C):
            s = probs[:, c]
            t = y[:, c]
            P = t.sum()

            # Degenerate cases
            if P == 0:
                thresholds[c] = torch.tensor(1.0, device=probs.device, dtype=probs.dtype)
                continue
            if (t.numel() - int(P.item())) == 0:
                thresholds[c] = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
                continue

            _, J, s_sorted = cls._auc_binary(s, t)  # J aligned with s_sorted (descending)
            best_i = int(torch.argmax(J).item())
            thresholds[c] = s_sorted[best_i]
        return thresholds

    @staticmethod
    def _prepare_loss(criterion: Union[nn.Module, Callable]):
        """
        Returns (loss_fn, is_cross_entropy)
        Supports: class, instance, functional, or zero-arg factory returning an nn.Module.
        """
        # Class (e.g., nn.CrossEntropyLoss)
        if isinstance(criterion, type) and issubclass(criterion, nn.Module):
            loss_fn = criterion()
            return loss_fn, isinstance(loss_fn, nn.CrossEntropyLoss)

        # Instance
        if isinstance(criterion, nn.Module):
            return criterion, isinstance(criterion, nn.CrossEntropyLoss)

        # Functional CE
        if criterion is F.cross_entropy:
            return criterion, True

        # Try zero-arg factory returning a Module
        is_factory = callable(criterion) and not isinstance(criterion, nn.Module)
        if is_factory:
            try:
                produced = criterion()
                if isinstance(produced, nn.Module):
                    return produced, isinstance(produced, nn.CrossEntropyLoss)
            except TypeError:
                # Not a zero-arg factory; fall through to generic callable
                pass

        # Generic callable loss (assume it takes (logits, target) directly)
        return criterion, False

    @torch.inference_mode()
    def _predict_proba_batched(self, X: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compute probabilities in batches to avoid OOM; toggles eval() and restores the previous mode."""
        was_training = self.training
        self.eval()
        try:
            N = X.shape[0]
            outs = []
            for i in range(0, N, batch_size):
                xb = X[i: i + batch_size].to(device=device, dtype=self.model_dtype)
                logits = self.forward(xb)
                outs.append(self._logits_to_probs(logits))
            return torch.cat(outs, dim=0)
        finally:
            if was_training:
                self.train()

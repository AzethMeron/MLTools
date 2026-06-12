# MLTools Codebase Audit — June 2026

Deep audit of every module under `MLTools/` for semantic and logical
correctness, followed by hardening and the introduction of a formal test
suite (`tests/`, 530+ tests — see `TESTING.md`). Every finding below is
covered by at least one regression test.

Severity legend: **C** = crash/data-corruption, **H** = wrong results,
**M** = robustness/compat, **L** = cosmetic/minor.

## Findings & fixes

### DataProcessing

| # | Sev | Module | Finding | Fix |
|---|-----|--------|---------|-----|
| 1 | H | `StandardScaler.partial_fit` | Despite the "Welford/Chan" docstring, the per-batch sum of squared deviations used the cancellation-prone identity `Σx² − B·mean²`. With `mean ≫ std` in float32 (e.g. mean 1e6, std 1) variances collapsed to 0 or went negative. | Two-pass centered computation `Σ(x − mean)²`, clamped at 0. Verified against a float64 reference at mean 1e6/std 1. |
| 2 | M | `StandardScaler.partial_fit` | Feeding a batch with a different feature count after initialization produced an obscure broadcast error. | Explicit `ValueError` with both dimensions named. Also: `batch_size <= 0` now rejected in `fit`. |
| 3 | H | `PCA` | `explained_ratio_` was a plain attribute, not the registered buffer the class declared (`explained_variance_values_` was registered but never written). Ratios were silently lost on `state_dict()` round-trips and did not follow `.to(device)`. | `explained_ratio_` is now a registered buffer; the dead buffer was removed; buffer/docs naming reconciled. |
| 4 | H | `IncrementalPCA.fit` | `fit()` did **not** reset accumulators — fitting twice doubled `n_samples_seen_` and blended stale moments, contradicting the advertised sklearn parity. | `fit()` calls `reset()` first; `partial_fit` remains cumulative. |
| 5 | H | `IncrementalPCA` | Raw-moment accumulation (`S1=Σx`, `S2=Σxxᵀ`) cancels catastrophically for `|mean| ≫ std` **even in float64** (found by an adversarial test: variance 0.73 instead of ~1.08 at mean 1e7). | Replaced with centered streaming moments (running mean + scatter matrix) merged via Chan's parallel update. |
| 6 | M | `IncrementalPCA.get_covariance` | Formula deviated from sklearn: noise was added on top of the full kept spectrum instead of `diag(max(λ−σ²,0))`. | Aligned with sklearn's formula. |
| 7 | C | `Letterbox.__call__` | Non-uniform `fill` (e.g. `(114, 0, 0)`) re-padded an **already padded** tensor and crashed on shape mismatch. The per-channel path was unreachable for uniform fills, so the bug was latent. | Rewritten: resized content is pasted onto a canvas pre-filled per channel. |
| 8 | M | `Letterbox` | Extreme aspect ratios could round the content size to 0 px (`F.interpolate` failure). No validation of `image_size`/`fill`. | Content size clamped to ≥ 1 px (mirrored in `transform_detections` to keep boxes aligned); constructor validation added. |
| 9 | H | `RandomJPEGCompression` | When `subsampling` was a list, the choice used the **global** `random` module instead of the instance's seeded RNG, breaking reproducibility. | Uses `self.random.choice`. |
| 10 | C | `RandomJPEGCompression` | RGBA/P/LA inputs crashed (`OSError: cannot write mode RGBA as JPEG`). | Non-JPEG-encodable modes round-trip through RGB and convert back (alpha becomes opaque — documented). Also: `assert` → `ValueError`, broken `__repr__` (printed a bound method) fixed, `p` validated. |
| 11 | C | `RandomScale` | `"area"` interpolation maps to `InterpolationMode.BOX`, which torchvision **cannot apply to tensors** — every use raised. | Routed through `F.interpolate(mode="area")`. Also: `min_scale/max_scale/p` validated; unknown algorithm names get a helpful `ValueError`. |
| 12 | C | `ParallelizedTransformer` | A transform raising inside a worker killed the worker process and `run()` **deadlocked forever** waiting for results; the instance was permanently poisoned. | Workers catch exceptions and ship them back; `run()` drains the queue, then raises `RuntimeError` carrying the worker traceback (original exception chained). Workers survive and the instance stays usable. `num_workers ≥ 1` validated. |

### Dataset

| # | Sev | Module | Finding | Fix |
|---|-----|--------|---------|-----|
| 13 | C | `DetectionDataset` | `import urllib` without `urllib.request` — `COCO.download()` worked only if some other module had imported the submodule first. | `import urllib.request`. |
| 14 | C/H | `DetectionDataset` | Augmentation parameter handling was type-fragile: `rotate=0` crashed (`randint(0, 0)`), `rotate=15.0` (float) and `translate_w=1` (int) were **silently ignored** by `isinstance` chains. | Unified `__sample_range` helper: `None` / scalar (int or float) → `U(−x, x)` / 2-tuple-or-list → `U(lo, hi)`; everything else raises. Scalar rotation now samples continuously instead of integer degrees. |
| 15 | C | `MNIST.download` | `url = url.rstrip("/") + "/" + fname` **mutated the base URL inside the loop** — the second and subsequent files were fetched from a garbage URL. | Separate `file_url` variable. |
| 16 | M | `CIFAR10.download` | `tarfile.extractall` without a filter is vulnerable to path-traversal archives. | `filter="data"` when available (Python ≥ 3.11.4). |
| 17 | C | `PreloadedDataset._ensure_pil` | `(1, H, W)` grayscale CHW tensors produced an `(H, W, 1)` array, which PIL rejects. | Squeezed to `(H, W)` → mode `L`. |
| 18 | C | `FlyingStuff.COCOPersonForeground` | `CocoDetection(..., download=True)` — that class has no `download` parameter; constructor always raised `TypeError`. | Argument removed (documented that data must exist locally). |
| 19 | L | `FlyingStuff` | Duplicated dead code after `return` in two factories; duplicate mid-file `import torch`; inner variable `I2` shadowing the rendered frame tensor. | Cleaned up / renamed. |

### Detections

| # | Sev | Module | Finding | Fix |
|---|-----|--------|---------|-----|
| 20 | C | `Detection.ToSupervision` | Referenced the undefined name `Detections` — every call raised `NameError`. | Calls `Detection.to_supervision`. |
| 21 | M | `Detection.from_supervision` | `supervision` allows `confidence`/`class_id` to be `None`; `zip(...)` then raised `TypeError`. | Defaults: confidence 1.0, class_id −1. |

### Network

| # | Sev | Module | Finding | Fix |
|---|-----|--------|---------|-----|
| 22 | C | `ResidualBlock` | With the default `norm_factory=None` and `in_channels != out_channels`, the skip path called `None(out_channels)` → `TypeError`. | Goes through `_make_norm` (AutoGroupNorm default), like every other block. |
| 23 | C | `AutoGroupNorm` | Small channel counts with `min_groups > 1` computed `natural = channels // pref = 0` and evaluated `channels % 0` → `ZeroDivisionError`. | `natural` floored at 1. |
| 24 | C | `Bottleneck` | Small `out_channels × reduction` truncated to **0 working channels** → invalid `Conv2d`. | `max(1, ...)`. |
| 25 | M | `Conv2dWithFeatures` | String paddings (`"same"`) crashed the feature path with an unpacking error. | Clear `NotImplementedError` (plain forward unaffected). |

### Utilities

| # | Sev | Module | Finding | Fix |
|---|-----|--------|---------|-----|
| 26 | H | `TrainingLoop.run` | `history[...]['test_metrics']` stored `SaveDump(train_metrics)` — **test metrics were never recorded**. | Stores `test_metrics`. |
| 27 | C/H | `TrainingLoop` / `DelayedStorage` | Batches beyond the last full `log_every_k` window were dropped; loaders with fewer than `log_every_k` batches reported loss 0.0 and crashed `torch.cat([])`. | `DelayedStorage.flush()` added; both epoch steps flush partial buffers; empty-epoch `cat` guarded. |
| 28 | H | `TrainingLoop.run` | First measured epoch set `best_val` without saving `best_path` — 1-epoch runs produced no best checkpoint; identity check `is math.inf` breaks after checkpoint resume. | Baseline epoch saves `best_path`; `math.isinf()` comparison. |
| 29 | M | `CosineAnnealingWithWarmup` | `total_epochs == warmup_epochs` → `CosineAnnealingLR(T_max=0)` → division by zero at step time. | Validated (`warmup ≥ 1`, `total > warmup`). |
| 30 | C | `Utilities.__init__` | Eagerly imported `Visualizer` (matplotlib) and `Webhooks` (requests), so `import MLTools` failed without the optional extras the README explicitly calls optional. | PEP 562 lazy module `__getattr__`; also exported `CosineAnnealingWithWarmup`. |
| 31 | H | `DiscordWebhook` | With file attachments, embeds/content were sent as plain form fields — Discord cannot parse embeds that way (requires `payload_json`). | Multipart sends `payload_json=json.dumps(body)`. |
| 32 | H | `Visualizer.images_grid` | `share_axes` used the condition `r*c > 0`, which excludes the whole first row and first column from axis sharing. | `(r + c) > 0`. |
| 33 | M | `Visualizer.draw_to_pil` | Non-tight path used `canvas.tostring_argb()`, removed in matplotlib 3.10. | `canvas.buffer_rgba()`. |
| 34 | L | `Tee` | `encoding` became `None` when the primary stream exposes `encoding=None` (e.g. `io.StringIO`), breaking consumers expecting a string; `write()` returned `None` instead of the character count. | `or "utf-8"` fallback; `write` returns the primary stream's count. |

## Behavioral changes worth knowing

- `TrainingLoop` now accounts for **all** batches in epoch losses/outputs
  (previously the trailing `< log_every_k` window was dropped). Long-run loss
  curves may differ very slightly; short runs become correct.
- `DetectionDataset(rotate=x)` with scalar `x` now samples continuous angles
  `U(−x, x)` instead of integer `randint(−x, x)`.
- `IncrementalPCA.fit()` resets state (sklearn semantics). Use `partial_fit`
  to accumulate.
- `import MLTools` no longer requires matplotlib/requests.

## Known limitations (documented, not changed)

- `Detection.IOU`/`NMS` operate on axis-aligned envelopes — not rotated-box
  IOU. `to_xyxy(anchor=...)`'s anchor parameter is reserved/unused.
- `Detection.Rescale` with non-uniform scaling does not preserve rotated
  rectangles exactly (fine for the axis-aligned case it is used for).
- `PCA`/`StandardScaler` `load_state_dict` requires the destination buffers
  to be allocated (fit once on dummy data of the right shape first).
- `FrameWriter` is hard-wired to the `avc1` fourcc and raises `IOError` where
  OpenH264 is unavailable (bundled binaries live in `codecs/`).
- Serialization helpers (`SaveBin`/`LoadBin`/...), `Metrics`, checkpoint
  loading (`weights_only=False`) and `TinyImageNetFeatures` use pickle —
  only load data you trust.
- `wip/` is intentionally outside the audit scope.

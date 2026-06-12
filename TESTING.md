# MLTools Test Suite

Formal pytest suite covering every subpackage. Tests are grouped in four
flavors throughout the files:

- **typical** — happy-path behavior and API contracts;
- **edge cases** — empty inputs, single samples, 1×1 images, constant data,
  degenerate aspect ratios, short dataloaders, prime channel counts, …;
- **randomized** — seeded fuzzing over shapes, magnitudes and configurations
  (`tests/test_stress.py` plus parametrized trials inside the unit files);
- **adversarial** — inputs designed to break the implementation: catastrophic
  cancellation magnitudes (mean ≫ std), corrupted dataset headers (wrong IDX
  magic, truncated pickle batches, bad `.flo` magic), JPEG-unencodable image
  modes, exceptions thrown inside worker processes, path-traversal-style
  concerns, removed-API compatibility (matplotlib ≥ 3.10).

## Setup

```bash
pip install torch torchvision          # match your platform/CUDA
pip install -r requirements.txt        # opencv, numpy, pillow, tqdm, supervision
pip install pytest matplotlib requests # test-only + optional extras
```

`matplotlib` and `requests` are optional for the library itself but needed to
run their test files (`test_visualizer.py`, `test_webhooks.py` — the webhook
tests mock the network and never make real requests).

## Running

```bash
pytest                       # everything (~530 tests, < 1 min on CPU)
pytest -m "not slow"         # skip multiprocessing + fuzzing batches (~3 s)
pytest -m slow               # only the stress/fuzz + multiprocessing tests
pytest tests/test_pca.py     # one module
pytest -k "letterbox and fill"  # by keyword
pytest -ra                   # show skip reasons (enabled by default via pytest.ini)
```

Configuration lives in `pytest.ini` (test discovery, markers, warning
filters).

### Markers

| Marker | Meaning |
|--------|---------|
| `slow` | Spawns worker processes, runs end-to-end training loops, or executes the 185-case fuzz battery. Deselect with `-m "not slow"` for a fast signal. |
| `network` | Reserved for tests needing internet. **None exist** — dataset loaders are tested against synthetic on-disk replicas of the official formats (IDX, CIFAR pickle batches, TinyImageNet layout, COCO JSON, `.flo`). |

### Environment-dependent skips

- `test_videoproc.py::test_writer_roundtrip` skips when no H.264 (`avc1`)
  encoder is available (e.g. `opencv-python-headless` without the bundled
  `codecs/`). The `FrameReader` tests use `mp4v` and run everywhere OpenCV
  can encode it.
- Everything runs on CPU; CUDA is never required.

## Layout

| File | Covers |
|------|--------|
| `tests/conftest.py` | Global deterministic seeding (every test), PIL image helpers. |
| `tests/test_standard_scaler.py` | StandardScaler: reference-vs-float64 stats, batch invariance, numerical-stability adversaries. |
| `tests/test_pca.py` | PCA: eigenstructure vs dense reference, orthonormality, buffer persistence, anisotropic scales. |
| `tests/test_incremental_pca.py` | IncrementalPCA: streaming == one-shot, whitening, covariance/precision, refit reset, huge-offset stability. |
| `tests/test_letterbox.py` | Letterbox geometry, per-channel fill, detection mapping (including a rendered-dot consistency check). |
| `tests/test_jpeg_transform.py`, `tests/test_random_scale.py` | Pixel transforms: determinism, probability gates, mode/interp matrix. |
| `tests/test_parallelized_transformer.py` | Worker ordering, exception propagation, reuse after failure (slow). |
| `tests/test_detections.py` | Box conversions, rotation/flip geometry, IOU/NMS, supervision bridge. |
| `tests/test_training_utils.py` | Counters, storages, scheduler, full TrainingLoop runs: learning, history schema, checkpoint/resume, early stopping. |
| `tests/test_utilities_misc.py` | Serialization, ClassMapper, Metrics, Cosmetics, Convertion, Tee. |
| `tests/test_layers.py`, `tests/test_fpn.py`, `tests/test_conv2d_with_features.py` | All network blocks: shapes, gradients, factory plumbing, feature-reconstruction identity. |
| `tests/test_preloaded_dataset.py` | Every storage mode, tensor/ndarray coercion paths. |
| `tests/test_dataset_loaders.py` | MNIST/CIFAR10/TinyImageNet against synthetic official-format files + corrupted variants. |
| `tests/test_detection_dataset.py` | DetectionDataset augmentation pipeline, COCO JSON parsing, save/load round-trip. |
| `tests/test_flyingstuff.py` | `.flo` IO, affine math, renderer flow correctness (analytic + photometric), motion bounds, loader. |
| `tests/test_visualizer.py`, `tests/test_webhooks.py`, `tests/test_videoproc.py` | Charts/grids/PIL export; mocked Discord payloads; real video round-trips. |
| `tests/test_package.py` | Import integrity, lazy optional dependencies, public name surface. |
| `tests/test_stress.py` | The fuzz battery (slow): 25 seeded trials per component over hostile magnitudes, sizes and configurations. |

## Conventions for new tests

1. Seed everything — the autouse fixture seeds `random`/`numpy`/`torch`, and
   randomized tests derive per-trial seeds from the parametrized index so
   failures reproduce exactly.
2. No network, no real datasets — synthesize the on-disk format in `tmp_path`.
3. Mark anything that forks processes or loops over many trials as `slow`.
4. When fixing a bug, encode it as a test whose docstring names the original
   failure mode (grep for "used to" to see the pattern).

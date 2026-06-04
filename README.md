# MLTools

A personal collection of machine-learning utilities for computer vision, built on PyTorch. It bundles dataset loaders, image transforms, network building blocks, a detection-box toolkit, and a set of training/IO helpers behind a single importable package.

The library is organized into five subpackages, all exposed under the top-level `MLTools` namespace:

```python
from MLTools import DataProcessing, Dataset, Detections, Network, Utilities
```

## Installation

Requires **Python ≥ 3.10**. PyTorch and torchvision are required but installed separately so you can match your CUDA/CPU build:

```bash
# 1. Install PyTorch for your platform (see https://pytorch.org)
pip install torch torchvision

# 2. Install MLTools straight from GitHub
pip install git+https://github.com/AzethMeron/MLTools.git
```

Or for local development:

```bash
git clone https://github.com/AzethMeron/MLTools.git
cd MLTools
pip install -e .
```

The packaged dependencies are `opencv-python`, `numpy`, `pillow`, `tqdm`, and `supervision`. A few modules also need `matplotlib` (`Visualizer`) and `requests` (`DiscordWebhook`); install them if you use those parts.

If you have trouble writing H.264/`avc1` video, copy the `codecs/` directory (bundled OpenH264 binaries for Linux and Windows) into your working directory.

## What's inside

### DataProcessing — transforms and feature preprocessing

GPU-capable preprocessing and augmentation built on torch ops.

- **StandardScaler** — feature standardization (mean/unit-variance) with numerically stable streaming statistics and `fit`/`partial_fit`. Device-agnostic, mirrors scikit-learn's default behavior.
- **PCA** — batched PCA in PyTorch that streams the Gram matrix for large datasets and runs on GPU. Assumes pre-centered input.
- **IncrementalPCA** — pure-PyTorch incremental PCA with API parity to `sklearn.decomposition.IncrementalPCA` (`partial_fit`, `transform`, `inverse_transform`, whitening).
- **Letterbox** — YOLO-style aspect-preserving resize with symmetric padding (PIL → PIL), backed by torch interpolation.
- **ParallelizedTransformer** — runs an arbitrary transform over a batch of PIL images across worker processes (`spawn` context). Intended for inference-time preprocessing rather than the training loop.
- **RandomJPEGCompression** — JPEG recompression augmentation with configurable quality and chroma subsampling.
- **RandomScale** — random down/up-rescaling augmentation, useful for simulating resolution loss and blur.

*Use for:* dimensionality reduction, feature normalization before classifiers, YOLO-style input pipelines, and degradation/robustness augmentation.

### Dataset — loaders and generators

PyTorch `Dataset` subclasses with built-in download helpers; image datasets return `(PIL.Image, one-hot)` pairs.

- **MNIST**, **CIFAR10**, **TinyImageNet** — classic image-classification datasets with one-call `download()`.
- **TinyImageNetFeatures** — precomputed ResNet-50 embeddings for TinyImageNet, for fast feature-space experiments.
- **PreloadedDataset** — eagerly caches another dataset in memory with selectable storage (raw PIL, zlib, PNG, or JPEG) to trade memory against access speed.
- **COCO / DetectionDataset** — COCO-format object-detection loading that converts annotations into `Detection` objects and integrates the `ClassMapper` and `Letterbox` utilities.
- **FlyingStuffDataset** — a FlyingChairs-style synthetic optical-flow generator that composes moving foreground cutouts over backgrounds and produces dense flow plus occlusion masks.

*Use for:* quick benchmarking, classification and detection training, optical-flow research, and reproducible in-memory dataset caching.

### Detections — bounding-box toolkit

- **Detection** — a rotated-box representation in `CXCYWH + angle` format with confidence and class id.

Supports conversions (`xyxy`, top-left `xywh`, `cxcywh`), geometric ops (rotate, horizontal/vertical flip, translate, rescale between image sizes), rotated-corner computation, IoU, per-class and class-agnostic **NMS**, drawing onto a PIL image, and round-trip interop with the `supervision` library.

*Use for:* post-processing detector outputs, dataset annotation handling, augmentation of labeled boxes, and visualization.

### Network — CNN building blocks

Composable `nn.Module` layers and feature-pyramid networks with factory-based norm/activation selection.

- **Layers** — `PureConv`, `NormActConv`, `ConvNormAct`, `PointwiseConv`, `DepthwiseConv`, attention blocks (`SqueezeExciteBlock`, `CBAM`), residual/CSP blocks (`ResidualBlock`, `Bottleneck`, `CSP1_X`, `CSP2_X`), and **AutoGroupNorm**, which picks a sensible group count automatically.
- **FPN** — `GenericFPN` (configurable sum/concat/CSP fusion, dynamic widths, extra levels) plus the `ClassicFPN` and `CspFPN` presets.
- **Conv2dWithFeatures** — a drop-in `nn.Conv2d` that can additionally return the unfolded receptive-field patches.

*Use for:* assembling detection/segmentation backbones and necks, experimenting with attention and CSP designs, and feature-extraction tasks.

### Utilities — training, IO, and visualization

- **TrainingLoop** — a full, subclassable training harness: checkpoint/resume, best-model tracking, early stopping, LR schedulers, custom per-batch logic, and pluggable metric computation. Comes with `RecentLossCounter`, `AverageLossCounter`, `TensorBatchStorage`, and `DelayedStorage`.
- **Serialization** — `SaveBin`/`LoadBin`/`SaveDump`/`LoadDump` (pickle + zlib) and `LoadOrCompute` for cached computation.
- **Metrics** — an append-only, compressed, persisted metric logger.
- **Visualizer** — Matplotlib charting helpers (line, multi-line, scatter, bar, image grids, loss curves) that can render directly to PIL.
- **VideoProc** — `FrameReader` and `FrameWriter` for batched video IO via OpenCV, returning PIL or BGR frames.
- **ClassMapper** — bidirectional mapping between dataset labels, contiguous class ids, and class names.
- **Convertion** — `PilToOpenCV` / `OpenCVToPil` conversions.
- **Cosmetics** — `CountParameters` (with scientific-notation formatting) and `SciNotation`.
- **Tee** — mirror `stdout`/`stderr` to log files via `TeeStdout` / `TeeStderr`.
- **DiscordWebhook** — post text, images, and embeds to a Discord channel (handy for long-run training notifications).

*Use for:* running and monitoring training, caching intermediate results, logging, video frame pipelines, and reporting.

## License

MIT — see [LICENSE](LICENSE).

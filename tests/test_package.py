"""Package-level integrity: imports, lazy optional deps, public surface."""
import importlib

import pytest


def test_top_level_import():
    import MLTools
    from MLTools import DataProcessing, Dataset, Detections, Network, Utilities  # noqa: F401


def test_utilities_lazy_exports():
    import MLTools.Utilities as U
    assert "Visualizer" in dir(U)
    assert "DiscordWebhook" in dir(U)
    # actually resolvable
    assert U.Visualizer is not None
    assert U.DiscordWebhook is not None


def test_utilities_unknown_attribute():
    import MLTools.Utilities as U
    with pytest.raises(AttributeError):
        U.DoesNotExist


@pytest.mark.parametrize("module,names", [
    ("MLTools.DataProcessing", ["StandardScaler", "PCA", "IncrementalPCA", "Letterbox",
                                 "ParallelizedTransformer", "RandomJPEGCompression", "RandomScale"]),
    ("MLTools.Dataset", ["TinyImageNet", "PreloadedDataset", "MNIST", "CIFAR10",
                          "TinyImageNetFeatures", "COCO", "DetectionDataset", "FlyingStuffDataset"]),
    ("MLTools.Detections", ["Detection"]),
    ("MLTools.Network", ["AutoGroupNorm", "PureConv", "NormActConv", "ConvNormAct",
                          "PointwiseConv", "DepthwiseConv", "SqueezeExciteBlock", "CBAM",
                          "ResidualBlock", "Bottleneck", "CSP1_X", "CSP2_X",
                          "GenericFPN", "ClassicFPN", "CspFPN", "Conv2dWithFeatures"]),
    ("MLTools.Utilities", ["SaveBin", "LoadBin", "SaveDump", "LoadDump", "LoadOrCompute",
                            "SciNotation", "CountParameters", "PilToOpenCV", "OpenCVToPil",
                            "ClassMapper", "Metrics", "RecentLossCounter", "AverageLossCounter",
                            "TensorBatchStorage", "DelayedStorage", "TrainingLoop",
                            "CosineAnnealingWithWarmup", "FrameReader", "FrameWriter",
                            "Tee", "TeeStdout", "TeeStderr"]),
])
def test_public_names_resolvable(module, names):
    mod = importlib.import_module(module)
    for name in names:
        assert getattr(mod, name) is not None, f"{module}.{name} missing"

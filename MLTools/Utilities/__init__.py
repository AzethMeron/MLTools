from .Serialization import SaveBin, LoadBin, SaveDump, LoadDump, LoadOrCompute
from .Cosmetics import SciNotation, CountParameters
from .Convertion import PilToOpenCV, OpenCVToPil
from .ClassMapper import ClassMapper
from .Metrics import Metrics
from .TrainingUtils import RecentLossCounter, AverageLossCounter, TensorBatchStorage, DelayedStorage, TrainingLoop, CosineAnnealingWithWarmup
from .VideoProc import FrameReader, FrameWriter
from .Tee import Tee, TeeStdout, TeeStderr

# Visualizer (matplotlib) and DiscordWebhook (requests) depend on optional
# packages, so they are imported lazily (PEP 562). `import MLTools` must not
# fail when those extras are missing.
_LAZY = {
    "Visualizer": ("MLTools.Utilities.Visualizer", "Visualizer"),
    "DiscordWebhook": ("MLTools.Utilities.Webhooks", "DiscordWebhook"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        module_name, attr = _LAZY[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))

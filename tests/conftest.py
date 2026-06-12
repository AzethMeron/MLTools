"""Shared fixtures and helpers for the MLTools test suite."""
import random

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture(autouse=True)
def _seed_everything():
    """Make every test deterministic by default. Tests that want different
    streams reseed locally."""
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    yield


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def make_pil(width=32, height=24, mode="RGB", seed=0):
    """Deterministic random PIL image."""
    r = np.random.default_rng(seed)
    channels = len(mode) if mode != "P" else 1
    if channels == 1:
        arr = r.integers(0, 256, size=(height, width), dtype=np.uint8)
    else:
        arr = r.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
    img = Image.fromarray(arr if channels > 1 else arr, mode="RGB" if channels == 3 else None)
    if img.mode != mode:
        img = img.convert(mode)
    return img


def solid_pil(width, height, color=(10, 20, 30)):
    return Image.new("RGB", (width, height), color)

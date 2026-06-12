"""Letterbox transform: geometry, fill colors, detection mapping."""
import numpy as np
import pytest
from PIL import Image

from MLTools.DataProcessing import Letterbox
from MLTools.Detections import Detection

from conftest import solid_pil


# ---------------------------------------------------------------- typical

def test_output_size():
    lb = Letterbox((640, 480))
    out = lb(solid_pil(320, 240))
    assert out.size == (640, 480)


def test_no_op_when_size_matches():
    img = solid_pil(64, 48, (200, 100, 50))
    out = Letterbox((64, 48))(img)
    assert out.size == (64, 48)
    assert np.array_equal(np.array(out), np.array(img))


def test_wide_image_pads_top_bottom():
    lb = Letterbox((100, 100), fill=(114, 114, 114))
    out = np.array(lb(solid_pil(200, 100, (255, 0, 0))))  # 2:1 -> 100x50 content
    # rows 0-24 and 75-99 are padding
    assert (out[:25] == 114).all()
    assert (out[75:] == 114).all()
    # center rows are (mostly) red content
    assert (out[40:60, :, 0] == 255).all()
    assert (out[40:60, :, 1] == 0).all()


def test_tall_image_pads_left_right():
    lb = Letterbox((100, 100))
    out = np.array(lb(solid_pil(50, 100, (0, 255, 0))))
    assert (out[:, :25] == 114).all()
    assert (out[:, 75:] == 114).all()
    assert (out[:, 40:60, 1] == 255).all()


def test_upscaling_small_image():
    out = Letterbox((128, 128))(solid_pil(16, 16, (1, 2, 3)))
    arr = np.array(out)
    assert out.size == (128, 128)
    assert (arr[:, :, 0] == 1).all() and (arr[:, :, 2] == 3).all()


def test_non_rgb_input_converted():
    img = Image.new("L", (40, 20), 128)
    out = Letterbox((80, 80))(img)
    assert out.mode == "RGB"
    assert out.size == (80, 80)


# ---------------------------------------------------------------- the fill bug

def test_non_uniform_fill_color():
    """fill=(R,G,B) with distinct channels used to crash with a shape
    mismatch (re-padding an already padded tensor). Verify per-channel fill."""
    lb = Letterbox((100, 100), fill=(10, 20, 30))
    out = np.array(lb(solid_pil(200, 100, (255, 255, 255))))
    pad = out[:20]  # top padding region
    assert (pad[:, :, 0] == 10).all()
    assert (pad[:, :, 1] == 20).all()
    assert (pad[:, :, 2] == 30).all()


# ---------------------------------------------------------------- edge cases

def test_extreme_aspect_ratio_min_1px():
    lb = Letterbox((32, 32))
    out = lb(solid_pil(1000, 1))  # would round content height to 0px
    assert out.size == (32, 32)


def test_single_pixel_input():
    out = Letterbox((10, 10))(solid_pil(1, 1, (9, 9, 9)))
    assert out.size == (10, 10)


def test_invalid_constructor():
    with pytest.raises(ValueError):
        Letterbox((0, 100))
    with pytest.raises(ValueError):
        Letterbox((100, 100), fill=(1, 2))


def test_repr():
    assert "Letterbox" in repr(Letterbox((64, 48)))


# ---------------------------------------------------------------- detections

def test_transform_detections_full_image_box():
    """A box covering the whole source must map exactly onto the content area."""
    lb = Letterbox((100, 100))
    ow, oh = 200, 100  # -> content 100x50, top offset 25
    det = Detection.from_cxcywh(ow / 2, oh / 2, ow, oh, class_id=0, confidence=1.0)
    out = lb.transform_detections([det], (ow, oh))[0]
    assert out.cx == pytest.approx(50.0)
    assert out.cy == pytest.approx(25 + 25.0)
    assert out.w == pytest.approx(100.0)
    assert out.h == pytest.approx(50.0)


def test_transform_detections_preserves_class_and_confidence():
    lb = Letterbox((64, 64))
    det = Detection.from_cxcywh(10, 10, 4, 4, class_id=7, confidence=0.5)
    out = lb.transform_detections([det], (32, 32))[0]
    assert out.class_id == 7 and out.confidence == 0.5
    # 32->64 is a 2x scale with no padding
    assert out.cx == pytest.approx(20.0) and out.w == pytest.approx(8.0)


def test_transform_detections_consistent_with_image():
    """Place a colored dot, letterbox the image, and confirm the transformed
    box lands on the dot."""
    ow, oh = 120, 60
    img = solid_pil(ow, oh, (0, 0, 0))
    px, py = 90, 30
    img.putpixel((px, py), (255, 0, 0))
    lb = Letterbox((60, 60), fill=(0, 0, 0))
    out = np.array(lb(img))

    det = Detection.from_cxcywh(px + 0.5, py + 0.5, 2, 2, 0, 1.0)
    mapped = lb.transform_detections([det], (ow, oh))[0]
    cx, cy = int(mapped.cx), int(mapped.cy)
    region = out[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3, 0]
    assert region.max() > 30, "transformed detection misses the red dot"

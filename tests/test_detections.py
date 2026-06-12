"""Detection geometry: constructors, transforms, NMS, IOU, supervision bridge."""
import math
import warnings

import numpy as np
import pytest

from MLTools.Detections import Detection


# ---------------------------------------------------------------- constructors

def test_from_xyxy():
    d = Detection.from_xyxy(10, 20, 30, 60, class_id=1, confidence=0.9)
    assert (d.cx, d.cy, d.w, d.h) == (20, 40, 20, 40)
    assert d.rotation == 0 and d.class_id == 1 and d.confidence == 0.9


def test_from_topleft_xywh_and_alias():
    d1 = Detection.from_topleft_xywh(10, 20, 20, 40, 1, 0.9)
    d2 = Detection.from_xywh(10, 20, 20, 40, 1, 0.9)
    assert (d1.cx, d1.cy) == (20, 40) == (d2.cx, d2.cy)


def test_from_cxcywh_and_explode():
    d = Detection.from_cxcywh(5, 6, 7, 8, 2, 0.5)
    (cx, cy, w, h, rot), cid, conf = d.Explode()
    assert (cx, cy, w, h, rot, cid, conf) == (5, 6, 7, 8, 0, 2, 0.5)


def test_xyxy_roundtrip():
    d = Detection.from_xyxy(1.5, 2.5, 10.5, 22.5, 0, 1.0)
    assert d.to_xyxy() == pytest.approx([1.5, 2.5, 10.5, 22.5])


def test_accessors_and_str():
    d = Detection(1, 2, 3, 4, 5, 6, 0.7)
    assert d.Center() == (1, 2)
    assert d.Width() == 3 and d.Height() == 4
    assert d.Rotation() == 5 and d.ClassId() == 6 and d.Confidence() == 0.7
    assert "Detection" in str(d) and "Detection" in repr(d)


# ---------------------------------------------------------------- geometry

def test_rescale():
    d = Detection.from_cxcywh(10, 20, 4, 8, 0, 1.0)
    out = d.Rescale((100, 100), (200, 50))
    assert (out.cx, out.cy, out.w, out.h) == (20, 10, 8, 4)


def test_rotate_90_about_origin():
    d = Detection.from_cxcywh(10, 0, 2, 2, 0, 1.0)
    out = d.Rotate(90, (0, 0))
    # screen coords: (x,y) -> (-y, x)
    assert out.cx == pytest.approx(0, abs=1e-9)
    assert out.cy == pytest.approx(10)
    assert out.rotation == 90


def test_rotate_360_identity():
    d = Detection.from_cxcywh(7, 9, 2, 4, 0, 1.0)
    out = d.Rotate(360, (3, 3))
    assert out.cx == pytest.approx(7) and out.cy == pytest.approx(9)
    assert out.rotation == pytest.approx(0)


def test_rotated_corners_axis_aligned():
    d = Detection.from_cxcywh(10, 10, 4, 2, 0, 1.0)
    corners = d.RotatedCorners()
    assert corners[0] == pytest.approx((8, 9))
    assert corners[1] == pytest.approx((12, 9))
    assert corners[2] == pytest.approx((12, 11))
    assert corners[3] == pytest.approx((8, 11))


def test_rotated_corners_90_degrees_swaps_extent():
    d = Detection(10, 10, 4, 2, 90, 0, 1.0)
    x1, y1, x2, y2 = d.to_xyxy()
    assert (x2 - x1) == pytest.approx(2)
    assert (y2 - y1) == pytest.approx(4)


def test_horizontal_flip():
    d = Detection.from_cxcywh(10, 30, 4, 6, 0, 1.0)
    out = d.HorizontalFlip(100)
    assert out.cx == 90 and out.cy == 30
    assert (out.w, out.h) == (4, 6)


def test_vertical_flip():
    d = Detection.from_cxcywh(10, 30, 4, 6, 0, 1.0)
    out = d.VerticalFlip(100)
    assert out.cx == 10 and out.cy == 70


def test_flip_twice_is_identity():
    d = Detection(10, 30, 4, 6, 25, 0, 1.0)
    h2 = d.HorizontalFlip(100).HorizontalFlip(100)
    v2 = d.VerticalFlip(50).VerticalFlip(50)
    for out in (h2, v2):
        assert out.cx == pytest.approx(d.cx) and out.cy == pytest.approx(d.cy)
        # box footprint must match (rotation may differ by 180 for a rectangle)
        assert np.allclose(out.to_xyxy(), d.to_xyxy())


def test_flip_preserves_box_footprint_of_rotated_box():
    """Mirroring a rotated box: the axis-aligned envelope is mirrored too."""
    d = Detection(30, 40, 10, 4, 30, 0, 1.0)
    W = 100
    flipped = d.HorizontalFlip(W)
    x1, y1, x2, y2 = d.to_xyxy()
    fx1, fy1, fx2, fy2 = flipped.to_xyxy()
    assert fx1 == pytest.approx(W - x2) and fx2 == pytest.approx(W - x1)
    assert fy1 == pytest.approx(y1) and fy2 == pytest.approx(y2)


def test_translate():
    d = Detection.from_cxcywh(5, 5, 2, 2, 0, 1.0)
    assert d.TranslateW(3).cx == 8
    assert d.TranslateH(-2).cy == 3


# ---------------------------------------------------------------- IOU / NMS

def test_iou_identical_boxes():
    d = Detection.from_xyxy(0, 0, 10, 10, 0, 1.0)
    assert d.IOU(d) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    a = Detection.from_xyxy(0, 0, 10, 10, 0, 1.0)
    b = Detection.from_xyxy(20, 20, 30, 30, 0, 1.0)
    assert a.IOU(b) == pytest.approx(0.0)


def test_iou_half_overlap():
    a = Detection.from_xyxy(0, 0, 10, 10, 0, 1.0)
    b = Detection.from_xyxy(5, 0, 15, 10, 0, 1.0)
    assert a.IOU(b) == pytest.approx(1 / 3, abs=1e-6)


def test_naive_nms_suppresses_overlaps():
    keep_me = Detection.from_xyxy(0, 0, 10, 10, 0, 0.9)
    drop_me = Detection.from_xyxy(1, 1, 11, 11, 1, 0.5)  # class ignored
    far = Detection.from_xyxy(50, 50, 60, 60, 0, 0.8)
    out = Detection.NaiveNMS([keep_me, drop_me, far], iou_threshold=0.5)
    assert keep_me in out and far in out and drop_me not in out


def test_classful_nms_keeps_separate_classes():
    a = Detection.from_xyxy(0, 0, 10, 10, 0, 0.9)
    b = Detection.from_xyxy(1, 1, 11, 11, 1, 0.5)  # overlaps but other class
    out = Detection.NMS([a, b], iou_threshold=0.5)
    assert len(out) == 2


def test_nms_empty_input():
    assert Detection.NaiveNMS([]) == []
    assert Detection.NMS([]) == []


def test_nms_single_box():
    d = Detection.from_xyxy(0, 0, 5, 5, 0, 0.7)
    assert Detection.NaiveNMS([d]) == [d]


# ---------------------------------------------------------------- supervision bridge

def test_supervision_roundtrip():
    sv = pytest.importorskip("supervision")
    dets = [
        Detection.from_xyxy(0, 0, 10, 10, 1, 0.9),
        Detection.from_xyxy(5, 5, 25, 35, 2, 0.4),
    ]
    sv_dets = Detection.to_supervision(dets)
    assert len(sv_dets) == 2
    back = Detection.from_supervision(sv_dets)
    for orig, restored in zip(dets, back):
        assert restored.to_xyxy() == pytest.approx(orig.to_xyxy())
        assert restored.class_id == orig.class_id
        assert restored.confidence == pytest.approx(orig.confidence)


def test_supervision_empty():
    sv = pytest.importorskip("supervision")
    out = Detection.to_supervision([])
    assert len(out) == 0


def test_supervision_nested_lists():
    sv = pytest.importorskip("supervision")
    batch = [[Detection.from_xyxy(0, 0, 4, 4, 0, 1.0)], []]
    out = Detection.to_supervision(batch)
    assert isinstance(out, list) and len(out) == 2


def test_from_supervision_none_confidence():
    """supervision allows confidence/class_id to be None — don't crash."""
    sv = pytest.importorskip("supervision")
    import numpy as np
    raw = sv.Detections(xyxy=np.array([[0.0, 0.0, 5.0, 5.0]]))
    out = Detection.from_supervision(raw)
    assert len(out) == 1
    assert out[0].confidence == pytest.approx(1.0)


def test_deprecated_to_supervision_alias_works():
    """ToSupervision used to reference an undefined name (NameError)."""
    sv = pytest.importorskip("supervision")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        out = Detection.ToSupervision([Detection.from_xyxy(0, 0, 4, 4, 0, 1.0)])
    assert len(out) == 1


def test_draw_runs():
    from PIL import Image
    img = Image.new("RGB", (64, 64))
    d = Detection.from_cxcywh(32, 32, 20, 10, 0, 1.0)
    out = d.Draw(img)
    assert np.asarray(out).sum() > 0  # something was drawn
    out2 = Detection(32, 32, 20, 10, 30, 1, 1.0).Draw(img, class_id_to_name=lambda c: f"cls{c}")
    assert out2 is img

"""DetectionDataset (+COCO json parsing, SimpleDataset save/load)."""
import json

import numpy as np
import pytest
from PIL import Image

from MLTools.Dataset import COCO, DetectionDataset
from MLTools.Dataset.DetectionDataset import SimpleDataset
from MLTools.Detections import Detection
from MLTools.Utilities import ClassMapper

from conftest import solid_pil


class StubDetectionSource:
    """Minimal (image, [Detection]) dataset."""
    def __init__(self, n=3, size=(80, 60)):
        self.n = n
        self.size = size
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        W, H = self.size
        img = solid_pil(W, H, (idx * 10 % 255, 50, 100))
        dets = [Detection.from_cxcywh(W / 2, H / 2, W / 4, H / 4, idx % 2, 1.0)]
        return img, dets


# ---------------------------------------------------------------- basic

def test_basic_letterbox_only():
    ds = DetectionDataset(StubDetectionSource(), ClassMapper(), image_size=(64, 64))
    assert len(ds) == 3
    img, dets = ds[0]
    assert img.size == (64, 64)
    assert len(dets) == 1
    d = dets[0]
    # source 80x60 -> scale 0.8 -> content 64x48, top pad 8
    assert d.cx == pytest.approx(32.0)
    assert d.cy == pytest.approx(8 + 24.0)
    assert d.w == pytest.approx(16.0)
    assert d.h == pytest.approx(12.0)


def test_augmentation_callable_applied():
    calls = []
    def aug(img):
        calls.append(1)
        return img
    ds = DetectionDataset(StubDetectionSource(), ClassMapper(), image_size=(32, 32), augmentation=aug)
    ds[0]
    assert calls


def test_flips_move_detections():
    np.random.seed(0)
    ds = DetectionDataset(StubDetectionSource(), ClassMapper(), image_size=(64, 64),
                          horizontal_flip=1.0, vertical_flip=1.0)
    _, dets = ds[0]
    d = dets[0]
    # center box flips onto itself
    assert d.cx == pytest.approx(32.0)
    assert d.cy == pytest.approx(32.0)


def test_translation_scalar_and_tuple():
    np.random.seed(0)
    ds = DetectionDataset(StubDetectionSource(), ClassMapper(), image_size=(64, 64),
                          translate_w=(0.25, 0.25), translate_h=(-0.25, -0.25))
    _, dets = ds[0]
    d = dets[0]
    assert d.cx == pytest.approx(32 + 16.0)
    assert d.cy == pytest.approx(32 - 16.0)


def test_rotation_int_zero_no_crash():
    """rotate=0 used to crash np.random.randint(0, 0)."""
    ds = DetectionDataset(StubDetectionSource(), ClassMapper(), image_size=(32, 32), rotate=0)
    ds[0]


def test_rotation_float_scalar_now_works():
    """rotate=15.0 (float) used to be silently ignored."""
    np.random.seed(1)
    ds = DetectionDataset(StubDetectionSource(), ClassMapper(), image_size=(64, 64), rotate=15.0)
    img, dets = ds[0]
    assert dets[0].rotation != 0  # rotation actually applied


def test_translate_int_scalar_now_works():
    np.random.seed(2)
    ds = DetectionDataset(StubDetectionSource(), ClassMapper(), image_size=(64, 64), translate_w=1)
    _, dets = ds[0]
    assert dets[0].cx != pytest.approx(32.0)


def test_invalid_range_spec():
    ds = DetectionDataset(StubDetectionSource(), ClassMapper(), image_size=(32, 32),
                          rotate=(1, 2, 3))
    with pytest.raises(ValueError):
        ds[0]
    ds2 = DetectionDataset(StubDetectionSource(), ClassMapper(), image_size=(32, 32),
                           rotate="lots")
    with pytest.raises(TypeError):
        ds2[0]


def test_generate_once_is_deterministic():
    np.random.seed(0)
    ds = DetectionDataset(StubDetectionSource(), ClassMapper(), image_size=(32, 32),
                          horizontal_flip=0.5, rotate=20, generate_once=True)
    img_a, dets_a = ds[1]
    img_b, dets_b = ds[1]
    assert img_a is img_b
    assert dets_a is dets_b


def test_class_stats():
    ds = DetectionDataset(StubDetectionSource(n=5), ClassMapper(), image_size=(32, 32))
    stats = ds.GetClassStats()
    assert stats == {0: 3, 1: 2}
    assert ds.GetClassStats() is stats  # cached


# ---------------------------------------------------------------- save / load

def test_save_and_load_augmented_dataset(tmp_path):
    ds = DetectionDataset(StubDetectionSource(), ClassMapper(), image_size=(32, 32))
    ds.SaveAugmentedDataset(str(tmp_path))
    loaded = DetectionDataset.LoadAugmentedDataset(str(tmp_path))
    assert isinstance(loaded, SimpleDataset)
    assert len(loaded) == 3
    img, dets = loaded[0]
    assert img.size == (32, 32)
    assert len(dets) == 1


# ---------------------------------------------------------------- COCO json

@pytest.fixture
def coco_dir(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(2):
        solid_pil(40, 30).save(img_dir / f"img_{i}.png")
    anno = {
        "images": [{"id": i, "file_name": f"img_{i}.png"} for i in range(2)],
        "categories": [{"id": 10, "name": "cat"}, {"id": 20, "name": "dog"}],
        "annotations": [
            {"image_id": 0, "category_id": 10, "bbox": [5, 5, 10, 10]},
            {"image_id": 0, "category_id": 20, "bbox": [0, 0, 20, 15]},
            {"image_id": 1, "category_id": 10, "bbox": [1, 2, 3, 4]},
        ],
    }
    anno_path = tmp_path / "annotations.json"
    anno_path.write_text(json.dumps(anno))
    return img_dir, anno_path


def test_coco_loading(coco_dir):
    img_dir, anno_path = coco_dir
    mapper = ClassMapper()
    ds = COCO(str(img_dir), str(anno_path), mapper)
    assert len(ds) == 2
    assert len(mapper) == 2
    img, dets = ds[0]
    assert img.size == (40, 30)
    assert len(dets) == 2
    d = dets[0]  # bbox [5,5,10,10] topleft-xywh -> center (10,10)
    assert (d.cx, d.cy, d.w, d.h) == (10, 10, 10, 10)
    assert d.class_id == mapper.LabelToClass(10)

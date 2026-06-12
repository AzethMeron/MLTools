"""Serialization, ClassMapper, Metrics, Cosmetics, Convertion, Tee."""
import io
import sys

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image

from MLTools.Utilities import (
    ClassMapper,
    CountParameters,
    LoadBin,
    LoadDump,
    LoadOrCompute,
    Metrics,
    OpenCVToPil,
    PilToOpenCV,
    SaveBin,
    SaveDump,
    SciNotation,
    Tee,
    TeeStdout,
    TeeStderr,
)

from conftest import make_pil


# ---------------------------------------------------------------- Serialization

def test_save_load_bin_roundtrip(tmp_path):
    obj = {"a": [1, 2, 3], "b": np.arange(5), "c": "text"}
    path = tmp_path / "obj.bin"
    SaveBin(str(path), obj)
    loaded = LoadBin(str(path))
    assert loaded["a"] == [1, 2, 3]
    assert np.array_equal(loaded["b"], obj["b"])


def test_save_load_dump_roundtrip():
    obj = {"x": torch.arange(4)}
    blob = SaveDump(obj)
    assert isinstance(blob, bytes)
    loaded = LoadDump(blob)
    assert torch.equal(loaded["x"], obj["x"])


def test_load_or_compute_caches(tmp_path):
    path = tmp_path / "cache.bin"
    calls = []

    def expensive():
        calls.append(1)
        return 42

    assert LoadOrCompute(str(path), expensive) == 42
    assert LoadOrCompute(str(path), expensive) == 42
    assert len(calls) == 1  # second call was served from disk


def test_load_dump_rejects_garbage():
    with pytest.raises(Exception):
        LoadDump(b"definitely not zlib")


# ---------------------------------------------------------------- ClassMapper

def test_class_mapper_register_and_lookup():
    m = ClassMapper()
    m.Register(101, "cat")
    m.Register(202, "dog")
    assert len(m) == 2
    assert m.LabelToClass(101) == 0 and m.LabelToClass(202) == 1
    assert m.ClassToName(0) == "cat" and m.ClassToLabel(1) == 202
    assert m.NameToClass("dog") == 1


def test_class_mapper_duplicate_register_ignored():
    m = ClassMapper()
    m.Register(1, "cat")
    m.Register(1, "feline")  # same label_id -> no-op
    assert len(m) == 1
    assert m.ClassToName(0) == "cat"


def test_class_mapper_copy_independent():
    m = ClassMapper()
    m.Register(1, "cat")
    c = m.Copy()
    c.Register(2, "dog")
    assert len(m) == 1 and len(c) == 2


def test_class_mapper_unknown_raises():
    with pytest.raises(KeyError):
        ClassMapper().LabelToClass(7)


# ---------------------------------------------------------------- Metrics

def test_metrics_update_save_load(tmp_path, capsys):
    path = tmp_path / "metrics.pkl"
    m = Metrics(str(path))
    m.update({"loss": 0.5, "epoch": 1})
    m.update({"loss": 0.25, "epoch": 2})
    assert len(m) == 2
    assert m[0]["loss"] == 0.5
    assert [e["epoch"] for e in m] == [1, 2]

    m2 = Metrics(str(path))
    m2.load()
    assert len(m2) == 2 and m2[1]["loss"] == 0.25


def test_metrics_without_path():
    m = Metrics(None)
    m.update({"v": 1})  # save() is a no-op without path
    assert m[0] == {"v": 1}


# ---------------------------------------------------------------- Cosmetics

def test_sci_notation_basic():
    assert SciNotation(12345.0) == "1.23 × 10⁴"
    assert SciNotation(0.00123) == "1.23 × 10⁻³"


def test_sci_notation_simplify():
    assert SciNotation(1.0) == "1.00"           # exp == 0
    assert SciNotation(1000.0) == "10³"         # base == 1.00
    assert SciNotation(1000.0, simplify=False) == "1.00 × 10³"


def test_count_parameters():
    model = nn.Linear(10, 5)  # 10*5 + 5 = 55
    assert CountParameters(model) == 55
    model.bias.requires_grad_(False)
    assert CountParameters(model, trainable_only=True) == 50
    assert CountParameters(model, trainable_only=False) == 55
    assert isinstance(CountParameters(model, return_sci_notation=True), str)


# ---------------------------------------------------------------- Convertion

def test_pil_opencv_roundtrip():
    img = make_pil(20, 10, seed=5)
    cv = PilToOpenCV(img)
    assert cv.shape == (10, 20, 3)
    back = OpenCVToPil(cv)
    assert np.array_equal(np.asarray(back), np.asarray(img))


def test_pil_to_opencv_converts_modes():
    gray = Image.new("L", (8, 8), 100)
    cv = PilToOpenCV(gray)
    assert cv.shape == (8, 8, 3)


# ---------------------------------------------------------------- Tee

def test_tee_mirrors_writes():
    a, b = io.StringIO(), io.StringIO()
    t = Tee(a, b)
    n = t.write("hello")
    assert a.getvalue() == "hello" == b.getvalue()
    assert n == 5


def test_tee_requires_streams():
    with pytest.raises(ValueError):
        Tee()


def test_tee_passthrough_attributes():
    a = io.StringIO()
    t = Tee(a)
    assert t.encoding is not None
    assert t.isatty() is False
    assert callable(t.getvalue)  # passthrough via __getattr__


def test_tee_stdout_context(tmp_path, capsys):
    log = tmp_path / "out.log"
    with TeeStdout(str(log)):
        print("captured-line")
    assert "captured-line" in log.read_text()
    assert "captured-line" in capsys.readouterr().out


def test_tee_stderr_context(tmp_path, capsys):
    log = tmp_path / "err.log"
    with TeeStderr(str(log)):
        print("err-line", file=sys.stderr)
    assert "err-line" in log.read_text()


def test_tee_restores_stdout_on_exception(tmp_path):
    old = sys.stdout
    with pytest.raises(RuntimeError):
        with TeeStdout(str(tmp_path / "x.log")):
            raise RuntimeError("boom")
    assert sys.stdout is old

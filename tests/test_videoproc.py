"""FrameReader / FrameWriter against a real temp video file (mp4v codec).

FrameWriter is hard-wired to 'avc1' (OpenH264); in environments without the
codec the constructor raises IOError and those tests are skipped.
"""
import cv2
import numpy as np
import pytest
from PIL import Image

from MLTools.Utilities import FrameReader, FrameWriter


N_FRAMES, W, H, FPS = 10, 64, 48, 20.0


@pytest.fixture
def video_path(tmp_path):
    path = str(tmp_path / "clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, FPS, (W, H))
    if not writer.isOpened():
        pytest.skip("mp4v codec unavailable")
    for i in range(N_FRAMES):
        frame = np.full((H, W, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


# ---------------------------------------------------------------- FrameReader

def test_reader_metadata(video_path):
    with FrameReader(video_path, batch_size=4) as r:
        assert r.width == W and r.height == H
        assert r.fps == pytest.approx(FPS)
        assert r.frame_count == N_FRAMES


def test_reader_batching(video_path):
    with FrameReader(video_path, batch_size=4) as r:
        batches = list(r)
    sizes = [len(b) for b in batches]
    assert sizes == [4, 4, 2]
    assert all(isinstance(f, Image.Image) for b in batches for f in b)
    assert batches[0][0].size == (W, H)


def test_reader_bgr_mode(video_path):
    with FrameReader(video_path, batch_size=100, use_pil=False) as r:
        batch = r.read()
    assert isinstance(batch[0], np.ndarray)
    assert batch[0].shape == (H, W, 3)


def test_reader_read_returns_empty_when_done(video_path):
    r = FrameReader(video_path, batch_size=100)
    assert len(r.read()) == N_FRAMES
    assert r.read() == []


def test_reader_reset(video_path):
    r = FrameReader(video_path, batch_size=N_FRAMES)
    first = r.read()
    assert r.reset()
    again = r.read()
    assert len(first) == len(again) == N_FRAMES
    r.close()


def test_reader_invalid_source(tmp_path):
    with pytest.raises(IOError):
        FrameReader(str(tmp_path / "missing.mp4"))


def test_reader_invalid_batch_size(video_path):
    with pytest.raises(ValueError):
        FrameReader(video_path, batch_size=0)


def test_reader_close_idempotent(video_path):
    r = FrameReader(video_path)
    r.close()
    r.close()
    assert r.read() == []


# ---------------------------------------------------------------- FrameWriter

def test_writer_roundtrip(tmp_path):
    out_path = str(tmp_path / "out.mp4")
    try:
        writer = FrameWriter(out_path, fps=10, width=W, height=H)
    except IOError:
        pytest.skip("avc1 codec unavailable in this environment")
    frames = [Image.new("RGB", (W, H), (i * 25, 0, 0)) for i in range(5)]
    with writer:
        writer.write(frames)
        writer.write(frames[0])  # single frame path
    with FrameReader(out_path, batch_size=100) as r:
        assert len(r.read()) == 6

from typing import Iterable, List, Union, Optional
import cv2
import numpy as np
from PIL import Image
from Convertion import PilToOpenCV, OpenCVToPil

class FrameReader:
    """
    Read frames from a video source in fixed-size batches.

    - Based on cv2.VideoCapture
    - Returns PIL.Image (RGB) if use_pil=True, otherwise returns BGR np.ndarray
    - Works with `with` or standalone
    """

    def __init__(
        self,
        source: Union[str, int],
        batch_size: int = 64,
        use_pil: bool = True,
        default_fps: float = 30.0
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        self.source = source
        self.batch_size = int(batch_size)
        self.use_pil = bool(use_pil)
        self.default_fps = float(default_fps)

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video source: {source}")

        fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 0.0
        self.fps: float = fps if np.isfinite(fps) and fps > 0 else self.default_fps

        self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        raw_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_count: Optional[int] = int(raw_count) if raw_count and raw_count > 0 else None

        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration

        batch = []
        for _ in range(self.batch_size):
            ok, frame = self.cap.read()
            if not ok:
                break

            # Convert to PIL (RGB)
            if self.use_pil:
                frame = OpenCVToPil(frame)

            batch.append(frame)

        if not batch:
            self.close()
            raise StopIteration

        return batch

    def read(self):
        """Read a single batch, [] if finished."""
        try:
            return next(self)
        except StopIteration:
            return []

    def reset(self):
        """Seek to first frame (works only for real files)."""
        if self._closed:
            return False
        return bool(self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0))

    def close(self):
        if not self._closed:
            self.cap.release()
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

class FrameWriter:
    def __init__(self, filename, fps = 30, width = 640, height = 480, use_pil=True):
        self.use_pil = use_pil
        self.fps = fps
        self.width = width
        self.height = height
        self.fourcc = cv2.VideoWriter_fourcc(*'avc1')
        self.cap = cv2.VideoWriter(filename, self.fourcc, self.fps, (self.width, self.height))
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video target: {filename}")
        self._closed = False
    
    def write(self, frames):
        if self._closed: print(f"Trying to write frames, but FrameWriter is already closed")
        if not isinstance(frames, list): frames = [frames]
        for i, frame in enumerate(frames):
            if self.use_pil: frame = PilToOpenCV(frame)
            frame = cv2.resize(frame, (self.width, self.height), interpolation = cv2.INTER_LINEAR)
            self.cap.write(frame)
    
    def close(self):
        if not self._closed:
            self.cap.release()
            self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self:
        self.close()
        
    def __del__(self):
        try:
            self.close()
        except:
            pass
    
    
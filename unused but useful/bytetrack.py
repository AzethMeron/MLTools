"""
ByteTrack (minimal, NumPy-only) tracker with Supervision `Detections` adapter.

- No premade tracking libs are used.
- Only builtins + numpy.
- Accepts detections from `supervision` (https://github.com/roboflow/supervision)
  via the `update()` method. If `supervision` is not installed at runtime,
  you can still pass raw arrays to `update()`.

Key ideas (ByteTrack in a nutshell):
1) Two-stage association using IoU:
   - Stage A: match existing tracks with *high-confidence* detections.
   - Stage B: match *remaining* unmatched tracks with *low-confidence* detections.
2) Constant-velocity Kalman filter in (x, y, a, h) box-space (as in SORT/ByteTrack):
   state = [cx, cy, a, h, vx, vy, va, vh].T where a = aspect_ratio = w/h.
3) Track states: Tracked, Lost, Removed. Lost tracks are kept for `track_buffer`
   frames before removal.
4) Newly matched detections spawn tracks; unmatched high/low dets can be queued
   (per ByteTrack style) with probation via `min_hits`.

This is a compact, readable reference implementation intended for integration
in pipelines that produce `supervision.Detections`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Iterable
import numpy as np

# ---------------------------- Utility functions ----------------------------- #

def xyxy_to_xyah(xyxy: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] -> [cx, cy, a, h].
    a = w / h. Shape preserved along leading dims.
    """
    x1, y1, x2, y2 = xyxy[..., 0], xyxy[..., 1], xyxy[..., 2], xyxy[..., 3]
    w = np.maximum(0.0, x2 - x1)
    h = np.maximum(0.0, y2 - y1)
    cx = x1 + w * 0.5
    cy = y1 + h * 0.5
    a = np.divide(w, np.maximum(h, 1e-6))
    return np.stack([cx, cy, a, h], axis=-1)


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    """Convert [cx, cy, a, h] -> [x1, y1, x2, y2]."""
    cx, cy, a, h = xyah[..., 0], xyah[..., 1], xyah[..., 2], xyah[..., 3]
    w = a * h
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=-1)


def iou_matrix(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of boxes in [x1, y1, x2, y2].
    Returns an (N, M) matrix.
    """
    if b1.size == 0 or b2.size == 0:
        return np.zeros((b1.shape[0], b2.shape[0]), dtype=np.float32)

    N, M = b1.shape[0], b2.shape[0]
    ious = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        x1 = np.maximum(b1[i, 0], b2[:, 0])
        y1 = np.maximum(b1[i, 1], b2[:, 1])
        x2 = np.minimum(b1[i, 2], b2[:, 2])
        y2 = np.minimum(b1[i, 3], b2[:, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area1 = np.maximum(0.0, b1[i, 2] - b1[i, 0]) * np.maximum(0.0, b1[i, 3] - b1[i, 1])
        area2 = np.maximum(0.0, b2[:, 2] - b2[:, 0]) * np.maximum(0.0, b2[:, 3] - b2[:, 1])
        union = area1 + area2 - inter + 1e-6
        ious[i] = inter / union
    return ious


# --------------------------- Hungarian (NumPy) ------------------------------ #

def hungarian(cost: np.ndarray) -> List[Tuple[int, int]]:
    """A simple Hungarian algorithm for rectangular cost matrices.
    Returns list of (row, col) assignments. Complexity ~ O(n^3).
    Note: expects finite costs. Pads to square internally.
    """
    cost = np.asarray(cost, dtype=np.float64)
    n_rows, n_cols = cost.shape
    n = max(n_rows, n_cols)

    # Pad to square with large cost
    pad_value = cost.max() + 1e6 if cost.size else 1e6
    padded = np.full((n, n), pad_value, dtype=np.float64)
    padded[:n_rows, :n_cols] = cost

    # Row reduction
    padded -= padded.min(axis=1, keepdims=True)
    # Column reduction
    padded -= padded.min(axis=0, keepdims=True)

    # Masks
    starred = np.zeros_like(padded, dtype=bool)
    primed = np.zeros_like(padded, dtype=bool)
    covered_rows = np.zeros(n, dtype=bool)
    covered_cols = np.zeros(n, dtype=bool)

    # Step 1: star zeros greedily
    for i in range(n):
        for j in range(n):
            if padded[i, j] == 0 and not covered_rows[i] and not covered_cols[j]:
                starred[i, j] = True
                covered_rows[i] = True
                covered_cols[j] = True
    covered_rows[:] = False
    covered_cols[:] = False

    def cover_starred_columns():
        covered_cols[:] = starred.any(axis=0)

    def find_a_zero():
        for i in range(n):
            if covered_rows[i]:
                continue
            for j in range(n):
                if not covered_cols[j] and padded[i, j] == 0:
                    return i, j
        return None

    def find_star_in_row(row):
        col = np.where(starred[row])[0]
        return (col[0] if col.size else None)

    def find_star_in_col(col):
        row = np.where(starred[:, col])[0]
        return (row[0] if row.size else None)

    def find_prime_in_row(row):
        col = np.where(primed[row])[0]
        return (col[0] if col.size else None)

    def augment_path(path):
        for r, c in path:
            if primed[r, c]:
                primed[r, c] = False
                starred[r, c] = True
            else:
                starred[r, c] = False

    def clear_covers_and_primes():
        covered_rows[:] = False
        covered_cols[:] = False
        primed[:, :] = False

    cover_starred_columns()
    while covered_cols.sum() < n:
        z = find_a_zero()
        while z is None:
            # Adjust the matrix
            uncovered = ~covered_rows[:, None] & ~covered_cols[None, :]
            m = padded[uncovered].min() if np.any(uncovered) else 0.0
            padded[~covered_rows] -= m
            padded[:, covered_cols] += m
            z = find_a_zero()

        i, j = z
        primed[i, j] = True
        star_col = find_star_in_row(i)
        if star_col is None:
            # Augmenting path
            path = [(i, j)]
            col = j
            row = find_star_in_col(col)
            while row is not None:
                path.append((row, col))
                col = find_prime_in_row(row)
                path.append((row, col))
                row = find_star_in_col(col)
            augment_path(path)
            clear_covers_and_primes()
            cover_starred_columns()
        else:
            covered_rows[i] = True
            covered_cols[star_col] = False

    # Build assignments within original dims
    matches: List[Tuple[int, int]] = []
    for i in range(n_rows):
        j = np.where(starred[i])[0]
        if j.size and j[0] < n_cols:
            matches.append((i, int(j[0])))
    return matches


# --------------------------- Kalman Filter (CV) ----------------------------- #

class KalmanFilter:
    """8D constant-velocity Kalman filter in box-space.
    State: [cx, cy, a, h, vx, vy, va, vh].T
    Measurement: [cx, cy, a, h].T
    """

    def __init__(self):
        dt = 1.0
        self._motion_mat = np.eye(8)
        for i in range(4):
            self._motion_mat[i, i + 4] = dt
        self._update_mat = np.eye(4, 8)

        # Base uncertainties (tuned for stability)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros(4, dtype=np.float64)
        mean = np.r_[mean_pos, mean_vel]

        std = np.array([
            2 * self._std_weight_position * measurement[3],  # cx
            2 * self._std_weight_position * measurement[3],  # cy
            1e-2,                                           # a
            2 * self._std_weight_position * measurement[3],  # h
            10 * self._std_weight_velocity * measurement[3], # vx
            10 * self._std_weight_velocity * measurement[3], # vy
            1e-3,                                           # va
            10 * self._std_weight_velocity * measurement[3], # vh
        ])
        P = np.diag(std ** 2)
        return mean, P

    def predict(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = np.array([
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ])
        std_vel = np.array([
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-3,
            self._std_weight_velocity * mean[3],
        ])
        Q = np.diag(np.r_[std_pos, std_vel] ** 2)

        mean = self._motion_mat @ mean
        cov = self._motion_mat @ cov @ self._motion_mat.T + Q
        return mean, cov

    def project(self, mean: np.ndarray, cov: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std = np.array([
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ])
        R = np.diag(std ** 2)
        mean_proj = self._update_mat @ mean
        cov_proj = self._update_mat @ cov @ self._update_mat.T + R
        return mean_proj, cov_proj

    def update(self, mean: np.ndarray, cov: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_proj, cov_proj = self.project(mean, cov)
        S = cov_proj
        K = cov @ self._update_mat.T @ np.linalg.inv(S)
        y = measurement - mean_proj
        mean = mean + K @ y
        cov = cov - K @ S @ K.T
        return mean, cov


# ------------------------------- Track class -------------------------------- #

_TRACK_ID_COUNTER = 0

def _next_id() -> int:
    global _TRACK_ID_COUNTER
    _TRACK_ID_COUNTER += 1
    return _TRACK_ID_COUNTER


@dataclass
class Track:
    mean: np.ndarray
    cov: np.ndarray
    track_id: int
    score: float
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    state: str = "Tracked"  # Tracked | Lost | Removed
    class_id: Optional[int] = None

    def to_xyxy(self) -> np.ndarray:
        xyah = self.mean[:4]
        return xyah_to_xyxy(xyah)

    def predict(self, kf: KalmanFilter):
        self.mean, self.cov = kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: KalmanFilter, det_xyah: np.ndarray, score: float, class_id: Optional[int]):
        self.mean, self.cov = kf.update(self.mean, self.cov, det_xyah)
        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0
        self.state = "Tracked"
        if class_id is not None:
            self.class_id = class_id


# ------------------------------ ByteTrack core ------------------------------ #

@dataclass
class ByteTrack:
    track_thresh: float = 0.5         # high confidence threshold (tau)
    match_thresh: float = 0.8         # IoU thresh for Stage A
    low_thresh: float = 0.1           # low-score floor for Stage B
    fuse_score: bool = True           # update track score with matched det score
    track_buffer: int = 30            # max age before removing lost track
    min_hits: int = 1                 # promote to confirmed after this many hits

    kf: KalmanFilter = field(default_factory=KalmanFilter)
    tracks: List[Track] = field(default_factory=list)
    lost: List[Track] = field(default_factory=list)
    removed: List[Track] = field(default_factory=list)

    frame_id: int = 0

    # ---------------------------- Public API -------------------------------- #
    def reset(self):
        self.tracks.clear()
        self.lost.clear()
        self.removed.clear()
        self.frame_id = 0

    def update(self,
               detections: "DetectionsLike",
               classes: Optional[np.ndarray] = None) -> List[Track]:
        """Run one frame update.

        Parameters
        ----------
        detections: DetectionsLike
            Either a `supervision.Detections` object or a tuple (xyxy, scores)
            where xyxy is (N,4) and scores is (N,).
        classes: Optional[np.ndarray]
            Optional class ids (N,) to attach to tracks.
        """
        self.frame_id += 1

        det_xyxy, det_scores = self._extract_detections(detections)
        if classes is None and hasattr(detections, "class_id") and detections.class_id is not None:
            try:
                classes = np.asarray(detections.class_id)
            except Exception:
                classes = None

        # Split high / low confidence
        high_mask = det_scores >= self.track_thresh
        low_mask = (det_scores >= self.low_thresh) & (~high_mask)

        high_xyxy, high_scores = det_xyxy[high_mask], det_scores[high_mask]
        low_xyxy, low_scores = det_xyxy[low_mask], det_scores[low_mask]
        high_cls = classes[high_mask] if classes is not None else None
        low_cls = classes[low_mask] if classes is not None else None

        # Predict all current tracks (Tracked + Lost)
        for t in self.tracks:
            t.predict(self.kf)
        for t in self.lost:
            t.predict(self.kf)

        # Stage A: match tracked tracks with high dets
        matches_a, u_tracks, u_high = self._associate(self.tracks, high_xyxy, self.match_thresh)
        self._apply_matches(self.tracks, matches_a, high_xyxy, high_scores, high_cls)

        # Unmatched tracked go to Lost (if already confirmed) or stay pending
        newly_lost = []
        still_tracked = []
        for idx in u_tracks:
            t = self.tracks[idx]
            if t.hits >= self.min_hits:
                t.state = "Lost"
                newly_lost.append(t)
            else:
                # Not yet confirmed; keep it but don't promote
                still_tracked.append(t)
        # Update current tracked list to those still matched or pending new
        self.tracks = [t for i, t in enumerate(self.tracks) if i not in u_tracks]
        self.lost.extend(newly_lost)

        # Stage B: try to recover with low-score detections using Lost + pending
        pool_tracks = self.lost + still_tracked
        matches_b, u_pool, u_low = self._associate(pool_tracks, low_xyxy, self.match_thresh)
        self._apply_matches(pool_tracks, matches_b, low_xyxy, low_scores, low_cls)

        # Rebuild track lists after Stage B
        # - Any in pool that got updated should be (or remain) Tracked
        new_tracks = []
        new_lost = []
        for t in pool_tracks:
            if t.time_since_update == 0:
                t.state = "Tracked"
                new_tracks.append(t)
            else:
                new_lost.append(t)
        # Existing already-updated tracks from Stage A were kept; combine
        self.tracks.extend(new_tracks)
        self.lost = new_lost

        # Spawn new tracks from unmatched high detections
        for j in u_high:
            mean, cov = self.kf.initiate(xyxy_to_xyah(high_xyxy[j]))
            cls = int(high_cls[j]) if high_cls is not None else None
            self.tracks.append(Track(mean, cov, _next_id(), float(high_scores[j]), hits=1,
                                     state="Tracked" if self.min_hits <= 1 else "Tracked"))
            self.tracks[-1].class_id = cls

        # Age and remove old lost tracks
        kept_lost = []
        for t in self.lost:
            if t.time_since_update > self.track_buffer:
                t.state = "Removed"
                self.removed.append(t)
            else:
                kept_lost.append(t)
        self.lost = kept_lost

        # Return currently active tracks (state == Tracked)
        return [t for t in self.tracks if t.state == "Tracked"]

    # ---------------------------- Internal helpers -------------------------- #

    def _extract_detections(self, detections: "DetectionsLike") -> Tuple[np.ndarray, np.ndarray]:
        """Supports supervision.Detections or (xyxy, scores)."""
        # Try Supervision Detections
        if hasattr(detections, "xyxy") and hasattr(detections, "confidence"):
            xyxy = np.asarray(detections.xyxy, dtype=np.float64)
            scores = np.asarray(detections.confidence, dtype=np.float64)
            return xyxy, scores
        # Tuple / list fallback
        if isinstance(detections, (tuple, list)) and len(detections) >= 2:
            xyxy = np.asarray(detections[0], dtype=np.float64)
            scores = np.asarray(detections[1], dtype=np.float64)
            return xyxy, scores
        raise TypeError("detections must be supervision.Detections or (xyxy, scores)")

    def _associate(self,
                   tracks: List[Track],
                   det_xyxy: np.ndarray,
                   iou_thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(tracks) == 0 or det_xyxy.shape[0] == 0:
            return [], list(range(len(tracks))), list(range(det_xyxy.shape[0]))

        track_boxes = np.stack([t.to_xyxy() for t in tracks], axis=0)
        ious = iou_matrix(track_boxes, det_xyxy)
        cost = 1.0 - ious
        matches = hungarian(cost)

        # Filter by IoU threshold
        final_matches = []
        matched_tracks = set()
        matched_dets = set()
        for r, c in matches:
            if ious[r, c] >= iou_thresh:
                final_matches.append((r, c))
                matched_tracks.add(r)
                matched_dets.add(c)

        u_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        u_dets = [j for j in range(det_xyxy.shape[0]) if j not in matched_dets]
        return final_matches, u_tracks, u_dets

    def _apply_matches(self,
                       tracks: List[Track],
                       matches: List[Tuple[int, int]],
                       det_xyxy: np.ndarray,
                       det_scores: np.ndarray,
                       det_classes: Optional[np.ndarray]):
        for r, c in matches:
            det_xyah = xyxy_to_xyah(det_xyxy[c])
            score = det_scores[c]
            cls = int(det_classes[c]) if det_classes is not None else None
            tracks[r].update(self.kf, det_xyah, score if self.fuse_score else tracks[r].score, cls)


# ------------------------------ Convenience -------------------------------- #

DetectionsLike = object  # runtime duck-typing


def tracks_to_supervision(tracks: Iterable[Track]):
    """(Optional) Helper to turn active tracks into `supervision.Detections`.
    Only fills xyxy and tracker_id (plus class_id if present). Requires supervision installed.
    """
    try:
        import supervision as sv  # type: ignore
    except Exception as e:
        raise RuntimeError("supervision is required for tracks_to_supervision()") from e

    xyxy = np.stack([t.to_xyxy() for t in tracks], axis=0) if tracks else np.zeros((0, 4), dtype=np.float32)
    tracker_ids = np.array([t.track_id for t in tracks], dtype=int) if tracks else np.zeros((0,), dtype=int)
    class_ids = np.array([t.class_id if t.class_id is not None else -1 for t in tracks], dtype=int) if tracks else np.zeros((0,), dtype=int)

    dets = sv.Detections(xyxy=xyxy)
    dets.tracker_id = tracker_ids
    dets.class_id = class_ids
    return dets


# ------------------------------- Demo usage -------------------------------- #
if False: #__name__ == "__main__":
    # Minimal dry-run with fake data
    # Each frame: a few boxes (x1,y1,x2,y2) with scores
    bt = ByteTrack(track_thresh=0.5, match_thresh=0.7, low_thresh=0.1, track_buffer=20)

    frames = [
        (np.array([[10, 10, 60, 80], [200, 50, 260, 140]], dtype=float), np.array([0.9, 0.85])),
        (np.array([[13, 12, 63, 82], [198, 50, 258, 140]], dtype=float), np.array([0.92, 0.2])),  # second det low
        (np.array([[16, 14, 66, 84]], dtype=float), np.array([0.4])),  # low det only
        (np.array([[20, 16, 70, 86], [196, 48, 256, 138]], dtype=float), np.array([0.88, 0.88])),
    ]

    for f, (boxes, scores) in enumerate(frames, 1):
        active = bt.update((boxes, scores))
        print(f"Frame {f}")
        for t in active:
            x1, y1, x2, y2 = t.to_xyxy()
            print(f"  ID {t.track_id:2d} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) hits={t.hits} score={t.score:.2f}")

#!/usr/bin/env python3
"""
Standalone script: reads a video, runs YOLOv8-medium, converts to Supervision
Detections, tracks with minimal NumPy ByteTrack, and writes an H.264 (if
available) MP4 with boxes + track IDs.

No argparse: configure parameters below as plain variables.

Requirements:
  pip install ultralytics supervision opencv-python numpy

Place `bytetrack_minimal_numpy_supervision.py` next to this file or on PYTHONPATH.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple
from ultralytics import YOLO
import supervision as sv
import torch

# ---------------------- USER PARAMETERS (edit these) ------------------------ #
SOURCE: str = "input.mp4"              # Path to input video
OUT: str = "tracked.mp4"               # Output path (MP4)
MODEL: str = "yolov8m.pt"              # YOLO weights
CONF: float = 0.25                      # YOLO confidence threshold
IMGSZ: int = 1280                       # YOLO image size

# ByteTrack parameters
TRACK_THRESH: float = 0.5               # High conf threshold (tau)
LOW_THRESH: float = 0.1                 # Low conf floor for recovery stage
MATCH_THRESH: float = 0.7               # IoU threshold for association
TRACK_BUFFER: int = 60                  # Frames to keep lost tracks

# Keep only these class names (None to keep all). Example: ["person", "car"]
CLASSES: Optional[list[str]] = None
# -------------------- END USER PARAMETERS ---------------------------------- #


def get_video_info(path: str) -> Tuple[int, int, float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 1e-3:
        fps = 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, float(fps), frames


def make_writer(out_path: str, size: Tuple[int, int], fps: float) -> cv2.VideoWriter:
    w, h = size
    for fcc in ("avc1", "H264", "X264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*fcc)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if writer.isOpened():
            if fcc != "mp4v":
                print(f"[i] Using codec {fcc} for output: {out_path}")
            else:
                print("[!] Falling back to 'mp4v' (your OpenCV may lack H.264 encoder support)")
            return writer
    raise RuntimeError("Failed to create a VideoWriter with H.264 or mp4v")


def main():
    W, H, FPS, N = get_video_info(SOURCE)
    print(f"[i] Input: {SOURCE} | {W}x{H} @ {FPS:.2f} FPS | {N} frames (reported)")

    model = YOLO(MODEL)
    model = model.to(torch.device("cuda"))
    names = model.model.names if hasattr(model.model, "names") else {}

    keep_class_ids = None
    if CLASSES:
        wanted = {s.strip().lower() for s in CLASSES if s.strip()}
        keep_class_ids = {cid for cid, n in names.items() if str(n).lower() in wanted}
        print(f"[i] Filtering classes to: {sorted(list(keep_class_ids))} ({CLASSES})")

    tracker = ByteTrack(track_thresh=TRACK_THRESH,
                        match_thresh=MATCH_THRESH,
                        low_thresh=LOW_THRESH,
                        track_buffer=TRACK_BUFFER)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    cap = cv2.VideoCapture(SOURCE)
    writer = make_writer(OUT, (W, H), FPS)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            results = model.predict(source=frame, imgsz=IMGSZ, conf=CONF, verbose=False)
            r0 = results[0]

            dets = sv.Detections.from_ultralytics(r0)
            if keep_class_ids is not None and dets.class_id is not None:
                mask = np.array([cid in keep_class_ids for cid in dets.class_id], dtype=bool)
                dets = dets[mask]

            active_tracks = tracker.update(dets, classes=dets.class_id if hasattr(dets, 'class_id') else None)
            tracked_dets = tracks_to_supervision(active_tracks)

            # Labels: ID + class name (if available)
            labels = []
            tids = getattr(tracked_dets, 'tracker_id', None)
            cids = getattr(tracked_dets, 'class_id', None)
            for i in range(len(tracked_dets)):
                tid = int(tids[i]) if tids is not None and len(tids) > 0 else -1
                label = f"ID {tid}"
                if cids is not None and len(cids) > 0:
                    cid = int(cids[i])
                    cname = names.get(cid, str(cid))
                    label = f"{label} - {cname}"
                labels.append(label)

            frame = box_annotator.annotate(scene=frame, detections=tracked_dets)
            frame = label_annotator.annotate(scene=frame, detections=tracked_dets, labels=labels)

            writer.write(frame)

            if frame_idx % 50 == 0:
                print(f"[i] Frame {frame_idx} | active tracks: {len(active_tracks)}")
    finally:
        cap.release()
        writer.release()

    print(f"[i] Done. Wrote: {OUT}")


if __name__ == "__main__":
    main()

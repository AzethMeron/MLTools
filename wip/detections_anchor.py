# Requirements:
#   pip install matplotlib pillow
#
# Run this script; it will save "anchor_types_animation.gif" in the working directory.

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---- paste your Detection class here OR import it if it's in a module ----
# from your_module import Detection
# (Assuming the class from your message is already available in scope.)
from MLTools.Detections import Detection

# ---------- helpers ----------
def rect_patch_from_xyxy(ax, xyxy, **kwargs):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    return ax.add_patch(plt.Rectangle((x1, y1), w, h, fill=False, linewidth=2, **kwargs))

def polygon_patch_from_points(ax, pts, **kwargs):
    from matplotlib.patches import Polygon
    return ax.add_patch(Polygon(pts, closed=True, fill=False, linewidth=2, **kwargs))

# ---------- setup ----------
# Base detection: center (0,0), w=4, h=2, rotation=0
det0 = Detection(0.0, 0.0, 4.0, 2.0, 0.0, class_id=0, confidence=1.0)

# Output GIF settings
filename = "anchor_types_animation.gif"
fps = 10  # playback fps
angles = list(range(0, 361, 1))  # 0..360 inclusive

# Figure/axes
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect("equal", adjustable="box")
# generous limits so everything stays visible
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.grid(True, linestyle="--", alpha=0.3)

# Artists to update each frame
# Rotated rectangle (outline)
rot_poly = None
# Axis-aligned bboxes for anchors
box_max = None
box_min = None
box_mean = None
# Center point + annotation
center_pt, = ax.plot([], [], marker="o", markersize=5)
title = ax.set_title("")

# Legend (static; add dummy handles so legend stays fixed)
legend_handles = [
    plt.Line2D([0], [0], color="black", lw=2, label="Rotated rect"),
    plt.Line2D([0], [0], color="red",   lw=2, label="anchor = 'max'"),
    plt.Line2D([0], [0], color="blue",  lw=2, label="anchor = 'min'"),
    plt.Line2D([0], [0], color="green", lw=2, label="anchor = 'mean'"),
]
ax.legend(handles=legend_handles, loc="upper right")

def init():
    # initialize empty artists so blitting works
    global rot_poly, box_max, box_min, box_mean
    # create empty placeholders (will be replaced on first frame)
    rot_poly, = ax.plot([], [], color="black")  # temporary; replaced with Polygon
    return []

def draw_frame(angle_deg):
    """Update all artists for the given angle."""
    global rot_poly, box_max, box_min, box_mean

    # Make a fresh detection for this frame; set both .rotation and .angle for compatibility
    det = Detection(det0.cx, det0.cy, det0.w, det0.h, rotation=angle_deg,
                    class_id=det0.class_id, confidence=det0.confidence)

    # Remove previous frame's patches (if any)
    for p in [rot_poly, box_max, box_min, box_mean]:
        if p is not None and hasattr(p, "remove"):
            try:
                p.remove()
            except Exception:
                pass

    # Rotated rectangle outline (true oriented box)
    rotated_pts = det.RotatedCorners()
    rot_poly = polygon_patch_from_points(ax, rotated_pts, color="black")

    # Axis-aligned XYXY for each anchor
    xyxy_max  = det.to_xyxy(anchor="max")
    xyxy_min  = det.to_xyxy(anchor="min")
    xyxy_mean = det.to_xyxy(anchor="mean")

    box_max  = rect_patch_from_xyxy(ax, xyxy_max,  color="red")
    box_min  = rect_patch_from_xyxy(ax, xyxy_min,  color="blue")
    box_mean = rect_patch_from_xyxy(ax, xyxy_mean, color="green")

    # Center marker
    center_pt.set_data([det.cx], [det.cy])

    # Title
    title.set_text(f"Rotation: {angle_deg}°")

    return [rot_poly, box_max, box_min, box_mean, center_pt, title]

# Build animation
anim = FuncAnimation(
    fig,
    func=lambda i: draw_frame(angles[i]),
    frames=len(angles),
    init_func=init,
    interval=1000 / fps,
    blit=False,          # blit=False to keep legend/grid stable
    repeat=True,
)

# Save GIF
writer = PillowWriter(fps=fps)
anim.save(filename, writer=writer)
print(f"Saved: {filename}")

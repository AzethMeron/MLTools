"""Visualizer: chart construction, image grids, PIL export."""
import matplotlib
import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from MLTools.Utilities import Visualizer

from conftest import make_pil


@pytest.fixture
def viz():
    return Visualizer(figsize=(3, 2), dpi=60)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def test_line(viz):
    fig, ax = viz.line([1, 2, 3], label="l", title="t", xlabel="x", ylabel="y", marker="o")
    assert isinstance(fig, Figure)
    assert ax.get_title() == "t"
    assert len(ax.lines) == 1


def test_line_with_explicit_x(viz):
    fig, ax = viz.line([1, 4, 9], x=[1, 2, 3])
    xdata = ax.lines[0].get_xdata()
    assert list(xdata) == [1, 2, 3]


def test_line_length_mismatch(viz):
    with pytest.raises(ValueError):
        viz.line([1, 2, 3], x=[1, 2])


def test_multi_line(viz):
    fig, ax = viz.multi_line({"a": [1, 2], "b": [3, 4]}, markers=["o", None])
    assert len(ax.lines) == 2


def test_multi_line_empty_rejected(viz):
    with pytest.raises(ValueError):
        viz.multi_line({})


def test_scatter(viz):
    fig, ax = viz.scatter([1, 2, 3], label="pts")
    assert len(ax.collections) == 1


def test_bar(viz):
    fig, ax = viz.bar([3, 1, 2], labels=["a", "b", "c"], rotate_xticks=True)
    assert len(ax.patches) == 3
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]


def test_images_grid_flat(viz):
    imgs = [make_pil(8, 8, seed=i) for i in range(4)]
    fig, axes = viz.images_grid(imgs, nrows=2, ncols=2, titles=["a", "b", "c", "d"])
    assert axes.shape == (2, 2)


def test_images_grid_nested_ragged(viz):
    rows = [[make_pil(8, 8)], [make_pil(8, 8), make_pil(8, 8)]]
    fig, axes = viz.images_grid(rows, suptitle="grid")
    assert axes.shape == (2, 2)
    # missing cell hidden
    assert not axes[0, 1].axison


def test_images_grid_flat_requires_shape(viz):
    with pytest.raises(ValueError):
        viz.images_grid([make_pil(8, 8)])


def test_images_grid_count_mismatch(viz):
    with pytest.raises(ValueError):
        viz.images_grid([make_pil(8, 8)] * 3, nrows=2, ncols=2)


def test_images_grid_share_axes_first_row(viz):
    """share_axes used to skip row 0 / column 0 (r*c>0 bug)."""
    imgs = [make_pil(8, 8, seed=i) for i in range(4)]
    fig, axes = viz.images_grid(imgs, nrows=2, ncols=2, share_axes=True)
    ref = axes[0, 0]
    for r in range(2):
        for c in range(2):
            if (r, c) == (0, 0):
                continue
            assert axes[r, c].get_shared_x_axes().joined(ref, axes[r, c]), \
                f"axes[{r},{c}] not x-shared with axes[0,0]"


def test_loss_chart(viz):
    history = [
        {"epoch": 0, "train_loss": 1.0, "test_loss": 1.5},
        {"epoch": 1, "train_loss": 0.5, "test_loss": 0.7},
    ]
    fig, ax = viz.loss_chart(history)
    assert len(ax.lines) == 2


def test_draw_to_pil_tight(viz):
    fig, _ = viz.line([1, 2, 3])
    img = Visualizer.draw_to_pil(fig)
    assert isinstance(img, Image.Image) and img.mode == "RGBA"
    assert img.width > 10 and img.height > 10


def test_draw_to_pil_non_tight(viz):
    """non-tight path used tostring_argb(), removed in matplotlib 3.10."""
    fig, _ = viz.line([1, 2, 3])
    img = Visualizer.draw_to_pil(fig, tight=False, bg_color="white")
    assert isinstance(img, Image.Image) and img.mode == "RGBA"
    arr = np.asarray(img)
    assert arr.shape[2] == 4 and arr[..., 3].max() == 255

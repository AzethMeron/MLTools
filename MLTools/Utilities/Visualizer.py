from typing import Iterable, Optional, Sequence, Tuple, Dict, Any, Union, List
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg

from PIL import Image, ImageOps
import numpy as np


Number = Union[int, float]


@dataclass
class Visualizer:
    """
    Simple charting helper built on Matplotlib.
    """
    figsize: Tuple[float, float] = (6.0, 4.0)
    dpi: int = 120
    style: Optional[str] = None  # e.g., "ggplot", etc.
    rcparams: Dict[str, Any] = field(default_factory=dict)

    # ---------- Charts ----------

    def line(
        self,
        y: Sequence[Number],
        x: Optional[Sequence[Number]] = None,
        *,
        label: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        marker: Optional[str] = None,
        linewidth: float = 2.0,
        grid: bool = True,
        legend: bool = True,
        minor_grid: bool = True,  # minor grid ON by default
    ) -> Tuple[Figure, Axes]:
        x_arr, y_arr = _to_arrays(x, y)
        fig, ax = self._new_figure()
        ax.plot(x_arr, y_arr, label=label, marker=marker, linewidth=linewidth)
        _decorate(ax, title, xlabel, ylabel, grid, legend and (label is not None), minor_grid)
        return fig, ax

    def multi_line(
        self,
        series: Dict[str, Sequence[Number]],
        x: Optional[Sequence[Number]] = None,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        markers: Optional[Sequence[Optional[str]]] = None,
        linewidth: float = 2.0,
        grid: bool = True,
        legend: bool = True,
        minor_grid: bool = True,  # minor grid ON by default
    ) -> Tuple[Figure, Axes]:
        if not series:
            raise ValueError("`series` must contain at least one (label -> y) pair.")
        any_y = next(iter(series.values()))
        x_arr, _ = _to_arrays(x, any_y)

        fig, ax = self._new_figure()
        for i, (name, y_vals) in enumerate(series.items()):
            _, y_arr = _to_arrays(x_arr, y_vals)
            marker = markers[i] if markers and i < len(markers) else None
            ax.plot(x_arr, y_arr, label=name, marker=marker, linewidth=linewidth)
        _decorate(ax, title, xlabel, ylabel, grid, legend, minor_grid)
        return fig, ax

    def scatter(
        self,
        y: Sequence[Number],
        x: Optional[Sequence[Number]] = None,
        *,
        label: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        s: float = 20.0,
        grid: bool = True,
        legend: bool = True,
        minor_grid: bool = True,  # minor grid ON by default
    ) -> Tuple[Figure, Axes]:
        x_arr, y_arr = _to_arrays(x, y)
        fig, ax = self._new_figure()
        ax.scatter(x_arr, y_arr, label=label, s=s)
        _decorate(ax, title, xlabel, ylabel, grid, legend and (label is not None), minor_grid)
        return fig, ax

    def bar(
        self,
        values: Sequence[Number],
        labels: Optional[Sequence[str]] = None,
        *,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        grid: bool = True,
        rotate_xticks: bool = False,
        minor_grid: bool = True,  # minor grid ON by default
    ) -> Tuple[Figure, Axes]:
        y_arr = np.asarray(values, dtype=float)
        n = y_arr.shape[0]
        x_idx = np.arange(n)
        if labels is None:
            labels = [str(i) for i in range(n)]

        fig, ax = self._new_figure()
        ax.bar(x_idx, y_arr)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(labels, rotation=45 if rotate_xticks else 0,
                           ha="right" if rotate_xticks else "center")
        _decorate(ax, title, xlabel, ylabel, grid, legend=False, minor_grid=minor_grid)
        return fig, ax

    # ---------- NEW: Image grid ----------

    def images_grid(
        self,
        images: Union[Sequence[Image.Image], Sequence[Sequence[Image.Image]]],
        *,
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        titles: Optional[Sequence[str]] = None,
        share_axes: bool = True,
        suptitle: Optional[str] = None,
        tight: bool = True,
        minor_grid: bool = True,   # minor grid ON by default
        grid: bool = True,         # major grid ON by default
        keep_aspect: bool = True,
    ) -> Tuple[Figure, np.ndarray]:
        """
        Display a matrix of PIL images.

        Parameters
        ----------
        images : list[Image] | list[list[Image]]
            Either a flat list with nrows & ncols specified, or a nested list (rows of images).
        nrows, ncols : int, optional
            Required for flat list; ignored for nested list.
        titles : list[str], optional
            Per-cell titles for flat list; for nested list, titles are row-major.
        share_axes : bool
            If True, subplots share x/y; keeps grids aligned.
        suptitle : str, optional
            Figure-level title.
        tight : bool
            Use tight_layout to reduce padding.
        minor_grid, grid : bool
            Control minor/major grid visibility on each subplot.
        keep_aspect : bool
            If True, preserves image aspect (no stretching).

        Returns
        -------
        (fig, axes) where axes is an array of Axes with shape (nrows, ncols).
        """
        # Normalize input to nested list form and determine grid size
        nested = isinstance(images, (list, tuple)) and images and isinstance(images[0], (list, tuple))
        if nested:
            rows_list: List[List[Image.Image]] = [list(map(_ensure_pil, row)) for row in images]  # type: ignore[arg-type]
            _nrows = len(rows_list)
            _ncols = max((len(r) for r in rows_list), default=0)
        else:
            if nrows is None or ncols is None:
                raise ValueError("For a flat list of images, please provide both nrows and ncols.")
            flat = list(map(_ensure_pil, images))  # type: ignore[arg-type]
            if len(flat) != nrows * ncols:
                raise ValueError(f"Expected nrows*ncols images ({nrows*ncols}), got {len(flat)}.")
            # reshape row-major
            rows_list = [flat[i*ncols:(i+1)*ncols] for i in range(nrows)]
            _nrows, _ncols = nrows, ncols

        # Create figure
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        if suptitle:
            fig.suptitle(suptitle)

        # Build subplots
        axes = np.empty((_nrows, _ncols), dtype=object)
        for r in range(_nrows):
            for c in range(_ncols):
                ax: Axes = fig.add_subplot(_nrows, _ncols, r*_ncols + c + 1, sharex=axes[0,0] if (share_axes and r*c>0) else None, sharey=axes[0,0] if (share_axes and r*c>0) else None)  # type: ignore[index]
                axes[r, c] = ax
                # Place image if exists, else hide axis
                if c < len(rows_list[r]):
                    img = rows_list[r][c]
                    arr = np.array(img)
                    if keep_aspect:
                        ax.imshow(arr)
                    else:
                        ax.imshow(arr, aspect="auto")
                    # Optional per-cell title
                    if titles is not None:
                        idx = r * _ncols + c
                        if idx < len(titles):
                            ax.set_title(titles[idx])
                    # Apply grid decoration
                    _apply_grid(ax, grid=grid, minor_grid=minor_grid)
                else:
                    ax.axis("off")

        if tight:
            fig.tight_layout()

        return fig, axes
    
    def loss_chart(self, history):
        epochs = [ entry['epoch'] for entry in history ]
        train_losses = [ entry['train_loss'] for entry in history ]
        test_losses = [ entry['test_loss'] for entry in history ]
        return self.multi_line(series = {'train loss': train_losses, 'test loss': test_losses}, x = epochs)
    
    # ---------- Convert to PIL ----------

    @staticmethod
    def draw_to_pil(
        fig: Figure,
        *,
        dpi: Optional[int] = None,
        tight: bool = True,
        bg_color: Optional[str] = None,
    ) -> Image.Image:
        if bg_color is not None:
            fig.patch.set_facecolor(bg_color)

        canvas = FigureCanvasAgg(fig)
        if dpi is not None:
            fig.set_dpi(dpi)

        if tight:
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
            buf.seek(0)
            return Image.open(buf).convert("RGBA")

        canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
        buf = buf[:, :, [1, 2, 3, 0]]  # ARGB -> RGBA
        return Image.fromarray(buf, mode="RGBA")

    # ---------- Internals ----------

    def _new_figure(self) -> Tuple[Figure, Axes]:
        if self.style:
            plt.style.use(self.style)
        cm = plt.rc_context(self.rcparams)
        cm.__enter__()
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111)
        fig._rc_cm = cm  # type: ignore[attr-defined]
        fig.canvas.mpl_connect("close_event", lambda evt: cm.__exit__(None, None, None))
        return fig, ax


# ---------- Helpers ----------

def _to_arrays(
    x: Optional[Sequence[Number]],
    y: Sequence[Number],
) -> Tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(list(y), dtype=float)
    if x is None:
        x_arr = np.arange(len(y_arr))
    else:
        x_arr = np.asarray(list(x), dtype=float)
        if x_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(f"x and y must have the same length (got {x_arr.shape[0]} vs {y_arr.shape[0]}).")
    return x_arr, y_arr


def _decorate(
    ax: Axes,
    title: Optional[str],
    xlabel: Optional[str],
    ylabel: Optional[str],
    grid: bool,
    legend: bool,
    minor_grid: bool,
) -> None:
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

    # Major grid
    if grid:
        ax.grid(True, linestyle="--", alpha=0.5)

    # Minor grid ON by default
    if minor_grid:
        try:
            ax.minorticks_on()
        except Exception:
            pass
        ax.grid(True, which="minor", linestyle=":", alpha=0.3)

    if legend:
        ax.legend()

    ax.tick_params(axis="both", which="both", direction="out")
    ax.margins(x=0.02, y=0.05)


def _apply_grid(ax: Axes, *, grid: bool = True, minor_grid: bool = True) -> None:
    if grid:
        ax.grid(True, linestyle="--", alpha=0.5)
    if minor_grid:
        try:
            ax.minorticks_on()
        except Exception:
            pass
        ax.grid(True, which="minor", linestyle=":", alpha=0.3)


def _ensure_pil(img: Union[Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(img)

"""Shared matplotlib styling for the paper figures.

The settings here aim for a NeurIPS-submission-ready aesthetic with
larger, vector-friendly typography and an axes treatment that prints
cleanly in greyscale: top/right spines hidden, ``#888`` left/bottom
spines, dashed grid drawn behind data, scatter markers with white
edges.

Two styling profiles are exported:

- ``apply()``         — the default, used by the exploratory figures in
                        ``analysis/compute.py``. Compact NeurIPS column
                        sizing.
- ``apply_paper()``   — the larger, hand-tuned profile used by the paper
                        figures in ``analysis/paper_figures.py``. 24 pt
                        axis labels, 18 pt ticks/legend/annotations,
                        designed at 8" so LaTeX scales down to a
                        comfortable size at \\textwidth.

Each figure-producing function calls ``apply()`` or ``apply_paper()``
once at module load. ``save(fig, path)`` writes both .png (300 dpi)
and .pdf — paper figures use the PDF for vector embedding.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.colors import to_rgba


# --- page geometry --------------------------------------------------

# Default NeurIPS column geometry (used by the exploratory figs).
FIG_W = 5.5
COL_W = 2.65

# Paper-figure geometry: square-ish, large fonts, designed for LaTeX
# `\includegraphics[width=\textwidth]` scaling.
PAPER_W = 8.0
PAPER_H = 6.0     # default; override per-figure as needed
PAPER_H_TALL = 7.5
PAPER_H_SHORT = 4.0


# --- color and marker maps ------------------------------------------

# Scaffold encodes both color and marker so the figure survives in
# greyscale (color disambiguates in print, marker disambiguates in BW).
SCAFFOLD = {
    "codex":       "#1f77b4",
    "claude_code": "#d62728",
    "core_agent":  "#2ca02c",
    "opencode":    "#ff7f0e",
}
SCAFFOLD_MARKER = {
    "codex":       "s",
    "claude_code": "o",
    "core_agent":  "^",
    "opencode":    "D",
}

# Vendor palette (kept available even though current paper figures
# encode by scaffold). The colors / markers come from the
# pass_k_scatterplots reference.
VENDOR_COLORS  = {"Anthropic": "#D97757", "OpenAI": "#10A37F", "Google": "#4285F4"}
VENDOR_MARKERS = {"Anthropic": "o",       "OpenAI": "s",       "Google": "^"}


# Auxiliary colors.
GUIDE_GRAY      = "#888888"
SPINE_GRAY      = "#888888"
HIGHLIGHT_GOLD  = "#c89b1f"
GRID_DASH_ALPHA = 0.35


_APPLIED_PROFILE: str | None = None


def _common_rc() -> dict:
    return {
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "axes.unicode_minus": False,
        # Spines.
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.edgecolor":     SPINE_GRAY,
        "axes.linewidth":     1.0,
        "xtick.color":        SPINE_GRAY,
        "ytick.color":        SPINE_GRAY,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        # Grid behind data (we will set_axisbelow per ax for safety).
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "grid.color":         "gray",
        "grid.linestyle":     "--",
        "grid.alpha":         GRID_DASH_ALPHA,
        "grid.linewidth":     0.8,
        # Saving.
        "figure.dpi":         120,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.04,
        # Embed TrueType, not Type-3 (NeurIPS requires).
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    }


def apply() -> None:
    """Compact NeurIPS-column profile (used by exploratory figures)."""
    global _APPLIED_PROFILE
    if _APPLIED_PROFILE == "default":
        return
    rc = _common_rc()
    rc.update({
        "font.size":         9,
        "axes.titlesize":    9,
        "axes.labelsize":    9,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "figure.titlesize":  10,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon":      False,
        "legend.handlelength": 1.5,
        "legend.borderpad":    0.3,
        "legend.labelspacing": 0.3,
    })
    mpl.rcParams.update(rc)
    _APPLIED_PROFILE = "default"


def apply_paper() -> None:
    """Hand-tuned paper-figure profile (large fonts, designed at 8")."""
    global _APPLIED_PROFILE
    if _APPLIED_PROFILE == "paper":
        return
    rc = _common_rc()
    rc.update({
        "font.size":         16,
        "axes.titlesize":    18,
        "axes.labelsize":    20,
        "xtick.labelsize":   16,
        "ytick.labelsize":   16,
        "legend.fontsize":   16,
        "figure.titlesize":  20,
        "xtick.major.size":  5,
        "ytick.major.size":  5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        # Paper-style legend: light gray frame, sharp corners.
        "legend.frameon":      True,
        "legend.framealpha":   0.95,
        "legend.edgecolor":    "#cccccc",
        "legend.fancybox":     False,
        "legend.handlelength": 1.5,
        "legend.borderpad":    0.4,
        "legend.labelspacing": 0.4,
    })
    mpl.rcParams.update(rc)
    _APPLIED_PROFILE = "paper"


# --- axes / annotation helpers --------------------------------------

def style_axes(ax: plt.Axes) -> None:
    """Apply per-axes touches that rcParams alone can't carry: grid
    drawn beneath data, spine color, and a tiny tick padding bump so
    18 pt ticks don't sit flush against the spine."""
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_GRAY)
        ax.spines[side].set_linewidth(1.8)
    ax.tick_params(colors=SPINE_GRAY, which="both", pad=4)


def add_y_axis_break(ax: plt.Axes, frac: float = 0.075, width: float = 0.026) -> None:
    """Draw a small zigzag near the bottom of the left spine to signal
    that the y-axis doesn't start at zero. ``frac`` is the vertical
    extent of the zigzag in axes coordinates; ``width`` is the
    horizontal extent. Call after ``set_ylim`` is final."""
    from matplotlib.lines import Line2D
    # White wash to mask the spine where the zigzag sits.
    ax.add_line(Line2D(
        [0, 0], [0, frac],
        transform=ax.transAxes, color="white", linewidth=4.0,
        solid_capstyle="butt", zorder=4, clip_on=False))
    for y0 in (0.020, 0.050):
        ax.add_line(Line2D(
            [-width / 2, width / 2], [y0, y0 + frac * 0.34],
            transform=ax.transAxes, color=SPINE_GRAY, linewidth=1.8,
            solid_capstyle="butt", zorder=5, clip_on=False))


def annotate_points(ax: plt.Axes,
                    points: list[tuple[float, float, str, str]],
                    *, fontsize: int = 15) -> None:
    """Draw labels for a curated list of points with curved arrows.

    Each entry is ``(x_data, y_data, text, color)`` where ``color`` is
    the vendor / scaffold color used at α=0.66 for the text and
    α=0.70 for the arrow's white-bbox label background.

    The label sits at ``(x_data, y_data)`` itself by default; call sites
    that need to disambiguate clusters should pass pre-offset
    coordinates instead of relying on this function for placement.
    """
    for x, y, text, color in points:
        ax.annotate(
            text, xy=(x, y), xytext=(x, y), textcoords="data",
            fontsize=fontsize, fontweight="medium",
            color=to_rgba(color, 0.66),
            bbox=dict(facecolor="white", alpha=0.82,
                      pad=1.8, edgecolor="none"),
            arrowprops=dict(
                arrowstyle="->,head_length=0.5,head_width=0.2",
                connectionstyle="arc3,rad=0.35",
                color=to_rgba(color, 0.82),
                shrinkA=3, shrinkB=7, lw=1.25,
            ),
            annotation_clip=False,
        )


def annotate_with_arrow(ax: plt.Axes,
                        anchor_xy: tuple[float, float],
                        label_xy: tuple[float, float],
                        text: str, color: str,
                        *, fontsize: int = 15) -> None:
    """Curved-arrow annotation: ``anchor_xy`` is the data point, and
    ``label_xy`` is where the label text goes (also in data coords).
    Useful for crowded scatters where the label has to sit elsewhere."""
    ax.annotate(
        text, xy=anchor_xy, xytext=label_xy, textcoords="data",
        fontsize=fontsize, fontweight="medium",
        color=to_rgba(color, 0.74),
        bbox=dict(facecolor="white", alpha=0.84,
                  pad=1.8, edgecolor="none"),
        arrowprops=dict(
            arrowstyle="->,head_length=0.5,head_width=0.2",
            connectionstyle="arc3,rad=0.35",
            color=to_rgba(color, 0.84),
            shrinkA=3, shrinkB=7, lw=1.25,
        ),
        annotation_clip=False,
    )


# --- legend ----------------------------------------------------------

def legend_below(ax: plt.Axes, *, handles=None, labels=None,
                 ncol: int | None = None, fontsize: int = 14,
                 y: float = -0.18, columnspacing: float = 1.6,
                 handlelength: float = 1.5) -> mpl.legend.Legend:
    """Place a horizontal legend below the axes.

    Defaults to 14 pt (vs. the 18 pt body) so the legend sits visually
    secondary to the data. ``ncol`` defaults to one column per item so
    the legend stays a single row.
    """
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    elif labels is None:
        labels = [h.get_label() for h in handles]
    if ncol is None:
        ncol = max(1, len(handles))
    leg = ax.legend(
        handles=handles, labels=labels,
        loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
        fontsize=fontsize, frameon=False, handlelength=handlelength,
        borderpad=0.3, labelspacing=0.4, columnspacing=columnspacing,
    )
    return leg


# --- saving ----------------------------------------------------------

def save(fig: plt.Figure, path: Path | str) -> None:
    """Write ``path`` as PNG and the same stem as PDF. Closes the figure."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)


def save_fixed(fig: plt.Figure, path: Path | str) -> None:
    """Write PNG/PDF using the figure canvas exactly as sized.

    Use this for multi-panel LaTeX composites where several source
    figures need identical bounding boxes. The regular ``save`` helper
    uses tight bounding boxes, which is ideal for standalone figures but
    can make equal-width subfigures look different once arranged in TeX.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(p)
        fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)


def save_fixed_crop(
    fig: plt.Figure,
    path: Path | str,
    *,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> None:
    """Write PNG/PDF with a shared crop box in figure inches."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    bbox = Bbox.from_bounds(left, bottom, width, height)
    with mpl.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        fig.savefig(p, bbox_inches=bbox)
        fig.savefig(p.with_suffix(".pdf"), bbox_inches=bbox)
    plt.close(fig)

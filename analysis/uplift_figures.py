"""Paper-style figures for the uplift RCT analysis.

The source RMarkdown currently reads from Google Sheets. This module is
the repository-native plotting layer: export the sheet as CSV, place it
at ``figs/data/uplift_rct.csv`` (or pass ``--data``), and rerun.

Run:
    python -m analysis.uplift_figures --data figs/data/uplift_rct.csv

While waiting on final data, use:
    python -m analysis.uplift_figures --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analysis import style


style.apply_paper()


DEFAULT_DATA = Path("figs/data/uplift_rct.csv")
DEFAULT_OUT = Path("figs/paper/uplift_duration_by_condition.png")

# Wide panel for the RCT duration histogram.
_LANDSCAPE_CROP = dict(left=-0.35, bottom=0.20, width=7.65, height=5.25)

# Seaborn "colorblind" palette from the original RMarkdown.
COLORBLIND_PALETTE = [
    "#0173B2",
    "#DE8F05",
    "#029E73",
    "#D55E00",
    "#CC78BC",
    "#CA9161",
    "#FBAFE4",
    "#949494",
    "#ECE133",
    "#56B4E9",
]


def _find_duration_column(df: pd.DataFrame) -> str:
    matches = [c for c in df.columns if "duration" in str(c).lower()]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one duration column, found: "
            + ", ".join(map(str, matches))
        )
    return matches[0]


def _histogram_breaks(durations: pd.Series, *, bin_width: int = 15) -> np.ndarray:
    vals = pd.to_numeric(durations, errors="coerce").dropna()
    if vals.empty:
        raise ValueError("No numeric session durations found.")
    lo = np.floor(vals.min() / bin_width) * bin_width
    hi = np.ceil(vals.max() / bin_width) * bin_width
    return np.arange(lo, hi + bin_width, bin_width)


def fig_duration_by_condition(
    df: pd.DataFrame,
    path: Path | str = DEFAULT_OUT,
    *,
    condition_col: str = "Condition",
    duration_col: str | None = None,
    bin_width: int = 15,
) -> None:
    """Overlayed histogram of reproduction-session duration by condition."""
    if condition_col not in df.columns:
        raise ValueError(f"Missing condition column: {condition_col}")
    if duration_col is None:
        duration_col = _find_duration_column(df)
    if duration_col not in df.columns:
        raise ValueError(f"Missing duration column: {duration_col}")

    work = df[[condition_col, duration_col]].copy()
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")
    work = work.dropna(subset=[condition_col, duration_col])
    if work.empty:
        raise ValueError("No non-missing condition/duration rows found.")

    conditions = list(pd.unique(work[condition_col]))
    breaks = _histogram_breaks(work[duration_col], bin_width=bin_width)
    hist_xlim = (
        min(float(breaks.min()), 0.0),
        max(float(breaks.max()), 180.0),
    )
    tick_start = np.floor(hist_xlim[0] / 30) * 30
    tick_end = np.ceil(hist_xlim[1] / 30) * 30
    xticks = np.arange(tick_start, tick_end + 30, 30)
    xticks = xticks[(xticks >= hist_xlim[0]) & (xticks <= hist_xlim[1])]

    max_count = 0
    for condition in conditions:
        counts, _ = np.histogram(
            work.loc[work[condition_col] == condition, duration_col],
            bins=breaks,
        )
        max_count = max(max_count, int(counts.max(initial=0)))

    fig = plt.figure(figsize=(7.25, 5.6))
    ax = fig.add_axes([0.14, 0.19, 0.77, 0.72])
    style.style_axes(ax)

    handles: list[Patch] = []
    for i, condition in enumerate(conditions):
        vals = work.loc[work[condition_col] == condition, duration_col]
        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        ax.hist(
            vals,
            bins=breaks,
            color=color,
            alpha=0.45,
            edgecolor="white",
            linewidth=1.25,
            label=str(condition),
            zorder=3,
        )
        handles.append(Patch(facecolor=color, edgecolor="white",
                             alpha=0.45, label=str(condition)))

    ax.set_xlim(*hist_xlim)
    ax.set_ylim(0, max(1, max_count) * 1.18)
    ax.set_xticks(xticks)
    ax.set_xlabel("Session duration (minutes)")
    ax.set_ylabel("Number of sessions")
    ax.tick_params(labelsize=17)
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", alpha=0.20)

    ncol = min(3, max(1, len(handles)))
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=ncol,
        fontsize=15,
        frameon=True,
        framealpha=0.92,
        borderpad=0.35,
        labelspacing=0.35,
        columnspacing=1.0,
        handlelength=1.4,
    )

    style.save_fixed_crop(fig, path, **_LANDSCAPE_CROP)


def _demo_data(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_control = 72
    n_treatment = 70
    control = np.clip(rng.normal(72, 24, n_control), 12, 180)
    treatment = np.clip(rng.normal(88, 28, n_treatment), 15, 210)
    return pd.DataFrame({
        "Condition": ["Control"] * n_control + ["Treatment"] * n_treatment,
        "Session duration (minutes)": np.r_[control, treatment],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--condition-col", default="Condition")
    parser.add_argument("--duration-col", default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        df = _demo_data()
        out = args.out.with_name(args.out.stem + "_demo" + args.out.suffix)
    else:
        df = pd.read_csv(args.data)
        out = args.out

    fig_duration_by_condition(
        df,
        out,
        condition_col=args.condition_col,
        duration_col=args.duration_col,
    )
    print(f"Wrote {out} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

"""Paper-style figures for the uplift RCT analysis.

The source RMarkdown currently reads from Google Sheets. This module is
the repository-native plotting layer that reads from the published CSV.

Run:
    python -m analysis.uplift_figures --data data/RCT_responses_cleaned.csv

While waiting on final data, use:
    python -m analysis.uplift_figures --demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from analysis import style


style.apply_paper()


DEFAULT_DATA = Path("data/RCT_responses_cleaned.csv")
DEFAULT_OUT = Path("figs/uplift_duration_by_condition.png")
DEFAULT_OUT_PER_PAPER = Path("figs/duration_per_paper.png")
DEFAULT_OUT_STRIP     = Path("figs/duration_strip.png")

# Seaborn "colorblind" palette from the original RMarkdown.
LABEL_REMAP = {"AI-assisted": "Human-agent"}

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


# Maps lowercase CSV paper titles → properly-cased display titles.
PAPER_TITLE_REMAP: dict[str, str] = {
    "antinormative messaging group cues and the nuclear ban treaty":
        "Antinormative messaging, group cues, and the nuclear ban treaty",
    "beyond accuracy behavioral testing of nlp models with checklist":
        "Beyond accuracy: Behavioral testing of NLP models with CheckList",
    "cant we all just get along how women mps can ameliorate affective polarization in western publics":
        "Can't we all just get along? How women MPs can ameliorate affective polarization in western publics",
    "changing tides public attitudes on climate migration":
        "Changing tides: Public attitudes on climate migration",
    "decentralization can increase cooperation among public officials":
        "Decentralization can increase cooperation among public officials",
    "dropmessage unifying random dropping for graph neural networks":
        "DropMessage: Unifying random dropping for graph neural networks",
    "entertaining beliefs in economic mobility":
        "Entertaining beliefs in economic mobility",
    "fantastically ordered prompts and where to find them overcoming fewshot prompt order sensitivity":
        "Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity",
    "improving evaluation of machine translation quality estimation":
        "Improving evaluation of machine translation quality estimation",
    "indecent disclosures anticorruption reforms and political selection":
        "Indecent disclosures: Anticorruption reforms and political selection",
    "informer beyond efficient transformer for long sequence timeseries forecasting":
        "Informer: Beyond efficient transformer for long sequence time-series forecasting",
    "latxa an open language model and evaluation suite for basque":
        "Latxa: An open language model and evaluation suite for basque",
    "multiracial identity and political preferences":
        "Multiracial identity and political preferences",
    "multiwoz a largescale multidomain wizardofoz dataset for taskoriented dialogue modelling":
        "MultiWOZ: A large-scale multi-domain wizard-of-oz dataset for task-oriented dialogue modelling",
    "obfuscated gradients give a false sense of security circumventing defenses to adversarial examples":
        "Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples",
    "policy deliberation and voter persuasion experimental evidence from an election in the philippines":
        "Policy deliberation and voter persuasion: Experimental evidence from an election in the Philippines",
    "reliable conflictive multiview learning":
        "Reliable conflictive multi-view learning",
    "semisupervised neural protolanguage reconstruction":
        "Semisupervised neural proto-language reconstruction",
    "talking shops the effects of caucus discussion on policy coalitions":
        "Talking shops: The effects of caucus discussion on policy coalitions",
    "yellow vests pessimistic beliefs and carbon tax aversion":
        "Yellow vests, pessimistic beliefs, and carbon tax aversion",
}


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

    fig = plt.figure(figsize=(6, 3.5))
    ax = fig.add_axes([0.22, 0.18, 0.69, 0.74])
    style.style_axes(ax)

    handles: list[Patch] = []
    for i, condition in enumerate(conditions):
        vals = work.loc[work[condition_col] == condition, duration_col]
        color = COLORBLIND_PALETTE[i % len(COLORBLIND_PALETTE)]
        display_label = LABEL_REMAP.get(str(condition), str(condition))
        ax.hist(
            vals,
            bins=breaks,
            color=color,
            alpha=0.45,
            edgecolor="white",
            linewidth=1.25,
            label=display_label,
            zorder=3,
        )
        handles.append(Patch(facecolor=color, edgecolor="white",
                             alpha=0.45, label=display_label))

    did_not_complete = work[work[duration_col] >= 180]
    if not did_not_complete.empty:
        dnc_color = "#E63946"
        ax.hist(
            did_not_complete[duration_col],
            bins=breaks,
            color=dnc_color,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.25,
            zorder=4,
        )
        handles.append(Patch(facecolor=dnc_color, edgecolor="white",
                             alpha=0.85, label="Manual, did not complete"))

    ax.set_xlim(*hist_xlim)
    ax.set_ylim(0, max(1, max_count) * 1.18)
    ax.set_xticks(xticks)
    ax.set_xlabel("Session duration (minutes)")
    ax.set_ylabel("Number of sessions")
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(labelsize=17)
    ax.xaxis.label.set_size(17)
    ax.yaxis.label.set_size(17)
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", alpha=0.20)

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=len(handles),
        fontsize=14,
        frameon=False,
        borderpad=0.35,
        labelspacing=0.35,
        columnspacing=1.0,
        handlelength=1.4,
    )

    style.save(fig, path)


def fig_duration_per_paper(
    df: pd.DataFrame,
    path: Path | str = DEFAULT_OUT_PER_PAPER,
    *,
    condition_col: str = "Condition",
    title_col: str = "Paper Title",
    duration_col: str | None = None,
    dnf_threshold: float = 180.0,
) -> None:
    """Dot plot of replication duration per paper, Manual vs Human-agent."""
    if duration_col is None:
        duration_col = _find_duration_column(df)

    work = df[[title_col, condition_col, duration_col]].copy()
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")
    work = work.dropna(subset=[title_col, condition_col, duration_col])

    # collect all (duration, is_dnf) per paper/condition
    grouped: dict[str, dict[str, list[float]]] = {}
    for _, row in work.iterrows():
        title = str(row[title_col]).strip()
        cond = str(row[condition_col]).strip()
        dur = float(row[duration_col])
        grouped.setdefault(title, {}).setdefault(cond, []).append(dur)

    papers = sorted(grouped, key=lambda t: max(grouped[t].get("Manual", [0])))

    def _display(raw: str) -> str:
        proper = PAPER_TITLE_REMAP.get(raw, raw)
        return proper[:45] + "..." if len(proper) > 45 else proper

    short_labels = [_display(t) for t in papers]

    color_manual = COLORBLIND_PALETTE[1]       # #DE8F05
    color_agent  = COLORBLIND_PALETTE[0]       # #0173B2
    color_dnf    = "#E63946"

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_axes([0.05, 0.34, 0.92, 0.58])
    style.style_axes(ax)

    for i, paper in enumerate(papers):
        conds = grouped[paper]
        for dur in conds.get("Manual", []):
            if dur >= dnf_threshold:
                ax.scatter(i, dur, color=color_dnf, s=70, zorder=4)
            else:
                ax.scatter(i, dur, color=color_manual, s=70, zorder=3)
        for dur in conds.get("AI-assisted", []):
            ax.scatter(i, dur, color=color_agent, s=70, zorder=3)
        for m in conds.get("Manual", []):
            for a in conds.get("AI-assisted", []):
                ax.plot([i, i], [m, a], color="gray", linewidth=0.7, alpha=0.4, zorder=2)

    ax.set_xticks(range(len(papers)))
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Paper", fontsize=12)
    ax.set_ylabel("Duration (minutes)")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_manual,
               markersize=8, label="Without AI"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_agent,
               markersize=8, label="With AI"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_dnf,
               markersize=8, label="Without AI, capped at 180 min"),
    ]
    style.legend_below(ax, handles=handles, ncol=3, fontsize=14, y=-0.52)

    style.save(fig, path)


def fig_duration_strip(
    df: pd.DataFrame,
    path: Path | str = DEFAULT_OUT_STRIP,
    *,
    condition_col: str = "Condition",
    duration_col: str | None = None,
    dnf_threshold: float = 180.0,
) -> None:
    """Horizontal beeswarm plot of session durations by condition."""
    if duration_col is None:
        duration_col = _find_duration_column(df)

    work = df[[condition_col, duration_col]].copy()
    work[duration_col] = pd.to_numeric(work[duration_col], errors="coerce")
    work = work.dropna(subset=[condition_col, duration_col])

    work[condition_col] = work[condition_col].astype(str).str.strip()
    work = work[work[condition_col].isin(["AI-assisted", "Manual"])].copy()
    if work.empty:
        raise ValueError("No AI-assisted or Manual duration rows found.")

    work["display_condition"] = work[condition_col].replace(LABEL_REMAP)
    work["display_condition"] = work["display_condition"].replace(
        {"Human-agent": "Human-\nagent"}
    )
    work["point_type"] = work[condition_col]
    work.loc[work[duration_col] >= dnf_threshold, "point_type"] = "Did not complete"

    color_agent = COLORBLIND_PALETTE[0]
    color_manual = COLORBLIND_PALETTE[1]
    color_dnf = "#E63946"

    # ── figure / axes geometry ──────────────────────────────────────
    fig_w = style.PAPER_W * 0.5
    fig_h = fig_w / 1.6
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0.25, 0.28, 0.71, 0.64])
    style.style_axes(ax)

    # ── beeswarm placement per condition ───────────────────────────
    sns.swarmplot(
        data=work,
        x=duration_col,
        y="display_condition",
        hue="point_type",
        order=["Human-\nagent", "Manual"],
        palette={
            "AI-assisted": color_agent,
            "Manual": color_manual,
            "Did not complete": color_dnf,
        },
        dodge=False,
        size=5,
        alpha=0.92,
        edgecolor="white",
        linewidth=0.4,
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()

    # ── axes formatting ────────────────────────────────────────────
    ax.set_xlim(0, dnf_threshold * 1.06)
    ax.set_ylabel("")
    ax.set_xlabel("Session duration (minutes)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=10, length=0)
    ax.spines["left"].set_visible(False)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.set_xticks(range(0, int(dnf_threshold) + 1, 30))

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_dnf,
               markersize=5, label="did not complete (capped at 180 min)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.43),
              fontsize=8, frameon=False, handlelength=0.8)

    style.save(fig, path)


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
    parser.add_argument("--out-per-paper", type=Path, default=DEFAULT_OUT_PER_PAPER)
    parser.add_argument("--out-strip", type=Path, default=DEFAULT_OUT_STRIP)
    parser.add_argument("--condition-col", default="Condition")
    parser.add_argument("--duration-col", default=None)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        df = _demo_data()
        out = args.out.with_name(args.out.stem + "_demo" + args.out.suffix)
        out_per_paper = args.out_per_paper.with_name(
            args.out_per_paper.stem + "_demo" + args.out_per_paper.suffix
        )
        out_strip = args.out_strip.with_name(
            args.out_strip.stem + "_demo" + args.out_strip.suffix
        )
    else:
        df = pd.read_csv(args.data)
        out = args.out
        out_per_paper = args.out_per_paper
        out_strip = args.out_strip

    fig_duration_by_condition(
        df,
        out,
        condition_col=args.condition_col,
        duration_col=args.duration_col,
    )
    print(f"Wrote {out} and {out.with_suffix('.pdf')}")

    fig_duration_per_paper(
        df,
        out_per_paper,
        condition_col=args.condition_col,
        duration_col=args.duration_col,
    )
    print(f"Wrote {out_per_paper} and {out_per_paper.with_suffix('.pdf')}")

    fig_duration_strip(
        df,
        out_strip,
        condition_col=args.condition_col,
        duration_col=args.duration_col,
    )
    print(f"Wrote {out_strip} and {out_strip.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

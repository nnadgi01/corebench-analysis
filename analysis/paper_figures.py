"""Paper-supporting figures (NeurIPS narratives N1-N6 + cost).

Separated from ``analysis/compute.py`` so the figures targeted at the
§3 narratives are easy to find, edit, and rerun in isolation. The
exploratory figures used during development (calibration,
risk-coverage, errors-vs-pass, the variance-decomposition bar, OOD
pair plot) stay in compute.py.

Aesthetic conventions, applied via ``style.apply_paper()``:
- Square-ish ~8" figures with 24 pt axis labels and 18 pt ticks /
  legend / annotations. LaTeX scales the figure down at include time.
- Scatter markers ``s=200`` with ``edgecolor='white'``, drawn above
  the dashed grid (``zorder=3``).
- Scaffolds encode both color and marker shape (so the figure
  survives in greyscale).
- Subset labeling only — annotated points are hand-picked and
  labeled with curved arrows; identification of the rest is left to
  the caption / table.
"""

from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis import style
from analysis.compute import (
    _agent_id, short_label,
    _correct_core_agent_cost, _auroc, _per_agent_metrics, _fmt_dollars,
)

style.apply_paper()


# ----- shared constants ---------------------------------------------

# Canonical (model, scaffold) cell mapping for main39. Each value is a
# single config_dir we consider representative of that cell, picked to
# avoid mixing reasoning-effort or thread-count sweeps when comparing
# scaffolds. Used by fig_accuracy_by_scaffold_model,
# fig_tokens_by_scaffold_per_model, fig_cost_by_scaffold_per_model, and
# fig_scaffold_vs_model_decomposition (in compute.py).
CANONICAL_CELLS = {
    ("opus_4_5", "claude_code"): "corebench_hard_claude_code_agent_anthropic_claude_opus_4_5_max_thinking_tokens_10000_k0",
    ("opus_4_6", "claude_code"): "corebench_hard_claude_code_agent_anthropic_claude_opus_4_6_baseline_k0",
    ("opus_4_5", "opencode"):    "corebench_hard_opencode_agent_anthropic_claude_opus_4_5_thinking_budget_10000",
    ("opus_4_6", "opencode"):    "corebench_hard_opencode_agent_anthropic_claude_opus_4_6_baseline",
    ("gpt_5_4",  "opencode"):    "corebench_hard_opencode_agent_openai_gpt_5_4_reasoning_effort_high",
    ("opus_4_5", "core_agent"):  "corebench_hard_core_agent_anthropic_claude_opus_4_5_max_steps_200",
    ("opus_4_6", "core_agent"):  "corebench_hard_core_agent_anthropic_claude_opus_4_6_max_steps_200",
    ("gpt_5_4",  "core_agent"):  "corebench_hard_core_agent_gpt_5_4_max_steps_200",
    ("gpt_5_4",  "codex"):       "corebench_hard_codex_agent_gpt_5_4_reasoning_effort_medium",
}

MODEL_DISPLAY = {"opus_4_5": "Opus 4.5", "opus_4_6": "Opus 4.6", "gpt_5_4": "GPT-5.4"}
SCAFFOLD_DISPLAY = {"claude_code": "Claude Code", "codex": "Codex CLI",
                    "core_agent": "CORE-Agent", "opencode": "OpenCode"}

# Figure-only exclusion list. GPT-5.1 stays in all data tables, but is
# omitted from plots while the main-vs-reliability discrepancy remains
# under investigation.
FIGURE_EXCLUDE_AGENT_PATTERNS = (
    "gpt_5_1",
    "GPT-5 1",
    "GPT-5.1",
)


# ----- helpers ------------------------------------------------------

def filter_figure_agents(df: pd.DataFrame) -> pd.DataFrame:
    """Return a figure-only copy with uncertain agents removed.

    This intentionally does not mutate or rewrite any source tables.
    It checks the columns used across the pipeline: raw run tables use
    ``config_dir``, aggregate tables use ``agent_id`` / ``label``.
    """
    out = df.copy()
    mask = pd.Series(False, index=out.index)
    for col in ("agent_id", "config_dir", "label"):
        if col not in out.columns:
            continue
        s = out[col].fillna("").astype(str)
        for pattern in FIGURE_EXCLUDE_AGENT_PATTERNS:
            mask |= s.str.contains(pattern, regex=False)
    return out.loc[~mask].copy()

def _canonical_cell_table(runs: pd.DataFrame) -> pd.DataFrame:
    df = runs[runs["split"] == "main39"].copy()
    df["cost"] = df.apply(_correct_core_agent_cost, axis=1)
    rows = []
    for (model, scaffold), cfg in CANONICAL_CELLS.items():
        sub = df[df["config_dir"] == cfg]
        if len(sub) == 0:
            continue
        rows.append({
            "model": model,
            "scaffold": scaffold,
            "config_dir": cfg,
            "n": len(sub),
            "accuracy": sub["successful"].astype("boolean").astype(float).mean(),
            "tokens_mean": sub["total_tokens"].astype(float).mean(),
            "cost_mean": sub["cost"].astype(float).mean(),
        })
    return pd.DataFrame(rows)


def _fmt_tokens(v: float) -> str:
    if v >= 1e6: return f"{v/1e6:.1f}M"
    if v >= 1e3: return f"{v/1e3:.0f}k"
    return f"{v:.0f}"


def _set_token_log_ticks(ax: plt.Axes) -> None:
    """Use readable token ticks on log-scale resource plots."""
    lo, hi = ax.get_xlim()
    ticks = [
        2.5e5, 5e5, 1e6, 2e6, 5e6, 1e7, 2e7,
    ]
    ticks = [t for t in ticks if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_tokens(t) for t in ticks])


def _set_token_linear_ticks(ax: plt.Axes) -> None:
    """Use readable token ticks on linear-scale resource plots."""
    lo, hi = ax.get_xlim()
    ticks = [0, 5e5, 1e6, 1.5e6, 2e6, 2.5e6, 3e6, 4e6]
    ticks = [t for t in ticks if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_tokens(t) for t in ticks])


def _scaffold_legend_handles(scaffolds: list[str]) -> list[Line2D]:
    """Proxy artists so the legend shows clean marker glyphs."""
    return [
        Line2D([0], [0], marker=style.SCAFFOLD_MARKER[sc], color="none",
               markerfacecolor=style.SCAFFOLD[sc], markeredgecolor="white",
               markeredgewidth=1.2, markersize=12,
               label=SCAFFOLD_DISPLAY[sc])
        for sc in scaffolds
    ]


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12, 1.03, label, transform=ax.transAxes,
        ha="left", va="bottom", fontsize=15, fontweight="bold",
    )


# ----- dumbbell helper ----------------------------------------------

def _dumbbell(
    ax: plt.Axes,
    cells: pd.DataFrame,
    *,
    value_col: str,
    models: list[str],
    scaffolds: list[str],
    log_x: bool = False,
    marker_size: float = 240,
):
    """Per-model row: thin gray segment from min→max scaffold value,
    scaffold dots placed on it. Returns the per-model spread (max/min for
    log scales, max-min otherwise) for caller-side annotation.
    """
    y = np.arange(len(models))
    spreads = {}
    for yi, m in zip(y, models):
        sub = cells[cells["model"] == m].dropna(subset=[value_col])
        if len(sub) < 1:
            continue
        vals = sub[value_col].astype(float)
        lo, hi = vals.min(), vals.max()
        if len(sub) >= 2:
            ax.hlines(yi, lo, hi, colors=style.GUIDE_GRAY,
                      linewidth=2.0, alpha=0.5, zorder=2)
            spreads[m] = (hi / lo) if log_x else (hi - lo)
        for _, r in sub.iterrows():
            sc = r["scaffold"]
            ax.scatter(
                r[value_col], yi, s=marker_size, color=style.SCAFFOLD[sc],
                marker=style.SCAFFOLD_MARKER[sc],
                edgecolor="white", linewidth=1.5, zorder=4,
            )

    ax.set_yticks(y, [MODEL_DISPLAY[m] for m in models])
    ax.set_ylim(-0.6, len(models) - 0.4)
    if log_x:
        ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    return spreads


def _format_dumbbell_panel(
    ax: plt.Axes,
    cells: pd.DataFrame,
    *,
    value_col: str,
    x_label: str,
    models: list[str],
    scaffolds: list[str],
    log_x: bool = False,
    xlim: tuple[float, float] | None = None,
    spread_kind: str | None = None,
    marker_size: float = 240,
):
    style.style_axes(ax)
    spreads = _dumbbell(
        ax, cells, value_col=value_col,
        models=models, scaffolds=scaffolds, log_x=log_x,
        marker_size=marker_size,
    )
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(x_label, labelpad=8)
    if value_col == "accuracy":
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{int(round(v*100))}%"))

    if spread_kind == "pp":
        x_text = ax.get_xlim()[1]
        for yi, m in zip(np.arange(len(models)), models):
            if m in spreads:
                ax.text(x_text, yi, rf"$\Delta = {spreads[m]*100:.0f}$ pp",
                        ha="left", va="center", fontsize=11,
                        color=style.GUIDE_GRAY, style="italic")
    elif spread_kind == "ratio":
        x_text = ax.get_xlim()[1] / 1.06
        for yi, m in zip(np.arange(len(models)), models):
            if m in spreads:
                ax.text(x_text, yi, rf"${spreads[m]:.1f}\times$",
                        ha="right", va="center", fontsize=11,
                        color=style.GUIDE_GRAY, style="italic")
    return spreads


# ----- N1: accuracy by scaffold × model ------------------------------

def fig_accuracy_by_scaffold_model(runs: pd.DataFrame, path: Path):
    """N1. Same model, different scaffolds → different accuracy."""
    cells = _canonical_cell_table(runs)
    models = ["opus_4_5", "opus_4_6", "gpt_5_4"]
    scaffolds = ["claude_code", "opencode", "core_agent", "codex"]

    fig, ax = plt.subplots(figsize=(style.PAPER_W, 4.9))
    style.style_axes(ax)
    spreads = _dumbbell(ax, cells, value_col="accuracy",
                        models=models, scaffolds=scaffolds, log_x=False)

    ax.set_xlim(0.40, 1.04)
    ax.set_xlabel("Accuracy", labelpad=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(round(v*100))}%"))

    # Per-row spread annotation at the right end.
    for yi, m in zip(np.arange(len(models)), models):
        if m in spreads:
            ax.text(1.02, yi, rf"$\Delta = {spreads[m]*100:.0f}$ pp",
                    ha="left", va="center", fontsize=15,
                    color=style.GUIDE_GRAY, style="italic")

    style.legend_below(ax, handles=_scaffold_legend_handles(scaffolds), y=-0.26)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    style.save(fig, path)
    return cells


# ----- N2: tokens per scaffold per model ----------------------------

def fig_tokens_by_scaffold_per_model(runs: pd.DataFrame, path: Path):
    """N2. Per-model spread of tokens across scaffolds (now within ~2×
    after the cost-extraction fix; prior to the fix CoreAgent looked
    10× higher because per-step values were summed cumulatively)."""
    cells = _canonical_cell_table(runs)
    models = ["opus_4_5", "opus_4_6", "gpt_5_4"]
    scaffolds = ["claude_code", "opencode", "core_agent", "codex"]

    fig, ax = plt.subplots(figsize=(style.PAPER_W, 4.9))
    style.style_axes(ax)
    spreads = _dumbbell(ax, cells, value_col="tokens_mean",
                        models=models, scaffolds=scaffolds, log_x=True)

    xmin = cells["tokens_mean"].min() / 1.6
    xmax = cells["tokens_mean"].max() * 5.0
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Tokens per task", labelpad=10)

    for yi, m in zip(np.arange(len(models)), models):
        if m in spreads:
            ax.text(xmax / 1.05, yi, rf"${spreads[m]:.1f}\times$",
                    ha="right", va="center", fontsize=15,
                    color=style.GUIDE_GRAY, style="italic")

    style.legend_below(ax, handles=_scaffold_legend_handles(scaffolds), y=-0.26)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    style.save(fig, path)


def fig_scaffold_model_comparison_3panel(runs: pd.DataFrame, path: Path):
    """Three-panel scaffold × model comparison: accuracy, tokens, cost."""
    cells = _canonical_cell_table(runs)
    models = ["opus_4_5", "opus_4_6", "gpt_5_4"]
    scaffolds = ["claude_code", "opencode", "core_agent", "codex"]

    fig, axes = plt.subplots(
        1, 3, figsize=(style.PAPER_W * 1.85, 4.7),
        sharey=True,
    )

    _format_dumbbell_panel(
        axes[0], cells, value_col="accuracy",
        x_label="Accuracy", models=models, scaffolds=scaffolds,
        xlim=(0.40, 1.04), spread_kind="pp", marker_size=110,
    )
    _format_dumbbell_panel(
        axes[1], cells, value_col="tokens_mean",
        x_label="Tokens per task", models=models, scaffolds=scaffolds,
        log_x=True,
        xlim=(cells["tokens_mean"].min() / 1.6,
              cells["tokens_mean"].max() * 5.0),
        spread_kind="ratio", marker_size=110,
    )
    _format_dumbbell_panel(
        axes[2], cells, value_col="cost_mean",
        x_label="Cost per task (USD)", models=models, scaffolds=scaffolds,
        log_x=True,
        xlim=(cells["cost_mean"].min() / 1.6,
              cells["cost_mean"].max() * 5.0),
        spread_kind="ratio", marker_size=110,
    )

    for ax, title, letter in zip(
        axes, ["Accuracy", "Token use", "Cost"], ["A", "B", "C"]
    ):
        ax.set_title(title, loc="left", fontsize=15)
        _panel_label(ax, letter)
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    fig.legend(
        handles=_scaffold_legend_handles(scaffolds),
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=4, fontsize=12, frameon=False,
        handlelength=1.4, columnspacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    style.save(fig, path)
    return cells


# ----- N3: tokens / cost vs accuracy --------------------------------

# Per-figure manual annotation positions. Each entry maps a substring
# match against agent_id to (x_offset_log_units, y_offset, label_text).
# Substrings let us annotate by family without enumerating every config.
TOKEN_ACC_LABELS_MAIN = {
    "core_agent_anthropic_claude_opus_4_6":  (+0.10, -0.14, "CORE-Agent · Opus 4.6"),
    "core_agent_gpt_5_4":                    (-0.38, +0.10, "CORE-Agent · GPT-5.4"),   # shifted right
    "codex_agent_gpt_5_3_codex":             (-0.25, +0.08, "Codex · GPT-5.3 Codex"),  # top-left corner
    "opencode_agent_openai_gpt_5_4":          (-0.24, -0.16, "OpenCode · GPT-5.4"),    # shifted down
}

COST_ACC_LABELS_MAIN = {
    "core_agent_anthropic_claude_opus_4_6":  (-0.45, -0.10, "CORE-Agent · Opus 4.6"),
    "core_agent_gpt_5_4":                    (-0.16, +0.10, "CORE-Agent · GPT-5.4"),
    "codex_agent_gpt_5_3_codex":             (-0.20, +0.06, "Codex · GPT-5.3 Codex"),  # top-left corner
    "opencode_agent_openai_gpt_5_4":          (-0.30, -0.17, "OpenCode · GPT-5.4"),
}

_COMPOSITE_SQUARE_CROP = dict(left=-0.62, bottom=0.20, width=7.22, height=6.02)


def _resource_accuracy_handles(
    df: pd.DataFrame, fit_excludes: tuple[str, ...], x_scale: str = "log"
) -> list[Line2D]:
    fit_label = "OLS linear fit" if x_scale == "linear" else "OLS log-linear fit"
    if "core_agent" in fit_excludes:
        fit_label = "OLS fit (excl. CORE-Agent)"
    return [
        Line2D([0], [0], color=style.GUIDE_GRAY, linestyle="--",
               label=fit_label),
        *_scaffold_legend_handles([s for s in
                                   ["claude_code", "codex", "core_agent", "opencode"]
                                   if s in df["scaffold"].unique()]),
    ]


def _plot_resource_accuracy_panel(
    ax: plt.Axes,
    eff: pd.DataFrame,
    *,
    x_col: str = "tot_tok_mean",
    x_label: str = "Mean tokens per task (log scale)",
    x_scale: str = "log",
    fit_line: bool = True,
    fit_excludes: tuple[str, ...] = (),
    label_substrings: dict[str, tuple[float, float, str]] | None = None,
    marker_size: float = 135,
    annotation_fontsize: int = 13,
) -> pd.DataFrame:
    df = eff.copy()
    df = df.dropna(subset=[x_col]).copy()
    df["log_x"] = np.log10(df[x_col].astype(float))
    df["fit_x"] = df["log_x"] if x_scale == "log" else df[x_col].astype(float)

    style.style_axes(ax)

    if fit_line:
        fit_pts = df[~df["scaffold"].isin(fit_excludes)]
        if len(fit_pts) >= 3:
            slope, intercept = np.polyfit(fit_pts["fit_x"], fit_pts["accuracy"], 1)
            x_line = np.linspace(df["fit_x"].min(), df["fit_x"].max(), 50)
            plot_x = 10 ** x_line if x_scale == "log" else x_line
            ax.plot(
                plot_x, slope * x_line + intercept,
                color=style.GUIDE_GRAY, linewidth=1.65, linestyle="--",
                zorder=2,
            )

    for sc, sub in df.groupby("scaffold"):
        ax.scatter(
            sub[x_col], sub["accuracy"],
            s=marker_size, color=style.SCAFFOLD[sc],
            marker=style.SCAFFOLD_MARKER[sc],
            edgecolor="white", linewidth=1.55, zorder=3, alpha=0.90,
        )

    if label_substrings is None:
        label_substrings = TOKEN_ACC_LABELS_MAIN
    for substr, (dx_log, dy, text) in label_substrings.items():
        match = df[df["agent_id"].fillna("").str.contains(substr, regex=False)]
        if len(match) == 0:
            continue
        r = match.iloc[0]
        anchor = (r[x_col], r["accuracy"])
        label_xy = (10 ** (np.log10(anchor[0]) + dx_log), anchor[1] + dy)
        style.annotate_with_arrow(
            ax, anchor_xy=anchor, label_xy=label_xy,
            text=text, color=style.SCAFFOLD[r["scaffold"]],
            fontsize=annotation_fontsize,
        )

    ax.set_xscale(x_scale)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ymin = max(0.30, df["accuracy"].min() - 0.10)
    ax.set_ylim(ymin, 1.10)
    ax.set_yticks([t for t in np.arange(0.5, 1.01, 0.1) if ymin <= t <= 1.0])
    if ymin > 0.05:
        style.add_y_axis_break(ax)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(round(v*100))}%"))
    xmin, xmax = df[x_col].min(), df[x_col].max()
    if x_scale == "log":
        ax.set_xlim(xmin / 2.5, xmax * 2.5)
    else:
        ax.set_xlim(0, xmax * 1.35)
    if x_col in {"tot_tok_mean", "tokens_mean", "total_tokens"}:
        if x_scale == "log":
            _set_token_log_ticks(ax)
        else:
            _set_token_linear_ticks(ax)
    return df


def fig_tokens_vs_accuracy(eff: pd.DataFrame, path: Path, *, title: str | None = None,
                           x_col: str = "tot_tok_mean",
                           x_label: str = "Mean tokens per task (log scale)",
                           x_scale: str = "log",
                           fit_line: bool = True,
                           fit_excludes: tuple[str, ...] = (),
                           label_substrings: dict[str, tuple[float, float, str]] | None = None):
    """Scatter of resource-usage (log) vs accuracy. Each scaffold gets
    a distinct color *and* marker. Only points whose ``agent_id``
    matches a key in ``label_substrings`` get a curved-arrow label;
    everything else identifies via the legend and the caption.

    Pass ``x_col="cost_mean"`` and a matching ``x_label`` to render
    the dollar analog.
    """
    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    df = _plot_resource_accuracy_panel(
        ax, eff, x_col=x_col, x_label=x_label,
        x_scale=x_scale, fit_line=fit_line, fit_excludes=fit_excludes,
        label_substrings=label_substrings,
    )
    handles = _resource_accuracy_handles(df, fit_excludes, x_scale=x_scale)
    style.legend_below(
        ax, handles=handles, ncol=3, y=-0.20,
        columnspacing=1.2, handlelength=1.25,
    )
    fig.tight_layout()
    style.save(fig, path)
    return df


def fig_tokens_vs_accuracy_square(
    eff: pd.DataFrame,
    path: Path,
    *,
    x_col: str = "tot_tok_mean",
    x_label: str = "Mean tokens per task (log scale)",
    x_scale: str = "log",
    fit_line: bool = True,
    fit_excludes: tuple[str, ...] = (),
    label_substrings: dict[str, tuple[float, float, str]] | None = None,
):
    """Fixed-canvas square variant for composite LaTeX layouts."""
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_axes([0.15, 0.16, 0.70, 0.76])
    df = _plot_resource_accuracy_panel(
        ax, eff, x_col=x_col, x_label=x_label,
        x_scale=x_scale, fit_line=fit_line, fit_excludes=fit_excludes,
        label_substrings=label_substrings,
        marker_size=115, annotation_fontsize=14,
    )
    ax.tick_params(labelsize=17)
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    style.save_fixed_crop(fig, path, **_COMPOSITE_SQUARE_CROP)
    return df


def fig_resource_accuracy_2panel(eff: pd.DataFrame, path: Path):
    """Two-panel mean tokens/cost vs accuracy comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(style.PAPER_W * 1.65, 6.2),
                             sharey=True)
    df_tok = _plot_resource_accuracy_panel(
        axes[0], eff, x_col="tot_tok_mean",
        x_label="Mean tokens per task (log scale)",
        label_substrings={},
        marker_size=90, annotation_fontsize=11,
    )
    df_cost = _plot_resource_accuracy_panel(
        axes[1], eff, x_col="cost_mean",
        x_label="Mean cost per task (\\$, log scale)",
        label_substrings={},
        marker_size=90, annotation_fontsize=11,
    )
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=True)
    for ax, title, letter in zip(
        axes, ["Token use", "Cost"], ["A", "B"]
    ):
        ax.set_title(title, loc="left", fontsize=15)
        _panel_label(ax, letter)
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(15)
        ax.yaxis.label.set_size(15)
    handles = _resource_accuracy_handles(df_tok, ())
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=3, fontsize=12, frameon=False,
        handlelength=1.25, columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    style.save(fig, path)
    return df_tok, df_cost


def fig_resource_accuracy_landscape(eff: pd.DataFrame, path: Path):
    """Combined landscape two-panel: tokens and cost vs accuracy, shared legend."""
    fig, axes = plt.subplots(
        1, 2, figsize=(style.PAPER_W * 1.85, 4.6), sharey=True,
    )
    df_tok = _plot_resource_accuracy_panel(
        axes[0], eff,
        x_col="tot_tok_mean",
        x_label="Mean tokens per task (log scale)",
        label_substrings=TOKEN_ACC_LABELS_MAIN,
        marker_size=115, annotation_fontsize=16,
    )
    df_cost = _plot_resource_accuracy_panel(
        axes[1], eff,
        x_col="cost_mean",
        x_label="Mean cost per task (\\$, log scale)",
        label_substrings=COST_ACC_LABELS_MAIN,
        marker_size=115, annotation_fontsize=16,
    )
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", labelleft=False)
    handles = _resource_accuracy_handles(df_tok, ())
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
        ncol=len(handles), fontsize=14, frameon=False,
        handlelength=1.25, columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    style.save(fig, path)
    return df_tok, df_cost


def fig_tokens_vs_accuracy_landscape(
    eff: pd.DataFrame,
    path: Path,
    *,
    x_col: str = "tot_tok_mean",
    x_label: str = "Mean tokens per task (log scale)",
    x_scale: str = "log",
    fit_line: bool = True,
    fit_excludes: tuple[str, ...] = (),
    label_substrings: dict[str, tuple[float, float, str]] | None = None,
) -> pd.DataFrame:
    """Landscape single-panel for individual paper figures.

    Wider than tall (PAPER_W × 5.2") so each figure reads cleanly as a
    standalone panel at roughly 0.9\\textwidth in LaTeX.
    """
    fig, ax = plt.subplots(figsize=(style.PAPER_W, 5.0))
    df = _plot_resource_accuracy_panel(
        ax, eff, x_col=x_col, x_label=x_label,
        x_scale=x_scale, fit_line=fit_line, fit_excludes=fit_excludes,
        label_substrings=label_substrings,
        marker_size=135, annotation_fontsize=14,
    )
    handles = _resource_accuracy_handles(df, fit_excludes, x_scale=x_scale)
    style.legend_below(
        ax, handles=handles, ncol=3, y=-0.22,
        columnspacing=1.2, handlelength=1.25,
    )
    fig.tight_layout()
    style.save(fig, path)
    return df


# ----- N4: Codex variants on main vs OOD ----------------------------

def fig_codex_main_vs_ood(runs: pd.DataFrame, path: Path):
    """N4. Token usage on main39 vs OOD19 per Codex variant. Identity
    line marks 'OOD costs the same as main'; points above use more
    tokens on OOD, below use fewer. Codex · GPT-5.3 sits at the bottom-
    left corner (cheapest on both splits)."""
    df = runs[runs["scaffold"] == "codex"].copy()
    df = df[df["split"].isin(["main39", "ood19"])]
    df["agent_id"] = df["config_dir"].map(_agent_id)

    agg = (df.groupby(["agent_id", "split"])["total_tokens"]
             .mean().reset_index())
    wide = agg.pivot(index="agent_id", columns="split",
                     values="total_tokens").dropna()
    wide["label"] = wide.index.map(short_label).map(
        lambda s: re.sub(r"^Codex ", "", s))

    is_53 = wide.index.str.contains("gpt_5_3")

    fig, ax = plt.subplots(figsize=(style.PAPER_W, style.PAPER_H))
    style.style_axes(ax)

    # Identity line spanning the data range (with padding).
    lo = min(wide["main39"].min(), wide["ood19"].min()) / 1.6
    hi = max(wide["main39"].max(), wide["ood19"].max()) * 1.6
    ax.plot([lo, hi], [lo, hi], color=style.GUIDE_GRAY,
            linestyle="--", linewidth=1.0, zorder=2,
            label=r"$y = x$ (OOD = main)")

    # Two scatter passes so the gold Codex · GPT-5.3 sits visually on top.
    ax.scatter(wide.loc[~is_53, "main39"], wide.loc[~is_53, "ood19"],
               s=200, color=style.SCAFFOLD["codex"],
               marker=style.SCAFFOLD_MARKER["codex"],
               edgecolor="white", linewidth=1.5, zorder=3,
               label="Codex variant")
    ax.scatter(wide.loc[is_53, "main39"], wide.loc[is_53, "ood19"],
               s=240, color=style.HIGHLIGHT_GOLD,
               marker=style.SCAFFOLD_MARKER["codex"],
               edgecolor="white", linewidth=1.5, zorder=4,
               label="Codex · GPT-5.3")

    # Annotate only a curated subset with arrows; the cluster identifies
    # itself by sitting in the upper-right "Codex variant" cloud and the
    # outlier (Codex · GPT-5.3) is gold-highlighted.
    callouts = {
        "GPT-5 3 Codex":  (+0.20, -0.15),   # gold outlier
        "GPT-5 4 medium t9": (+0.05, +0.30),  # most expensive on OOD
        "GPT-5 4 medium t6": (-0.30, -0.30),  # cheapest non-5.3
        "GPT-5 4 high":   (+0.30, -0.05),   # below identity line
    }
    callout_label = {
        "GPT-5 3 Codex":     "Codex · GPT-5.3",
        "GPT-5 4 medium t9": "Codex · GPT-5.4 medium · t=0.9",
        "GPT-5 4 medium t6": "Codex · GPT-5.4 medium · t=0.6",
        "GPT-5 4 high":      "Codex · GPT-5.4 high",
    }
    for idx, row in wide.iterrows():
        lab = row["label"]
        if lab not in callouts:
            continue
        x, y = row["main39"], row["ood19"]
        dx_log, dy_log = callouts[lab]
        anchor = (x, y)
        label_xy = (10 ** (np.log10(x) + dx_log),
                    10 ** (np.log10(y) + dy_log))
        is_53 = "5 3" in lab
        color = style.HIGHLIGHT_GOLD if is_53 else style.SCAFFOLD["codex"]
        style.annotate_with_arrow(
            ax, anchor_xy=anchor, label_xy=label_xy,
            text=callout_label[lab], color=color, fontsize=15,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Tokens / task on main39")
    ax.set_ylabel("Tokens / task on OOD19")

    handles = [
        Line2D([0], [0], color=style.GUIDE_GRAY, linestyle="--",
               linewidth=1.0, label=r"$y = x$"),
        Line2D([0], [0], marker=style.SCAFFOLD_MARKER["codex"],
               markerfacecolor=style.SCAFFOLD["codex"],
               markeredgecolor="white", markeredgewidth=1.2,
               markersize=12, color="none", label="Codex variant"),
        Line2D([0], [0], marker=style.SCAFFOLD_MARKER["codex"],
               markerfacecolor=style.HIGHLIGHT_GOLD,
               markeredgecolor="white", markeredgewidth=1.2,
               markersize=13, color="none", label="Codex · GPT-5.3"),
    ]
    style.legend_below(
        ax, handles=handles, ncol=2, y=-0.20,
        columnspacing=1.3, handlelength=1.25,
    )
    fig.tight_layout()
    style.save(fig, path)
    return wide


# ----- N5: consistency vs accuracy ----------------------------------

# Per-panel manual offsets for label placement.
CONSISTENCY_LABEL_OFFSETS = {
    # leftmost (low-accuracy) point goes left; the cluster goes right.
    "left":  (-0.05,  0.00),
    "right": (+0.04,  0.00),
    "down":  (+0.04, -0.05),
    "up":    (+0.04, +0.05),
}


def fig_consistency_vs_accuracy(rel: pd.DataFrame, path: Path):
    """N5. Capability ↔ consistency correlation."""
    df = rel.dropna(subset=["pass_at_1"]).copy()
    if len(df) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(style.PAPER_W * 1.4, style.PAPER_H))
    panels = [
        ("outcome_consistency",  "Outcome consistency",  "#2ca02c"),
        ("resource_consistency", "Resource consistency", "#1f77b4"),
    ]

    for ax, panel in zip(axes, panels):
        _plot_consistency_panel(ax, df, *panel)
    fig.tight_layout()
    style.save(fig, path)


def fig_outcome_consistency_vs_accuracy(rel: pd.DataFrame, path: Path):
    """Single-panel outcome consistency vs reliability-sample accuracy."""
    df = rel.dropna(subset=["pass_at_1"]).copy()
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(style.PAPER_W * 0.78, style.PAPER_H))
    _plot_consistency_panel(
        ax, df, "outcome_consistency", "Outcome consistency", "#2ca02c")
    fig.tight_layout()
    style.save(fig, path)


def fig_resource_consistency_vs_accuracy(rel: pd.DataFrame, path: Path):
    """Single-panel resource consistency vs reliability-sample accuracy."""
    df = rel.dropna(subset=["pass_at_1"]).copy()
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(style.PAPER_W * 0.78, style.PAPER_H))
    _plot_consistency_panel(
        ax, df, "resource_consistency", "Resource consistency", "#1f77b4")
    fig.tight_layout()
    style.save(fig, path)


def fig_outcome_consistency_vs_accuracy_square(rel: pd.DataFrame, path: Path):
    """Fixed-canvas square variant for composite LaTeX layouts."""
    df = rel.dropna(subset=["pass_at_1"]).copy()
    if len(df) == 0:
        return
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_axes([0.15, 0.16, 0.70, 0.76])
    _plot_consistency_panel(
        ax, df, "outcome_consistency", "Outcome consistency", "#2ca02c",
        annotation_fontsize=16, r_fontsize=25,
        label_only={"Codex · GPT-5", "Codex · GPT-5.4"})
    ax.tick_params(labelsize=17)
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    style.save_fixed_crop(fig, path, **_COMPOSITE_SQUARE_CROP)


def fig_resource_consistency_vs_accuracy_square(rel: pd.DataFrame, path: Path):
    """Fixed-canvas square variant for composite LaTeX layouts."""
    df = rel.dropna(subset=["pass_at_1"]).copy()
    if len(df) == 0:
        return
    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_axes([0.15, 0.16, 0.70, 0.76])
    _plot_consistency_panel(
        ax, df, "resource_consistency", "Resource consistency", "#1f77b4",
        annotation_fontsize=16, r_fontsize=25,
        label_only={"Codex · GPT-5", "Codex · GPT-5.4"})
    ax.tick_params(labelsize=17)
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    style.save_fixed_crop(fig, path, **_COMPOSITE_SQUARE_CROP)


def _plot_consistency_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    col: str,
    label: str,
    color: str,
    *,
    annotation_fontsize: int = 12,
    r_fontsize: int = 18,
    show_labels: bool = True,
    label_only: set[str] | None = None,
) -> None:
    """Shared consistency panel with data-aware scaling and (optionally) labeled points.

    ``label_only`` restricts annotations to the named short-form labels; pass
    ``None`` (default) to label every point.
    """

    def padded_limits(s: pd.Series, *, floor: float = 0.0,
                      ceil: float = 1.0, min_pad: float = 0.01) -> tuple[float, float]:
        lo, hi = float(s.min()), float(s.max())
        span = max(hi - lo, min_pad)
        pad = max(span * 0.18, min_pad)
        return max(floor, lo - pad), min(ceil, hi + pad)

    def short(s: str) -> str:
        # Codex GPT-5 4 medium → Codex · GPT-5.4
        # Codex GPT-5 3 Codex medium → Codex · GPT-5.3
        if "GPT-5 3 Codex" in s:
            return "Codex · GPT-5.3 Codex"
        was_codex = s.startswith("Codex ")
        s = re.sub(r"^Codex ", "", s)
        m = re.match(r"GPT-5(?:\s+(\d))?\s*(?:Codex\s+)?medium$", s)
        if m:
            digit = m.group(1) or ""
            base = f"GPT-5.{digit}" if digit else "GPT-5"
            return f"Codex · {base}" if was_codex or "Codex" in s else base
        return s

    label_offsets = {
        "outcome_consistency": {
            "Codex · GPT-5":   (-0.08, +0.18),
            "Codex · GPT-5.2": (+0.06, +0.10),
            "Codex · GPT-5.3 Codex": (+0.04, +0.12),
            "Codex · GPT-5.4": (-0.20, +0.04),
        },
        "resource_consistency": {
            "Codex · GPT-5":         (+0.06, -0.32),  # below the bottom-left point
            "Codex · GPT-5.2":       (-0.05, +0.35),  # above the left cluster (unused when label_only set)
            "Codex · GPT-5.3 Codex": (-0.38, +0.20),  # left+up (unused when label_only set)
            "Codex · GPT-5.4":       (-0.22, +0.18),  # upper-left of top-right point
        },
    }

    style.style_axes(ax)
    sub = df.dropna(subset=[col]).sort_values("pass_at_1").reset_index(drop=True)
    xlim = padded_limits(sub["pass_at_1"], min_pad=0.008)
    ylim = padded_limits(sub[col], min_pad=0.015)

    if len(sub) >= 2:
        r = np.corrcoef(sub["pass_at_1"], sub[col])[0, 1]
        ax.text(0.04, 0.95, rf"$r = {r:+.2f}$",
                transform=ax.transAxes, fontsize=r_fontsize,
                color=style.GUIDE_GRAY, va="top")
        xs = np.linspace(xlim[0], xlim[1], 50)
        slope, intercept = np.polyfit(sub["pass_at_1"], sub[col], 1)
        ax.plot(xs, slope * xs + intercept,
                color=style.GUIDE_GRAY, linewidth=1.55,
                linestyle="--", zorder=2)
    ax.scatter(sub["pass_at_1"], sub[col], s=230, color=color,
               edgecolor="white", linewidth=1.7, zorder=3)

    if show_labels:
        x_span = xlim[1] - xlim[0]
        y_span = ylim[1] - ylim[0]
        for _, row in sub.iterrows():
            anchor = (row["pass_at_1"], row[col])
            text = short(row["label"])
            if label_only is not None and text not in label_only:
                continue
            dx_frac, dy_frac = label_offsets[col].get(text, (+0.04, +0.10))
            dx = dx_frac * x_span
            dy = dy_frac * y_span
            style.annotate_with_arrow(
                ax, anchor_xy=anchor,
                label_xy=(anchor[0] + dx, anchor[1] + dy),
                text=text, color=color, fontsize=annotation_fontsize,
            )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Accuracy (pass@1)")
    ax.set_ylabel(label)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(round(v*100))}%"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(round(v*100))}%"))


# ----- N6: predictability -------------------------------------------

# Shared bins for both predictability figures.
_ERR_EDGES   = [-0.5, 0.5, 2.5, 5.5, 10.5, 20.5, np.inf]
_ERR_CENTERS = np.array([0, 1.5, 4, 8, 15.5, 25])
_ERR_LABELS  = ["0", "1–2", "3–5", "6–10", "11–20", "21+"]


def _short_agent_label(a: str) -> str:
    if "gpt_5_3_codex" in a:
        return "Codex · GPT-5.3 Codex"
    label = short_label(a)
    was_codex = label.startswith("Codex ")
    s = re.sub(r"^Codex ", "", label)
    m = re.match(r"GPT-5(?:\s+(\d))?\s*(?:Codex\s+)?medium$", s)
    if not m:
        return label if was_codex else s
    digit = m.group(1) or ""
    base = f"GPT-5.{digit}" if digit else "GPT-5"
    return f"Codex · {base}" if was_codex or "Codex" in s else base


def _bin_metrics(sub: pd.DataFrame) -> pd.DataFrame:
    """Per-error-bin pass rate, mean confidence, and standard errors."""
    s = sub.copy()
    s["bin"] = pd.cut(s["num_errors"], bins=_ERR_EDGES, labels=False,
                      include_lowest=True)
    def se(x): return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return s.groupby("bin").agg(
        pass_mean=("pass", "mean"),
        pass_se=("pass", se),
        conf_mean=("confidence", "mean"),
        conf_se=("confidence", se),
        n=("pass", "size"),
    ).reindex(range(len(_ERR_CENTERS)))


_PASS_COLOR = "#2ca02c"
_CONF_COLOR = "#9467bd"


def fig_discrimination_bar(df: pd.DataFrame, path: Path):
    """Paper-style confidence discrimination panel.

    AUROC is 0.5 at chance, so the axis is centered tightly around the
    informative range instead of using the exploratory 0-1 scale.
    """
    metrics = (
        _per_agent_metrics(df)
        .dropna(subset=["P_AUROC"])
        .sort_values("P_AUROC", ascending=True)
        .copy()
    )
    if len(metrics) == 0:
        return
    metrics["label"] = (
        metrics["agent_id"]
        .map(_short_agent_label)
        .str.replace(r"^Codex · ", "", regex=True)
    )

    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_axes([0.23, 0.16, 0.62, 0.76])
    style.style_axes(ax)

    y = np.arange(len(metrics))
    se = metrics["P_AUROC_se"].fillna(0).to_numpy()
    vals = metrics["P_AUROC"].to_numpy()
    ax.barh(
        y, vals, xerr=se,
        height=0.58, color="#17becf", alpha=0.90,
        edgecolor="white", linewidth=1.45,
        error_kw=dict(ecolor=style.GUIDE_GRAY, capsize=5, lw=1.45),
        zorder=3,
    )
    ax.axvline(
        0.5, color=style.GUIDE_GRAY, linestyle="--",
        linewidth=1.55, zorder=2,
    )
    ax.text(
        0.502, len(metrics) - 0.55, "chance",
        ha="left", va="center", fontsize=17,
        color=style.GUIDE_GRAY,
    )
    for yi, v, e in zip(y, vals, se):
        ax.text(
            min(v + e + 0.012, 0.695), yi, f"{v:.2f}",
            va="center", ha="left", fontsize=17,
        )

    ax.set_yticks(y, metrics["label"])
    ax.set_xlim(0.30, 0.70)
    ax.set_xticks(np.arange(0.30, 0.71, 0.10))
    ax.set_xlabel("Discrimination (AUROC)")
    ax.set_ylabel("")
    ax.tick_params(labelsize=17)
    ax.xaxis.label.set_size(25)
    ax.grid(axis="y", visible=False)
    style.save_fixed_crop(fig, path, **_COMPOSITE_SQUARE_CROP)


def fig_calibration(df: pd.DataFrame, path: Path, *, n_bins: int = 5):
    """Paper-style confidence calibration panel."""
    d = df.dropna(subset=["confidence", "successful"]).copy()
    if len(d) == 0:
        return
    d["pass"] = d["successful"].astype(float)
    d["agent_id"] = d["config_dir"].map(_agent_id)
    agents = sorted(d["agent_id"].unique())
    cmap = plt.get_cmap("plasma", len(agents) + 2)

    fig = plt.figure(figsize=(6.4, 6.4))
    ax = fig.add_axes([0.15, 0.16, 0.70, 0.76])
    style.style_axes(ax)

    ax.plot(
        [0, 1], [0, 1],
        color=style.GUIDE_GRAY, linestyle="--",
        linewidth=1.55, zorder=1, label="Perfect calibration",
    )
    bins = np.linspace(0, 1, n_bins + 1)
    size_per_run = 2.4
    for i, agent_id in enumerate(agents):
        sub = d[d["agent_id"] == agent_id].copy()
        bin_idx = np.clip(
            np.digitize(sub["confidence"], bins) - 1,
            0, n_bins - 1,
        )
        sub = sub.assign(bin=bin_idx)
        agg = sub.groupby("bin").agg(
            conf_mean=("confidence", "mean"),
            pass_mean=("pass", "mean"),
            n=("pass", "size"),
        ).reset_index()
        color = cmap(i + 1)
        legend_label = re.sub(r"^Codex · ", "", _short_agent_label(agent_id))
        ax.plot(
            agg["conf_mean"], agg["pass_mean"],
            color=color, alpha=0.76, linewidth=2.2,
            zorder=2, label=legend_label,
        )
        ax.scatter(
            agg["conf_mean"], agg["pass_mean"],
            s=agg["n"] * size_per_run,
            color=color, alpha=0.90,
            edgecolors="white", linewidths=1.15,
            zorder=3,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Empirical pass rate in bin")
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(round(v*100))}%"))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(round(v*100))}%"))
    ax.tick_params(labelsize=17)
    ax.xaxis.label.set_size(25)
    ax.yaxis.label.set_size(25)
    ax.legend(
        loc="lower right", fontsize=15, ncol=1,
        frameon=True, framealpha=0.92,
        borderpad=0.40, labelspacing=0.35,
        handlelength=1.3,
    )
    style.save_fixed_crop(fig, path, **_COMPOSITE_SQUARE_CROP)


def _predictability_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], color=_PASS_COLOR, marker="o", markersize=11,
               markeredgecolor="white", markeredgewidth=1.2,
               linewidth=2.4, label="Empirical pass rate"),
        Line2D([0], [0], color=_CONF_COLOR, marker="s", markersize=11,
               markeredgecolor="white", markeredgewidth=1.2,
               linewidth=2.4, label="Self-rated confidence"),
    ]


def _plot_predictability_agent_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    agent_id: str,
    *,
    title_fontsize: int = 14,
    tick_fontsize: int = 10,
    show_ylabel: bool = False,
    strip_codex_prefix: bool = False,
) -> None:
    style.style_axes(ax)
    agg = _bin_metrics(df[df["agent_id"] == agent_id])
    x = _ERR_CENTERS[agg.index.values]

    ax.fill_between(x, agg["pass_mean"] - agg["pass_se"].fillna(0),
                    agg["pass_mean"] + agg["pass_se"].fillna(0),
                    color=_PASS_COLOR, alpha=0.18, zorder=2)
    ax.plot(x, agg["pass_mean"], color=_PASS_COLOR, marker="o",
            markersize=8, markeredgecolor="white", markeredgewidth=1.1,
            linewidth=2.15, zorder=3, label="pass rate")
    ax.fill_between(x, agg["conf_mean"] - agg["conf_se"].fillna(0),
                    agg["conf_mean"] + agg["conf_se"].fillna(0),
                    color=_CONF_COLOR, alpha=0.18, zorder=2)
    ax.plot(x, agg["conf_mean"], color=_CONF_COLOR, marker="s",
            markersize=7, markeredgecolor="white", markeredgewidth=1.1,
            linewidth=2.15, zorder=3, label="confidence")

    title = _short_agent_label(agent_id)
    if strip_codex_prefix:
        title = re.sub(r"^Codex · ", "", title)
    ax.set_title(title, fontsize=title_fontsize, loc="left")
    ax.set_xticks(_ERR_CENTERS, _ERR_LABELS)
    ax.tick_params(axis="x", labelsize=tick_fontsize, rotation=30, pad=2)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{int(round(v*100))}%"))
    if show_ylabel:
        ax.set_ylabel("Rate")


def fig_predictability_per_agent(df: pd.DataFrame, path: Path):
    """N6a. Small multiples (one panel per agent). Each panel plots
    pass rate and self-rated confidence vs binned tool-error count.
    The vertical gap is the grader's penalty for errors that don't
    actually predict failure."""
    df = df.dropna(subset=["confidence", "successful", "num_errors"]).copy()
    df["pass"] = df["successful"].astype(float)
    df["agent_id"] = df["config_dir"].map(_agent_id)
    agents = sorted(df["agent_id"].unique())

    n = len(agents)
    ncols = min(2, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(style.PAPER_W * 1.05, 6.4),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).flatten()

    for i, a in enumerate(agents):
        _plot_predictability_agent_panel(
            axes[i], df, a, show_ylabel=(i % ncols == 0),
        )

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.text(
        0.5, 0.09, "Failed bash commands per run",
        ha="center", va="center", fontsize=20,
    )
    fig.legend(handles=_predictability_handles(), loc="lower center",
               bbox_to_anchor=(0.5, 0.00), ncol=2,
               fontsize=12, frameon=False, handlelength=2.0)
    fig.subplots_adjust(
        left=0.10, right=0.99, top=0.94, bottom=0.22,
        wspace=0.18, hspace=0.50,
    )
    style.save(fig, path)


def fig_predictability_per_agent_vertical(df: pd.DataFrame, path: Path):
    """N6a vertical variant. Same per-agent predictability panels as
    ``fig_predictability_per_agent``, stacked for narrow layouts."""
    df = df.dropna(subset=["confidence", "successful", "num_errors"]).copy()
    df["pass"] = df["successful"].astype(float)
    df["agent_id"] = df["config_dir"].map(_agent_id)
    agents = sorted(df["agent_id"].unique())

    n = len(agents)
    if n == 0:
        return

    fig, axes = plt.subplots(
        n, 1,
        figsize=(style.PAPER_W * 0.78, 2.35 * n + 1.05),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes).flatten()

    for i, a in enumerate(agents):
        _plot_predictability_agent_panel(
            axes[i], df, a,
            title_fontsize=14, tick_fontsize=11,
            show_ylabel=False, strip_codex_prefix=True,
        )
        if i < n - 1:
            axes[i].tick_params(axis="x", labelbottom=False)

    fig.text(
        0.045, 0.54, "Rate",
        ha="center", va="center", rotation="vertical", fontsize=17,
    )
    fig.text(
        0.5, 0.075, "Failed bash commands per run",
        ha="center", va="center", fontsize=16,
    )
    fig.legend(handles=_predictability_handles(), loc="lower center",
               bbox_to_anchor=(0.5, 0.015), ncol=2,
               fontsize=12, frameon=False, handlelength=2.0)
    fig.subplots_adjust(
        left=0.16, right=0.985, top=0.975, bottom=0.16,
        hspace=0.34,
    )
    style.save(fig, path)


def fig_reliability_3panel(rel: pd.DataFrame, runs: pd.DataFrame, path: Path):
    """Three-panel reliability figure: two consistency panels plus
    predictability by agent as a 2x2 block."""
    rel = rel.dropna(subset=["pass_at_1"]).copy()
    runs = runs.dropna(subset=["confidence", "successful", "num_errors"]).copy()
    runs["pass"] = runs["successful"].astype(float)
    runs["agent_id"] = runs["config_dir"].map(_agent_id)
    agents = sorted(runs["agent_id"].unique())
    if len(rel) == 0 or len(agents) == 0:
        return

    fig = plt.figure(figsize=(style.PAPER_W * 1.35, 9.4))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 1.32], hspace=0.42, wspace=0.28,
    )

    ax_out = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[0, 1])
    _plot_consistency_panel(
        ax_out, rel, "outcome_consistency", "Outcome consistency", "#2ca02c")
    _plot_consistency_panel(
        ax_res, rel, "resource_consistency", "Resource consistency", "#1f77b4")
    _panel_label(ax_out, "A")
    _panel_label(ax_res, "B")

    pred_gs = gs[1, :].subgridspec(2, 2, hspace=0.55, wspace=0.18)
    pred_axes = [fig.add_subplot(pred_gs[i, j]) for i in range(2) for j in range(2)]
    for i, a in enumerate(agents[:4]):
        _plot_predictability_agent_panel(
            pred_axes[i], runs, a, title_fontsize=13, tick_fontsize=9,
            show_ylabel=(i % 2 == 0),
        )
    for j in range(len(agents), len(pred_axes)):
        pred_axes[j].set_visible(False)
    _panel_label(pred_axes[0], "C")

    fig.text(
        0.5, 0.055, "Failed bash commands per run",
        ha="center", va="center", fontsize=18,
    )
    fig.legend(
        handles=_predictability_handles(), loc="lower center",
        bbox_to_anchor=(0.5, 0.005), ncol=2,
        fontsize=12, frameon=False, handlelength=2.0,
    )
    fig.subplots_adjust(left=0.08, right=0.99, top=0.985, bottom=0.12)
    style.save(fig, path)


def fig_predictability_consolidated(df: pd.DataFrame, path: Path):
    """N6b. Consolidated predictability figure: pass rate, confidence,
    and confidence calibration by agent."""
    df = df.dropna(subset=["confidence", "successful", "num_errors"]).copy()
    df["pass"] = df["successful"].astype(float)
    df["agent_id"] = df["config_dir"].map(_agent_id)
    agents = sorted(df["agent_id"].unique())
    cmap = plt.get_cmap("plasma", len(agents) + 2)

    fig, (axL, axR, axC) = plt.subplots(
        1, 3, figsize=(style.PAPER_W * 1.85, style.PAPER_H_SHORT + 1.15),
    )
    style.style_axes(axL); style.style_axes(axR); style.style_axes(axC)

    for i, a in enumerate(agents):
        agg = _bin_metrics(df[df["agent_id"] == a])
        x = _ERR_CENTERS[agg.index.values]
        axL.plot(x, agg["pass_mean"], color=cmap(i + 1), marker="o",
                 markersize=9, markeredgecolor="white", markeredgewidth=1.0,
                 linewidth=2.0, zorder=3, label=_short_agent_label(a))
        axR.plot(x, agg["conf_mean"], color=cmap(i + 1), marker="s",
                 markersize=9, markeredgecolor="white", markeredgewidth=1.0,
                 linewidth=2.0, zorder=3, label=_short_agent_label(a))

    for ax, ylab in [(axL, "Empirical pass rate"),
                     (axR, "Self-rated confidence")]:
        ax.set_xticks(_ERR_CENTERS, _ERR_LABELS)
        ax.tick_params(axis="both", labelsize=11)
        ax.tick_params(axis="x", rotation=30, pad=2)
        ax.set_xlabel("Failed bash commands per run", fontsize=13)
        ax.set_ylabel(ylab, fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: f"{int(round(v*100))}%"))

    # Calibration panel: same structure as figs/calibration.png, but
    # compact enough to live beside the predictability curves.
    axC.plot([0, 1], [0, 1], color=style.GUIDE_GRAY, linestyle="--",
             linewidth=1.0, zorder=1)
    bins = np.linspace(0, 1, 6)
    size_per_run = 2.0
    for i, a in enumerate(agents):
        sub = df[df["agent_id"] == a].copy()
        bin_idx = np.clip(np.digitize(sub["confidence"], bins) - 1, 0, 4)
        sub = sub.assign(bin=bin_idx)
        agg = sub.groupby("bin").agg(
            conf_mean=("confidence", "mean"),
            pass_mean=("pass", "mean"),
            n=("pass", "size"),
        ).reset_index()
        axC.plot(agg["conf_mean"], agg["pass_mean"], color=cmap(i + 1),
                 alpha=0.70, linewidth=1.5, zorder=2)
        axC.scatter(agg["conf_mean"], agg["pass_mean"],
                    s=agg["n"] * size_per_run,
                    color=cmap(i + 1), alpha=0.90,
                    edgecolors="white", linewidths=0.8, zorder=3)
    axC.set_xlim(0, 1)
    axC.set_ylim(0, 1.05)
    axC.set_xlabel("Mean confidence in bin", fontsize=13)
    axC.set_ylabel("Empirical pass rate in bin", fontsize=14)
    axC.tick_params(axis="both", labelsize=11)
    axC.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(round(v*100))}%"))
    axC.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(round(v*100))}%"))

    for ax, title, letter in zip(
        [axL, axR, axC],
        ["Pass rate vs errors", "Confidence vs errors", "Calibration"],
        ["A", "B", "C"],
    ):
        ax.set_title(title, loc="left", fontsize=15)
        _panel_label(ax, letter)

    handles = [
        Line2D([0], [0], color=cmap(i + 1), marker="o", markersize=9,
               markeredgecolor="white", markeredgewidth=1.0,
               linewidth=2.0, label=_short_agent_label(a))
        for i, a in enumerate(agents)
    ]
    leg = fig.legend(handles=handles, loc="lower center",
                     bbox_to_anchor=(0.5, -0.01),
                     ncol=3, fontsize=12, frameon=False,
                     handlelength=2.0, columnspacing=1.6)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30, wspace=0.34)
    style.save(fig, path)


# ----- variance decomposition (accuracy + log-tokens) ---------------

def fig_variance_decomposition(runs: pd.DataFrame, path: Path):
    """Type-II variance shares for ``successful`` (0/1) and
    log(tokens), attributable to scaffold / model / capsule / residual.
    Restricted to the canonical-cell crossing in main39 where we have
    multiple models per scaffold and multiple scaffolds per model.
    """
    import statsmodels.api as sm  # noqa: F401  (statsmodels OLS via formula)
    from statsmodels.formula.api import ols

    df = runs[runs["split"] == "main39"].copy()
    keep = []
    for (model, scaffold), exact_cfg in CANONICAL_CELLS.items():
        sub = df[df["config_dir"] == exact_cfg].copy()
        if len(sub) == 0:
            continue
        sub["model_f"] = model
        sub["scaffold_f"] = scaffold
        keep.append(sub)
    work = pd.concat(keep, ignore_index=True)
    work["accuracy"] = work["successful"].astype("boolean").astype(float)
    work["log_tokens"] = np.log(work["total_tokens"].astype(float).clip(lower=1))

    terms = ["C(model_f)", "C(scaffold_f)", "C(capsule_id)"]
    targets = [("accuracy", "Accuracy"), ("log_tokens", "log(tokens)")]
    factor_order = ["C(scaffold_f)", "C(model_f)", "C(capsule_id)", "residual"]
    factor_labels = {
        "C(scaffold_f)": "Scaffold",
        "C(model_f)": "Model",
        "C(capsule_id)": "Task",
        "residual": "Residual",
    }
    factor_colors = {
        "Scaffold": style.SCAFFOLD["codex"],
        "Model":    style.SCAFFOLD["opencode"],
        "Task":     "#7f7f7f",
        "Residual": "#cccccc",
    }
    light_text = {"Scaffold", "Model", "Task"}

    rows = []
    for col, _ in targets:
        d = work.dropna(subset=[col]).copy()
        full = ols(f"{col} ~ " + " + ".join(terms), data=d).fit()
        ss_total = ((d[col] - d[col].mean()) ** 2).sum()
        ss_resid = (full.resid ** 2).sum()
        for t in terms:
            others = [x for x in terms if x != t]
            reduced = ols(f"{col} ~ " + " + ".join(others), data=d).fit()
            ss_term = (reduced.resid ** 2).sum() - ss_resid
            rows.append({"target": col, "factor": t,
                         "share": ss_term / ss_total})
        rows.append({"target": col, "factor": "residual",
                     "share": ss_resid / ss_total})
    decomp = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(style.PAPER_W * 1.1, 4.5))
    style.style_axes(ax)
    y = np.arange(len(targets))
    left = np.zeros(len(targets))
    for f in factor_order:
        lab = factor_labels[f]
        vals = [decomp[(decomp["target"] == col) & (decomp["factor"] == f)]
                ["share"].iloc[0] for col, _ in targets]
        ax.barh(y, vals, left=left, color=factor_colors[lab],
                edgecolor="white", linewidth=1.0, label=lab, zorder=3)
        for i, v in enumerate(vals):
            if v > 0.06:
                ax.text(left[i] + v / 2, y[i], f"{int(round(v*100))}%",
                        ha="center", va="center", fontsize=14,
                        color="white" if lab in light_text else "black",
                        weight="bold")
        left += np.array(vals)
    # Give the y-axis a touch more headroom so floated labels never clip.
    ax.set_ylim(-0.7, len(targets) - 0.3)

    ax.set_yticks(y, [label for _, label in targets])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Share of variance (Type II)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda v, _: f"{int(round(v*100))}%"))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.36)
    style.legend_below(ax, ncol=4, y=-0.55)
    style.save(fig, path)
    return decomp


# ----- cost analog of N2 --------------------------------------------

def fig_cost_by_scaffold_per_model(runs: pd.DataFrame, path: Path):
    """Dollars-per-task analog of fig_tokens_by_scaffold_per_model."""
    cells = _canonical_cell_table(runs)
    models = ["opus_4_5", "opus_4_6", "gpt_5_4"]
    scaffolds = ["claude_code", "opencode", "core_agent", "codex"]

    fig, ax = plt.subplots(figsize=(style.PAPER_W, 4.9))
    style.style_axes(ax)
    spreads = _dumbbell(ax, cells, value_col="cost_mean",
                        models=models, scaffolds=scaffolds, log_x=True)

    xmin = cells["cost_mean"].min() / 1.6
    xmax = cells["cost_mean"].max() * 5.0
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Cost per task (USD)", labelpad=10)

    for yi, m in zip(np.arange(len(models)), models):
        if m in spreads:
            ax.text(xmax / 1.05, yi, rf"${spreads[m]:.1f}\times$",
                    ha="right", va="center", fontsize=15,
                    color=style.GUIDE_GRAY, style="italic")

    style.legend_below(ax, handles=_scaffold_legend_handles(scaffolds), y=-0.26)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    style.save(fig, path)

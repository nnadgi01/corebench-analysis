"""Regenerate every §3 figure from the shared data tables in ``data/``.

Run from the repo root: python -m analysis.regenerate_figures
"""

from __future__ import annotations
from pathlib import Path

import pandas as pd

from analysis import paper_figures as P
from analysis import uplift_figures as U


DATA        = Path("data")
PAPER_OUT   = Path("figs")
SCRATCH_OUT = Path("figs/scratch")
FIGS_OUT    = Path("figs")
UPLIFT_DATA = DATA / "RCT_responses_cleaned.csv"
PAPER_OUT.mkdir(parents=True, exist_ok=True)
SCRATCH_OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    runs = pd.read_parquet(DATA / "runs.parquet")
    eff  = pd.read_csv(DATA / "efficiency_per_agent.csv")
    rel  = pd.read_csv(DATA / "reliability_per_agent.csv")

    # Figure-only exclusion: keep GPT-5.1 in source tables, but omit it
    # from plots while its main-vs-reliability discrepancy is unsettled.
    runs_fig = P.filter_figure_agents(runs)
    eff_fig = P.filter_figure_agents(eff)
    rel_fig = P.filter_figure_agents(rel)

    eff_main = eff_fig[eff_fig["split"] == "main39"]
    eff_ood  = eff_fig[eff_fig["split"] == "ood19"]
    rel_runs = runs_fig[runs_fig["split"] == "reliability"]

    # ----- §3 narratives N1–N6 --------------------------------------
    P.fig_accuracy_by_scaffold_model(
        runs_fig, SCRATCH_OUT / "accuracy_by_scaffold_model.png")
    P.fig_tokens_by_scaffold_per_model(
        runs_fig, SCRATCH_OUT / "tokens_by_scaffold_per_model.png")
    P.fig_scaffold_model_comparison_3panel(
        runs_fig, SCRATCH_OUT / "scaffold_model_comparison_3panel.png")
    P.fig_resource_accuracy_landscape(
        eff_main, PAPER_OUT / "resource_accuracy.png",
    )
    P.fig_tokens_vs_accuracy_landscape(
        eff_main, SCRATCH_OUT / "tokens_vs_accuracy.png",
    )
    P.fig_tokens_vs_accuracy_square(
        eff_main, SCRATCH_OUT / "tokens_vs_accuracy_square.png",
    )
    P.fig_tokens_vs_accuracy(
        eff_main, SCRATCH_OUT / "tokens_vs_accuracy_linear.png",
        x_label="Mean tokens per task (linear scale)",
        x_scale="linear",
    )
    P.fig_resource_accuracy_2panel(
        eff_main, SCRATCH_OUT / "resource_accuracy_2panel.png",
    )
    P.fig_resource_accuracy_2panel(
        eff_main, SCRATCH_OUT / "resource_accuracy_2panel_nolabels.png")
    P.fig_codex_main_vs_ood(
        runs_fig, SCRATCH_OUT / "codex_main_vs_ood.png")
    P.fig_consistency_vs_accuracy(
        rel_fig, SCRATCH_OUT / "consistency_vs_accuracy.png")
    P.fig_outcome_consistency_vs_accuracy_square(
        rel_fig, PAPER_OUT / "outcome_consistency_vs_accuracy.png")
    P.fig_resource_consistency_vs_accuracy_square(
        rel_fig, PAPER_OUT / "resource_consistency_vs_accuracy.png")
    P.fig_predictability_per_agent(
        rel_runs, SCRATCH_OUT / "predictability_per_agent.png")
    P.fig_predictability_per_agent_vertical(
        rel_runs, PAPER_OUT / "predictability_per_agent_vertical.png")
    P.fig_reliability_3panel(
        rel_fig, rel_runs, SCRATCH_OUT / "reliability_3panel.png")
    P.fig_predictability_consolidated(
        rel_runs, SCRATCH_OUT / "predictability_consolidated.png")
    P.fig_calibration(
        rel_runs, PAPER_OUT / "calibration.png")
    P.fig_discrimination_bar(
        rel_runs, PAPER_OUT / "discrimination_bar.png")
    # Also overwrite the legacy locations so ``figs/calibration.pdf`` and
    # ``figs/discrimination_bar.pdf`` have the same panel dimensions.
    P.fig_calibration(
        rel_runs, FIGS_OUT / "calibration.png")
    P.fig_discrimination_bar(
        rel_runs, FIGS_OUT / "discrimination_bar.png")
    P.fig_variance_decomposition(
        runs_fig, SCRATCH_OUT / "variance_decomposition.png")

    # ----- uplift RCT -----------------------------------------------
    # Optional until the released CSV is added to figs/data/.
    if UPLIFT_DATA.exists():
        uplift = pd.read_csv(UPLIFT_DATA)
        U.fig_duration_by_condition(
            uplift, PAPER_OUT / "uplift_duration_by_condition.png")

    # ----- cost analogs ---------------------------------------------
    P.fig_cost_by_scaffold_per_model(
        runs_fig, SCRATCH_OUT / "cost_by_scaffold_per_model.png")
    P.fig_tokens_vs_accuracy_landscape(
        eff_main, SCRATCH_OUT / "cost_vs_accuracy.png",
        x_col="cost_mean",
        x_label="Mean cost per task (\\$, log scale)",
        label_substrings=P.COST_ACC_LABELS_MAIN,
    )
    P.fig_tokens_vs_accuracy_square(
        eff_main, SCRATCH_OUT / "cost_vs_accuracy_square.png",
        x_col="cost_mean",
        x_label="Mean cost per task (\\$, log scale)",
        label_substrings=P.COST_ACC_LABELS_MAIN,
    )

    # ----- OOD variants ---------------------------------------------
    P.fig_tokens_vs_accuracy(
        eff_ood, SCRATCH_OUT / "tokens_vs_accuracy_ood.png",
    )
    P.fig_tokens_vs_accuracy(
        eff_ood, SCRATCH_OUT / "cost_vs_accuracy_ood.png",
        x_col="cost_mean",
        x_label="Mean cost per task (\\$, log scale)",
        fit_excludes=(),
        label_substrings=P.COST_ACC_LABELS_MAIN,
    )

    print(f"Wrote {len(list(PAPER_OUT.glob('*.png')))} active figures "
          f"to {PAPER_OUT}/ and {len(list(SCRATCH_OUT.glob('*.png')))} "
          f"scratch figures to {SCRATCH_OUT}/")


if __name__ == "__main__":
    main()

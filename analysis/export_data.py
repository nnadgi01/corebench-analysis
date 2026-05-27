"""Export the small set of general tables that back every figure in
``figs/paper/`` and ``figs/teammate/``. A teammate can validate any
claim by re-aggregating one of these.

Three tables cover everything:

- ``runs.parquet`` / ``runs.csv``    one row per (run_id, config_dir,
  capsule_id, rep_idx). Adds ``cost`` (the corrected per-row cost
  used by all dollar figures) and ``agent_id`` (config_dir with the
  reliability ``_kN`` suffix stripped) on top of the canonical
  schema.
- ``efficiency_per_agent.csv``        per (agent_id, split) aggregate:
  accuracy, mean / median / std of tokens and cost. Backs every bar
  chart and every tokens/cost-vs-accuracy scatter.
- ``reliability_per_agent.csv``       per agent_id over the k=5
  reliability split: pass@1, pass-at-least-1-of-k, pass-all-k,
  outcome / resource consistency, mean / median confidence. Backs the
  consistency and predictability figures.

Run: python -m analysis.export_data
"""

from __future__ import annotations
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from analysis.compute import (
    _correct_core_agent_cost, _agent_id,
    efficiency_table, reliability_table,
)


OUT = Path("figs/data")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    runs = pd.read_parquet("data/runs.parquet").copy()
    runs["cost"]     = runs.apply(_correct_core_agent_cost, axis=1)
    runs["agent_id"] = runs["config_dir"].map(_agent_id)

    # --- runs (single source of truth) -------------------------------
    shutil.copy2("data/runs.parquet", OUT / "runs.parquet")
    runs.to_csv(OUT / "runs.csv", index=False)

    # --- efficiency: per (agent_id, split) ---------------------------
    eff_main = efficiency_table(runs[runs["split"] == "main39"], "main39")
    eff_ood  = efficiency_table(runs[runs["split"] == "ood19"],  "ood19")
    eff = pd.concat([eff_main, eff_ood], ignore_index=True)
    # Keep a stable column order; some columns differ between splits,
    # missing values get NaN.
    front = ["split", "agent_id", "scaffold", "model", "n", "accuracy"]
    rest = [c for c in eff.columns if c not in front]
    eff = eff[front + rest]
    eff.to_csv(OUT / "efficiency_per_agent.csv", index=False)

    # --- reliability: per agent_id (k=5) -----------------------------
    rel, _ = reliability_table(runs[runs["split"] == "reliability"], k=5)
    rel.to_csv(OUT / "reliability_per_agent.csv", index=False)

    # --- README: what each file contains -----------------------------
    (OUT / "README.md").write_text(_readme())

    print(f"Wrote {len(runs):,} runs, "
          f"{len(eff)} (agent × split) cells, "
          f"{len(rel)} reliability agents to {OUT}/")


def _readme() -> str:
    return (
        "# Figure data\n\n"
        "Three tables back every figure in `figs/paper/` and "
        "`figs/teammate/`. Regenerate with "
        "`python -m analysis.export_data`.\n\n"
        "## `runs.parquet` / `runs.csv`\n"
        "One row per (run_id, config_dir, capsule_id, rep_idx). The "
        "canonical schema is documented in `RECAP.md`; this export "
        "adds two derived columns:\n\n"
        "- `cost` — corrected per-row cost used by every dollar "
        "figure. For `core_agent` rows where the smolagents stdout "
        "cost is broken (< 0.5% of token-priced) we substitute the "
        "token-priced rate-card cost. See `analysis/cost_audit.md`.\n"
        "- `agent_id` — `config_dir` with the reliability `_kN` "
        "rep-suffix stripped, so reliability reps collapse to one "
        "agent identity.\n\n"
        "## `efficiency_per_agent.csv`\n"
        "Per `(agent_id, split)` cell. Columns: `n` (capsules with a "
        "non-null outcome), `accuracy` (pass@1), `tot_tok_mean / "
        "_median / _std`, `cost_mean / _median / _std`. Backs every "
        "bar chart and the tokens / cost-vs-accuracy scatters.\n\n"
        "## `reliability_per_agent.csv`\n"
        "Per `agent_id` over the k=5 reliability split. Columns: "
        "`pass_at_1`, `pass_at_least_1_of_k`, `pass_all_k`, "
        "`outcome_consistency`, `resource_consistency`, "
        "`confidence_mean`, `confidence_median`. Backs "
        "`consistency_vs_accuracy.png` and "
        "`predictability_per_agent.png` / "
        "`predictability_consolidated.png`.\n"
    )


if __name__ == "__main__":
    main()

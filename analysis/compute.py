"""Compute §3.1 (reliability) and §3.2 (efficiency) tables from
data/runs.parquet. Outputs CSVs + summary.md in analysis/ and headline
figures in figs/.

Plots emitted:
  figs/tokens_vs_accuracy.png               — main39, all scaffolds
  figs/reliability_outcome_consistency.png  — (2·p̂−1)² per agent
  figs/reliability_resource_consistency.png — exp(−CV_tokens) per agent
  figs/calibration.png                      — binned mean confidence vs accuracy
  figs/calibration_bar.png                  — P_cal = 1 − ECE per agent
  figs/risk_coverage.png                    — risk vs coverage (top-confidence first)
  figs/discrimination_bar.png               — P_AUROC per agent
"""

from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from extractor import pricing
from analysis import style

style.apply()

RUNS = Path("data/runs.parquet")
OUT = Path("analysis")
FIGS = Path("figs")
PAPER_FIGS = FIGS / "paper"
OUT.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)
PAPER_FIGS.mkdir(exist_ok=True)

PALETTE = style.SCAFFOLD

# Canonical (model, scaffold) cells live in analysis.paper_figures;
# fig_scaffold_vs_model_decomposition imports them locally below to
# avoid an import cycle (paper_figures depends on compute helpers).


# ----- agent identity -------------------------------------------------

def _agent_id(config_dir: str) -> str:
    """Strip trailing _kN and _codex_version_X.Y.Z so reps and version-pinned
    re-runs collapse to the same agent identity."""
    s = re.sub(r"_k\d+$", "", config_dir)
    s = re.sub(r"_codex_version_[\d.]+", "", s)
    return s


def short_label(agent_id: str) -> str:
    s = agent_id.replace("corebench_hard_", "")
    s = s.replace("_agent_", " ")
    s = s.replace("anthropic_", "")
    s = s.replace("openai_", "")
    s = s.replace("_reasoning_effort_", " ")
    s = s.replace("_max_thinking_tokens_10000", "")
    s = s.replace("_thinking_budget_10000", " think10k")
    s = s.replace("_max_steps_200", "")
    s = s.replace("_baseline", "")
    s = s.replace("_max_threads_", " t")
    s = s.replace("claude_code", "CC")
    s = s.replace("opencode", "OC")
    s = s.replace("core", "CA")
    s = s.replace("codex", "Codex")
    s = s.replace("claude_opus_", "Opus ")
    s = s.replace("gpt_", "GPT-")
    s = s.replace("_codex", "-codex")
    s = s.replace("_", " ").strip()
    return s


# ----- §3.2 efficiency -----------------------------------------------

def _token_priced_cost(row) -> float | None:
    """Token-priced fallback cost for a single run.

    All four parsers store ``input_tokens`` as the *uncached* input
    (verified in extractor/parse_*.py). Cache reads live in
    ``cached_input_tokens`` and are billed at the cached-input rate
    via pricing.cost_usd. Earlier versions of this function passed
    ``cached_input_tokens=0`` here, which silently dropped cache
    cost — see analysis/cost_audit.md for the impact.
    """
    in_t = row.get("input_tokens")
    out_t = row.get("output_tokens")
    cached = row.get("cached_input_tokens")
    in_t = int(in_t) if pd.notna(in_t) else 0
    out_t = int(out_t) if pd.notna(out_t) else 0
    cached = int(cached) if pd.notna(cached) else 0
    if in_t == 0 and out_t == 0 and cached == 0:
        return None
    return pricing.cost_usd(
        row["model"],
        input_tokens=in_t,
        cached_input_tokens=cached,
        output_tokens=out_t,
    )


def _correct_core_agent_cost(row) -> float | None:
    # Smolagents' stdout cost is unreliable: missing rate-card entries for
    # some Opus model ids zero out the number entirely (Opus 4.5 here),
    # while in-card models report a value that depends on a price table we
    # don't control. Always recompute from token totals using pricing.py
    # so 4.5 and 4.6 are priced by the same method.
    if row["scaffold"] != "core_agent":
        return row["total_cost_usd"]
    est = _token_priced_cost(row)
    return est if est is not None else row.get("total_cost_usd")


def efficiency_table(df: pd.DataFrame, split_label: str) -> pd.DataFrame:
    df = df.copy()
    df["agent_id"] = df["config_dir"].map(_agent_id)
    df["total_cost_usd_stdout"] = df["total_cost_usd"]
    df["total_cost_usd_estimated"] = df.apply(_token_priced_cost, axis=1)
    df["total_cost_usd"] = df.apply(_correct_core_agent_cost, axis=1)

    agg = df.groupby("agent_id").agg(
        n=("successful", "size"),
        n_pass=("successful", "sum"),
        accuracy=("successful", "mean"),

        cost_mean=("total_cost_usd", "mean"),
        cost_median=("total_cost_usd", "median"),
        cost_total=("total_cost_usd", "sum"),
        cost_n=("total_cost_usd", lambda s: s.notna().sum()),

        in_tok_mean=("input_tokens", "mean"),
        cached_in_mean=("cached_input_tokens", "mean"),
        out_tok_mean=("output_tokens", "mean"),
        tot_tok_mean=("total_tokens", "mean"),

        dur_mean_s=("duration_s", "mean"),
        dur_median_s=("duration_s", "median"),
        dur_n=("duration_s", lambda s: s.notna().sum()),

        turns_mean=("num_turns", "mean"),
        tools_mean=("num_tool_calls", "mean"),
        tools_median=("num_tool_calls", "median"),
    ).reset_index()

    pass_only = df[df["successful"] == True]
    pass_cost = pass_only.groupby("agent_id")["total_cost_usd"].mean().rename("cost_mean_passed")
    agg = agg.merge(pass_cost, on="agent_id", how="left")

    scaffold = df.groupby("agent_id")["scaffold"].first().rename("scaffold")
    model = df.groupby("agent_id")["model"].first().rename("model")
    agg = agg.merge(scaffold, on="agent_id").merge(model, on="agent_id")

    agg["label"] = agg["agent_id"].map(short_label)
    agg["split"] = split_label
    cols = [
        "split", "agent_id", "label", "scaffold", "model",
        "n", "n_pass", "accuracy",
        "cost_mean", "cost_median", "cost_mean_passed", "cost_total", "cost_n",
        "in_tok_mean", "cached_in_mean", "out_tok_mean", "tot_tok_mean",
        "dur_mean_s", "dur_median_s", "dur_n",
        "turns_mean", "tools_mean", "tools_median",
    ]
    return agg[cols]


# ----- §3.1 reliability ----------------------------------------------

def reliability_table(df: pd.DataFrame, k: int = 5):
    df = df.copy()
    df["agent_id"] = df["config_dir"].map(_agent_id)
    df["pass"] = df["successful"].astype("boolean").astype(float)

    def _cv(s):
        m = s.mean()
        return s.std() / m if pd.notna(m) and m > 0 else np.nan

    by_task = df.groupby(["agent_id", "capsule_id"]).agg(
        n_reps=("pass", "size"),
        n_pass=("pass", "sum"),
        cost_mean=("total_cost_usd", "mean"),
        cost_std=("total_cost_usd", "std"),
        cost_cv=("total_cost_usd", _cv),
        tokens_mean=("total_tokens", "mean"),
        tokens_std=("total_tokens", "std"),
        tokens_cv=("total_tokens", _cv),
        duration_mean=("duration_s", "mean"),
        duration_cv=("duration_s", _cv),
        tools_mean=("num_tool_calls", "mean"),
        tools_cv=("num_tool_calls", _cv),
    ).reset_index()

    # Per-task outcome consistency = (2 * p_hat - 1)^2, normalized Bernoulli
    # variance: 1.0 when reps unanimously agree, 0.0 at p_hat = 0.5.
    p_hat = by_task["n_pass"] / by_task["n_reps"]
    by_task["outcome_consistency"] = (2 * p_hat - 1) ** 2
    by_task.loc[by_task["n_reps"] < 2, "outcome_consistency"] = np.nan

    # Per-task resource consistency: C_res = exp(-CV_tokens), the single
    # resource form of C_res = exp(-mean_r CV_r) with R = {total tokens}.
    by_task["resource_consistency"] = np.exp(-by_task["tokens_cv"])

    rows = []
    for agent_id, sub in by_task.groupby("agent_id"):
        full = sub[sub["n_reps"] == k]
        if len(full) == 0:
            continue
        pass_at_least_one = (full["n_pass"] >= 1).mean()  # pass@5
        pass_all = (full["n_pass"] == k).mean()           # pass^5
        pass_zero = (full["n_pass"] == 0).mean()
        discordant = ((full["n_pass"] > 0) & (full["n_pass"] < k)).mean()
        pass_at_1 = full["n_pass"].mean() / k

        oc = full["outcome_consistency"]
        rc = full["resource_consistency"].dropna()
        rows.append({
            "agent_id": agent_id,
            "label": short_label(agent_id),
            "n_tasks": len(sub),
            "n_tasks_full_k": len(full),
            "pass_at_1": pass_at_1,
            "pass_at_least_1_of_k": pass_at_least_one,
            "pass_all_k": pass_all,
            "pass_zero_k": pass_zero,
            "discordant_frac": discordant,
            "outcome_consistency": oc.mean(),
            "outcome_consistency_se": oc.std(ddof=1) / np.sqrt(len(oc)) if len(oc) > 1 else np.nan,
            "resource_consistency": rc.mean(),
            "resource_consistency_se": rc.std(ddof=1) / np.sqrt(len(rc)) if len(rc) > 1 else np.nan,
            "cost_cv_mean": full["cost_cv"].mean(),
            "tokens_cv_mean": full["tokens_cv"].mean(),
            "tools_cv_mean": full["tools_cv"].mean(),
        })
    out = pd.DataFrame(rows).sort_values("pass_at_1", ascending=False)
    return out, by_task


# ----- figures --------------------------------------------------------


def fig_outcome_consistency(rel: pd.DataFrame, k: int, path: Path):
    """Bar per agent: mean per-task outcome consistency (2*p_hat - 1)^2,
    where p_hat = n_pass / K. 1.0 means reps unanimously agree on every
    task; 0.0 means reps split 50/50 on every task. Error bars are
    SEM across tasks."""
    df = rel.dropna(subset=["outcome_consistency"]).copy()
    df = df.sort_values("outcome_consistency").copy()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(df) + 1.2)))
    ax.barh(df["label"], df["outcome_consistency"],
            xerr=df["outcome_consistency_se"].fillna(0),
            color="#2ca02c", edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="black", capsize=3, lw=1))
    for y, (v, se) in enumerate(zip(df["outcome_consistency"],
                                    df["outcome_consistency_se"].fillna(0))):
        ax.text(v + se + 0.01, y, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.10)
    ax.set_xlabel(f"Outcome consistency  (mean of (2·p̂−1)² across tasks, k={k} reps; error = SEM)")
    ax.set_title(f"Outcome consistency (1 = unanimous, 0 = 50/50 split)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    style.save(fig, path)


def fig_resource_consistency(rel: pd.DataFrame, path: Path):
    """Bar per agent: mean across tasks of C_res = exp(−CV_tokens), the
    single-resource form (R = {total tokens}) of the C_res formula. 1.0
    means reps consume identical tokens; → 0 means wildly variable.
    Error bars are SEM across tasks."""
    df = rel.dropna(subset=["resource_consistency"]).copy()
    df = df.sort_values("resource_consistency").copy()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(df) + 1.2)))
    ax.barh(df["label"], df["resource_consistency"],
            xerr=df["resource_consistency_se"].fillna(0),
            color="#1f77b4", edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="black", capsize=3, lw=1))
    for y, (v, se) in enumerate(zip(df["resource_consistency"],
                                    df["resource_consistency_se"].fillna(0))):
        ax.text(v + se + 0.01, y, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.10)
    ax.set_xlabel("Resource consistency  C_res = exp(−CV$_{\\mathrm{tokens}}$)  (error = SEM)")
    ax.set_title("Resource consistency on token usage (1 = identical, 0 = wildly variable)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    style.save(fig, path)


def fig_calibration(df: pd.DataFrame, path: Path, *, n_bins: int = 5):
    """Calibration plot: bin runs by post-hoc confidence and compare
    mean confidence vs empirical pass rate in each bin. Diagonal is
    perfect calibration. Marker area ∝ bin frequency."""
    df = df.dropna(subset=["confidence", "successful"]).copy()
    df["pass"] = df["successful"].astype(float)
    df["agent_id"] = df["config_dir"].map(_agent_id)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--",
            label="perfect calibration", linewidth=1)
    bins = np.linspace(0, 1, n_bins + 1)
    cmap = plt.get_cmap("tab10")
    SIZE_PER_RUN = 2.5  # marker area (pt²) per run

    for i, (agent_id, sub) in enumerate(df.groupby("agent_id")):
        label = short_label(agent_id)
        bin_idx = np.clip(np.digitize(sub["confidence"], bins) - 1, 0, n_bins - 1)
        sub = sub.assign(bin=bin_idx)
        agg = sub.groupby("bin").agg(
            conf_mean=("confidence", "mean"),
            pass_mean=("pass", "mean"),
            n=("pass", "size"),
        ).reset_index()
        color = cmap(i % 10)
        ax.plot(agg["conf_mean"], agg["pass_mean"], color=color,
                alpha=0.65, linewidth=1.2, label=label, zorder=2)
        ax.scatter(agg["conf_mean"], agg["pass_mean"],
                   s=agg["n"] * SIZE_PER_RUN,
                   color=color, alpha=0.85,
                   edgecolors="black", linewidths=0.4, zorder=3)

    # Size legend
    for ref_n in (20, 60, 120):
        ax.scatter([], [], s=ref_n * SIZE_PER_RUN, color="lightgray",
                   edgecolors="black", linewidths=0.4,
                   label=f"n = {ref_n}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mean confidence in bin")
    ax.set_ylabel("Empirical pass rate in bin")
    ax.set_title(f"Confidence calibration  ({n_bins} bins, marker area ∝ bin n)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2,
              labelspacing=1.2, handletextpad=1.0,
              borderpad=0.6, columnspacing=1.2,
              framealpha=0.95)
    fig.tight_layout()
    style.save(fig, path)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """AUROC via the rank-sum identity. labels in {0,1}."""
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for kk in range(i, j + 1):
                ranks[order[kk]] = avg
        i = j + 1
    rank_pos_sum = ranks[pos].sum()
    return float((rank_pos_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _roc_curve(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores, kind="mergesort")
    s = scores[order]
    y = labels[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    tpr = tps / max(1, n_pos)
    fpr = fps / max(1, n_neg)
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return fpr, tpr


def _risk_coverage(scores: np.ndarray, labels: np.ndarray):
    """Sort runs by confidence descending; for each prefix of size k,
    coverage = k/N and risk = error rate among the top-k. Returns
    (coverage, risk) arrays."""
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    n = len(y)
    if n == 0:
        return np.array([0.0]), np.array([0.0])
    err = (1 - y).astype(float)
    cum_err = np.cumsum(err)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = cum_err / k
    return coverage, risk


def fig_risk_coverage(df: pd.DataFrame, path: Path):
    """Risk-coverage panel plot: one subplot per agent. Each panel shows
    three curves:
      - model      = sort by post-hoc confidence (descending)
      - ideal      = oracle ranking (all passes first, then failures)
      - random     = expected risk under random ranking (flat at overall risk)

    Sorting by confidence descending: at coverage c, the curve plots the
    failure rate among the most-confident c·N runs. A useful confidence
    signal hugs the ideal curve; a useless one hugs the random line."""
    df = df.dropna(subset=["confidence", "successful"]).copy()
    df["pass"] = df["successful"].astype(int)
    df["agent_id"] = df["config_dir"].map(_agent_id)

    agents = sorted(df["agent_id"].unique())
    n = len(agents)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, agent_id in enumerate(agents):
        ax = axes[i]
        sub = df[df["agent_id"] == agent_id]
        scores = sub["confidence"].to_numpy()
        labels = sub["pass"].to_numpy()
        overall_acc = labels.mean()

        # Model: sort by confidence descending
        cov_m, risk_m = _risk_coverage(scores, labels)
        # Ideal: oracle uses the labels themselves as scores
        cov_i, risk_i = _risk_coverage(labels.astype(float), labels)
        # Random: expected accuracy is flat at the overall pass rate
        ax.axhline(overall_acc, color="#d62728", linestyle=":",
                   linewidth=1.6, label=f"random (E = {overall_acc:.2f})")
        ax.plot(cov_i, 1 - risk_i, color="#2ca02c", linestyle="--",
                linewidth=1.6, label="ideal (oracle)")
        ax.plot(cov_m, 1 - risk_m, color="#1f77b4", linewidth=2.2,
                label="post-hoc confidence")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{short_label(agent_id)}  (n={len(sub)})", fontsize=10)
        ax.grid(alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel("Accuracy (pass rate)")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Coverage")
        if i == 0:
            ax.legend(loc="lower left", fontsize=8)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Selective accuracy vs coverage  (model vs ideal vs random)",
                 y=1.0, fontsize=12)
    fig.tight_layout()
    style.save(fig, path)


def _ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE = sum_b (n_b/N) * |mean(c_b) - mean(y_b)|."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores, bins) - 1, 0, n_bins - 1)
    n = len(scores)
    out = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        out += (mask.sum() / n) * abs(scores[mask].mean() - labels[mask].mean())
    return out


def _bootstrap_se(scores: np.ndarray, labels: np.ndarray, fn,
                  n_boot: int = 1000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(scores)
    if n < 2:
        return float("nan")
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = fn(scores[idx], labels[idx])
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            vals.append(v)
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals, ddof=1))


def _per_agent_metrics(df: pd.DataFrame, n_bins: int = 5,
                       n_boot: int = 1000,
                       conf_col: str = "confidence") -> pd.DataFrame:
    df = df.dropna(subset=[conf_col, "successful"]).copy()
    df["pass"] = df["successful"].astype(int)
    df["agent_id"] = df["config_dir"].map(_agent_id)
    rows = []
    for agent_id, sub in df.groupby("agent_id"):
        scores = sub[conf_col].to_numpy()
        labels = sub["pass"].to_numpy()
        p_cal = 1.0 - _ece(scores, labels, n_bins=n_bins)
        p_auroc = _auroc(scores, labels)
        p_cal_se = _bootstrap_se(scores, labels,
                                 lambda s, y: 1.0 - _ece(s, y, n_bins=n_bins),
                                 n_boot=n_boot, seed=hash(agent_id) & 0xFFFF)
        p_auroc_se = _bootstrap_se(scores, labels, _auroc,
                                   n_boot=n_boot, seed=(hash(agent_id) & 0xFFFF) + 1)
        rows.append({
            "agent_id": agent_id,
            "label": short_label(agent_id),
            "P_cal": p_cal,
            "P_cal_se": p_cal_se,
            "P_AUROC": p_auroc,
            "P_AUROC_se": p_auroc_se,
            "n": len(scores),
        })
    return pd.DataFrame(rows)


def fig_calibration_bar(df: pd.DataFrame, path: Path):
    metrics = _per_agent_metrics(df).dropna(subset=["P_cal"]).sort_values("P_cal")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(metrics) + 1.2)))
    ax.barh(metrics["label"], metrics["P_cal"],
            xerr=metrics["P_cal_se"].fillna(0),
            color="#9467bd", edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="black", capsize=3, lw=1))
    for y, (v, se) in enumerate(zip(metrics["P_cal"],
                                    metrics["P_cal_se"].fillna(0))):
        ax.text(v + se + 0.01, y, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.10)
    ax.set_xlabel("Calibration  P_cal = 1 − ECE  (error = bootstrap SE, 1000 reps)")
    ax.set_title("Confidence calibration metric per agent")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    style.save(fig, path)


def fig_discrimination_bar(df: pd.DataFrame, path: Path):
    metrics = _per_agent_metrics(df).dropna(subset=["P_AUROC"]).sort_values("P_AUROC")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(metrics) + 1.2)))
    ax.barh(metrics["label"], metrics["P_AUROC"],
            xerr=metrics["P_AUROC_se"].fillna(0),
            color="#17becf", edgecolor="black", linewidth=0.5,
            error_kw=dict(ecolor="black", capsize=3, lw=1))
    for y, (v, se) in enumerate(zip(metrics["P_AUROC"],
                                    metrics["P_AUROC_se"].fillna(0))):
        ax.text(v + se + 0.01, y, f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xlim(0, 1.10)
    ax.set_xlabel("Discrimination  P_AUROC  (error = bootstrap SE, 1000 reps)")
    ax.set_title("Confidence discrimination metric per agent")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    style.save(fig, path)


def fig_ood_vs_main(runs: pd.DataFrame, path: Path | None):
    """Pair the Codex configs that ran on both main39 and ood19.
    Three-panel scatter: accuracy, mean tokens, mean cost — main39 on
    x, ood19 on y. Diagonal = 'OOD matches in-distribution'. Off-axis
    excursions reveal which configs spend more or generalize better
    on OOD."""
    import re
    def canon(c): return re.sub(r"_k0$", "", c)

    df = runs.copy()
    df["cfg"] = df["config_dir"].map(canon)

    rows = []
    for split in ("main39", "ood19"):
        sub = df[df["split"] == split].copy()
        sub["pass"] = sub["successful"].astype("boolean").astype(float)
        sub["tot_tok"] = sub["total_tokens"].astype(float)
        # use corrected cost for core_agent / token-priced fallback
        sub["cost"] = sub.apply(_correct_core_agent_cost, axis=1)
        per_cfg = sub.groupby("cfg").agg(
            acc=("pass", "mean"),
            tok=("tot_tok", "mean"),
            cost=("cost", "mean"),
        ).reset_index()
        per_cfg["split"] = split
        rows.append(per_cfg)
    long = pd.concat(rows)
    wide = long.pivot(index="cfg", columns="split",
                      values=["acc", "tok", "cost"]).dropna()
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    # only Codex paired configs survive the dropna (claude_code/oc/ca aren't in OOD)
    wide["label"] = wide["cfg"].map(lambda c: short_label(_agent_id(c)))

    wide = wide.sort_values("acc_main39").reset_index(drop=True)
    wide["num"] = range(1, len(wide) + 1)

    fig = plt.figure(figsize=(16, 5.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.55],
                          wspace=0.3)

    def panel(gs_idx, x, y, color, xlabel, ylabel, title, log=False):
        ax = fig.add_subplot(gs[0, gs_idx])
        if log:
            lo = min(wide[x].min(), wide[y].min()) * 0.7
            hi = max(wide[x].max(), wide[y].max()) * 1.4
            ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1)
            ax.set_xscale("log"); ax.set_yscale("log")
        else:
            ax.plot([0.5, 1.05], [0.5, 1.05], color="gray", linestyle="--", linewidth=1)
            ax.set_xlim(0.5, 1.05); ax.set_ylim(0.5, 1.05)
        ax.scatter(wide[x], wide[y], s=160, color=color,
                   edgecolors="black", linewidths=0.5, zorder=3, alpha=0.85)
        for _, r in wide.iterrows():
            ax.annotate(str(r["num"]), (r[x], r[y]),
                        fontsize=8, color="white", weight="bold",
                        ha="center", va="center", zorder=4)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")

    panel(0, "acc_main39", "acc_ood19", "#1f77b4",
          "Accuracy on main39", "Accuracy on OOD19", "Accuracy")
    panel(1, "tok_main39", "tok_ood19", "#2ca02c",
          "Tokens / task (main39)", "Tokens / task (OOD19)", "Tokens", log=True)
    panel(2, "cost_main39", "cost_ood19", "#d62728",
          "Cost / task (main39)", "Cost / task (OOD19)", "Cost", log=True)

    # legend panel
    ax_leg = fig.add_subplot(gs[0, 3])
    ax_leg.axis("off")
    legend_lines = [f"{r['num']:>2}. {r['label']}" for _, r in wide.iterrows()]
    ax_leg.text(0, 1, "\n".join(legend_lines), fontsize=9,
                family="monospace", verticalalignment="top",
                bbox=dict(facecolor="white", edgecolor="lightgray",
                          boxstyle="round,pad=0.5"))

    fig.suptitle(
        f"Codex configs: OOD generalization ({len(wide)} paired configs; "
        "diagonal = 'OOD matches main39')",
        fontsize=12, y=1.0,
    )
    if path is None:
        plt.close(fig)
    else:
        style.save(fig, path)
    return wide


def fig_scaffold_vs_model_decomposition(runs: pd.DataFrame, path: Path | None):
    """Two-way variance decomposition on log(tokens) and log(cost)
    over (model, scaffold, capsule). Restricted to the crossed cells
    in main39 where we have multiple models per scaffold AND multiple
    scaffolds per model. Reports share of total SS attributable to
    each factor (Type II SS via statsmodels OLS)."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    from analysis.paper_figures import CANONICAL_CELLS  # avoid import cycle

    df = runs[runs["split"] == "main39"].copy()
    df["cost"] = df.apply(_correct_core_agent_cost, axis=1)

    keep = []
    for (model, scaffold), exact_cfg in CANONICAL_CELLS.items():
        sub = df[df["config_dir"] == exact_cfg].copy()
        if len(sub) == 0:
            continue
        sub["model"] = model
        sub["scaffold_f"] = scaffold
        keep.append(sub)
    work = pd.concat(keep, ignore_index=True)
    work["log_tokens"] = np.log(work["total_tokens"].astype(float).clip(lower=1))
    work["log_cost"] = np.log(work["cost"].astype(float).clip(lower=1e-6))

    # cells observed
    cells = work.groupby(["model", "scaffold_f"]).size().reset_index(name="n")
    print("\nVariance-decomposition cells (main39):")
    print(cells.to_string(index=False))

    # Type II SS: fit full and reduced models per term, take SS difference
    def ss_decomp(formula_full, data, terms):
        full = ols(formula_full, data=data).fit()
        ss_total = ((data[formula_full.split("~")[0].strip()] -
                     data[formula_full.split("~")[0].strip()].mean()) ** 2).sum()
        ss_resid = (full.resid ** 2).sum()
        # Type II for each term: refit without it (keeping other main effects)
        out = {}
        for t in terms:
            others = [x for x in terms if x != t]
            reduced = ols(f"{formula_full.split('~')[0]} ~ " + " + ".join(others),
                          data=data).fit()
            out[t] = (reduced.resid ** 2).sum() - ss_resid
        out["residual"] = ss_resid
        out["total"] = ss_total
        return out

    terms = ["C(model)", "C(scaffold_f)", "C(capsule_id)"]
    rows = []
    for y in ("log_tokens", "log_cost"):
        d = work.dropna(subset=[y]).copy()
        ss = ss_decomp(f"{y} ~ C(model) + C(scaffold_f) + C(capsule_id)",
                       d, terms)
        for t in terms + ["residual"]:
            rows.append({"target": y, "factor": t, "SS": ss[t],
                         "share": ss[t] / ss["total"]})
    decomp = pd.DataFrame(rows)
    print("\nVariance shares (Type II):")
    pretty = decomp.pivot(index="factor", columns="target", values="share")
    print(pretty.round(3))

    # ---- figure ----
    target_labels = {"log_tokens": "log(tokens)", "log_cost": "log(cost)"}
    factor_order = ["C(scaffold_f)", "C(model)", "C(capsule_id)", "residual"]
    factor_labels = {"C(scaffold_f)": "scaffold", "C(model)": "model",
                     "C(capsule_id)": "task (capsule)", "residual": "residual"}
    colors = {"scaffold": "#1f77b4", "model": "#ff7f0e",
              "task (capsule)": "#7f7f7f", "residual": "#cccccc"}

    fig, ax = plt.subplots(figsize=(10, 3.5))
    targets = list(target_labels)
    y = np.arange(len(targets))
    left = np.zeros(len(targets))
    for f in factor_order:
        vals = [decomp[(decomp["target"] == t) & (decomp["factor"] == f)]["share"].iloc[0]
                for t in targets]
        ax.barh(y, vals, left=left, color=colors[factor_labels[f]],
                edgecolor="black", linewidth=0.5,
                label=factor_labels[f])
        for i, v in enumerate(vals):
            if v > 0.04:
                ax.text(left[i] + v / 2, y[i], f"{v*100:.0f}%",
                        ha="center", va="center", fontsize=10,
                        color="white" if factor_labels[f] in ("scaffold", "model") else "black",
                        weight="bold")
        left += np.array(vals)

    ax.set_yticks(y, [target_labels[t] for t in targets])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Share of total sum-of-squares (Type II)")
    ax.set_title("Variance decomposition: scaffold vs. model contribution to efficiency"
                 f"\n(crossed cells, n={len(work)} observations across {len(cells)} cells)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4),
              ncol=4, fontsize=9, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if path is None:
        plt.close(fig)
    else:
        style.save(fig, path)
    return decomp


def fig_errors_vs_pass_and_confidence(df: pd.DataFrame, path: Path):
    """Small multiples (one panel per agent). Each panel plots two
    series vs num_errors on shared integer bins:
      - empirical pass rate (does errors predict failure?)
      - mean post-hoc confidence (does the grader use errors?)
    The vertical gap between the two lines = the grader's penalty
    for errors that don't actually predict failure."""
    df = df.dropna(subset=["confidence", "successful", "num_errors"]).copy()
    df["pass"] = df["successful"].astype(float)
    df["agent_id"] = df["config_dir"].map(_agent_id)

    # Shared integer bins so x-positions align across agents
    edges = [-0.5, 0.5, 2.5, 5.5, 10.5, 20.5, np.inf]
    centers = [0, 1.5, 4, 8, 15.5, 25]
    labels_x = ["0", "1–2", "3–5", "6–10", "11–20", "21+"]

    agents = sorted(df["agent_id"].unique())
    n = len(agents)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.5 * ncols, 3.4 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for i, agent_id in enumerate(agents):
        ax = axes[i]
        sub = df[df["agent_id"] == agent_id].copy()
        sub["bin"] = pd.cut(sub["num_errors"], bins=edges, labels=False,
                            include_lowest=True)
        agg = sub.groupby("bin").agg(
            pass_mean=("pass", "mean"),
            pass_se=("pass", lambda s: s.std(ddof=1)/np.sqrt(len(s)) if len(s)>1 else np.nan),
            conf_mean=("confidence", "mean"),
            conf_se=("confidence", lambda s: s.std(ddof=1)/np.sqrt(len(s)) if len(s)>1 else np.nan),
            n=("pass", "size"),
        ).reindex(range(len(centers)))
        x = np.array(centers)[agg.index.values]

        ax.fill_between(x,
                        agg["pass_mean"] - agg["pass_se"].fillna(0),
                        agg["pass_mean"] + agg["pass_se"].fillna(0),
                        color="#2ca02c", alpha=0.18)
        ax.plot(x, agg["pass_mean"], color="#2ca02c", marker="o",
                linewidth=2.0, label="empirical pass rate")
        ax.fill_between(x,
                        agg["conf_mean"] - agg["conf_se"].fillna(0),
                        agg["conf_mean"] + agg["conf_se"].fillna(0),
                        color="#9467bd", alpha=0.18)
        ax.plot(x, agg["conf_mean"], color="#9467bd", marker="s",
                linewidth=2.0, label="mean confidence")

        # n annotations under the x-axis line
        for cx, nb in zip(x, agg["n"].fillna(0).astype(int)):
            if nb > 0:
                ax.text(cx, -0.03, f"n={nb}", ha="center", va="top",
                        fontsize=7, color="gray")

        ax.set_title(short_label(agent_id), fontsize=10)
        ax.set_xticks(centers)
        ax.set_xticklabels(labels_x)
        ax.set_ylim(-0.08, 1.05)
        ax.grid(alpha=0.3)
        if i % ncols == 0:
            ax.set_ylabel("rate / confidence")
        if i // ncols == nrows - 1:
            ax.set_xlabel("num_errors (failed bash commands)")
        if i == 0:
            ax.legend(loc="lower left", fontsize=8)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Pass rate vs. self-rated confidence, binned by num_errors",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    style.save(fig, path)


def fig_errors_discrimination(df: pd.DataFrame, path: Path,
                              n_boot: int = 1000):
    """Per-agent AUROC bars comparing post-hoc confidence with two
    deterministic features derived from the trajectory:
      - −num_errors  (fewer failed commands → higher score)
      - −error_rate  (lower failure fraction among commands)
    Negation makes 'higher score = healthier', so AUROC is comparable
    to confidence (where higher = more confident in success)."""
    df = df.dropna(subset=["confidence", "successful",
                           "num_errors", "num_commands"]).copy()
    df = df[df["num_commands"] > 0]
    df["pass"] = df["successful"].astype(int)
    df["agent_id"] = df["config_dir"].map(_agent_id)
    df["neg_errors"] = -df["num_errors"].astype(float)
    df["neg_error_rate"] = -(df["num_errors"] / df["num_commands"]).astype(float)

    rows = []
    for agent_id, sub in df.groupby("agent_id"):
        labels = sub["pass"].to_numpy()
        for col, label in [("confidence", "post-hoc confidence"),
                           ("neg_errors", "−num_errors"),
                           ("neg_error_rate", "−error_rate")]:
            scores = sub[col].to_numpy()
            au = _auroc(scores, labels)
            au_se = _bootstrap_se(scores, labels, _auroc, n_boot=n_boot,
                                  seed=hash((agent_id, col)) & 0xFFFF)
            rows.append({"agent_id": agent_id, "label": short_label(agent_id),
                         "feature": label, "AUROC": au, "AUROC_se": au_se})
    res = pd.DataFrame(rows)

    # bar grouped by agent, three bars (features) per agent
    agents = sorted(res["agent_id"].unique(),
                    key=lambda a: -res[(res.agent_id==a) & (res.feature=="post-hoc confidence")]["AUROC"].iloc[0])
    feats = ["post-hoc confidence", "−num_errors", "−error_rate"]
    colors = {"post-hoc confidence": "#17becf",
              "−num_errors": "#ff7f0e",
              "−error_rate": "#9467bd"}
    n = len(agents)
    y = np.arange(n)
    h = 0.26

    fig, ax = plt.subplots(figsize=(10, max(3, 0.7 * n + 1.5)))
    for j, feat in enumerate(feats):
        vals = [res[(res.agent_id==a) & (res.feature==feat)]["AUROC"].iloc[0] for a in agents]
        ses = [res[(res.agent_id==a) & (res.feature==feat)]["AUROC_se"].iloc[0] for a in agents]
        offset = (j - 1) * h
        ax.barh(y + offset, vals, height=h,
                xerr=[s if pd.notna(s) else 0 for s in ses],
                color=colors[feat], edgecolor="black", linewidth=0.5,
                error_kw=dict(ecolor="black", capsize=3, lw=1), label=feat)
        for i, (v, se) in enumerate(zip(vals, ses)):
            se_show = se if pd.notna(se) else 0
            ax.text(v + se_show + 0.01, y[i] + offset, f"{v:.2f}",
                    va="center", fontsize=8)

    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y, [short_label(a) for a in agents])
    ax.set_xlim(0, 1.10)
    ax.set_xlabel("AUROC  (P[score(success) > score(failure)])")
    ax.set_title("Discrimination: LLM-elicited confidence vs deterministic error features")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    style.save(fig, path)


def fig_confidence_v1_vs_v2(df: pd.DataFrame, path: Path):
    """Side-by-side comparison of P_cal and P_AUROC under v1 (original)
    vs v2 (re-elicited) confidence. Two subplots, grouped horizontal bars
    per agent. Error bars are bootstrap SEs."""
    m1 = _per_agent_metrics(df, conf_col="confidence")
    m2 = _per_agent_metrics(df, conf_col="confidence_v2")
    merged = m1.merge(m2, on=["agent_id", "label"], suffixes=("_v1", "_v2"))
    # Order rows by v1 P_AUROC ascending so labels read consistently across panels
    merged = merged.sort_values("P_AUROC_v1").reset_index(drop=True)

    n = len(merged)
    y = np.arange(n)
    h = 0.4
    fig, (axc, axd) = plt.subplots(
        1, 2, figsize=(13, max(3, 0.6 * n + 1.5)), sharey=True)

    # P_cal panel
    axc.barh(y - h/2, merged["P_cal_v1"], height=h,
             xerr=merged["P_cal_se_v1"].fillna(0),
             color="#9467bd", edgecolor="black", linewidth=0.5,
             error_kw=dict(ecolor="black", capsize=3, lw=1), label="v1")
    axc.barh(y + h/2, merged["P_cal_v2"], height=h,
             xerr=merged["P_cal_se_v2"].fillna(0),
             color="#d4b3ff", edgecolor="black", linewidth=0.5,
             error_kw=dict(ecolor="black", capsize=3, lw=1), label="v2")
    for i, (v1, v2, s1, s2) in enumerate(zip(
            merged["P_cal_v1"], merged["P_cal_v2"],
            merged["P_cal_se_v1"].fillna(0), merged["P_cal_se_v2"].fillna(0))):
        axc.text(v1 + s1 + 0.01, i - h/2, f"{v1:.2f}", va="center", fontsize=8)
        axc.text(v2 + s2 + 0.01, i + h/2, f"{v2:.2f}", va="center", fontsize=8)
    axc.set_yticks(y, merged["label"])
    axc.set_xlim(0, 1.10)
    axc.set_xlabel("P_cal = 1 − ECE")
    axc.set_title("Calibration  (v1 vs v2)")
    axc.grid(axis="x", alpha=0.3)
    axc.legend(loc="lower right", fontsize=9)

    # P_AUROC panel
    axd.barh(y - h/2, merged["P_AUROC_v1"], height=h,
             xerr=merged["P_AUROC_se_v1"].fillna(0),
             color="#17becf", edgecolor="black", linewidth=0.5,
             error_kw=dict(ecolor="black", capsize=3, lw=1), label="v1")
    axd.barh(y + h/2, merged["P_AUROC_v2"], height=h,
             xerr=merged["P_AUROC_se_v2"].fillna(0),
             color="#a8e0e8", edgecolor="black", linewidth=0.5,
             error_kw=dict(ecolor="black", capsize=3, lw=1), label="v2")
    for i, (v1, v2, s1, s2) in enumerate(zip(
            merged["P_AUROC_v1"], merged["P_AUROC_v2"],
            merged["P_AUROC_se_v1"].fillna(0), merged["P_AUROC_se_v2"].fillna(0))):
        axd.text(v1 + s1 + 0.01, i - h/2, f"{v1:.2f}", va="center", fontsize=8)
        axd.text(v2 + s2 + 0.01, i + h/2, f"{v2:.2f}", va="center", fontsize=8)
    axd.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    axd.set_xlim(0, 1.10)
    axd.set_xlabel("P_AUROC")
    axd.set_title("Discrimination  (v1 vs v2)")
    axd.grid(axis="x", alpha=0.3)
    axd.legend(loc="lower right", fontsize=9)

    fig.suptitle("Post-hoc confidence: original (v1) vs re-elicited (v2)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    style.save(fig, path)


# ----- markdown summary -----------------------------------------------

def _fmt_dollars(x):
    return f"${x:.2f}" if pd.notna(x) else "—"


def _fmt_pct(x):
    return f"{x*100:.1f}%" if pd.notna(x) else "—"


def write_markdown_summary(eff_main, eff_ood, rel, path: Path):
    lines = []
    lines.append("# §3.1 + §3.2 results — auto-generated from data/runs.parquet\n")

    lines.append("## §3.2 Efficiency on CORE-Bench v1.1 (39 tasks, k=0)\n")
    lines.append("Sorted by accuracy. core_agent cost for Opus 4.5 / GPT-5.4 is a "
                 "token-priced uncached upper bound (smolagents stdout cost is buggy "
                 "or missing for those configs).\n")
    cols = ["label", "scaffold", "accuracy",
            "cost_mean", "cost_median", "tot_tok_mean",
            "dur_mean_s", "tools_mean"]
    df = eff_main.sort_values("accuracy", ascending=False)[cols].copy()
    df["accuracy"] = df["accuracy"].map(_fmt_pct)
    df["cost_mean"] = df["cost_mean"].map(_fmt_dollars)
    df["cost_median"] = df["cost_median"].map(_fmt_dollars)
    df["tot_tok_mean"] = df["tot_tok_mean"].map(lambda v: f"{v/1e6:.1f}M" if pd.notna(v) else "—")
    df["dur_mean_s"] = df["dur_mean_s"].map(lambda v: f"{int(v)}s" if pd.notna(v) else "—")
    df["tools_mean"] = df["tools_mean"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
    df.columns = ["agent", "scaffold", "acc",
                  "cost mean", "cost median", "tokens mean",
                  "duration mean", "tool calls mean"]
    lines.append(df.to_markdown(index=False))
    lines.append("")

    lines.append("\n## §3.2 Efficiency on CORE-Bench OOD (19 tasks)\n")
    df = eff_ood.sort_values("accuracy", ascending=False)[cols].copy()
    df["accuracy"] = df["accuracy"].map(_fmt_pct)
    df["cost_mean"] = df["cost_mean"].map(_fmt_dollars)
    df["cost_median"] = df["cost_median"].map(_fmt_dollars)
    df["tot_tok_mean"] = df["tot_tok_mean"].map(lambda v: f"{v/1e6:.1f}M" if pd.notna(v) else "—")
    df["dur_mean_s"] = df["dur_mean_s"].map(lambda v: f"{int(v)}s" if pd.notna(v) else "—")
    df["tools_mean"] = df["tools_mean"].map(lambda v: f"{v:.1f}" if pd.notna(v) else "—")
    df.columns = ["agent", "scaffold", "acc",
                  "cost mean", "cost median", "tokens mean",
                  "duration mean", "tool calls mean"]
    lines.append(df.to_markdown(index=False))
    lines.append("")

    lines.append("\n## §3.1 Reliability (k=5 reps × 39 tasks per agent)\n")
    lines.append("- **outcome consistency** = mean over tasks of (2·p̂−1)², where "
                 "p̂ = n_pass / K. 1.0 = unanimous reps on every task; "
                 "0.0 = 50/50 splits.\n"
                 "- **resource consistency** = mean over tasks of "
                 "exp(−CV_tokens) (single-resource form on total tokens). "
                 "1.0 = identical token use; → 0 = wildly variable.\n")
    cols = ["label", "outcome_consistency", "resource_consistency",
            "tokens_cv_mean"]
    df = rel.sort_values("resource_consistency", ascending=False)[cols].copy()
    df["outcome_consistency"] = df["outcome_consistency"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    df["resource_consistency"] = df["resource_consistency"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    df["tokens_cv_mean"] = df["tokens_cv_mean"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    df.columns = ["agent", "outcome consistency", "resource consistency",
                  "token CV"]
    lines.append(df.to_markdown(index=False))
    lines.append("")

    path.write_text("\n".join(lines))


def write_confidence_summary(runs: pd.DataFrame, path: Path):
    """Per-agent P_cal + P_AUROC summary."""
    df = runs[runs["split"] == "reliability"].copy()
    df = df.dropna(subset=["confidence", "successful"])
    df["pass"] = df["successful"].astype(int)
    df["agent_id"] = df["config_dir"].map(_agent_id)

    rows = []
    for agent_id, sub in df.groupby("agent_id"):
        scores = sub["confidence"].to_numpy()
        labels = sub["pass"].to_numpy()
        rows.append({
            "agent": short_label(agent_id),
            "P_cal": 1.0 - _ece(scores, labels, n_bins=5),
            "P_AUROC": _auroc(scores, labels),
            "mean_conf": scores.mean(),
            "mean_pass": labels.mean(),
            "n": len(scores),
        })
    out = pd.DataFrame(rows).sort_values("P_AUROC", ascending=False)

    lines = ["\n## Confidence (post-hoc, reliability split)\n"]
    lines.append("- **P_cal = 1 − ECE** (1.0 = perfectly calibrated; ECE is "
                 "the n-weighted gap between mean confidence and mean accuracy "
                 "over 5 confidence bins).\n"
                 "- **P_AUROC** = P[score(success) > score(failure)] across all "
                 "(success, failure) pairs (0.5 = chance).\n"
                 "- **mean conf − mean pass** flags over/under-confidence.\n")
    fmt = out.copy()
    fmt["P_cal"] = fmt["P_cal"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    fmt["P_AUROC"] = fmt["P_AUROC"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    fmt["mean_conf"] = fmt["mean_conf"].map(_fmt_pct)
    fmt["mean_pass"] = fmt["mean_pass"].map(_fmt_pct)
    fmt.columns = ["agent", "P_cal", "P_AUROC", "mean conf", "mean pass", "n"]
    lines.append(fmt.to_markdown(index=False))
    lines.append("")

    with open(path, "a") as f:
        f.write("\n".join(lines))
    return out


# ----- main -----------------------------------------------------------

def main():
    runs = pd.read_parquet(RUNS)
    print(f"Loaded {len(runs)} runs.")

    eff_main = efficiency_table(runs[runs["split"] == "main39"], "main39")
    eff_ood = efficiency_table(runs[runs["split"] == "ood19"], "ood19")
    eff_main.to_csv(OUT / "efficiency_main39.csv", index=False)
    eff_ood.to_csv(OUT / "efficiency_ood19.csv", index=False)

    rel, _ = reliability_table(runs[runs["split"] == "reliability"], k=5)
    rel.to_csv(OUT / "reliability_summary.csv", index=False)

    # Paper-supporting figures live in analysis/paper_figures.py;
    # imported lazily to avoid the import cycle from
    # paper_figures → compute (helpers).
    from analysis import paper_figures as P

    # Figure-only exclusion: keep GPT-5.1 in summary/data tables, but
    # omit it from plots while the main-vs-reliability discrepancy is
    # unsettled.
    runs_fig = P.filter_figure_agents(runs)
    eff_main_fig = P.filter_figure_agents(eff_main)
    eff_ood_fig = P.filter_figure_agents(eff_ood)
    rel_fig = P.filter_figure_agents(rel)

    # OOD tokens_vs_accuracy lives under figs/paper/.
    P.fig_tokens_vs_accuracy(
        eff_ood_fig, PAPER_FIGS / "tokens_vs_accuracy_ood.png",
    )
    fig_outcome_consistency(
        rel_fig, k=5, path=FIGS / "reliability_outcome_consistency.png",
    )
    fig_resource_consistency(
        rel_fig, path=FIGS / "reliability_resource_consistency.png",
    )

    # Calibration + discrimination — uses reliability split (full coverage)
    rel_runs = runs[runs["split"] == "reliability"]
    rel_runs_fig = runs_fig[runs_fig["split"] == "reliability"]
    fig_calibration(rel_runs_fig, FIGS / "calibration.png")
    fig_calibration_bar(rel_runs_fig, FIGS / "calibration_bar.png")
    fig_risk_coverage(rel_runs_fig, FIGS / "risk_coverage.png")
    fig_discrimination_bar(rel_runs_fig, FIGS / "discrimination_bar.png")
    fig_confidence_v1_vs_v2(rel_runs_fig, FIGS / "confidence_v1_vs_v2.png")
    fig_errors_vs_pass_and_confidence(
        rel_runs_fig, FIGS / "errors_vs_pass_and_confidence.png")
    fig_errors_discrimination(
        rel_runs_fig, FIGS / "errors_discrimination.png")
    fig_ood_vs_main(runs_fig, FIGS / "ood_vs_main.png")
    ood = fig_ood_vs_main(runs, None)
    ood.to_csv(OUT / "ood_vs_main.csv", index=False)
    fig_scaffold_vs_model_decomposition(
        runs_fig, FIGS / "scaffold_vs_model_decomposition.png")
    decomp = fig_scaffold_vs_model_decomposition(runs, None)
    decomp.to_csv(OUT / "scaffold_vs_model_decomposition.csv", index=False)

    # ----- paper-supporting figures (NeurIPS narratives N1–N6) -----
    # Routed to figs/paper/ so they're easy to grab for the manuscript;
    # exploratory figures stay in figs/.
    P.fig_accuracy_by_scaffold_model(
        runs_fig, PAPER_FIGS / "accuracy_by_scaffold_model.png")
    cells = P._canonical_cell_table(runs)
    cells.to_csv(OUT / "canonical_cells.csv", index=False)
    P.fig_tokens_by_scaffold_per_model(
        runs_fig, PAPER_FIGS / "tokens_by_scaffold_per_model.png")
    P.fig_tokens_vs_accuracy(
        eff_main_fig, PAPER_FIGS / "tokens_vs_accuracy.png",
    )
    P.fig_tokens_vs_accuracy(
        eff_main_fig, PAPER_FIGS / "tokens_vs_accuracy_linear.png",
        x_label="Mean tokens per task (linear scale)",
        x_scale="linear",
    )
    P.fig_codex_main_vs_ood(runs_fig, PAPER_FIGS / "codex_main_vs_ood.png")
    P.fig_consistency_vs_accuracy(
        rel_fig, PAPER_FIGS / "consistency_vs_accuracy.png")
    P.fig_outcome_consistency_vs_accuracy(
        rel_fig, PAPER_FIGS / "outcome_consistency_vs_accuracy.png")
    P.fig_resource_consistency_vs_accuracy(
        rel_fig, PAPER_FIGS / "resource_consistency_vs_accuracy.png")
    P.fig_predictability_per_agent(
        rel_runs_fig, PAPER_FIGS / "predictability_per_agent.png")
    P.fig_reliability_3panel(
        rel_fig, rel_runs_fig, PAPER_FIGS / "reliability_3panel.png")
    P.fig_predictability_consolidated(
        rel_runs_fig, PAPER_FIGS / "predictability_consolidated.png")
    P.fig_variance_decomposition(
        runs_fig, PAPER_FIGS / "variance_decomposition.png")

    # ----- cost figures (after audit) -------------------------------
    P.fig_tokens_vs_accuracy(
        eff_main_fig, PAPER_FIGS / "cost_vs_accuracy.png",
        x_col="cost_mean",
        x_label="Mean cost per task (\\$, log scale)",
        label_substrings=P.COST_ACC_LABELS_MAIN,
    )
    P.fig_tokens_vs_accuracy(
        eff_ood_fig, PAPER_FIGS / "cost_vs_accuracy_ood.png",
        x_col="cost_mean",
        x_label="Mean cost per task (\\$, log scale)",
        fit_excludes=(),
        label_substrings=P.COST_ACC_LABELS_MAIN,
    )
    P.fig_cost_by_scaffold_per_model(
        runs_fig, PAPER_FIGS / "cost_by_scaffold_per_model.png")

    write_markdown_summary(eff_main, eff_ood, rel, OUT / "summary.md")
    conf = write_confidence_summary(runs, OUT / "summary.md")
    conf.to_csv(OUT / "confidence_summary.csv", index=False)

    print("\n=== §3.2 main39 (top by accuracy) ===")
    cols = ["label", "accuracy", "tot_tok_mean", "cost_mean"]
    print(eff_main.sort_values("accuracy", ascending=False)[cols].to_string(index=False))

    print("\n=== §3.1 reliability ===")
    cols = ["label", "pass_at_1", "pass_at_least_1_of_k", "pass_all_k"]
    print(rel[cols].to_string(index=False))


if __name__ == "__main__":
    main()

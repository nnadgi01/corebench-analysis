# Life After Benchmark Saturation: A Case Study of CORE-Bench
<p>
<a href="https://huggingface.co/collections/agent-evals/core-bench-v11">
<img alt="Dataset" src="https://img.shields.io/badge/Hugging%20Face-Dataset-yellow.svg"> 
</p>

Analysis of results from CORE-Bench v1.1, CORE-Bench OOD, and a human-agent collaboration uplift study on computational reproducibility.

Run CORE-Bench v1.1 through the [Holistic Agent Leaderboard](https://github.com/princeton-pli/hal-harness/tree/feat/corebenchv2-prefect) harness. You can find the CORE-Bench v1.1 dataset [here](https://huggingface.co/datasets/agent-evals/core-bench-v1.1-mainline) and the CORE-Bench OOD dataset [here](https://huggingface.co/datasets/agent-evals/core-bench-v1.1-ood).

## Repository structure

```
.
├── data/              # data tables behind §3/§4 figures
├── analysis/          # scripts to regenerate §3/§4 figures
├── notebooks/         # §3.3 model/scaffold + §4 uplift analysis
├── acc_saturation/    # §2 accuracy & saturation metrics
├── sankey/            # §2 construction-pipeline diagrams
├── docent/            # Docent rubrics & runners
├── extractor/         # raw-log extraction helpers
├── figs/              # generated figures
└── requirements.txt
```

Each `## Paper section` below maps the data files and scripts to the figures and tables they produce.

## Setup

```bash
git clone <repo-url> && cd corebench-analysis
conda create -n core-bench python -y && conda activate core-bench
pip install -r requirements.txt

# Only needed to re-fetch raw logs/rubrics from Docent (§3.3):
cp .env.example .env   # then fill in DOCENT_API_KEY
```

Regenerating the reliability, efficiency, and uplift figures (§3.1, §3.2, §4) needs only the committed data tables — no Docent access or extraction pipeline required:

```bash
python -m analysis.regenerate_figures
```

The R Markdown notebook (`./notebooks/uplift_analysis.Rmd`, §4) requires R.

## Paper section 2: Construct validity
- [CORE-Bench v1.1 logs](https://docent.transluce.org/dashboard/f739ce50-eec8-4d8e-86b3-2c3dd9f42ab7) ([truncated](https://docent.transluce.org/dashboard/1d88d50a-7990-4528-aaf9-4b721d53b43d) tool call outputs version for log analysis)
- [CORE-Bench OOD logs](https://docent.transluce.org/dashboard/6fcaee2b-844f-4930-b62f-617ebf924b35) ([truncated](https://docent.transluce.org/dashboard/94497783-2245-4613-8d5f-73ab653079ec) tool call outputs version for log analysis)

`./acc_saturation/accuracies.ipynb` computes the accuracies of all agent configurations and accuracy saturation metrics

`./sankey/sankey_main.py` generates the construction pipeline for CORE-Bench v1.1

`./sankey/sankey_ood.py` generates the construction pipeline for CORE-Bench OOD

## Paper section 3: Multidimensional evaluation of agent performance

**Data tables** (`./data/`) — back every §3 figure; anything more granular can be re-aggregated from `runs.parquet`.

- `runs.parquet` (also exported as `runs.csv`) — source of truth, one row per run × config × capsule × rep
- `efficiency_per_agent.csv` — backs the tokens / cost-vs-accuracy figures (§3.2)
- `reliability_per_agent.csv` — backs the consistency and predictability figures (§3.1)
- `RCT_responses_cleaned.csv` — uplift questionnaire data, used by `./analysis/uplift_figures.py` (§4)

<details>
<summary>Column reference</summary>

- `runs.parquet`: keys `(run_id, config_dir, capsule_id, rep_idx)`; derived `cost` (post-correction per-row cost) and `agent_id` (`config_dir` with the reliability `_kN` rep-suffix stripped so reps collapse to one identity).
- `efficiency_per_agent.csv`: per `(agent_id, split)` — `n`, `accuracy`, `tot_tok_mean/_median/_std`, `cost_mean/_median/_std`.
- `reliability_per_agent.csv`: per `agent_id` over the k=5 split — `pass_at_1`, `pass_at_least_1_of_k`, `pass_all_k`, `outcome_consistency`, `resource_consistency`, `confidence_mean`, `confidence_median`.

</details>

**Analysis scripts** (`./analysis/`, run from the repo root):

`regenerate_figures.py` — regenerates the §3 and §4 figures from the committed data tables; no extraction pipeline needed.

```bash
python -m analysis.regenerate_figures   # writes the figures below to ./figs/
```

- `resource_accuracy.pdf` — tokens / cost vs. accuracy landscape (§3.2)
- `outcome_consistency_vs_accuracy.pdf` — outcome consistency vs. pass@1 (§3.1)
- `resource_consistency_vs_accuracy.pdf` — resource consistency vs. pass@1 (§3.1)
- `predictability_per_agent_vertical.pdf` — per-agent confidence and AUROC (§3.1)
- `calibration.pdf` — confidence calibration curves (§3.1)
- `discrimination_bar.pdf` — AUROC discrimination bar chart (§3.1)
- `uplift_duration_by_condition.pdf` — distribution of reproduction session durations (§4)

`export_data.py` — re-exports the data tables from `runs.parquet`; only needed when re-running from raw extraction outputs (`python -m analysis.export_data`).

<details>
<summary>Supporting modules (imported, not run directly)</summary>

- `paper_figures.py` — per-figure functions called by `regenerate_figures.py`
- `uplift_figures.py` — builds the uplift duration figure from `RCT_responses_cleaned.csv`
- `compute.py` — shared data transforms and metric computations
- `style.py` — shared Matplotlib styling (paper-mode formatting, colors, markers)

</details>

### Paper section 3.3: Decoupling model and scaffold

Three notebooks in `./notebooks/` produce the analysis behind §3.3 — the root-cause taxonomy of the accuracy failures, the representative trajectory-level disagreements across scaffolds, and the three findings (similar accuracies mask different failures; scaffolds induce distinct solution strategies; direct fixes outperform rewrites), all built on Docent rubrics over the agent logs.

- **`model_scaffold_decomposition.ipynb`** — per-capsule pass/fail comparisons across the model × scaffold grid (oracle-router and scaffold-disagreement analysis), the root-cause taxonomy of the failures (Docent failure rubric), and the answer-source / direct-fix-vs-rewrite success rates (Docent success rubric).
- **`model_scaffold_case_studies.ipynb`** — qualitative trajectory deep-dives behind the representative-disagreement table: CORE-Agent vs. first-party scaffold, OpenCode / Codex CLI as third-party controls, and the vision-read vs. code-output answer-source split.
- **`failure_mode_taxonomy.ipynb`** — behavioral failure-mode analysis: the root-cause taxonomy, wrong-value / wrong-answer breakdowns, and the answer-source / resolution-strategy / verification-pattern analyses (rubric v2) behind the "scaffolds induce distinct solution strategies" finding.

**Data:** the rubric-v2 analysis reads `./data/rubric_v2_results.json`. The raw agent logs and the Docent judge rubrics are fetched live from the [Docent v1.1 collection](https://docent.transluce.org/dashboard/1d88d50a-7990-4528-aaf9-4b721d53b43d) by the loader cells; these require `DOCENT_API_KEY` to be set in `.env`.

## Paper section 4: Human-agent collaboration uplift

`./data/RCT_responses_cleaned.csv`: Questionnaire responses 
- Entered by the evaluators (RCT participants) after each reproduction run
- Includes links to Docent logs for these runs
- Retrieved from Google Forms, with some data cleaning and redaction of private information (like email addresses) 
- Used by the three data analysis scripts described below

`./notebooks/uplift_analysis.Rmd`:

- Generates a version of Figure 3 ("Distribution of durations of reproduction sessions")
- Contains the fixed effects model to estimate the uplift factor and CR2 standard error reported in section 4.2 and the appendix

`./notebooks/eda_questionnaire_res.ipynb`: Notebook to conduct an exploratory data analysis of the questionnaire data

Generates several other results reported in section 4.2 and the appendix:
- Table 6 ("Observed collaboration patterns across 25 human-AI collaboration reproduction runs")
- Table 17 ("Where the agent was perceived to be useful for human-agent collaborative reproduction runs")
- Table 18 ("Where the agent encountered difficulties across human-AI collaboration reproduction runs")

`./notebooks/rct_results_analysis.ipynb`: Notebook with some additional data analysis

Generates several other results reported in the appendix, including:
- Table 14 ("Overview of Reproduction Outcomes by Step")

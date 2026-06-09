# corebench-analysis

TKTK

## Paper section 3: Multidimensional evaluation of agent performance

**Data tables** — three files in `./data/` back every §3 figure; anything more granular can be re-aggregated from `runs.parquet`.

`./data/runs.parquet` (also exported as `./data/runs.csv`): one row per (run\_id, config\_dir, capsule\_id, rep\_idx)
- Source of truth for all §3 figures
- Adds two derived columns: `cost` (post-correction per-row cost) and `agent_id` (`config_dir` with the reliability `_kN` rep-suffix stripped so reliability reps collapse to one identity)

`./data/efficiency_per_agent.csv`: per `(agent_id, split)` cell
- Columns: `n`, `accuracy`, `tot_tok_mean / _median / _std`, `cost_mean / _median / _std`
- Backs the bar charts and tokens / cost-vs-accuracy scatters (§3.2)

`./data/reliability_per_agent.csv`: per `agent_id` over the k=5 reliability split
- Columns: `pass_at_1`, `pass_at_least_1_of_k`, `pass_all_k`, `outcome_consistency`, `resource_consistency`, `confidence_mean`, `confidence_median`
- Backs the consistency and predictability figures (§3.1)

`./data/RCT_responses_cleaned.csv` is also used by `./analysis/uplift_figures.py` to generate the uplift duration figure (§4)

**Analysis scripts** — all live in `./analysis/` and are run from the repo root.

`./analysis/regenerate_figures.py`: regenerates every §3 paper figure from the data tables above; no extraction pipeline needed

```bash
python -m analysis.regenerate_figures
```

Generates figures reported in §3 and written to `./figs/`:
- `resource_accuracy.pdf` — tokens / cost vs. accuracy landscape (§3.2)
- `outcome_consistency_vs_accuracy.pdf` — outcome consistency vs. pass@1 (§3.1)
- `resource_consistency_vs_accuracy.pdf` — resource consistency vs. pass@1 (§3.1)
- `predictability_per_agent_vertical.pdf` — per-agent confidence and AUROC (§3.1)
- `calibration.pdf` — confidence calibration curves (§3.1)
- `discrimination_bar.pdf` — AUROC discrimination bar chart (§3.1)
- `uplift_duration_by_condition.pdf` — distribution of reproduction session durations (§4)

`./analysis/paper_figures.py`: individual figure functions called by `regenerate_figures.py`

`./analysis/uplift_figures.py`: generates the uplift duration figure from `./data/RCT_responses_cleaned.csv`

`./analysis/style.py`: shared Matplotlib aesthetics (paper-mode formatting, colors, marker shapes)

`./analysis/compute.py`: core data transformations and metric computations shared across figure scripts

`./analysis/export_data.py`: re-exports the three data tables from `runs.parquet`; only needed if re-running from raw extraction outputs

```bash
python -m analysis.export_data
```

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

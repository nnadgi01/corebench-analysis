# corebench-analysis

TKTK

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

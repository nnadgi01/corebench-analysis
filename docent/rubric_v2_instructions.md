# Rubric v2 Instructions — Behavioral Diagnostic for CORE-Bench Agent Runs

## Domain context

The grader's job is to read the full trace of a single agent run on a CORE-Bench computational reproducibility task. In each task, an AI agent is placed inside a scientific code capsule (a self-contained repository with code, data, and a manuscript) and asked to reproduce a specific numeric result — typically a statistic, p-value, count, or model metric reported in the associated paper. The agent has a 45-minute wall-clock budget and must write its final answer to `answer.json`.

The grader must interpret the final numeric score (0 = failure, 1 = success) and produce a compact but informative diagnosis: whether the run was a success or failure, where the final answer came from, what single obstacle most shaped the outcome, how the agent responded to that obstacle, whether the answer was verified, what overall strategy pattern the agent followed, and a short narrative explaining how the run unfolded.

---

## Decision procedure

### 1. Understand the inputs and basic setup.

- Confirm that the trace describes one continuous run of a single agent attempting a computational reproducibility task inside a code capsule.
- Read the entire trace, including:
  - The initial task prompt (which specifies what value to reproduce and where to write it).
  - The agent's messages and any internal reasoning traces.
  - All tool calls (shell commands, Python/R execution, file reads/writes, web search, vision model queries) and their outputs.
- Locate the final numeric score:
  - Identify the explicit final score in the transcript (e.g., "score: 1" or "score: 0").
  - Treat 1 as success and 0 as failure.
- You may make light causal inferences (e.g., "because dependency X failed to install, the main analysis script could not run") when clearly grounded in the trace.

### 2. Assign the label from the final numeric score.

- Find the last explicit numeric score in the transcript that is clearly designated as the final outcome.
- If the final score is 1: set label = "success".
- If the final score is 0: set label = "failure".
- Do not override or reinterpret the numeric score based on your own judgment of the transcript; always follow the explicit final score.

### 3. Determine the answer_source (where the final answer primarily came from).

Focus on the primary source of the final value(s) written to `answer.json`, not intermediate steps. If the agent used multiple sources, pick the one that most directly produced the submitted value.

Assign answer_source to exactly one of:

- **"original_pipeline"**
  - Use when the agent ran the capsule's code as-is, or with only dependency/environment fixes (installing packages, fixing file paths, setting environment variables, granting permissions). The computational logic itself was not changed.
  - Examples: running `run.sh` after installing missing packages; executing the original R/Python scripts after fixing a path variable; running a Makefile after installing system dependencies.
  - Key test: if you removed the agent's changes, would the code produce a different result? If no (the changes only made it runnable), this is original_pipeline.

- **"modified_pipeline"**
  - Use when the agent wrote simplified scripts, changed parameters, extracted subsets of the original code, or rewrote parts of the pipeline (e.g., translating R analysis to Python, creating a stripped-down computation script). The agent ran code, but not the original code as written.
  - Examples: rewriting an R script in Python to avoid R dependency issues; extracting only the relevant statistical test from a larger pipeline; changing model parameters or subsetting data to make code run faster.
  - Key test: the agent's code computes something, but the computational logic differs from what the capsule authors wrote.

- **"repo_artifacts"**
  - Use when the agent found the answer in pre-existing output files already present in the repository — cached results, saved model outputs, log files, result CSVs, `.RData` files, pickled objects, or previous run outputs — without re-running the code that produced them.
  - Examples: reading a value from `results/output.csv` that was already in the repo; loading a saved `.rds` file and extracting a statistic; finding the answer in a log file from a previous run.
  - Key test: the agent never ran the analysis code; it just read from files that already contained the answer.

- **"manuscript_or_readme"**
  - Use when the agent extracted the answer from the capsule's manuscript, README, paper PDF, or other documentation files within the repo. The agent read the reported value rather than computing it.
  - Examples: opening the PDF and finding "the Kruskal-Wallis p-value was 0.034"; reading a results table in the manuscript; extracting a number from the README's summary section.
  - Key test: the answer came from human-written text describing results, not from running or reading code/data outputs.

- **"figure_computational"**
  - Use when the agent generated the figure by running code, then extracted the value programmatically from the figure's underlying data (e.g., reading plot data structures, parsing saved figure data, extracting coordinates from plot objects).
  - Examples: running the plotting script and then reading the y-axis values from the plot object in R/Python; generating a figure and extracting the data points used to create it.

- **"figure_visual"**
  - Use when the agent used a vision model (e.g., `query_vision_language_model`) or visual inspection to read a value from a figure image, rather than extracting it from code output or data structures.
  - Examples: calling a vision model on a saved PNG to read a bar height; using OCR or visual estimation to extract a number from a chart image.
  - Key test: the value was read from pixels/visual representation rather than from numeric data.

- **"external_source"**
  - Use when the agent used web search, visited external webpages, or relied on general LLM knowledge to find the answer from sources outside the capsule repository (published papers online, academic databases, etc.).
  - Examples: searching for the paper on Google Scholar and reading the abstract; using `web_search` to find the published result; relying on training knowledge about the paper's findings.

- **"no_answer"**
  - Use when the agent did not produce a final answer: timed out before writing `answer.json`, explicitly gave up, or failed to write any meaningful value.
  - Also use when the agent wrote `answer.json` with an empty, null, or clearly placeholder value (e.g., "TODO", empty string, 0 used as a placeholder).

**Tie-breaking:**
- If the agent ran original code AND also checked the manuscript to verify, the answer_source is `original_pipeline` (the manuscript was verification, not the source).
- If the agent found a value in repo artifacts and also ran code that confirmed it, prefer `repo_artifacts` if the repo artifact was found first and the code run was confirmatory.
- If the agent ran modified code that reads from existing output files, prefer `repo_artifacts` (the code was just a file-reading wrapper, not a computation).

### 4. Scan the run for challenges and obstacles.

As you read, identify concrete points where the agent encountered friction:
- Errors from package installation (pip, conda, apt, R install.packages).
- Version conflicts or deprecated API calls during code execution.
- Long compilation times consuming the time budget.
- System-level constraints (no sudo, no GPU, disk full, externally-managed-environment).
- Bugs in the capsule's own code or bugs the agent introduced.
- Missing data files, broken symlinks, wrong paths.
- Ambiguous task questions or unclear metrics.
- Tool limitations (no R interpreter available, can't render PDFs, etc.).

For each notable challenge, note what triggered it, how the agent responded, and whether it was resolved.

### 5. Choose the primary_obstacle (single most impactful obstacle).

Pick the obstacle that consumed the most agent effort or most directly affected the outcome. If the agent encountered no significant obstacles, use "none".

Assign primary_obstacle to exactly one of:

- **"none"**
  - No significant obstacle; the agent proceeded smoothly from start to finish.
  - Appropriate mainly for label = "success" runs where setup and execution went cleanly.

- **"dependency_install_failure"**
  - A required package could not be installed at all: not available in the package index, build compilation failed permanently, no network access to download it, or the package doesn't exist for the platform.
  - Key distinction from version_conflict: the package never successfully installs.

- **"dependency_version_conflict"**
  - A package installed successfully but produced errors during execution due to version incompatibility: deprecated APIs, changed function signatures, removed arguments, renamed modules, or behavior changes between versions.
  - Key distinction from install_failure: the package is installed, but the code breaks when calling it.

- **"compilation_timeout"**
  - A C/C++/Fortran package or extension module compilation took so long it consumed a significant portion of the 45-minute time budget, even if it eventually succeeded.
  - Examples: compiling Stan models, building R packages from source with heavy C++ code, Cython compilation.

- **"environment_constraint"**
  - System-level limitations that blocked progress: no root/sudo access, no GPU when GPU is required, insufficient disk space, memory limits, Python externally-managed-environment blocking pip, read-only filesystem.
  - Key distinction from dependency issues: the constraint is at the OS/system level, not at the package level.

- **"code_bug_original"**
  - A bug or error in the capsule's own code that the agent had to diagnose: the code as written doesn't work even with correct dependencies, has hardcoded paths that don't exist, references undefined variables, or has logical errors.
  - Key test: the bug existed before the agent touched anything.

- **"code_bug_introduced"**
  - The agent introduced a bug while modifying or rewriting code: wrong variable names, incorrect logic in a rewritten script, off-by-one errors, wrong file paths in agent-written code.
  - Key test: the code worked (or would have worked) before the agent's modification broke it.

- **"data_missing_or_inaccessible"**
  - Required data files were missing from the capsule, symlinks were broken, paths in the code pointed to nonexistent locations, or files were in an unexpected/corrupted format.
  - Key distinction from code_bug: the code logic is fine, but the data it expects isn't there.

- **"task_ambiguity"**
  - The task question was ambiguous, had multiple valid interpretations, or referenced unclear metrics. The agent had to guess which value was being asked for, which table/figure to reproduce, or what format to use.
  - Examples: "report the p-value" when multiple tests are run; unclear which model's results to report; ambiguous rounding/formatting requirements.

- **"tool_limitation"**
  - The agent's available tools couldn't handle a required operation: no R interpreter in the environment, can't render or read PDF figures natively, no GUI for interactive plots, shell tool can't handle interactive prompts.
  - Key test: the limitation is in what the agent CAN do, not in what the capsule's code does.

**Tie-breaking:**
- If a dependency fails to install because of an environment constraint (e.g., no sudo for system libraries), prefer `environment_constraint` as the root cause.
- If the agent introduces a bug while trying to work around a dependency issue, the primary obstacle is the dependency issue (the introduced bug is a consequence).
- Choose the obstacle that consumed the most effort or most directly caused the final outcome, not merely the first one encountered.
- For label = "failure", avoid "none" — there is always something that prevented success.

### 6. Determine obstacle_resolution (how the agent responded to the primary obstacle).

This field captures the agent's *method* of response, not just whether it worked. If primary_obstacle is "none", use "not_applicable".

Assign obstacle_resolution to exactly one of:

- **"direct_fix"**
  - The agent diagnosed the issue and applied a targeted fix, then continued with the original pipeline: installed the correct package version, fixed the broken path, patched the specific bug, set the right environment variable, granted the needed permission.
  - Key test: the fix addressed the root cause and the original code subsequently ran.

- **"selective_execution"**
  - The agent identified that only part of the pipeline was needed for the answer, skipped the obstacle-causing parts entirely, and ran only the relevant code sections.
  - Examples: the full pipeline fails on step 3 of 5, but the answer only needs steps 4-5 which can run independently; the agent extracts just the statistical test code and runs it standalone.

- **"alternative_code"**
  - The agent wrote new code from scratch to replace the blocked section: rewrote the R analysis in Python, created a simplified computation script, implemented the statistical test manually.
  - Key distinction from selective_execution: the agent wrote NEW code rather than running a subset of EXISTING code.

- **"external_fallback"**
  - The agent abandoned code execution entirely and turned to web search, vision models, or general LLM knowledge to find the answer without computing it.
  - Examples: after failing to install R packages, the agent searches for the paper online; after code fails repeatedly, the agent reads the manuscript PDF or uses a vision model on a figure.

- **"brute_force_retry"**
  - The agent kept retrying variations of the same failing approach without clear diagnosis: trying multiple package versions sequentially, restarting the same computation, repeating failed commands with minor tweaks, or reinstalling the same packages.
  - Key test: the agent didn't diagnose the root cause — it just tried again (and again).

- **"gave_up"**
  - The agent stopped attempting to overcome the obstacle: submitted no answer, submitted a low-confidence guess without further effort, or explicitly declared it couldn't solve the problem.
  - Also use when the agent timed out while still stuck on the obstacle without making progress.

- **"not_applicable"**
  - No significant obstacle was encountered. Use only when primary_obstacle is "none".

**Tie-breaking:**
- If the agent tried brute_force_retry first and then switched to alternative_code, choose the approach that most directly produced (or failed to produce) the final answer.
- If the agent applied a direct_fix that partially worked and then used selective_execution for the remaining issue, prefer the dominant strategy.

### 7. Determine answer_verification (did the agent verify its answer before submitting?).

Look for evidence in the trace of the agent comparing its result to expected values, re-running computations, checking units/format, or expressing confidence/doubt.

Assign answer_verification to exactly one of:

- **"cross_checked"**
  - The agent explicitly verified the answer through an independent method: re-ran the computation a second time, compared the result against the manuscript's reported value, validated format/units against the task requirements, checked the result against a different code path, or ran the capsule's own validation/test scripts.
  - Key test: there are at least two independent pieces of evidence pointing to the same answer.

- **"single_pass"**
  - The agent computed the answer once and submitted it without explicit verification, but showed no signs of doubt. The answer came from a single code execution or single source, and the agent accepted it directly.
  - Key test: one computation, one answer, no cross-checking, no expressed uncertainty.

- **"expressed_uncertainty"**
  - The agent submitted an answer but explicitly noted uncertainty, caveats, or that it might be wrong. The agent may have flagged potential issues with precision, format, or interpretation.
  - Examples: "I'm not 100% sure this is the right metric"; "this might need more decimal places"; "I'll submit this but the computation may have been affected by the version issue."

- **"no_verification"**
  - No evidence of answer checking; the agent may have grabbed the first plausible value it found or submitted without any reflection on correctness. Also use when available evidence contradicts the submitted answer (e.g., error messages visible, but the agent submits a number anyway).

**Tie-breaking:**
- If the agent runs code and then briefly notes "this matches the expected format," that's `single_pass` (format matching is not substantive verification).
- If the agent explicitly compares its computed value to the manuscript and notes agreement, that's `cross_checked`.
- If the agent expresses uncertainty but also cross-checks, prefer `cross_checked` (the verification happened regardless of confidence level).

### 8. Determine the overall strategy_pattern.

Review the agent's behavior over the entire run and identify the dominant strategic pattern.

Assign strategy_pattern to exactly one of:

- **"mostly_linear_progress"**
  - The agent followed a mostly sequential plan — explore the repo, understand the task, install dependencies, run code, extract the answer — with only minor detours or easily-resolved issues.
  - No major backtracking; adjustments are incremental and generally effective.

- **"multiple_distinct_strategies"**
  - The agent tried two or more fundamentally different approaches to get the answer: e.g., tried running the original code, then switched to writing a custom script, then tried web search. Each strategy is substantively different.

- **"late_strategy_shift"**
  - The agent committed to one approach for most of the run, then pivoted to a substantially different strategy near the end — often when running out of time or after realizing the first approach won't work.
  - The shift is temporally concentrated in the final ~20% of the run.

- **"repetitive_or_looping"**
  - The agent repeated the same or very similar actions multiple times without making meaningful progress: reinstalling the same failing packages, re-running the same broken script, making trivial variations without diagnosing the root cause.

- **"gave_up_early"**
  - The agent abandoned the task after minimal exploration: made only a few attempts, declared the task impossible quickly, or produced a guess after superficial investigation.

- **"unclear_or_mixed"**
  - The pattern doesn't fit the above categories, or multiple patterns are equally present without a clear dominant one.

**Tie-breaking:**
- If the run starts linear but then loops extensively on one obstacle, prefer `repetitive_or_looping` if the looping dominated the outcome.
- If the agent tries multiple strategies but each attempt is brief and superficial, prefer `gave_up_early` if the total effort was minimal.
- Choose the pattern most diagnostically informative for understanding why the outcome occurred.

### 9. Construct the explanation narrative.

Write a concise, self-contained narrative (3-5 sentences) that a reader unfamiliar with the transcript can understand. The explanation must cover:

1. **Initial approach**: What the agent did first (explored the repo structure, read the task, identified relevant scripts).
2. **Primary obstacle** (if any): What went wrong and where in the trace it appeared.
3. **Response to obstacle**: How the agent reacted — did it fix it, work around it, retry, fall back to external sources, or give up?
4. **Answer source**: Where the final submitted value actually came from (ran original code, read from manuscript, extracted from cached output, etc.).
5. **Verification**: Whether and how the agent checked its answer before submitting.

Keep the narrative evidence-based. Reference specific trace blocks where relevant using citations. Use light causal language grounded in the trace (e.g., "Because the R dependency failed to compile, the agent rewrote the key statistical test in Python and submitted that result without cross-checking against the manuscript.").

### 10. Produce the final output object.

Populate all required fields with exactly one value from each enum:

- **label**: "success" | "failure"
- **answer_source**: "original_pipeline" | "modified_pipeline" | "repo_artifacts" | "manuscript_or_readme" | "figure_computational" | "figure_visual" | "external_source" | "no_answer"
- **primary_obstacle**: "none" | "dependency_install_failure" | "dependency_version_conflict" | "compilation_timeout" | "environment_constraint" | "code_bug_original" | "code_bug_introduced" | "data_missing_or_inaccessible" | "task_ambiguity" | "tool_limitation"
- **obstacle_resolution**: "direct_fix" | "selective_execution" | "alternative_code" | "external_fallback" | "brute_force_retry" | "gave_up" | "not_applicable"
- **answer_verification**: "cross_checked" | "single_pass" | "expressed_uncertainty" | "no_verification"
- **strategy_pattern**: "mostly_linear_progress" | "multiple_distinct_strategies" | "late_strategy_shift" | "repetitive_or_looping" | "gave_up_early" | "unclear_or_mixed"
- **explanation**: A non-empty narrative string (3-5 sentences) consistent with all other field assignments.

**Consistency checks before submitting:**
- If primary_obstacle is "none", then obstacle_resolution must be "not_applicable".
- If obstacle_resolution is "not_applicable", then primary_obstacle must be "none".
- If answer_source is "no_answer", label should almost always be "failure".
- If label is "failure" and answer_source is not "no_answer", the agent produced an answer but it was wrong — explain why in the narrative.
- The explanation should reference the assigned answer_source, primary_obstacle, and obstacle_resolution coherently.

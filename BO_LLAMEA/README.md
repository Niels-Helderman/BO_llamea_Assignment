# LLaMEA-Generated Bayesian Optimization

This repository implements a two-stage search and evaluation framework for automatically designed Bayesian optimization (BO) algorithms. [LLaMEA](https://github.com/XAI-liacs/LLaMEA) generates Python optimizer classes for two six-dimensional minimization problems: a [COCO-BBOB](https://numbbo.github.io/coco/) function via IOH and a problem from [Practical_Problems](https://github.com/BayesOptApp/Practical_Problems). The selected algorithms are benchmarked on **both** problems against CMA-ES and BoTorch.

The outer evolutionary search **maximizes mean AOCC**; each generated BO algorithm **minimizes the black-box objective**. The code targets LLaMEA **1.4.2**.

## 1. Project structure

```text
BO_LLAMEA/
├── configs/
│   └── experiment.yaml       # Problem, LLM, ES and evaluation settings
├── prompts/
│   ├── bbob.txt              # BBOB task prompt
│   ├── realworld.txt         # Real-world task prompt
│   └── operators/            # Seven mutation/crossover prompts + common rules
├── runs/
│   ├── stage1.py             # Generate, evaluate, select, checkpoint and resume
│   └── stage2.py             # Cross-problem benchmark and figures
├── src/
│   ├── problems.py           # Problem factories, bounds, references, calibration
│   ├── performance.py        # Objective guard, AOCC, traces, parallel seed runs
│   ├── baselines.py          # CMA-ES, BoTorch, random-search utility
│   └── utils.py              # Configuration, seeds, saving and plotting
├── results/                  # Created during execution
├── requirements.txt
└── README.md
```

`runs/` controls the experiments. `src/` provides the shared evaluation infrastructure. `prompts/` contains the text supplied to the LLM. `configs/experiment.yaml` selects the experimental conditions. The LLaMEA library itself is an installed dependency.

## 2. Installation

**Requirements:** Python 3.11–3.12, Git, SWIG, and access to Gemini or a local Ollama model. Run commands from the repository root.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

- **Windows PowerShell:** `.\.venv\Scripts\Activate.ps1`
- **Linux/macOS:** `source .venv/bin/activate`

Install SWIG using the operating system's package manager, or use Conda with `conda install -c conda-forge swig`. Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install "git+https://github.com/BayesOptApp/Practical_Problems.git"
```

### Gemini API access

The default `llm.provider` is `gemini`. Obtain an API key from [Google AI Studio](https://aistudio.google.com/apikey) and set `GEMINI_API_KEY` in the environment **without saving it in the repository**.

- **Windows PowerShell:** copy the API key, then run `$env:GEMINI_API_KEY = (Get-Clipboard -Raw).Trim()` in the same terminal as the experiment. Avoid copying another command between copying the key and executing the clipboard command.
- **Linux/macOS (Bash):** run `read -rs GEMINI_API_KEY; export GEMINI_API_KEY` and paste the key at the prompt.

For a local model, run Ollama, pull the desired model, and set `llm.provider: ollama` and `llm.model` in YAML. LLaMEA uses the configured provider to generate initial algorithms and offspring.

## 3. Reproduce the main experiment

The uploaded configuration defines **group 9**, **BBOB function 1 / instance 1**, **real-world problem 1124**, and **dimension 6**. The evaluation seeds are therefore **90–99** for every method and target. Stage 1 is configured for `(2+2)` elitist LLaMEA with roulette parent selection, novelty operators, rich feedback, and a 60-call budget per seeded run.

After installation and provider setup, run these commands **in order**:

```bash
# Stage 1: independent 20-candidate searches on both targets
python -m runs.stage1 --target bbob --limit 20
python -m runs.stage1 --target realworld --limit 20

# Stage 2: both winners and both baselines, on both targets
python -m runs.stage2 --budget 600 --workers 8
```

This is the recorded command configuration for the 20-candidate searches and the 600-call Stage 2 benchmark. Stage 1 uses **10 seeds × at most 60 objective calls** per candidate; Stage 2 evaluates **4 methods × 2 targets × 10 seeds**, each with **up to 600 calls**. `--workers 8` changes seed-level parallelism, not the seed assignments. LLM sampling and some numerical routines may still vary across environments.

**Output behavior:** An untagged Stage 1 command starts a fresh search and replaces only `results/stage1/<target>/`. Stage 2 reads the winners from those two default directories. Running Stage 2 again with the same budget replaces `results/stage2/budget_0600/`. Archive any result directories that must be retained before repeating those commands.

### Optional run controls

| Purpose | Command or setting | Behavior |
|---|---|---|
| Continue Stage 1 | `python -m runs.stage1 --target bbob --resume --limit 100` | The limit is **cumulative candidates**, not extra generations |
| Short Stage 2 run | `python -m runs.stage2 --budget 50 --workers 4` | Separate `results/stage2/budget_0050/` output |
| Tagged Stage 1 repetition | `python -m runs.stage1 --target bbob --limit 20 --tag P1 --rep 1` | Writes `results/stage1_runs/P1_r1/bbob/` |
| ES / prompting variants | Stage 1: `--parents`, `--offspring`, `--selection`, `--operators`, `--feedback` | Overrides matching YAML settings for that run |
| Change LLM or AOCC | `configs/experiment.yaml` | Requires a **new** search if the saved resume signature changes |

`--rep` requires `--tag`. Tagged searches are independent experiments; Stage 2 does **not** automatically load their winners. Without a tag, the `stage1_runs/` directory is not created by Stage 1.

## 4. End-to-end workflow (pseudocode)

The search budget counts **individual generated candidates**, including the initial parent population. It does not count generations. Each candidate is scored from the same ten problem seeds. Tagged repetitions and shortened benchmarking budgets are optional run parameters, not changes to the optimization objective.

```text
read experiment.yaml
seeds = [10*group_id, ..., 10*group_id + 9]

FOR target IN {bbob, realworld}:
    load problem, target-specific task prompt and eligible operator prompts
    metric = fixed AOCC reference + optional cached calibration scale

    FOR each requested independent search repetition rep:
        output = stage1/<target>                       if untagged
                 stage1_runs/<tag>_r<rep>/<target>     if tagged
        initialize n_parents LLaMEA individuals

        WHILE evaluated_candidate_count < candidate_limit:
            generate initial individuals or up to n_offspring new candidates
            (shorten final batch to respect candidate_limit)

            FOR each generated Python optimizer:
                FOR seed IN seeds, in parallel (up to stage1.seed_workers):
                    create fresh black-box instance and seed the RNGs
                    run optimizer with guarded objective(x)
                    record each accepted x and f(x), up to 60 calls
                    calculate best-so-far curve and AOCC for that seed
                fitness = mean(seed AOCC) if all ten runs succeed, else 0
                save code, per-seed traces, fitness and evaluation feedback

            update elitist ES population using configured parent selection
            save an atomic population checkpoint and fitness history
        export the best valid generated algorithm as best.py / best.json

methods = {LLaMEA_bbob, LLaMEA_realworld, CMA-ES, BoTorch}
budget = Stage 2 --budget override, otherwise stage2.budget
workers = Stage 2 --workers override, otherwise stage2.seed_workers

FOR target IN {bbob, realworld}:
    FOR method IN methods:
        FOR seed IN seeds, in parallel (up to workers):
            run a fresh optimizer with a guarded budget of budget calls
            save accepted objective observations and per-run status
        save per-evaluation and per-seed data; update summary statistics
    plot mean best-so-far convergence +/- standard error
    save raw and reference-relative convergence figures under
        results/stage2/budget_<budget>/
```

Stage 1 population checkpoints support `--resume` after committed generations. Stage 2 writes results as method/target combinations finish but has no independent resume command.

## 5. Configuration and experimental choices

The central settings are in `configs/experiment.yaml`. CLI options override only the corresponding parameters for the current invocation.

| Section | Key settings | Effect |
|---|---|---|
| `problems` | BBOB `fid`, `instance`; real-world `problem_id`; `dimension` | Selects the two fixed minimization tasks |
| `llm` | `provider`, `model` | Selects the code-generation backend |
| `llamea` | `n_parents`, `n_offspring`, `elitism`, `parent_selection` | Controls the evolutionary strategy |
| `llamea` | `operators`, `operator_weights`, `feedback` | Controls offspring prompts and feedback detail |
| `stage1` | `max_candidates`, `budget`, `seed_workers`, `timeout_seconds` | Candidate count and seeded evaluation limits |
| `stage1.aocc` | `lower_log10_error`, `upper_log10_error`, references, scale | Defines the fitness transformation |
| `stage2` | `budget`, `seed_workers`, `timeout_seconds`, `botorch_initial_points`, `ioh_logging` | Controls benchmarking and optional native logging |

**Prompts.** `prompts/bbob.txt` and `prompts/realworld.txt` are task instructions used during initialization and offspring generation. In `novelty` mode, seven operator files specify refinement, surrogate, acquisition, trust-region, hybrid, initialization and crossover mutations; `common.txt` supplies shared constraints. The YAML operator weights determine sampling probabilities. Crossover is eligible only when at least two parents exist. In `baseline` mode, LLaMEA uses its default mutation prompt. `basic` and `rich` feedback are both computed from actual run statistics; `rich` adds convergence checkpoints, seed variability and stagnation summaries without changing fitness.

**AOCC.** For budget \(B\), let \(b_t\) be the best-so-far objective, \(r\) the fixed problem reference, and \(S\) the optional problem scale. With lower/upper logarithmic clipping bounds \(L,U\), the implemented fitness for each seed is:

\[
\operatorname{AOCC}=\frac{1}{B}\sum_{t=1}^{B}\operatorname{clip}\!\left(\frac{U-\log_{10}(\max((b_t-r)/S,10^L))}{U-L},0,1\right).
\]

With `scale: null`, the implementation uses the unscaled error \(b_t-r\) instead of dividing by \(S\). The current window is \(L=-8,\ U=2\). For BBOB, the reference comes from the instance-specific IOH optimum. For real-world problem 1124, zero is a lower bound of its squared-error objective. With `scale: auto`, a fixed scale is computed from **32 separately logged calibration points per target**, cached in `results/calibration/`, and reused across algorithms. Calibration calls are not part of any seeded BO run. The calibrated metric differs from the unscaled log-precision AOCC used in some LLaMEA studies, so the chosen mode must be reported alongside scores.

**Baseline implementations.** CMA-ES uses `pycma` ask/tell, initialized at the search-space midpoint with initial sigma equal to 25% of the mean domain width. BoTorch uses a `SingleTaskGP`, standardized outcomes, 12 scrambled Sobol initialization points, `LogExpectedImprovement`, five acquisition restarts and 64 raw samples. BoTorch models the negative objective so its internal maximization implements minimization. All initialization calls count toward the stated budget.

## 6. Results and analysis

```text
results/
├── calibration/
│   ├── bbob.json, realworld.json
├── stage1/
│   ├── bbob/
│   │   ├── best.py, best.json, fitness.png, experiment.json
│   │   ├── code/candidate_NNN.{py,json}
│   │   ├── traces/candidate_NNN.json
│   │   └── state.json, checkpoint.json, exp-*/   (LLaMEA history)
│   └── realworld/                              (same format)
├── stage1_runs/                                 (only for tagged searches)
│   └── <tag>_r<rep>/<target>/
└── stage2/
    └── budget_0600/
        ├── experiment.json, summary.csv
        ├── bbob_runs.csv, bbob_statuses.csv, bbob_results.json
        ├── realworld_runs.csv, realworld_statuses.csv, realworld_results.json
        └── figures/                            (raw and normalized convergence)
```

**Stage 1 analysis:** `fitness.png` shows each candidate's mean AOCC and the nondecreasing best-so-far fitness. `code/` contains every generated algorithm and its fitness/feedback record; `traces/` contains the ten seeded objective histories for each candidate. The native LLaMEA logs retain prompts, responses and evolutionary lineage. `best.py` is the chosen algorithm; `best.json` records its source candidate and fitness. Invalid candidate records are retained separately from valid scores.

**Stage 2 analysis:** Each `*_runs.csv` row is one accepted objective evaluation (including input coordinates and best-so-far objective); `*_statuses.csv` has one row per seeded optimizer run; `*_results.json` contains full traces; and `summary.csv` aggregates final objectives, AOCC, successful-run counts, warnings and standard errors. The figures show mean best-so-far trajectories with **± one standard error**, using raw and normalized/reference-relative views. Raw BBOB plots display best objective minus known optimum; raw real-world plots display the objective. The plot labels do not change the underlying data transformation. If enabled, native IOH logs supplement the custom traces and use configured event triggers rather than recording all observations by default.

**Describing the selected algorithms:** Since LLaMEA generates code stochastically, describe each final `best.py` from its actual implementation. Record its initialization, surrogate, acquisition/next-point rule, exploration mechanism, model updates, termination conditions and computational cost. Include a short, code-specific pseudocode for each winner and relate its choices to the Stage 1 score and its native-versus-transfer performance in Stage 2. The generated code and seed-level traces form the evidence for these descriptions.

**Reproducibility record:** `experiment.json` saves target IDs, seeds, budgets, prompt and configuration information, selected method names and software/platform metadata. Retain the source code, task/operator prompts, the selected searches' LLaMEA histories, winner files, per-seed results and figures. Also record the CPU model, logical cores, RAM, operating system and accelerator (if used) for the machine running the benchmark. LLM output can vary even when evaluation seeds are fixed. Serialized LLaMEA checkpoints may contain provider credentials and must not be published or loaded from untrusted sources.

## 7. Dependencies and attribution

The framework integrates [LLaMEA](https://github.com/XAI-liacs/LLaMEA), [IOHexperimenter / IOH](https://github.com/IOHprofiler/IOHexperimenter), [Practical_Problems](https://github.com/BayesOptApp/Practical_Problems), [pycma](https://github.com/CMA-ES/pycma), [BoTorch](https://github.com/pytorch/botorch), [GPyTorch](https://github.com/cornellius-gp/gpytorch) and [scikit-learn](https://scikit-learn.org/). These libraries remain independently licensed. Generated optimizers execute dynamically in subprocesses: **the subprocess timeout is not a security sandbox**; run untrusted generated code in an appropriately isolated environment.

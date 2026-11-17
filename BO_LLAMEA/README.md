# LLaMEA-Generated Bayesian Optimization

This repository implements a two-stage experiment in **automated Bayesian optimization (BO) algorithm design**. [LLaMEA](https://github.com/XAI-liacs/LLaMEA) generates executable Python optimizers. The optimizers are evaluated on a six-dimensional [COCO-BBOB](https://numbbo.github.io/coco/) problem and a six-dimensional problem from [Practical_Problems](https://github.com/BayesOptApp/Practical_Problems). The selected optimizers are then benchmarked on *both* problems against CMA-ES and a BoTorch baseline.

All black-box objectives are **minimized**. LLaMEA **maximizes** an aggregated Area Over the Convergence Curve (AOCC) fitness. This implementation uses LLaMEA **1.4.2**.

## 1. Repository layout

```text
BO_LLAMEA/
├── configs/
│   └── experiment.yaml          # Problem, LLM, ES, evaluation and metric settings
├── prompts/
│   ├── bbob.txt                 # BBOB task instructions
│   ├── realworld.txt            # Real-world task instructions
│   └── operators/              # Custom LLaMEA operator instructions
│       ├── common.txt
│       ├── refine.txt
│       ├── surrogate.txt
│       ├── acquisition.txt
│       ├── trust_region.txt
│       ├── hybrid.txt
│       ├── init.txt
│       └── crossover.txt
├── runs/
│   ├── stage1.py                # Evolutionary search, candidate scoring and resume
│   └── stage2.py                # Cross-problem benchmarking and reporting
├── src/
│   ├── problems.py              # Problem factories, bounds, references and calibration
│   ├── performance.py           # Guarded objective, AOCC and parallel seed evaluation
│   ├── baselines.py             # CMA-ES, BoTorch and random-search utility
│   └── utils.py                 # Configuration, persistence, seeding and plots
├── results/                     # Created by experiments
├── requirements.txt
└── README.md
```

**Division of responsibilities:** `runs/` coordinates experiments, `src/` implements shared evaluation and reporting, `prompts/` contains the editable LLM instructions, and `configs/experiment.yaml` specifies experiment choices. LLaMEA itself is an installed dependency, not copied into this repository.

## 2. Experimental workflow

### Stage 1 — generate and select two BO algorithms

Two **independent** searches are performed: one for BBOB and one for the real-world target.

```text
Task prompt + configured operator prompts
             │
             ▼
       LLaMEA / LLM
             │  Python optimizer class
             ▼
      Candidate evaluator
             │
             ├── seed 10g     → at most 60 objective evaluations
             ├── seed 10g+1   → at most 60 objective evaluations
             └── ...         → 10 independent runs
             │
             ▼
   Mean AOCC + validity feedback
             │
             ▼
      Evolutionary selection
             │
             └── next offspring, up to 100 candidates total
```

For group ID `g`, the same ten seeds `10g, ..., 10g+9` are used for **every** algorithm and problem. The configured starting ES is **(2+1)**: two initial parents, one offspring per generation, elitist survivor selection, roulette parent selection, and the weighted novelty-operator portfolio. LLaMEA's generated class must provide:

```python
class Optimizer:
    def __init__(self, budget: int, dim: int):
        ...

    def __call__(self, func):
        # All black-box evaluations must call func(x).
        # Return (best_y: float, best_x: array-like).
        ...
```

The objective's bounds are available as `func.bounds.lb` and `func.bounds.ub`. The budget includes **all** initialization evaluations. Candidates are evaluated in isolated worker processes, and the objective wrapper checks input shape, finiteness, bounds and evaluation count. All observed points and values are retained in per-seed traces.

A candidate receives the mean AOCC across ten seeds if all ten runs complete successfully; invalid candidates receive fitness zero. Generated source code, fitness feedback, per-seed histories and LLaMEA logs are persisted. The best valid candidate is exported as `best.py` and `best.json` for each target.

### Stage 2 — evaluate transfer and baselines

The selected BBOB-generated and real-world-generated algorithms are **each run on both problems**. The benchmark also includes CMA-ES and BoTorch: four methods × two problems × ten seeds = **80 seeded runs**. The per-run evaluation budget is at most **600**, including initialization; `--budget` can select a smaller development budget. Each seeded run uses the same guarded evaluator as Stage 1.

Stage 2 stores per-evaluation CSV data, per-run JSON traces and statuses, method-level summaries, and convergence figures with mean **± standard error** over successful runs. Failure counts are reported separately; failed runs are not silently averaged into success statistics.

## 3. Installation

**Prerequisites:** Python 3.11 or 3.12, Git, SWIG, and a supported LLM provider. A Python virtual environment is recommended. The experiment runs on CPU; more evaluation workers require more CPU and memory.

Open a shell in the repository root (the directory containing `requirements.txt`). Choose **one** environment method:

**Virtual environment**

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

**Conda alternative**

```bash
conda create -n bo-llamea python=3.12 -y
conda activate bo-llamea
conda install -c conda-forge swig
```

Install Git and SWIG separately if not provided by the environment. Check `git --version` and `swig -version`. Then install the Python dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/BayesOptApp/Practical_Problems.git"
python -m pip install -r requirements.txt
python -m pip check
```

The practical-problem package is installed directly from its upstream repository; LLaMEA 1.4.2 is pinned in `requirements.txt` for API and checkpoint compatibility. To verify the installation:

```bash
python -c "from llamea import LLaMEA; import ioh, Practical_Problems, cma, botorch; print('Imports OK')"
```

### LLM provider configuration

The default is the Google Gemini model selected under `llm` in YAML. Obtain a key at [Google AI Studio](https://aistudio.google.com/apikey) and set `GEMINI_API_KEY` **in the shell environment**, not in source files or committed YAML.

**PowerShell (clipboard-based):** type or paste the following command **without pressing Enter**; next copy the key from AI Studio, return to the shell, and press Enter. This reads the key without placing it in the command history.

```powershell
$env:GEMINI_API_KEY = (Get-Clipboard -Raw).Trim()
```

**Bash (Linux/macOS):**

```bash
read -rs -p 'Gemini API key: ' GEMINI_API_KEY; echo
export GEMINI_API_KEY
```

Check only whether the key exists (never print it):

```bash
python -c "import os; print('Gemini key configured:', bool(os.getenv('GEMINI_API_KEY')))"
```

Alternatively, install [Ollama](https://ollama.com/), start its service, pull a suitable local model, and set `llm.provider: ollama` and `llm.model` to the local model name. API/model availability and LLM rate limits depend on the selected provider.

## 4. Configuration and prompts

All experiment settings are in **`configs/experiment.yaml`**. Verify the assigned group ID and both problem IDs before a final run. The current configuration specifies group `9`, BBOB function `1` (instance `1`), real-world problem `1124`, and dimension `6`.

| YAML field | Default | Meaning |
|---|---|---|
| `llm.provider`, `llm.model` | `gemini`, `gemini-3.1-flash-lite` | LLM provider/model |
| `llamea.n_parents` | `2` | Parent population size |
| `llamea.n_offspring` | `1` | New candidates per generation |
| `llamea.elitism` | `true` | Retain higher-fitness individuals |
| `llamea.parent_selection` | `roulette` | Rank-based parent sampling |
| `llamea.operators` | `novelty` | Weighted custom operator portfolio (`baseline` uses LLaMEA's default) |
| `llamea.feedback` | `rich` | Adds convergence and seed-variation statistics to fitness feedback |
| `llamea.operator_weights` | See YAML | Sampling weights for the seven operators; `0` disables one |
| `stage1.max_candidates` | `100` | Maximum evaluated candidate slots per target search |
| `stage1.budget`, `seed_workers` | `60`, `4` | Inner budget and number of parallel seeded workers |
| `stage1.aocc` | See YAML | Fixed reference, log-error bounds, optional scale |
| `stage2.budget`, `seed_workers` | `600`, `4` | Benchmark budget and parallel workers |
| `stage2.botorch_initial_points` | `12` | Sobol initialization for BoTorch |
| `stage2.ioh_logging` | `false` | Optional native IOH logging for BBOB |

**Task prompts.** `prompts/bbob.txt` and `prompts/realworld.txt` describe the respective BO tasks. The runner appends the required optimizer interface and supplies the resulting prompt to LLaMEA for both initialization and mutation.

**Operator prompts.** In novelty mode, `prompts/operators/` provides seven named instructions: refine, surrogate, acquisition, trust region, hybrid, initialization, and crossover. Every active operator includes the shared constraints from `common.txt`. The YAML weights govern which operator is sampled for an offspring; crossover requires at least two parents. In baseline mode LLaMEA uses its built-in default mutation instruction.

**Evaluation feedback.** `basic` returns fitness, validity and initial/final performance; `rich` adds intermediate convergence checkpoints, AOCC variation and stagnation information. Both modes calculate fitness identically. Prompt content, weights, model and other search settings are included in the resume signature; altering them requires a fresh search.

**AOCC metric.** Let `b_t` be the best objective after evaluation `t`, `r` a fixed problem reference, and `s` the optional fixed scale. In the default scaled mode:

```text
error_t = max((b_t - r) / s, 10^L)
AOCC    = mean_t clip((U - log10(error_t)) / (U - L), 0, 1)
```

The current bounds are `L=-8`, `U=2`. Higher AOCC is better and values lie in `[0,1]`. For BBOB, `r` is the instance-specific IOH optimum. For real-world ID 1124, zero is a justified lower bound of its sum-of-squares objective; other practical problems require an explicitly justified reference. With `scale: auto`, `s` is obtained once from `calibration_points: 32` fixed-seed, separately logged calibration evaluations and reused for every algorithm. With `scale: null`, no scale is applied and the paper-style *unscaled error* is used. **Even unscaled-error AOCC has a score between 0 and 1.** The calibrated variant is a documented project-specific metric and should be reconciled with the instructors' AOCC convention before final reporting.

## 5. Running experiments

Run all commands from the repository root with the virtual environment activated.

### Stage 1

Start a short independent search for each target:

```bash
python -m runs.stage1 --target bbob --limit 10
python -m runs.stage1 --target realworld --limit 10
```

Run to the configured maximum by omitting `--limit`, or specify `--limit 100`. **`--limit` counts cumulative candidates, not generations**; for `(2+1)`, 100 candidates comprise two initial parents plus up to 98 offspring.

Resume from a committed checkpoint:

```bash
python -m runs.stage1 --target bbob --resume --limit 100
```

Without `--resume`, the selected target's previous Stage 1 output is **overwritten**. With `--resume`, the effective prompts, configuration and fitness metric must match the checkpoint. Only load locally trusted checkpoint pickle files.

To preserve a separately named ES/LLM experiment:

```bash
python -m runs.stage1 --target bbob --tag operator_exp --rep 0 --limit 20
```

Tagged experiments use `results/stage1_runs/operator_exp_r0/bbob/`; ordinary runs **do not create** `stage1_runs/`. `--rep` requires `--tag`. ES settings may also be overridden with `--parents`, `--offspring`, `--selection`, `--operators`, and `--feedback`. The Stage 2 runner reads winners from the default `results/stage1/<target>/` directories; tagged winners are not selected automatically.

### Stage 2

After both default Stage 1 runs have produced `best.py` and `best.json`:

```bash
python -m runs.stage2 --budget 50             # Development run
python -m runs.stage2 --budget 600 --workers 4 # Full evaluation horizon
```

The per-run budget can be overridden in the range `1..600`. Results are kept separately by budget, e.g. `results/stage2/budget_0050/` and `budget_0600/`; **rerunning the same budget overwrites its folder**. The four algorithm/problem combinations are processed sequentially, with the ten seeds parallelized within each combination. Stage 2 does not currently support resuming partially finished experiments.

## 6. Baselines and evaluation conditions

| Method | Main settings | Rationale |
|---|---|---|
| CMA-ES | `pycma` ask/tell; initial center at midpoint of bounds; initial sigma = 25% of mean domain width | Standard derivative-free optimizer and bounded exploration |
| BoTorch | `SingleTaskGP` with standardized targets; 12 scrambled Sobol initial points; `LogExpectedImprovement`; 5 acquisition restarts and 64 raw samples | Conventional GP-based BO benchmark with stable acquisition and moderate search cost |
| LLaMEA_bbob | Selected `results/stage1/bbob/best.py` | Algorithm evolved for BBOB, assessed on both targets |
| LLaMEA_realworld | Selected `results/stage1/realworld/best.py` | Algorithm evolved for the real-world target, assessed on both targets |

BoTorch maximizes the **negated** objective internally to implement minimization. Each method may use its own initialization design, but every initial objective query counts against the same per-run budget. The BoTorch hyperparameters above are implemented defaults, **not confirmation of instructors' externally specified baseline settings**; verify the course specification before final submission.

## 7. Results, plots, and logging

```text
results/
├── calibration/
│   ├── bbob.json
│   └── realworld.json
├── stage1/
│   ├── bbob/
│   │   ├── best.py, best.json
│   │   ├── fitness.png
│   │   ├── code/candidate_NNN.{py,json}
│   │   ├── traces/candidate_NNN.json
│   │   ├── experiment.json, state.json, checkpoint.json
│   │   └── exp-*/              # Native LLaMEA history and checkpoint
│   └── realworld/             # Same layout
├── stage1_runs/               # Created only by tagged runs
│   └── <tag>_r<rep>/<target>/
└── stage2/
    └── budget_0600/
        ├── experiment.json, summary.csv
        ├── bbob_results.json, bbob_runs.csv, bbob_statuses.csv
        ├── realworld_results.json, realworld_runs.csv, realworld_statuses.csv
        └── figures/
            ├── bbob_convergence.png, bbob_normalized.png
            └── realworld_convergence.png, realworld_normalized.png
```

**Stage 1 figures** are titled `LLaMEA search on BBOB` or `LLaMEA search on real world`, with `Candidates` on the x-axis and `Mean AOCC (normalized/unscaled)` on the y-axis. Candidate scores and the non-decreasing best-so-far fitness are plotted separately.

**Stage 2 figures** show `BBOB` or `real world` BO convergence with the seed range in the title. The raw y-axis is `Loss` for BBOB (`best - optimum`) and `Objective` for the real-world task. Supplementary normalized plots use `Normalized gap` (`(best - reference)/scale`) or `Gap` when unscaled. Legends show `LLaMEA_bbob`, `LLaMEA_realworld`, `CMA-ES` and `BoTorch`; shaded bands show ± one standard error of the mean. All plots are produced from the corresponding stored per-seed trajectories.

**Run records** contain proposed six-dimensional points, observed objective values, AOCC, final best objective, runtime, validity/error status and aggregated numerical warnings. Stage 1 also stores generated code, feedback, parent/evolution information and LLaMEA's prompt/response logs. Stage 2 stores one CSV row per accepted objective call, one status row per seeded run and a summary for each method/problem pair. Native IOH logging for BBOB can be enabled with `stage2.ioh_logging: true`; it supplements, rather than replaces, the detailed project traces.

## 8. Analysis of the two selected algorithms

**This section must be completed from the actual `best.py` files after the final two searches.** The generation process is stochastic, and the repository does not contain a fixed implementation of either winner. A component-level account cannot be accurately written before the final algorithms exist. For an assessment-ready submission, record the following for **both** winners:

| Component | BBOB-generated winner | Real-world-generated winner |
|---|---|---|
| Source / LLaMEA candidate ID | From `bbob/best.json` | From `realworld/best.json` |
| Initialization | Read `bbob/best.py` | Read `realworld/best.py` |
| Surrogate or predictive model | Identify model and training data | Identify model and training data |
| Acquisition / next-point rule | Identify scoring and optimization | Identify scoring and optimization |
| Exploration and fallback | Identify exploration and failure handling | Identify exploration and failure handling |
| Update / termination | Identify model updates and budget use | Identify model updates and budget use |
| Computation cost | Describe fitting/restart policy | Describe fitting/restart policy |

For each winner, include a **code-specific pseudocode listing** of initialization, surrogate fitting, point proposal, objective evaluation, incumbent update and termination. If a winner uses a nonstandard or surrogate-free method, document what it actually does rather than forcing it into a generic GP template. Explain how its components relate to the Stage 1 AOCC outcome, and compare its Stage 2 behavior on the native and transfer targets. Record the exact final candidate IDs and scores.

The required final results are the **full source code, saved search history and prompts/LLM responses, both winner implementations, all Stage 2 per-seed results, and plots with variability**—not only two `best.py` files.

## 9. Reproducibility, hardware, and attribution

- Fixed group seeds are computed in `src/utils.py`; seed selection, problem IDs and experiment settings are stored in `experiment.json`. Tagged repetition indices change outer-loop randomness, not the ten evaluation seeds. External LLM generation can still be nondeterministic.
- `metadata()` records Python, operating system and package versions. For the final experiment, also record **CPU model, logical core count, RAM, operating system, GPU (if any), and whether inference used Gemini API or local Ollama**. Enter the measured hardware details below rather than assuming all contributors use the same machine.
- Generated Python runs in subprocesses with bounded evaluation counts and timeouts, but **subprocesses are not a security sandbox**. Execute generated source only in a suitably restricted environment.
- AOCC scaling/calibration, timeout failures, missing LLM responses and model-service errors should be reported when relevant to the experiment's validity. Ensure the metric and baseline settings used in the final run are consistent across methods.

**Final-run machine record (complete after running experiments):**

| Field | Actual machine |
|---|---|
| CPU / logical cores | _To be recorded_ |
| RAM | _To be recorded_ |
| OS and Python version | _To be recorded_ |
| GPU / accelerator | _To be recorded or “none”_ |
| LLM provider/model | _To be recorded_ |
| Dependency versions | See `experiment.json` |

**External components and attribution:** [LLaMEA](https://github.com/XAI-liacs/LLaMEA), [COCO/IOH](https://github.com/IOHprofiler/IOHexperimenter), [Practical_Problems](https://github.com/BayesOptApp/Practical_Problems), [CMA-ES/pycma](https://github.com/CMA-ES/pycma), [BoTorch](https://github.com/pytorch/botorch), [GPyTorch](https://github.com/cornellius-gp/gpytorch), and [scikit-learn](https://scikit-learn.org/stable/). Their original software and licenses remain with their respective authors. This repository integrates these libraries through the shared BO evaluation interface; generated source is stored separately for inspection and reporting.

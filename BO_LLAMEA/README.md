# Bayesian Optimization with LLaMEA

This project implements the **two-stage Bayesian Optimization (BO) practical assignment**. It uses [LLaMEA](https://github.com/XAI-liacs/LLaMEA) to generate BO algorithms, then benchmarks the selected algorithms against **CMA-ES** and **BoTorch** on:

- One six-dimensional **COCO-BBOB** problem, supplied by [`ioh`](https://github.com/IOHprofiler/IOHexperimenter).
- One six-dimensional **real-world** problem, supplied by [`Practical_Problems`](https://github.com/BayesOptApp/Practical_Problems).

The BO algorithms **minimize** objective values. LLaMEA **maximizes** their average AOCC fitness. Importantly, the algorithm generated for each target is also tested on the *other* target to measure transferability.

## 1. Project structure

```text
BO_LLAMEA/
├── configs/
│   └── experiment.yaml       # All experiment and model settings
├── prompts/
│   ├── bbob.txt              # LLaMEA prompt for the BBOB search
│   └── realworld.txt         # LLaMEA prompt for the real-world search
├── runs/
│   ├── stage1.py             # LLaMEA generation, evaluation, selection, resume
│   └── stage2.py             # Cross-problem benchmarking and reports
├── src/
│   ├── problems.py           # Problem setup, bounds, optimum/reference, calibration
│   ├── performance.py        # Objective-call guard, AOCC, parallel seed evaluation
│   ├── baselines.py          # CMA-ES, BoTorch and a simple random-search utility
│   └── utils.py              # Seeds, configuration, saving results and plotting
├── results/
│   ├── calibration/          # Fixed metric calibration for each problem
│   ├── stage1/              # Generated algorithms, fitness, traces and checkpoints
│   └── stage2/              # Benchmark results grouped by evaluation budget
├── requirements.txt
└── README.md
```

**LLaMEA is an installed dependency, not code reimplemented here.** It handles LLM prompting, the evolutionary search and its internal logs. Our `runs/` modules provide the experiment entry points; `src/` implements the problem interfaces, evaluation, baselines and plots. The generated algorithms under `results/stage1/` are **outputs of experiments**, not hand-written framework code.

### Experiment workflow

```text
YAML configuration + target-specific prompt
                   |
                   v
Stage 1: LLaMEA generates a Python BO algorithm
                   |
      Evaluate on 10 fixed seeds × up to 60 objective calls
                   |
      Mean AOCC + failure feedback → evolutionary selection
                   |
       Save one best algorithm for each target
                   |
                   v
Stage 2: Evaluate FOUR methods on BOTH problems
  generated_bbob / generated_realworld / CMA-ES / BoTorch
                   |
    10 seeds per combination × chosen evaluation budget
                   |
           CSV, JSON and convergence plots
```

## 2. Installation

### Prerequisites

- **Python 3.11 or 3.12**, **Git**, and **SWIG**.
- A virtual environment (`venv` or Conda) is recommended.
- A supported LLM for Stage 1: **Gemini** (API key) or **Ollama** (locally installed model).
- CPU execution is supported; multiple evaluation workers require sufficient CPU/RAM.

Clone the repository (or extract its ZIP), then open a terminal **inside the folder containing `requirements.txt`**.

**Option A — `venv`:**

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

**Option B — Conda** (use this *instead of* `venv`):

```bash
conda create -n bo-llamea python=3.12 -y
conda activate bo-llamea
```

Install Git and SWIG using your operating system's package manager; with Conda, SWIG can be installed using `conda install -c conda-forge swig`. Check `git --version` and `swig -version`.

Install dependencies, in this order:

```bash
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/BayesOptApp/Practical_Problems.git"
python -m pip install -r requirements.txt
python -m pip check
```

The real-world problems come directly from GitHub. `requirements.txt` pins **`llamea==1.4.2`** because this project's checkpoint integration targets that version.

Verify the main imports if necessary:

```bash
python -c "from llamea import LLaMEA; import ioh, Practical_Problems, cma, botorch; print('Imports OK')"
```

### Configure the LLM

By default, `configs/experiment.yaml` selects Gemini:

```yaml
llm:
  provider: gemini
  model: gemini-3.1-flash-lite
```

Create an API key in [Google AI Studio](https://aistudio.google.com/apikey) and expose it as the environment variable **`GEMINI_API_KEY`**. Do not place API keys in YAML, code or shell history.

**Windows PowerShell:** place this command in the terminal **without executing it yet**. Copy your API key using AI Studio's **Copy** button, return to PowerShell and press **Enter** directly. Copying anything else in between replaces the clipboard contents.

```powershell
$env:GEMINI_API_KEY = (Get-Clipboard -Raw).Trim()
```

**Bash (Linux/macOS):**

```bash
read -rs -p 'Gemini API key: ' GEMINI_API_KEY; echo
export GEMINI_API_KEY
```

Optional check (does not display the key):

```bash
python -c "import os; print('Key configured:', bool(os.getenv('GEMINI_API_KEY')))"
```

The key is available only in the current terminal session. Provider model availability and free-tier limits may change.

To use **Ollama** instead, [install Ollama](https://ollama.com/), download a local model, set `llm.provider: ollama` and `llm.model` to its installed model name. No Gemini key is needed.

## 3. Configuration

The project's main settings live in **`configs/experiment.yaml`**. Set the correct group ID and selected problem IDs *before* running experiments.

| Setting | Purpose |
| --- | --- |
| `group_id` | Defines 10 fixed seeds: `10*g, ..., 10*g+9` |
| `dimension` | `6` for both problems |
| `problems.bbob.fid`, `instance` | BBOB function ID and instance (`1`) |
| `problems.realworld.problem_id` | Selected six-dimensional real-world problem |
| `llm.provider`, `llm.model` | Gemini or Ollama model for Stage 1 |
| `llamea.n_parents`, `n_offspring` | Evolution strategy population sizes, each `1..4` |
| `llamea.parent_selection` | `random`, `tournament` or `roulette` |
| `llamea.elitism` | Must remain `true` in this implementation |
| `stage1.max_candidates`, `budget` | Up to **100 candidates**, **60 objective calls per seed** |
| `stage1.seed_workers`, `timeout_seconds` | Parallel seed processes and per-run timeout |
| `stage1.aocc` | Fitness reference, normalization and calibration |
| `stage2.budget`, `seed_workers` | Benchmark budget (default **600**) and concurrency |
| `stage2.botorch_initial_points` | BoTorch baseline initialization size |
| `prompts.bbob`, `prompts.realworld` | Paths to the two independent search prompts |

**Evolution strategy:** the default is **two initial parents and two offspring per generation**, an elitist `(2+2)` search. `--limit` counts *total cumulative candidate attempts*, not generations: with `(2+2)`, `--limit 4` means 2 parents + 2 offspring. Without `--limit`, the configured maximum is 100. A final offspring batch is shortened to respect the candidate limit.

**How prompts are used:** Stage 1 loads the target's text file from `prompts/`, appends the required Python interface and passes it to LLaMEA. On later generations, LLaMEA also provides selected parent code and evaluation feedback. Both prompt files may contain different problem-specific guidance. Each generated algorithm must implement a class with `__init__(self, budget, dim)` and `__call__(self, func)`, returning `(best_y, best_x)`; `func(x)` is the **only** way to query the black-box objective and provides `func.bounds.lb` / `func.bounds.ub`.

**AOCC:** a candidate's fitness is the mean normalized AOCC over ten seeds; larger is better. BBOB uses its `ioh` optimum as the reference. Real-world problem **1124** uses a justified reference lower bound of zero; other practical problems require an explicit justified reference in YAML. With `stage1.aocc.scale: auto`, a fixed scale is calibrated from **32 additional, separately recorded objective queries per target** in `results/calibration/`, then reused for every candidate and benchmark. These setup calls are not part of any algorithm's 60/600-call budget. This is a **project-specific relative-log-error AOCC variant**; confirm the instructor's preferred normalization for the final graded analysis.

## 4. Running the experiments

Run commands **from the repository root** in the activated environment.

### Stage 1 — generate algorithms

```bash
python -m runs.stage1 --target bbob --limit 10
python -m runs.stage1 --target realworld --limit 10
```

This starts **two independent LLaMEA searches** (one per target). Each generated candidate is executed on **10 seeded runs × up to 60 objective calls**, and the mean AOCC is fed back to LLaMEA. A valid winner is saved as `results/stage1/<target>/best.py` with metadata in `best.json`.

To search up to the full **100-candidate limit**, omit `--limit` or set `--limit 100`.

To continue a previously committed search:

```bash
python -m runs.stage1 --target bbob --resume --limit 100
python -m runs.stage1 --target realworld --resume --limit 100
```

**Important:** without `--resume`, Stage 1 **overwrites that target's existing Stage 1 directory** and starts again at Candidate 1. With `--resume`, the limit is a *cumulative target*, not an additional number of candidates. Resume requires an unchanged prompt, model, problem, metric and configuration. The other target is not affected.

The generated-code evaluation is parallel **across seeds** (`stage1.seed_workers`, default `4`); LLM requests are not run in parallel. **Current limitation:** failed Gemini generation attempts (such as after repeated HTTP 503 responses) can be logged as invalid candidates and consume search slots. Inspect the candidate records if an API outage occurs.

### Stage 2 — benchmark both winners

After **both** Stage 1 targets have a valid `best.py` and `best.json`, run a quick comparison:

```bash
python -m runs.stage2 --budget 50
```

Or override the number of parallel seed processes:

```bash
python -m runs.stage2 --budget 50 --workers 8
```

For the final evaluation horizon:

```bash
python -m runs.stage2 --budget 600
```

Stage 2 runs **4 algorithms × 2 problems × 10 seeds = 80 seeded runs**. The `--budget` argument sets the maximum number of actual black-box calls *per run* (`1..600`); without it, the YAML value applies. Stage 2 evaluates each algorithm/problem pairing in turn and parallelizes its ten seeds (default 4 workers). It does **not** have `--resume`.

Each budget has a separate folder (`results/stage2/budget_0050/`, `budget_0600/`, etc.). Repeating Stage 2 with the **same** budget overwrites that folder. Use a smaller budget to preview runtime and figures; preserve the required final benchmark budget for the assignment. The current BoTorch baseline configuration is an example and should be checked against instructor requirements.

## 5. Results and plots

```text
results/
├── calibration/{bbob,realworld}.json
├── stage1/
│   ├── bbob/
│   │   ├── best.py, best.json           # Selected BO algorithm and fitness
│   │   ├── fitness.png                 # Candidate AOCC and best-so-far AOCC
│   │   ├── code/candidate_NNN.{py,json} # Candidate source and evaluation feedback
│   │   ├── traces/candidate_NNN.json   # Ten per-seed objective histories
│   │   ├── experiment.json             # Config, prompt, metric, seeds, environment
│   │   ├── state.json, checkpoint.json
│   │   └── exp-*/                      # Native LLaMEA logs and checkpoint pickle
│   └── realworld/                    # Same layout for the real-world search
└── stage2/
    └── budget_0050/
        ├── experiment.json, summary.csv
        ├── bbob_results.json, realworld_results.json
        ├── bbob_runs.csv, realworld_runs.csv
        ├── bbob_statuses.csv, realworld_statuses.csv
        └── figures/
            ├── bbob_convergence.png, realworld_convergence.png
            └── bbob_normalized.png, realworld_normalized.png
```

- **Stage 1 `fitness.png`:** mean normalized AOCC (**higher is better**) versus candidate count. Invalid candidates get fitness `-1` and are marked separately.
- **Stage 2 `*_convergence.png`:** best-so-far BBOB **loss** or real-world **objective**, averaged across valid seeds, with standard-error bands (**lower is better**).
- **Stage 2 `*_normalized.png`:** relative best-so-far gap on the same normalized logarithmic scale for both problems. These plots supplement, rather than replace, the raw convergence curves.
- **Stage 2 `*_statuses.csv`:** seeded run statuses, objective calls, runtime, failure messages and aggregated scikit-learn/GP warnings. Warnings are collected in the workers rather than flooding the terminal.

## 6. Reproducibility and limitations

The project saves experiment settings, prompts, objective references/scales, seeds, candidate code, fitness, per-run trajectories and environment metadata. For the report, document which **surrogate, acquisition strategy and initialization** each winner actually implements by reading its `best.py`; these details are generated, not fixed in `src/`.

Generated Python is executed in subprocesses with objective-call and wall-time limits. **A subprocess is not a security sandbox**: use a restricted environment for untrusted generated programs. Gemini errors (`429`/`503`) are distinct from invalid BO code, and the current runner can count failed LLM generations against the candidate limit. In Stage 2, Gaussian-process fitting cost can increase substantially at 600 evaluations, so short development runs do not predict full-run timing exactly.

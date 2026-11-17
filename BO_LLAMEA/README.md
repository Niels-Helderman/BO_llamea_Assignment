# LLaMEA-Generated Bayesian Optimization

This repository implements a two-stage experiment in **automated Bayesian optimization (BO) algorithm design**. [LLaMEA](https://github.com/XAI-liacs/LLaMEA) generates Python optimizers for two six-dimensional objectives: a [COCO-BBOB](https://numbbo.github.io/coco/) function accessed through IOH, and a task from [Practical_Problems](https://github.com/BayesOptApp/Practical_Problems). The two selected optimizers are then evaluated on **both objectives**, alongside CMA-ES and BoTorch.

All black-box problems are **minimized**. LLaMEA selects generated algorithms by **maximizing mean Area Over the Convergence Curve (AOCC)** across ten fixed evaluation seeds. The implementation targets LLaMEA **1.4.2** and runs on CPU.

## 1. Repository structure

```text
BO_LLAMEA/
├── configs/
│   └── experiment.yaml          # Problem, LLM, ES, budget and AOCC settings
├── prompts/
│   ├── bbob.txt                 # BBOB task prompt
│   ├── realworld.txt            # Real-world task prompt
│   └── operators/              # Weighted mutation/crossover instructions
│       ├── common.txt
│       ├── refine.txt
│       ├── surrogate.txt
│       ├── acquisition.txt
│       ├── trust_region.txt
│       ├── hybrid.txt
│       ├── init.txt
│       └── crossover.txt
├── runs/
│   ├── stage1.py                # Evolutionary search and checkpointing
│   └── stage2.py                # Cross-problem benchmark and reporting
├── src/
│   ├── problems.py              # Problem factories and metric calibration
│   ├── performance.py           # Budget guard, AOCC and parallel runs
│   ├── baselines.py             # CMA-ES, BoTorch and random search
│   └── utils.py                 # Configuration, seeds, I/O and plotting
├── results/                     # Generated at runtime; see Section 7
├── requirements.txt
└── README.md
```

`runs/` coordinates experiments, `src/` contains shared implementations, `prompts/` contains editable LLM instructions, and `configs/experiment.yaml` specifies experimental choices. LLaMEA is installed as a dependency; its evolutionary engine is not copied into the repository.

## 2. Experimental design

### Stage 1 — evolve one optimizer per target

A separate LLaMEA search runs for each problem:

```text
Target-specific task prompt + eligible mutation/crossover prompt
                              ↓
                     LLaMEA → LLM → Python class
                              ↓
              10 independent seeded evaluations
                (at most 60 objective calls each)
                              ↓
                    Mean AOCC and feedback
                              ↓
                     Evolutionary selection
                              ↓
                  Next candidate / best.py
```

Generated classes implement `__init__(self, budget, dim)` and `__call__(self, func) -> (best_y, best_x)`, where `func(x)` takes a finite length-`dim` NumPy vector and exposes bounds through `func.bounds.lb` and `func.bounds.ub`. The objective wrapper checks shape, bounds and the strict evaluation budget; every initialization point counts as a black-box call.

An individual is valid when all ten runs finish successfully. Its fitness is the mean AOCC; an invalid candidate receives zero fitness and its diagnostic traceback is recorded. LLaMEA keeps a population of high-fitness parents, generates offspring using the configured operators, and saves generated source, feedback, per-seed traces, prompts/responses and recoverable checkpoints.

In the **recorded experiments** (Section 6), the ES had **2 parents, 2 offspring, elitism enabled, roulette parent selection, novelty operators and rich feedback**. `--limit 20` means **20 total candidate slots per target**, not 20 generations.

### Stage 2 — transfer and baseline comparison

Stage 2 loads both saved winners and benchmarks four methods—`LLaMEA_bbob`, `LLaMEA_realworld`, `CMA-ES`, and `BoTorch`—on **both targets**, using the same ten seeds. A full 600-evaluation benchmark comprises 4 methods × 2 problems × 10 seeds = **80 seeded runs**. Each method gets the same maximum number of real black-box calls, including initialization.

The evaluator records every accepted `x` and `f(x)` call, then calculates best-so-far trajectories, AOCC, mean final performance, standard errors and failures. Stage 2 processes method/problem combinations sequentially while parallelizing independent seeds within each combination.

## 3. Installation

**Prerequisites:** Python 3.11 or 3.12, Git, SWIG, and access to a supported LLM provider. The code is CPU-compatible; higher worker counts require additional CPU and memory. Run commands from the directory containing `requirements.txt`.

Create and activate a virtual environment **or** use Conda:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

Alternatively, use Conda:

```bash
conda create -n bo-llamea python=3.12 -y
conda activate bo-llamea
conda install -c conda-forge swig
```

Install Git and SWIG through the operating system's package manager if needed. Verify both tools with `git --version` and `swig -version`, then install the project dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/BayesOptApp/Practical_Problems.git"
python -m pip install -r requirements.txt
python -m pip check
```

LLaMEA 1.4.2 is pinned in `requirements.txt`. A simple import check is:

```bash
python -c "from llamea import LLaMEA; import ioh, Practical_Problems, cma, botorch; print('Imports OK')"
```

### LLM provider

The configured provider is Gemini (`gemini-3.1-flash-lite`). Obtain an API key at [Google AI Studio](https://aistudio.google.com/apikey). **Do not place secrets in source, YAML, shell command arguments, screenshots or repository commits.**

For PowerShell, prepare the following command without executing it, copy the API key directly from AI Studio, then return to the shell and press Enter:

```powershell
$env:GEMINI_API_KEY = (Get-Clipboard -Raw).Trim()
```

For Bash:

```bash
read -rs -p 'Gemini API key: ' GEMINI_API_KEY; echo
export GEMINI_API_KEY
```

Check configuration without displaying the secret:

```bash
python -c "import os; print('Gemini key configured:', bool(os.getenv('GEMINI_API_KEY')))"
```

Alternatively, install [Ollama](https://ollama.com/), start its server and change `llm.provider` and `llm.model` in YAML to a locally available model. Model availability, provider rate limits and generated-code reproducibility depend on the chosen LLM.

## 4. Configuration and reproducibility

All settings are in `configs/experiment.yaml`. The **saved experiment configuration and Stage 1 metadata** contain the following values:

| Setting | Recorded value |
|---|---|
| Group / dimension | `9` / `6` |
| BBOB | Function `1`, instance `1` |
| Real-world | Practical problem `1124` |
| Model | `gemini-3.1-flash-lite` |
| ES | `(2+2)`, elitist, `roulette` selection |
| Operator / feedback mode | `novelty` / `rich` |
| Stage 1 candidate limit | Up to `100` per target; `20` used here |
| Stage 1 black-box budget | `60` evaluations per seed |
| Stage 1 parallel workers | `4` |
| Stage 2 configured budget | `600` evaluations per seed |
| Stage 2 default workers | `4` (`--workers` overrides) |
| AOCC log-precision bounds | `-8` to `2` |
| AOCC objective scaling | `auto`, with 32 calibration points and seed `2026` |

**Evaluation seeds.** The project derives ten seeds from group ID `g` as `10g, 10g+1, ..., 10g+9`. For **group 9, these are precisely `90, 91, 92, 93, 94, 95, 96, 97, 98, 99`**. Both saved Stage 1 `experiment.json` files and all winner traces confirm this range. Stage 2 uses the same group-derived seeds; `--workers 8` changes scheduling, not seed selection. Completed Stage 2 metadata independently records the actual worker count and seeds.

**Prompts and evolutionary search.** `prompts/bbob.txt` and `prompts/realworld.txt` define the two tasks. In `novelty` mode, LLaMEA samples one eligible operator from seven instruction files, using the weights in YAML; `common.txt` adds shared validity constraints. Crossover requires two parents. `baseline` mode uses LLaMEA's default operator. `basic` feedback includes candidate validity and fitness; `rich` adds convergence checkpoints and seed variability without changing the numeric fitness.

**AOCC.** For the best-so-far objective `b_t`, reference `r`, fixed scale `s`, and log window `[L,U]`, the project computes

```text
error_t = max((b_t - r) / s, 10^L)
AOCC    = mean_t clip((U - log10(error_t)) / (U - L), 0, 1)
```

For the recorded runs, `L=-8`, `U=2`, and `scale: auto`. The saved values are:

| Target | Reference `r` | Fixed scale `s` |
|---|---:|---:|
| BBOB f1, instance 1 | 79.48 | 58.9892050763 |
| Real-world problem 1124 | 0 | 7474.5273796173 |

The BBOB reference is its IOH-reported optimum. The real-world reference of zero is a lower bound justified by problem 1124's sum-of-squares objective; other real-world problems may require an explicit reference. The scale is estimated from independent fixed-seed calibration points, not from candidate outcomes. With `scale: null`, the error is unscaled (`b_t-r`); **both modes still map AOCC to `[0,1]`**. The calibrated metric is a project-specific variant and must be identified as such when comparing against published AOCC results.

## 5. Running the experiments

The commands below assume the configured targets and an activated environment.

### Stage 1

To run the **20-candidate searches** recorded in the Stage 1 results:

```bash
python -m runs.stage1 --target bbob --limit 20
python -m runs.stage1 --target realworld --limit 20
```

The archived experiment records show 20 evaluated candidates for each target. The CLI accepts a cumulative candidate limit up to the YAML maximum. Omit `--limit` to use `stage1.max_candidates`, or resume a committed run:

```bash
python -m runs.stage1 --target realworld --resume --limit 100
```

**Without `--resume`, Stage 1 overwrites the chosen target's existing `results/stage1/<target>/` directory.** Resume requires the same effective prompt, model, ES and AOCC settings. Only load trusted checkpoint pickle files.

Tagged experimental runs are opt-in:

```bash
python -m runs.stage1 --target bbob --tag example --rep 0 --limit 20
```

Tagged outputs go to `results/stage1_runs/example_r0/bbob/`. Untagged runs do not create `stage1_runs/`. `--rep` requires `--tag`. Stage 2 reads the default `results/stage1/` winners, not tagged winners automatically.

### Stage 2

After both `best.py` files have been saved, run the **full-budget benchmark** with:

```bash
python -m runs.stage2 --budget 600 --workers 8
```

The runner evaluates four methods on two targets over seeds **90–99**, allowing up to 600 objective calls per seeded run. Eight independent seeds execute concurrently when resources permit. Outputs are written to `results/stage2/budget_0600/`. Running the same budget again replaces that directory. A smaller development run can use `--budget 50`.

**Results included in this snapshot:** complete Stage 1 source and evaluation traces for both targets. Stage 2 outputs (`experiment.json`, `summary.csv`, individual results and convergence plots) are not included. No numerical Stage 2 performance claims are made; include those files after completing the benchmark.

## 6. Recorded Stage 1 results (17 November 2026)

The archived Stage 1 directories contain the complete 20-candidate searches for both targets, including source code, fitness figures, ten-seed traces and LLaMEA conversation/selection logs. All statistics below are calculated from recorded evaluations.

| Metric | BBOB | Real-world |
|---|---:|---:|
| Candidates recorded | 20 | 20 |
| Valid candidates | 18 | 20 |
| Winner | Candidate **15** | Candidate **14** |
| Winning mean AOCC | **0.473079** | **0.493305** |
| Winner mean final objective | **79.481841** | **2.871359** |
| Winner final-objective standard error | 0.000346 | 0.174540 |
| Winner successful seeded runs | 10/10 | 10/10 |
| Objective calls per winning seeded run | 60 | 60 |

For BBOB, the winning mean final objective is close to the IOH optimum of 79.48. On real-world problem 1124, the winning algorithm reduces the mean objective from approximately 139.3 after the first ten evaluations to approximately 2.87 after 60. These values are **within-target Stage 1 results**; they do not establish cross-problem generalization or superiority over either baseline.

The selected optimizers are saved in `results/stage1/bbob/best.py` and `results/stage1/realworld/best.py`. `best.json` records their candidate numbers, AOCC scores and metric parameters. The original generated code is retained unmodified for provenance.

## 7. Selected algorithms: component-level analysis

The following descriptions are based on the saved `best.py` files. Both generated implementations target bounded, continuous minimization and use scikit-learn Gaussian Process regression.

| Component | BBOB winner — candidate 15 | Real-world winner — candidate 14 |
|---|---|---|
| Initialization | Latin hypercube, up to 12 initial points at a 60-call budget | Scrambled Sobol, 16 initial points in dimension 6 |
| Surrogate | `GaussianProcessRegressor` with constant × Matérn-5/2 + white-noise kernel | GP with constant × Matérn-5/2 + white-noise kernel |
| Target scaling | `normalize_y=True` | Manually standardizes observations on every refit |
| Input scaling | Maps bounded inputs to the unit cube for GP/acquisition optimization | Uses original bounded coordinates |
| Acquisition | Expected Improvement (EI) for minimization | Expected Improvement (EI) for minimization |
| Acquisition optimization | Multiple L-BFGS-B starts, including a best-point start | Ten L-BFGS-B starts within a trust region |
| Adaptation | Reoptimizes GP hyperparameters every five iterations; checks near-duplicate proposals | Expands trust region after improvement and shrinks otherwise; resets if too small |
| Fallback | Uses a random proposal when an acquisition point is too close to previous observations | Attempts random sampling if GP/acquisition computation raises an exception |

**BBOB winner, simplified workflow:**

```text
1. Generate 12 Latin-hypercube points inside the objective bounds.
2. Evaluate them and maintain the best observed (x, y).
3. Until the evaluation budget is exhausted:
   a. Scale observed inputs to [0,1]^6 and fit a Matérn GP.
   b. Update kernel hyperparameters periodically.
   c. Maximize EI using several local optimizer starting points.
   d. If the proposed point is nearly duplicated, draw a random point.
   e. Evaluate exactly one objective point and update the incumbent.
4. Return the best observed objective and point.
```

**Real-world winner, simplified workflow:**

```text
1. Generate 16 scrambled-Sobol points within the bounds.
2. Evaluate them and track the best observed point.
3. Until the evaluation budget is exhausted:
   a. Standardize observed objective values and fit a Matérn GP.
   b. Construct a trust region around the incumbent.
   c. Optimize EI from 10 starting points inside the trust region.
   d. Evaluate the proposed point and update the incumbent.
   e. Expand the trust region after improvement; shrink it otherwise.
   f. Attempt a random fallback after an exception.
4. Return the best observed objective and point.
```

**Implementation caveat:** the real-world winner's exception fallback samples its stored input and its objective argument separately. If exercised, this can associate an observed objective with the wrong input vector inside the generated optimizer. The submitted generated source is retained as produced by LLaMEA; any correction should be separately documented and re-evaluated before benchmarking as a modified algorithm. The winners also rely on repeated GP fits, so their runtime at 600 evaluations may be substantially larger than at 60.

## 8. Baselines and Stage 2 reporting

| Method | Implemented settings |
|---|---|
| CMA-ES | `pycma` ask/tell; midpoint initialization; initial step size = 25% of mean search-domain width; bounded search |
| BoTorch | `SingleTaskGP`, standardized target, 12 scrambled Sobol initial points, `LogExpectedImprovement`, 5 acquisition restarts and 64 raw samples |
| LLaMEA_bbob | Generated BBOB winner, evaluated without changing its source |
| LLaMEA_realworld | Generated real-world winner, evaluated without changing its source |

BoTorch minimizes the original function by fitting to negated objective values. These baseline settings are the **implemented configuration**, not independent verification of any separately specified course baseline parameters.

Stage 2 saves `summary.csv`, `bbob_runs.csv`, `realworld_runs.csv`, per-seed statuses/results and figures. The BBOB convergence plot shows **best-so-far loss relative to the optimum**; the real-world plot shows **best-so-far objective**. Supplementary plots show normalized gaps under the fixed AOCC references/scales. Curves display mean ± standard error over successful seeds, and failures are reported separately. The final report should compare both winners across **both problems**, together with CMA-ES and BoTorch.

## 9. Result files and logging

```text
results/
├── calibration/                # Fixed reference/scale records when generated
├── stage1/
│   ├── bbob/
│   │   ├── best.py, best.json, fitness.png
│   │   ├── experiment.json, checkpoint.json, state.json
│   │   ├── code/candidate_NNN.{py,json}
│   │   ├── traces/candidate_NNN.json
│   │   └── exp-*/              # Native LLaMEA code and JSONL logs; local checkpoint may exist
│   └── realworld/             # Same structure
├── stage1_runs/               # Created only for tagged Stage 1 experiments
└── stage2/
    └── budget_0600/
        ├── experiment.json, summary.csv
        ├── bbob_results.json, bbob_runs.csv, bbob_statuses.csv
        ├── realworld_results.json, realworld_runs.csv, realworld_statuses.csv
        └── figures/
            ├── bbob_convergence.png, bbob_normalized.png
            └── realworld_convergence.png, realworld_normalized.png
```

Each Stage 1 candidate has a generated source file, summary record and an array of ten seeded traces. Each trace records actual objective inputs/outputs, AOCC, warnings, elapsed time and error status. `fitness.png` shows individual candidate scores and the non-decreasing best-so-far AOCC. The native LLaMEA logs preserve prompts, responses and evolutionary lineage. Local checkpoint pickles preserve resumable ES state; they are not included in the shareable archive. Stage 2 uses the same objective-call recorder and adds CSV summaries and convergence figures. Optional IOH Analyzer logging (`stage2.ioh_logging: true`) supplements the project's traces for BBOB; IOH logging itself does not determine the AOCC formula.

## 10. Reproducibility, safety and attribution

- The recorded Stage 1 searches used **group 9, seeds 90–99**, Python **3.12.14**, **Windows 11**, LLaMEA **1.4.2**, IOH **0.3.22**, `Practical_Problems` **0.1.0**, BoTorch **0.18.1** and CMA-ES (`cma`) **4.5.0**. Exact dependency versions are saved in each target's `experiment.json`; the submitted `requirements.txt` defines the supported installation.
- LLaMEA generation may differ between searches because hosted model outputs and remote service behavior are not guaranteed to be deterministic. Stage 1 tagged repetition indices affect outer-loop randomness while evaluation seeds remain fixed.
- Numerical output can vary with hardware, thread libraries or timeouts. The evaluation runner seeds Python, NumPy and PyTorch as applicable, and limits numerical-library threads within workers.
- Generated Python code executes in a subprocess with a time limit but is **not sandboxed** against arbitrary filesystem or network access. Use a restricted runtime when executing untrusted source.
- The code preserves optimizer outputs and reference metrics separately. The reported AOCC uses a calibrated scale; do not compare it numerically with an unscaled AOCC experiment unless the scores are recalculated under identical settings.
- Saved result metadata records operating system, Python and package versions. Hardware details absent from machine metadata must be recorded from the execution machines before submission.
- **Checkpoint privacy:** LLaMEA checkpoint pickles can serialize LLM client state, including API credentials. These binary files are excluded from the shareable project archive; retain them only in a trusted private environment. The copied Stage 1 histories remain useful for inspection, but resuming from the packaged results requires a separately retained trusted checkpoint.

**Hardware record for the final submission:**

| Item | Recorded value |
|---|---|
| Operating system | Windows 11 (Stage 1 metadata) |
| Python | 3.12.14 |
| CPU model / logical cores | Not recorded in snapshot metadata |
| RAM | Not recorded in snapshot metadata |
| GPU / accelerator | Not recorded in snapshot metadata |
| LLM provider/model | Gemini / `gemini-3.1-flash-lite` |
| Stage 2 machine and results | Requires Stage 2 experiment files |

**External software:** [LLaMEA](https://github.com/XAI-liacs/LLaMEA), [IOH/IOHexperimenter](https://github.com/IOHprofiler/IOHexperimenter), [COCO-BBOB](https://numbbo.github.io/coco/), [Practical_Problems](https://github.com/BayesOptApp/Practical_Problems), [CMA-ES/pycma](https://github.com/CMA-ES/pycma), [BoTorch](https://github.com/pytorch/botorch), [GPyTorch](https://github.com/cornellius-gp/gpytorch) and [scikit-learn](https://scikit-learn.org/). Source attribution and package licenses remain with their respective authors.

# Bayesian Optimization with LLaMEA

This repository contains the implementation structure for the Bayesian
Optimization practical assignment. The project uses LLaMEA to generate
Bayesian Optimization algorithms and evaluates them against established
baselines on a COCO-BBOB problem and a real-world problem.

## Repository structure

```text
BO_llamea_Assignment/
├── bo_assignment/
├── config/
├── artifacts/
├── results/
├── scripts/
├── tests/
├── requirements.txt
└── .env.example
```

### `bo_assignment/`

This is the main Python package:

- `problems.py` contains problem setup and problem-evaluation utilities.
- `algorithms.py` contains optimization algorithms and baseline implementations.
- `generation.py` configures and uses the installed LLaMEA package for Stage 1.
- `benchmarking.py` runs Stage 2 comparisons on both assignment problems.
- `analysis.py` contains AOCC utilities, loads results, aggregates runs, and
  creates tables and plots.

The package is an assignment-specific layer around the installed LLaMEA
implementation. It does not recreate LLaMEA's evolutionary loop, population
management, LLM abstractions, or internal logging. Copied blocks from the
supplied course notebooks are marked with comments identifying their source
notebook and cell. Locations for which the supplied files did not contain an
implementation are marked with `TODO` comments.

### `config/`

This directory is intended for experiment configuration:

- `assignment.yaml` combines experiment settings and prompts in one file.

### `artifacts/`

This directory stores outputs produced by LLaMEA and the benchmarking process:

- `generation/bbob/` stores the best algorithm generated for the BBOB target,
  together with its future metadata, prompts, and search logs.
- `generation/realworld/` stores the equivalent real-world target output.
- `benchmarking/` is reserved for benchmark execution artifacts.

The generated algorithms are experiment artifacts rather than hand-written
framework source code.

### `results/`

This directory stores experiment output:

- `raw/` contains per-run and per-seed data.
- `tables/` contains aggregated comparison tables.
- `plots/` contains convergence and AOCC plots.

### `scripts/`

These are the user-facing entry points:

- `generate_algorithms.py` runs the two LLaMEA generation experiments.
- `run_benchmarks.py` evaluates generated algorithms and baselines.
- `make_results.py` aggregates results and creates plots.

### `tests/`

`test_assignment.py` is reserved for tests covering problem setup, AOCC,
budget handling, generated algorithms, and reproducibility.

### Root files

- `requirements.txt` lists Python dependencies, including the installed
  LLaMEA package.
- `.env.example` documents the environment variables used for LLM API keys.

## Intended experiment flow

```text
Configure problems, seeds, budgets, and prompts
        ↓
Generate one algorithm for the BBOB problem with LLaMEA
        ↓
Generate one algorithm for the real-world problem with LLaMEA
        ↓
Benchmark both generated algorithms on both problems
        ↓
Compare them with CMA-ES and BoTorch
        ↓
Aggregate results and create plots
```

The two generated algorithms must be evaluated on both their original target
and the other problem to measure specialization and generalization.

## Relationship with LLaMEA

LLaMEA is an external dependency. The assignment code should use its existing
`LLaMEA`, LLM, `Solution`, `ExperimentLogger`, and namespace-preparation
implementations. This repository only supplies the problem-specific evaluator,
prompts, budgets, baselines, benchmark matrix, and analysis.

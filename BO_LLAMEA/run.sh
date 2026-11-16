#!/bin/bash

PYTHON_EXE="/tmp/${USER}_venv/bin/python"

for rep in 0 1 2; do
  # $PYTHON_EXE -m runs.stage1 --target bbob --limit 40 --tag A0 --rep $rep --parents 1 --offspring 1 --selection random
  # $PYTHON_EXE -m runs.stage1 --target bbob --limit 40 --tag A1 --rep $rep --parents 2 --offspring 2 --selection tournament --tournament-size 2
  # $PYTHON_EXE -m runs.stage1 --target bbob --limit 40 --tag B  --rep $rep --parents 2 --offspring 2 --selection tournament --tournament-size 2 --operators novelty
  # $PYTHON_EXE -m runs.stage1 --target bbob --limit 40 --tag C  --rep $rep --parents 2 --offspring 2 --selection tournament --tournament-size 2 --feedback rich
  $PYTHON_EXE -m runs.stage1 --target bbob --limit 100 --tag D  --rep $rep --parents 2 --offspring 2 --selection tournament --tournament-size 2 --operators novelty --feedback rich
done

$PYTHON_EXE runs/compare_stage1.py results/stage1_runs bbob
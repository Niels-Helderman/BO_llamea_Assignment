#!/bin/bash

PYTHON_EXE="/tmp/${USER}_venv/bin/python"

for target in bbob realworld; do
  $PYTHON_EXE -m runs.stage1 --target $target --limit 100 --tag D --rep 0 \
    --parents 2 --offspring 2 --selection tournament --tournament-size 2 \
    --operators novelty --feedback rich
done

for target in bbob realworld; do
  echo "=== $target ==="
  $PYTHON_EXE runs/compare_stage1.py results/stage1_runs $target
done
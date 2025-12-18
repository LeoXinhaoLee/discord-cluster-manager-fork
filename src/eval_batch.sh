#!/bin/bash

export MODAL_TOKEN_ID=
export MODAL_TOKEN_SECRET=

set -x

# modal deploy runners/modal_runner_archs.py

## Put kernel .py files under `submissions-dir`: 1.py, 2.py, 3.py, ...
## Adjust batch size for parallel evaluation
python run_trimul_modal_batch.py \
  --submissions-dir bioml/trimul/submission_samples/submission_h100_batch \
  --gpu H100 \
  --mode leaderboard \
  --batch-size 16 \
  --workers 16 \
  --output-dir results/H100/batch


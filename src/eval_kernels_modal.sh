#!/bin/bash

export MODAL_TOKEN_ID=
export MODAL_TOKEN_SECRET=

set -x

## First deploy Modal app
modal deploy runners/modal_runner_archs.py


sleep 60

## Then use Modal to eval leaderboard kernels one by one
for i in {1..9}; do

    for round in {1..3}; do
    out_dir="results/H100/kernel_${i}/round_${round}"
    mkdir -p ${out_dir}
    python run_trimul_modal.py --submission ./bioml/trimul/submission_h100/${i}.py \
                               --gpu "H100" \
                               --mode leaderboard > results/H100/kernel_${i}/test_${round}.out 2>&1
    
    done

done


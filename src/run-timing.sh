#!/bin/bash

export MODAL_TOKEN_ID=ak-bX6HABsAJ704DMyTPaNnGT
export MODAL_TOKEN_SECRET=as-X5ywHDIM8aSkx7WGfBhX10

set -x

modal deploy runners/modal_runner_archs.py


# for i in 1; do
    
#     out_dir="results/H100/5"
#     mkdir -p ${out_dir}
#     python run_trimul_modal.py --submission ./bioml/trimul/submission_h100/5.py --gpu "H100" --mode leaderboard > ${out_dir}/test_${i}.out 2>&1 &

# done
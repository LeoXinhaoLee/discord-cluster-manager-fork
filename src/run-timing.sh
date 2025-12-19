#!/bin/bash

export MODAL_TOKEN_ID=ak-EdEEItYIy3bcbiq4B8Z6k8
export MODAL_TOKEN_SECRET=as-MWhUCJtnn6C2Clxm3fu6oP

set -x

# modal deploy runners/modal_runn1er_archs.py



out_dir="results/H100/test10"
mkdir -p ${out_dir}
# python run_trimul_modal.py --jsonl ./samples_openai_gpt-oss-120b_gpt_oss_medium_reasoning_20251217_211304.jsonl --gpu "A100" --mode leaderboard #> ${out_dir}/test_${i}.out 2>&1
python run_trimul_modal.py --jsonl ./samples_openai_gpt-oss-120b_gpt_oss_medium_reasoning_20251219_103905.jsonl --gpu "H100" --mode leaderboard > ${out_dir}/test_${i}.out 2>&1

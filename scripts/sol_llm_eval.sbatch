#!/bin/bash
# LLM evaluation: prompt v1 and v2 on all 64 PMC dev cases, the 6 example cases and the 15-row regression set;
# rules / LLM / rules+LLM with both status policies. Submit from the repo root: sbatch scripts/sol_llm_eval.sbatch
#SBATCH -J hoofbeats-llm-eval
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57669240fall2026
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-02:00:00
#SBATCH -o hoofbeats-llm-eval-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load mamba/latest
source activate hoofbeats
export OMP_NUM_THREADS=8          # SapBERT runs on these CPU cores; the GPU is vLLM's
nvidia-smi --query-gpu=name,memory.total --format=csv
git log -1 --format='[sol] code at commit %h %s'
bash scripts/sol_run.sh llm-eval --prompt-versions v1 v2 --status-policy both --out results/llm_eval

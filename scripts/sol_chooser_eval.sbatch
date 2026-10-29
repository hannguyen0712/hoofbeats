#!/bin/bash
# Linker chooser v2 on PMC dev with the chosen configuration (rules + LLM, prompt v1, status policy llm). Extraction
# replies come from results/llm_cache; only the LLM-choice requests are new. Submit from the repo root:
#   sbatch scripts/sol_chooser_eval.sbatch
#SBATCH -J hoofbeats-chooser
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57669240fall2026
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-01:00:00
#SBATCH -o hoofbeats-chooser-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load mamba/latest
source activate hoofbeats
export OMP_NUM_THREADS=8
git log -1 --format='[sol] code at commit %h %s'
bash scripts/sol_run.sh --chooser-version v2 llm-eval --prompt-versions v1 --status-policy llm \
    --out results/llm_eval_chooser2

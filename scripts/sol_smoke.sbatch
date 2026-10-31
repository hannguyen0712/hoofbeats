#!/bin/bash
# Smoke run: LLM finding candidates on the first 5 PMC dev cases. Submit from the repo root: sbatch scripts/sol_smoke.sbatch
#SBATCH -J hoofbeats-smoke
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57669240fall2026
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-01:00:00
#SBATCH -o hoofbeats-smoke-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load mamba/latest
source activate hoofbeats
nvidia-smi --query-gpu=name,memory.total --format=csv
bash scripts/sol_run.sh llm-extract --limit 5 --out results/llm_smoke

#!/bin/bash
# THE ONE-TIME PMC TEST RUN with the frozen Stage 1 configuration (configs/stage1_v1.json, tag stage1-final-dev):
# rules, LLM alone and rules + LLM on PMC dev (cached replies) and test, with the dev-vs-test gap. pmc-test-run
# refuses if the frozen files changed since the tag or a frozen test run is already logged. Submit from the repo root:
#   sbatch scripts/sol_pmc_test.sbatch
#SBATCH -J hoofbeats-pmc-test
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57669240fall2026
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-01:30:00
#SBATCH -o hoofbeats-pmc-test-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load mamba/latest
source activate hoofbeats
export OMP_NUM_THREADS=8
git log -1 --format='[sol] code at commit %h %s'
git describe --tags --always
python -c "from hoofbeats.config import frozen_problems as f; p = f(); print('[sol] frozen check:', p or 'ok'); exit(1 if p else 0)"
bash scripts/sol_run.sh pmc-test-run --allow-test --out results/pmc_test

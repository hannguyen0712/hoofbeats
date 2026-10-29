#!/bin/bash
# Full Stage 1 + ranking on every example case: vLLM with Qwen (frozen configuration configs/stage1_v1.json), then
# diagnose on examples/ and examples/heldout/; per-case JSON in results/diagnose/. Submit from the repo root:
#   sbatch scripts/sol_diagnose.sbatch
#SBATCH -J hoofbeats-diagnose
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57669240fall2026
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-01:00:00
#SBATCH -o hoofbeats-diagnose-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load mamba/latest
source activate hoofbeats
export OMP_NUM_THREADS=8
git log -1 --format='[sol] code at commit %h %s'
cases=$(python -c "import json; print(' '.join(f'{d}/{f}' for d in ('examples', 'examples/heldout')
                   for f in json.load(open(d + '/intended_findings.json')) if not f.startswith('_')))")
# --common none: rare diseases only, as in the reported example ranks
bash scripts/sol_run.sh --common none diagnose $cases --out results

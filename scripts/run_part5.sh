#!/usr/bin/env bash
# Part 5: re-run every Part 5 number with the same seed and commands (about 5 minutes on a laptop, no GPU).
# Run from the repo root after `python -m hoofbeats download`. Use it again at the code freeze.
#   bash scripts/run_part5.sh                 # rare diseases only (--common none)
#   NOTE="final" bash scripts/run_part5.sh    # label the runs
set -euo pipefail
cd "$(dirname "$0")/.."
NOTE=${NOTE:-"Part 5"}
PY=${PY:-python}

$PY -m hoofbeats --common none --note "$NOTE main benchmark, 500" benchmark --sample 500 --out results/main
$PY -m hoofbeats --common none --note "$NOTE main benchmark, all cases" benchmark --sample 0 --out results/full
$PY -m hoofbeats --common none --note "$NOTE next-test loop, 300" sequential --sample 300 --out results/main
$PY eval_part5.py all                                   # results/part5/: data section and main tables
$PY -m hoofbeats report --results results --latex       # reports/report-<n>/: figures and LaTeX tables

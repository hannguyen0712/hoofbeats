#!/bin/bash
# Ranking fixes for image findings, tested on the same 99 hand-checked MedPix cases as sol_vision_models.sbatch.
# It reuses that run's descriptions and Stage 1 reads (results/vision_models/llm_cache), so most requests are answered
# from the cache; the servers only start for anything missing. For two describers (Qwen3.5-9B with 2 images per
# request, the best one, and Qwen2.5-VL-7B, the old default), five settings of the same ranker:
#   a_none      no fixes (--no-image-region-filter): the numbers of the vision-models run
#   b_region    body-part filter: image findings outside the imaged body part are not scored (now the default)
#   c_reliable  b + image findings count at 20% (--reliability image_context 0.2)
#   d_series    b + frequencies of features seen in one or two patients pulled down (--series-prior 4,0.3)
#   e_all       b + c + d
# Submit from the repo root, with the hoofbeats env active, after sol_vision_models.sbatch has run:
#   sbatch --export=ALL,PY=$CONDA_PREFIX/bin/python scripts/sol_ranking_fixes.sbatch
# The 20% and 4,0.3 were picked on 36 other MedPix cases (an earlier run), not on these.
#SBATCH -J hoofbeats-fixes
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57669240fall2026
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00:00
#SBATCH -o hoofbeats-fixes-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=${PY:-}
if [ -z "$PY" ] || ! "$PY" -c "import numpy, vllm, torch, transformers" 2>/dev/null; then
    echo "[fixes] give the hoofbeats env's python: sbatch --export=ALL,PY=\$CONDA_PREFIX/bin/python scripts/sol_ranking_fixes.sbatch"; exit 1
fi
BIN=$(dirname "$PY")
export PATH="$BIN:$PATH"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache} HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=8
MEDPIX=${MEDPIX:-data/medpix-src/MedPix-2-0}
LABELS=${LABELS:-overnight/medpix_rare_labels.csv}
READER=Qwen/Qwen2.5-7B-Instruct
QWEN25=Qwen/Qwen2.5-VL-7B-Instruct
QWEN35=Qwen/Qwen3.5-9B
NOTHINK='{"chat_template_kwargs": {"enable_thinking": false}}'
O=results/ranking_fixes
CACHE=${CACHE:-results/vision_models/llm_cache}
mkdir -p $O/logs
for f in "$MEDPIX/Case_topic.json" "$LABELS" "$CACHE"; do
    [ -e "$f" ] || { echo "[fixes] missing $f (set MEDPIX=, LABELS= or CACHE= when submitting)"; exit 1; }
done
git log -1 --format='[fixes] code at commit %h %s'
free_port() { "$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
serve() {   # tag, model, GPU share, extra vllm args -> prints the chat URL once the server answers
    local tag=$1 model=$2 share=$3 port log pid
    shift 3
    port=$(free_port); log=$O/logs/vllm-$tag.log
    "$BIN/vllm" serve "$model" --served-model-name "$model" --host 127.0.0.1 --port "$port" --max-model-len 8192 \
        --gpu-memory-utilization "$share" --max-num-seqs 8 --seed 0 --generation-config vllm "$@" > "$log" 2>&1 &
    pid=$!
    echo $pid > $O/logs/$tag.pid
    for _ in $(seq 1 240); do
        "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$port/v1/models', timeout=3)" 2>/dev/null \
            && { echo "http://127.0.0.1:$port/v1/chat/completions"; return 0; }
        kill -0 $pid 2>/dev/null || { echo "[fixes] $model failed to start; see $log" >&2; tail -30 "$log" >&2; return 1; }
        sleep 5
    done
    echo "[fixes] $model not ready after 20 minutes; see $log" >&2; return 1
}
stop() {
    local f=$O/logs/$1.pid
    [ -f "$f" ] || return 0
    kill "$(cat "$f")" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$(cat "$f")" 2>/dev/null || break; sleep 2; done
    kill -9 "$(cat "$f")" 2>/dev/null; rm -f "$f"; sleep 15
}
trap 'for f in $O/logs/*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; done' EXIT
step() {    # name, command...: log to $O/logs/<name>.txt, keep going on failure
    local name=$1; shift
    echo "[fixes] $(date +%H:%M) $name"
    "$@" > "$O/logs/$name.txt" 2>&1 && echo "[fixes]   ok" || echo "[fixes]   FAILED (see $O/logs/$name.txt)"
}
run() {     # name, describer, its URL, its extra JSON ('' = none), images per request, ranker options...
    local name=$1 model=$2 url=$3 ex=$4 per=$5 extra=()
    shift 5
    [ -n "$ex" ] && extra=(--context-extra "$ex")
    step "$name" "$PY" -m hoofbeats --common none --llm-cache $CACHE "$@" \
        --llm-endpoint "$RURL" --llm-model "$READER" \
        --context-model "$model" --context-endpoint "$url" --context-max-tokens 4096 "${extra[@]}" \
        --context-samples 5 --context-agree 2 \
        medpix-context --conditions text,context,consensus,reference --max-images 2 --images-per-call "$per" \
        --medpix $MEDPIX --labels $LABELS --labels-only --out "$O/$name"
}
settings() {  # tag, describer, URL, extra JSON, images per request: the five ranker settings
    run "$1_a_none" "$2" "$3" "$4" "$5" --no-image-region-filter
    run "$1_b_region" "$2" "$3" "$4" "$5"
    run "$1_c_reliable" "$2" "$3" "$4" "$5" --reliability '{"image_context": 0.2}'
    run "$1_d_series" "$2" "$3" "$4" "$5" --series-prior 4,0.3
    run "$1_e_all" "$2" "$3" "$4" "$5" --reliability '{"image_context": 0.2}' --series-prior 4,0.3
}

RURL=$(serve reader "$READER" 0.30) || exit 1
if "$PY" -c "import vllm.model_executor.models.qwen3_5" 2>/dev/null; then
    URL=$(serve qwen35 "$QWEN35" 0.40 --limit-mm-per-prompt '{"image": 2}') && {
        settings qwen35 "$QWEN35" "$URL" "$NOTHINK" 2
        stop qwen35
    }
fi
URL=$(serve qwen25 "$QWEN25" 0.40 --limit-mm-per-prompt '{"image": 2}') && {
    settings qwen25 "$QWEN25" "$URL" "" 1
    stop qwen25
}
stop reader

echo "[fixes] $(date +%H:%M) done. Rank tables:"
for f in $O/logs/qwen*.txt; do
    echo "=== $(basename "$f" .txt)"; grep -E "^\| (history|text|Stage)|outside the imaged" "$f"
done

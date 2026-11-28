#!/bin/bash
# Image findings on the 99 hand-checked MedPix cases, read by the FULL Stage 1 (rules + numbers + LLM reader + SapBERT
# linker, as in diagnose) instead of the rules alone. Same cases, same settings for every describing model:
#   2 images per case, 5 descriptions per image, findings kept when 2 of 5 agree (consensus), plus the "asked"
#   condition: yes/no questions about the top 5 diseases' imaging findings, answered by Lingshu-7B (another model
#   family than the Qwen describers, so it also checks them).
#   01  Qwen2.5-VL-7B (the current default; baseline under the full Stage 1)
#   02  Qwen3.5-9B, 1 image per request      03  Qwen3.5-9B, 2 images per request
#   04  Lingshu-7B (describes and asks)
# Three vLLM servers share the A100: the Stage 1 reader (Qwen2.5-7B-Instruct), the describer and Lingshu.
# Before submitting, download the models once on a login node (about 16 + 19 + 16 GB):
#   export HF_HOME=/scratch/$USER/hf_cache
#   python -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ('Qwen/Qwen2.5-7B-Instruct', \
#     'Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen3.5-9B', 'lingshu-medical-mllm/Lingshu-7B', \
#     'cambridgeltl/SapBERT-from-PubMedBERT-fulltext')]"
# Qwen3.5 needs a recent vLLM (0.17 or newer): check with  vllm --version  (runs 02-03 are skipped if it is older).
# Submit from the repo root, with the hoofbeats env active:
#   sbatch --export=ALL,PY=$CONDA_PREFIX/bin/python scripts/sol_vision_models.sbatch
# Pick --ask-threshold from the threshold table of one run's log and report it on another split, not on the same cases.
#SBATCH -J hoofbeats-vmodels
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57669240fall2026
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-12:00:00
#SBATCH -o hoofbeats-vmodels-%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=${PY:-}
if [ -z "$PY" ] || ! "$PY" -c "import numpy, vllm, torch, transformers" 2>/dev/null; then
    echo "[vmodels] give the hoofbeats env's python: sbatch --export=ALL,PY=\$CONDA_PREFIX/bin/python scripts/sol_vision_models.sbatch"; exit 1
fi
BIN=$(dirname "$PY")
export PATH="$BIN:$PATH"
export HF_HOME=${HF_HOME:-/scratch/$USER/hf_cache} HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=8
MEDPIX=${MEDPIX:-data/medpix-src/MedPix-2-0}
LABELS=${LABELS:-overnight/medpix_rare_labels.csv}
READER=Qwen/Qwen2.5-7B-Instruct
QWEN25=Qwen/Qwen2.5-VL-7B-Instruct
QWEN35=Qwen/Qwen3.5-9B
LINGSHU=lingshu-medical-mllm/Lingshu-7B
NOTHINK='{"chat_template_kwargs": {"enable_thinking": false}}'
O=results/vision_models
CACHE=${CACHE:-$O/llm_cache}   # CACHE=results/<earlier run>/llm_cache reuses descriptions already made
mkdir -p $O/logs
for f in "$MEDPIX/Case_topic.json" "$LABELS"; do
    [ -e "$f" ] || { echo "[vmodels] missing $f (set MEDPIX= or LABELS= when submitting)"; exit 1; }
done
git log -1 --format='[vmodels] code at commit %h %s'
"$BIN/vllm" --version 2>/dev/null | sed 's/^/[vmodels] vllm /'

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
        kill -0 $pid 2>/dev/null || { echo "[vmodels] $model failed to start; see $log" >&2; tail -30 "$log" >&2; return 1; }
        sleep 5
    done
    echo "[vmodels] $model not ready after 20 minutes; see $log" >&2; return 1
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
    echo "[vmodels] $(date +%H:%M) $name"
    "$@" > "$O/logs/$name.txt" 2>&1 && echo "[vmodels]   ok" || echo "[vmodels]   FAILED (see $O/logs/$name.txt)"
}
run() {     # name, describer, its URL, its extra JSON ('' = none), images per request, conditions
    local extra=()
    [ -n "$4" ] && extra=(--context-extra "$4")
    step "$1" "$PY" -m hoofbeats --common none --llm-cache $CACHE \
        --llm-endpoint "$RURL" --llm-model "$READER" \
        --context-model "$2" --context-endpoint "$3" --context-max-tokens 4096 "${extra[@]}" \
        --context-samples 5 --context-agree 2 \
        medpix-context --conditions "$6" --max-images 2 --images-per-call "$5" \
        --ask-model "$LINGSHU" --ask-endpoint "$LURL" \
        --medpix $MEDPIX --labels $LABELS --labels-only --out "$O/$1"
}
ALL=text,context,consensus,asked,reference

RURL=$(serve reader "$READER" 0.24) || exit 1
LURL=$(serve lingshu "$LINGSHU" 0.30 --limit-mm-per-prompt '{"image": 2}') || exit 1

URL=$(serve qwen25 "$QWEN25" 0.30 --limit-mm-per-prompt '{"image": 2}') && {
    run 01_qwen25vl7b "$QWEN25" "$URL" "" 1 "$ALL"
    stop qwen25
}

if "$PY" -c "import vllm.model_executor.models.qwen3_5" 2>/dev/null; then
    URL=$(serve qwen35 "$QWEN35" 0.34 --limit-mm-per-prompt '{"image": 2}') && {
        run 02_qwen35_9b_1img "$QWEN35" "$URL" "$NOTHINK" 1 "$ALL"
        run 03_qwen35_9b_2img "$QWEN35" "$URL" "$NOTHINK" 2 context,consensus,asked
        stop qwen35
    }
else
    echo "[vmodels] this vLLM has no Qwen3.5 support (needs 0.17 or newer): runs 02-03 skipped"
fi

run 04_lingshu7b "$LINGSHU" "$LURL" "" 1 "$ALL"
stop lingshu
stop reader

echo "[vmodels] $(date +%H:%M) done. Summary (ranks, image findings vs the radiologist, question thresholds):"
for n in 01_qwen25vl7b 02_qwen35_9b_1img 03_qwen35_9b_2img 04_lingshu7b; do
    [ -f $O/logs/$n.txt ] && { echo "=== $n"; grep -E "^\||precision|recall|belief >=|Questions" $O/logs/$n.txt; }
done

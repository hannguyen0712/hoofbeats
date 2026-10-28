#!/bin/bash
# Start a vLLM server on this GPU node (127.0.0.1 only), wait until it answers, run one hoofbeats command against it,
# and stop the server on exit. Run from the repo root inside an allocation, with the conda env active:
#   bash scripts/sol_run.sh llm-extract --limit 5 --out results/llm_smoke
# Settings (environment): MODEL, MAX_LEN, CACHE (reply cache folder), HF_HOME.
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
MAX_LEN=${MAX_LEN:-16384}
CACHE=${CACHE:-results/llm_cache}
export HF_HOME=${HF_HOME:-/scratch/vmallip2/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}          # the model is already downloaded; compute nodes need no network
export VLLM_USE_FLASHINFER_SAMPLER=0                  # no nvcc on the nodes

mkdir -p results/sol
PORT=$(python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
URL="http://127.0.0.1:$PORT"
LOG="results/sol/vllm-${SLURM_JOB_ID:-local}-$PORT.log"

# --generation-config vllm: ignore the model's own sampling defaults (e.g. a repetition penalty), so temperature 0
# means plain greedy decoding and runs are comparable across models
vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.85 \
    --seed 0 --generation-config vllm > "$LOG" 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true' EXIT

ready() { python -c "import urllib.request; urllib.request.urlopen('$URL/v1/models', timeout=3)" 2>/dev/null; }
for _ in $(seq 1 180); do                             # up to 15 minutes
    ready && break
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "[sol] vLLM exited during start-up; last lines of $LOG:"; tail -40 "$LOG"; exit 1
    fi
    sleep 5
done
ready || { echo "[sol] vLLM not ready after 15 minutes; last lines of $LOG:"; tail -40 "$LOG"; exit 1; }
echo "[sol] vLLM ready: $MODEL on $URL (log $LOG)"

python -m hoofbeats --llm-model "$MODEL" --llm-endpoint "$URL/v1/chat/completions" --llm-cache "$CACHE" "$@"

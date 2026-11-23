#!/bin/bash
#SBATCH -J part4-s6
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57669240fall2026
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-00:45:00
#SBATCH -o s6-%j.out

cd "${SLURM_SUBMIT_DIR:-$PWD}"
module load mamba/latest
source activate hoofbeats
PY="$CONDA_PREFIX/bin/python"
"$PY" -c "import numpy, vllm; print('numpy', numpy.__version__, '| vllm', vllm.__version__)" || exit 1

export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}" HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
"$CONDA_PREFIX/bin/vllm" serve Qwen/Qwen2.5-7B-Instruct --host 127.0.0.1 --port 8000 --max-model-len 16384 \
  --gpu-memory-utilization 0.85 --seed 0 --generation-config vllm > vllm-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 120); do
  curl -s 127.0.0.1:8000/v1/models > /dev/null && break
  sleep 5
done
curl -s 127.0.0.1:8000/v1/models > /dev/null || { echo "vLLM did not start"; tail -30 vllm-$SLURM_JOB_ID.log; exit 1; }
echo "model ready"

"$PY" llm_part4.py ${CASES:-part4_cases.txt} 2>&1 | grep -v "^ dropped"
kill $VLLM_PID

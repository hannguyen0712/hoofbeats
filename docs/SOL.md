# Running the LLM steps on Sol

The model runs in a vLLM server on the same GPU node, bound to 127.0.0.1; hoofbeats refuses any non-localhost
endpoint for extraction and linking. Replies are cached per model (`results/llm_cache/<model>.jsonl`), so every
number can be reproduced on a laptop with `--llm-offline` after copying the cache back.

## 1. Copy the repo (from the Mac, repo root)

The branch is not pushed, so the working copy is copied with rsync. Large data not needed for these steps
(MedPix, Phenopacket Store, the hand-check sheets) stays behind. `data/kb/.cache/` goes along: it holds the SapBERT
index of HPO (about 90 MB) and its calibration, so Sol uses the same vectors and linker thresholds as the Mac.

```bash
rsync -av --exclude '.venv/' --exclude 'results/' --exclude 'scratch/' --exclude '__pycache__/' \
  --exclude 'data/medpix-src/' --exclude 'data/phenopacket-store/' --exclude 'data/pmc/checks/' \
  --exclude 'docs/proposal.pdf' ./ vmallip2@sol.asu.edu:/scratch/vmallip2/hoofbeats/
```

## 2. One-time setup (on Sol)

Install the repo's requirements into the `hoofbeats` env without changing vLLM's own versions:

```bash
module load mamba/latest && source activate hoofbeats
cd /scratch/vmallip2/hoofbeats
pip freeze | grep -iE '^(torch|vllm|numpy|pydantic|transformers)==' > /tmp/keep.txt
pip install -r requirements.txt -c /tmp/keep.txt
python -m unittest tests 2>&1 | tail -3      # expect only the 4 PubMed errors
# SapBERT for the linker; compute nodes run with HF_HUB_OFFLINE=1, so it must be in HF_HOME beforehand
HF_HOME=/scratch/vmallip2/hf_cache python -c "from transformers import AutoModel, AutoTokenizer; \
m = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"
```

## 3. Runs

Batch (preferred), from the repo root:
- `sbatch scripts/sol_smoke.sbatch`: LLM candidates on the first 5 PMC dev cases;
- `sbatch scripts/sol_llm_eval.sbatch`: `llm-eval` with prompt v1 and v2 on the 64 PMC dev cases, the 6 example
  cases and the regression set (rules / LLM / rules+LLM, both status policies).

Follow a job with `tail -f hoofbeats-<name>-<jobid>.out`.

Interactive, inside `salloc -p public -q class -A class_cse57669240fall2026 -G a100:1 -c 8 --mem=64G -t 0-2`:

```bash
module load mamba/latest && source activate hoofbeats
cd /scratch/vmallip2/hoofbeats
bash scripts/sol_run.sh llm-extract --limit 5 --out results/llm_smoke
```

`scripts/sol_run.sh` starts vLLM on a free localhost port, waits up to 15 minutes for it, runs the hoofbeats
command given to it, and stops the server on exit. `MODEL=... bash scripts/sol_run.sh ...` switches the model;
the vLLM log goes to `results/sol/`.

## 4. Copy results back (from the Mac, repo root)

```bash
rsync -av vmallip2@sol.asu.edu:/scratch/vmallip2/hoofbeats/results/{llm_cache,llm_smoke,llm_eval,sol} results/
rsync -av 'vmallip2@sol.asu.edu:/scratch/vmallip2/hoofbeats/hoofbeats-*.out' results/sol/
# the run log: Sol's copy is the Mac's plus the new runs; check that, then take it
rsync -av vmallip2@sol.asu.edu:/scratch/vmallip2/hoofbeats/pmc_eval/runs.jsonl /tmp/runs_sol.jsonl
diff <(cat pmc_eval/runs.jsonl) <(head -n "$(wc -l < pmc_eval/runs.jsonl)" /tmp/runs_sol.jsonl) && cp /tmp/runs_sol.jsonl pmc_eval/runs.jsonl
```

Re-running a command on the Mac with `--llm-offline --llm-cache results/llm_cache` (and the same `--llm-model`)
then answers every request from the copied replies, e.g.
`python -m hoofbeats --llm-model Qwen/Qwen2.5-7B-Instruct --llm-offline --llm-cache results/llm_cache llm-eval --out
results/llm_eval_offline` (its runs.jsonl lines say `"offline": true`).

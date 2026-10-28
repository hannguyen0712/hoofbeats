# Running the LLM steps on Sol

The model runs in a vLLM server on the same GPU node, bound to 127.0.0.1; hoofbeats refuses any non-localhost
endpoint for extraction and linking. Replies are cached per model (`results/llm_cache/<model>.jsonl`), so every
number can be reproduced on a laptop with `--llm-offline` after copying the cache back.

## 1. Copy the repo (from the Mac, repo root)

The branch is not pushed, so the working copy is copied with rsync. Large data not needed for these steps
(MedPix, Phenopacket Store, the hand-check sheets) stays behind.

```bash
rsync -av --exclude '.venv/' --exclude 'results/' --exclude 'scratch/' --exclude '__pycache__/' \
  --exclude 'data/medpix-src/' --exclude 'data/phenopacket-store/' --exclude 'data/pmc/checks/' \
  --exclude 'docs/proposal.pdf' ./ vmallip2@login.sol.rc.asu.edu:/scratch/vmallip2/hoofbeats/
```

## 2. One-time setup (on Sol)

Install the repo's requirements into the `hoofbeats` env without changing vLLM's own versions:

```bash
module load mamba/latest && source activate hoofbeats
cd /scratch/vmallip2/hoofbeats
pip freeze | grep -iE '^(torch|vllm|numpy|pydantic|transformers)==' > /tmp/keep.txt
pip install -r requirements.txt -c /tmp/keep.txt
python -m unittest tests 2>&1 | tail -3      # expect only the 4 PubMed errors
```

## 3. Runs

Batch (preferred): `sbatch scripts/sol_smoke.sbatch`, then `tail -f hoofbeats-smoke-<jobid>.out`.

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
rsync -av vmallip2@login.sol.rc.asu.edu:/scratch/vmallip2/hoofbeats/results/{llm_cache,llm_smoke,sol} results/
rsync -av 'vmallip2@login.sol.rc.asu.edu:/scratch/vmallip2/hoofbeats/hoofbeats-smoke-*.out' results/sol/
```

Re-running a command on the Mac with `--llm-offline --llm-cache results/llm_cache` (and the same `--llm-model`)
then answers every request from the copied replies.

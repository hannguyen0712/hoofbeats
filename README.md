# hoofbeats

Diagnosis support for rare and common diseases. Given a patient's findings
(HPO terms, labs, notes, CT/MR images), the system ranks diseases in one list, explains each
rank with evidence it has checked, and suggests what to look for next.

## Architecture

```
inputs                    stage 1                 stage 2                  stage 3
------                    -------                 -------                  -------
phenopackets  ─┐
MIMIC labs     ├─>  findings list            ─>  score every disease  ─>  LLM re-ranks top 10
notes          │    (HPO term, status,           (likelihood ratios        cites evidence ids,
CT/MR images  ─┘     source, source text)         + prior)                 unsupported sentences dropped
                                                      │
                                                      ├─> evidence lines, re-checked
                                                      └─> next checks (information gain)
```

**Stage 1, findings.** Everything becomes one list. Each item is an HPO term, a status
(present / absent / family / uncertain), where it came from, and the text or value behind it.
- Phenopackets: already HPO.
- Notes: dictionary match on HPO names and exact synonyms. Negation, hedge and family rules set the status.
- Labs: 31 tests mapped to HPO, judged against the lab's own reference range.
- Images: an LLM writes a short findings report (no diagnosis allowed). The report goes through the same dictionary.
- LLM note extraction: every finding must quote the note word for word.

**Stage 2, scoring** (LIRICAL-style). Per disease, each finding adds a log likelihood ratio:
- present: `f(u,d) / bg(u)`, best match over the term and its ancestors, floor 0.05
- absent: `(1 - f) / (1 - bg)` if the disease has that feature, otherwise neutral
- plus a prevalence prior when common diseases are loaded
- `f` = how often the disease shows the feature (from `phenotype.hpoa`)
- `bg` = share of diseases that have the feature

Annotations backed only by the test case's own paper are removed first (leakage control).
A softmax with a fitted temperature turns scores into probabilities.

Rare diseases come from HPO annotations (8,465 OMIM; Orphanet optional). Common diseases are
lab profiles learned from MIMIC: share of admissions with each abnormal lab, smoothed as
(abnormal + 1) / (measured + 2).

**Stage 3, grounding.** Each score term becomes an evidence line, e.g.
`Has 'Focal motor seizure'; Disease X shows this in ~67% of cases. LR 190 [PMID:...]`.
Every line is re-derived from the record and the knowledge base. Verified PubMed sentences
can be added (E-utilities only, queried with disease names and HPO terms).

The LLM reorders the top 10 and explains them, citing line ids like `[C3]`. A sentence is
dropped if it cites nothing, cites another disease's evidence, or doesn't overlap enough
with what it cites. Faithfulness = sentences kept / sentences written.

**Next checks.** For each unobserved feature: how much would knowing it shrink the
uncertainty over the top 20 diseases? The top 5 by that gain get suggested.

## Files

```
hoofbeats/model.py     ontology, disease profiles, ranker, explanations, checks, next checks
hoofbeats/data.py      loaders (Phenopacket Store, MIMIC, MedPix, Orphanet), text extractor
hoofbeats/llm.py       LLM client, re-rank, note + image reading, reply cache, privacy guard
hoofbeats/context.py   Stage A: image context from a vision model; multimodal and medpix-context commands
hoofbeats/workup.py    test suggestions for every displayed disease (MAxO, LOINC2HPO, GTR); workup-eval
hoofbeats/followup.py  follow-up: re-analyse a case with new results; before/after and confirmation status
hoofbeats/pubmed.py    PubMed evidence
hoofbeats/report.py    figures, tables and notes from saved runs
hoofbeats/__main__.py  command line
docs/index.html        local results viewer
tests.py               42 offline tests
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest tests
python -m hoofbeats download                      # HPO + Phenopacket Store -> data/
python -m hoofbeats download-tests                # MAxO + LOINC2HPO + GTR -> data/kb (test suggestions)
git clone --depth 1 https://github.com/CHILab1/MedPix-2.0.git data/medpix-src
wget -r -N -c -np -nH --cut-dirs=1 -P data https://physionet.org/files/mimic-iv-demo/2.2/
curl -L -o data/kb/ORPHA_ICD10_mapping_en.xml \
  https://raw.githubusercontent.com/orphanet-rare-diseases-issues/RD-CODE/master/ORPHA_ICD10_mapping/ORPHA_ICD10_mapping_en.xml
```

LLM backends (global flags, put them before the command):

```bash
# MedGemma via Ollama (set its context length to 16k)
LLM=(--llm --llm-endpoint http://localhost:11434/v1/chat/completions --llm-model hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M)
# Gemini (free tier: public data only)
GEM=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
FLASH=(--llm --llm-endpoint $GEM --llm-model gemini-3.8-flash --llm-key-env GEMINI_API_KEY \
       --llm-max-tokens 8192 --llm-extra '{"reasoning_effort": "low"}')
```

## Running

```bash
python -m hoofbeats --common none diagnose data/phenopacket-store/0.1.27/KCNT1/PMID_23086397_Patient_4.json
python -m hoofbeats --common none benchmark  --sample 500 --out results/main
python -m hoofbeats --common none sequential --sample 300 --out results/main
python -m hoofbeats --common none "${FLASH[@]}" --llm-rpm 5 llm-bench --llm-cases 30 --out results/compare
python -m hoofbeats --common none "${LLM[@]}" medpix --medpix data/medpix-src/MedPix-2-0 \
  --only-labeled --limit 40 --max-images 2 --images-per-call 2 --out results/compare
python -m hoofbeats learn-common --mimic data/mimic-iv-demo/2.2 --min-adm 3
python -m hoofbeats mimic --mimic data/mimic-iv-demo/2.2 --orphadata-xml data/kb/ORPHA_ICD10_mapping_en.xml --out results/mimic
python -m hoofbeats report --results results --latex
```

- `llm-bench` runs the same cases three ways: ranker alone, ranker + LLM re-rank, LLM alone.
- `--note "..."` labels any run.
- `--dry-run` counts the LLM requests a command would send, without sending any.

Each run writes `<out>/runs/<id>.txt`, a readable log with settings, output, metrics and any error.
`runs/latest.txt` is always the newest. `report` makes a new numbered folder each time
(`reports/report-1`, `-2`, ...) with figures, CSV/LaTeX tables and a short findings list.

LLM replies are cached in `<out>/llm_cache/`, so a re-run, or a run that stopped on a quota,
only sends new requests. Re-ranking is skipped when the truth isn't in the top 10, since
reordering can't change that case's score.

Results viewer: `python -m http.server 8000 --bind 127.0.0.1`, then open `localhost:8000/docs/`.

### How to run Stage 1 (patient input to findings, then ranking)

Stage 1 turns a case file (`examples/*.json`: note text, labs, vitals, image findings) into HPO findings, each with
its quote, readers and grounding. Its configuration is frozen in `configs/stage1_v1.json` (rules + numbers + LLM
reader, Qwen2.5-7B-Instruct; tag `stage1-final-dev`).

**Locally, rules + numbers only** (no GPU, no model). `diagnose` says it is running without the LLM reader:

```bash
python -m hoofbeats --common none diagnose examples/marfan_note.json examples/heldout/dm1_note.json --out results
# per-case JSON in results/diagnose/<case_id>.json
```

**Full Stage 1 with the LLM reader, on ASU Sol.** The model runs in a vLLM server on the same GPU node, bound to
127.0.0.1; hoofbeats refuses any other endpoint for reading patient text. One-time setup and copying are in
[docs/SOL.md](docs/SOL.md). Then, from the repo root on Sol:

```bash
sbatch scripts/sol_diagnose.sbatch        # all example cases -> results/diagnose/
# or inside an allocation (salloc ... -G a100:1), any cases:
bash scripts/sol_run.sh --common none diagnose examples/marfan_note.json --out results
```

`scripts/sol_run.sh` starts vLLM, waits for it, runs the command with `--llm-endpoint` set, and stops the server.
Replies are cached per model in `results/llm_cache/`; after copying that folder back, the same command runs on a
laptop without a GPU: `python -m hoofbeats --common none --llm-offline --llm-cache results/llm_cache diagnose ...`
(needs `pip install -r requirements-llm.txt` for the SapBERT linker). Any local OpenAI-compatible server works the
same way with `--llm-endpoint http://127.0.0.1:<port>/v1/chat/completions`.

### Images: Stage A (image context)

When a case has images, a vision model first describes each image (Stage A). The description is added to the
case's `image_context`, and Stage 1 reads it with the note, so image findings go through the same extraction,
grounding and scoring as text. The vision model is told not to name a diagnosis, and a disease name in its text is
never turned into a finding.

- A case file lists its images under `"images"`: `{"path": ..., "modality": ..., "plane": ..., "location": ...}`;
  paths are relative to the case file. Up to `--context-max-images` (default 4) are described, one per request.
- Which vision model: `--context-model` if given; otherwise the main `--llm-model` if it reads images (e.g.
  Qwen2.5-VL, MedGemma, Gemini); otherwise MedGemma 1.5 4B on a local Ollama
  (`ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M`). A `local_only` case only goes to a local model.
- `--context-verify` (off by default): the vision model is shown the image again with each finding it described and
  answers yes / no / unsure; only "yes" findings are kept.
- `--context-confidence {all,probable,definite}` (default `all`): keep only findings the vision model rated at least
  that sure.
- Any kind of medical image: radiology (X-ray, CT, MRI, ultrasound, echocardiography, nuclear medicine), ECG and
  EEG tracings, clinical photographs (face, skin, limbs), eye images (fundus, slit lamp, OCT), microscopy and
  endoscopy. The image's `modality` (then its `location`) decides the kind, and the prompt asks for what a specialist
  in that kind of image would report (e.g. rate, intervals and axis for an ECG; dysmorphology terms for a face).

Any image with a short history, on ASU Sol (one vLLM server; Qwen2.5-VL reads both the text and the image). The
example uses a MedPix image, so clone MedPix first (Setup):

```bash
MODEL=Qwen/Qwen2.5-VL-7B-Instruct bash scripts/sol_run.sh --common none multimodal --case-id mpx2554 \
  --text "2 day old male with abdominal distention." --age 0 --sex male \
  --image data/medpix-src/MedPix-2-0/images/MPX2554_synpic46958.png \
  --modality "MR - T2 weighted" --plane coronal --location abdomen --out results
# case file in results/multimodal/mpx2554.json, result in results/diagnose/mpx2554.json
```

Without a GPU, the same command runs against MedGemma on a local Ollama (slow on a CPU): leave out
`MODEL=... bash scripts/sol_run.sh` and start it with `python -m hoofbeats`.

Image context on MedPix, side by side (same cases; Stage 1 with rules + numbers; the radiologist's findings are the
upper bound for what a description can add). `--labels` takes a CSV `U_id,truth` of hand-checked diagnoses; without
it, MedPix diagnoses are matched to the knowledge base by name:

```bash
MODEL=Qwen/Qwen2.5-VL-7B-Instruct bash scripts/sol_run.sh --common none medpix-context \
  --medpix data/medpix-src/MedPix-2-0 --conditions text,context,verified,definite,reference \
  --limit 40 --max-images 2 --out results/medpix_context
```

### Test suggestions

For every disease `diagnose` displays, the tests that would confirm or rule it out, with the probability of that
disease after a positive and after a negative result. Run `download-tests` once first (Setup); without it,
`diagnose` runs without tests and says so. Details: [docs/TEST_SUGGESTIONS.md](docs/TEST_SUGGESTIONS.md).

```bash
python -m hoofbeats --common none diagnose examples/marfan_note.json --out results   # tests under "tests" in the JSON
python -m hoofbeats --common none workup examples/marfan_note.json                    # the same tests, readable
python -m hoofbeats --common none workup-eval --sample 300 --out results/workup_eval  # evaluation (below)
```

- Pre-test probability: the ranker's score, calibrated on held-out Phenopacket cases.
- Tests come from MAxO (diagnostic tests and examinations for HPO findings) and LOINC2HPO (lab tests); each has a
  burden from 1 (history, examination) to 5 (invasive). Each test's likelihood ratios come from how often each
  disease shows the findings the test can reveal. Tests are ordered by how much they change the probability per
  unit of burden.
- Per disease: up to 3 tests to confirm it (LR+ >= 5), 3 to rule it out (LR- <= 0.2), its gene test, and images
  already taken that are worth reviewing. Per case: the tests that best tell the top 3 diseases apart, and findings
  the note may already answer.
- Gene tests use the method the disease needs, read from the labs' registered tests in GTR: repeat-expansion testing
  (e.g. myotonic dystrophy, Huntington disease, fragile X), methylation analysis first (e.g. Prader-Willi, Angelman),
  otherwise sequencing and deletion/duplication analysis.
- Exome or genome sequencing is always listed as the broad genetic test, and FIRST when the diseases shown add up to
  less than 30%, since the diagnosis is then often not among them. Genome rather than exome when a shown disease
  needs repeat-expansion testing; trio for developmental delay or intellectual disability.

`workup-eval` simulates the use of the tests on Phenopacket cases: half of each patient's findings are hidden, each
method (the workup, tests that tell the top 3 apart, and baselines: feature information gain, most frequent feature,
random) picks tests one at a time, and a test reveals the hidden findings it can show. It reports Top-1/Top-10, the
burden spent, and how often the causal gene is among the gene tests of the top diseases.

### Follow-up

When a suggested test has been done, or the patient comes back with anything new, `followup` analyses the case
again with the existing pipeline: the original case (before), the original plus the follow-up (after, from Stage A),
and what changed.

```bash
python -m hoofbeats --common none followup examples/marfan_note.json examples/followup/marfan_followup.json --out results
# results/followup/<case_id>/: before/, after/, merged_case.json, followup.json (comparison and status)
```

- The follow-up file takes the same fields as a case file (`note_text`, `images`, `labs`, `vitals`, `genes`,
  `hpo_terms`, `details`) plus an optional `test_results` list, e.g.
  `{"test": "angiography", "result": "negative"}` or
  `{"test": "FBN1 gene testing", "result": "positive", "genes": ["FBN1"], "classification": "pathogenic"}`.
  `test` is the test's name or id as the suggestions show it.
- Every test result is also written into the note, so Stage 1 reads it. A negative test marks absent the findings it
  was suggested for; a positive test marks its finding present when it looks for only one (otherwise `details` says
  which). A pathogenic or likely pathogenic gene result adds the gene to the case; a negative gene test makes the
  diseases of that gene 10x less likely (lower, never ruled out).
- Status per disease: **confirmed** (a pathogenic / likely pathogenic variant in its gene; if several displayed
  diseases share the gene, the best-ranked one; the others are "same gene, fits less"), **strongly supported**
  (probability >= 90% and one of its confirm tests came back positive), otherwise **not yet confirmed**, with its
  next tests. The ranker's probability alone never makes a disease confirmed.

## Current numbers

500 Phenopacket cases, 290 diseases (357 evaluable), HPO 2026-09-01, seed 13.

| condition | Top-1 | Top-10 |
|---|---|---|
| present findings only | 0.180 | 0.342 |
| + absent findings | 0.146 | 0.282 |
| + absent, no leakage control | 0.478 | 0.648 |
| + absent + causal gene | 0.330 | 0.474 |

- Leakage control matters most: without it Top-1 is about 3x higher.
- Absent findings hurt. Weight 0 to 0.25 is best (evaluable Top-1 0.252 vs 0.204 at full weight).
- Calibration: temperature 11.6, ECE 0.127.
- Next checks (pilot, 300 cases): Top-1 0.160 -> 0.180 with information gain, 0.163 with random questions.
- MedPix (163 mapped cases): Top-10 0.074 from text, 0.129 with the radiologist's findings added.
- The LLM comparison is still running.

## Rules we keep

- The model never sees the answer. The case's own paper is out of the knowledge base. Diagnoses are stripped from notes. MedPix radiologist findings and captions are only used for scoring.
- Credentialed MIMIC never leaves the machine: local models only, enforced in code. Keep it under `/secure`; don't commit or paste it anywhere.
- Gemini's free tier may use prompts for training, so public datasets only.
- Ties count against the true disease. Evaluable subsets are reported. LLM output is checked, never trusted as is.

## Known gaps

- Absent findings and how to weight them.
- The dictionary misses a lot of radiology wording.
- Common-disease profiles only see labs.
- Next-check gains are capped because few suggested terms can be answered from the record.
- LLMs may have seen these published cases in training, so their gains are upper bounds.
- The faithfulness check is word overlap. Hand labels are needed to validate it.

## Data and credits

HPO and annotations (HPO license), Phenopacket Store (BSD-3), Orphanet ICD-10 mapping (CC BY 4.0),
MIMIC-IV demo (ODbL), MIMIC-IV (PhysioNet credentialed), MedPix 2.0 (no license stated; ask before
redistributing), MedGemma (HAI-DEF terms). Scoring follows LIRICAL (Robinson et al., AJHG 2020).

Research tool. Not for clinical decisions.
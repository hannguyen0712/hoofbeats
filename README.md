# hoofbeats

hoofbeats supports the diagnosis of rare and common diseases. It takes a patient's notes, labs, images and HPO
terms, and ranks every disease in our knowledge base. Each disease on the screen comes with an explanation whose
sentences are checked against the evidence, the tests that would confirm or rule it out, and a way to re-score the
case when results come back.

## How it works

```
case file          Stage A            Stage 1               Stage 2                Stage 3
---------          -------            -------               -------                -------
notes        ─┐
labs, vitals  │    images are   ─>   findings:        ─>   every disease     ─>   explanations that cite
images        ├─>  described         HPO term, status,      scored with             evidence, checked
genes         │    as text           quote, grounding       likelihood ratios       sentence by sentence
HPO terms    ─┘                                             + calibrated             │
                                                            probabilities            ├─> tests to confirm or rule out
                                                                                     └─> follow-up: re-score with results
```

**Stage A: image context.** A local vision model describes each image in words. It is told not to name a
diagnosis, and any disease name it writes is never turned into a finding. The description is then read by the same
Stage 1 readers as the note, so image findings are grounded and scored like any other finding.

**Stage 1: findings.** Everything becomes one list. Each finding has an HPO term, a status (present, absent, family
history or uncertain), where it came from, and the words or value behind it.
- Rules read the note: dictionary matches on HPO names and synonyms, with negation, hedging and family-history
  rules.
- A local LLM reads the note too, mostly to catch paraphrases. Every finding it reports must quote the note word
  for word, and SapBERT links the quote to an HPO term.
- A numbers reader judges labs and vitals against their reference ranges.
- HPO terms given as input are used as they are.

**Stage 2: scoring.** The ranker follows LIRICAL. For each disease, every finding adds a log likelihood ratio:
- present: `f(u,d) / bg(u)`, the best match over the term and its ancestors, with a floor of 0.05
- absent: `(1 - f) / (1 - bg)` at half weight, and only for features the disease shows in at least 80% of cases
- `f` is how often the disease shows the feature (from `phenotype.hpoa`), and `bg` is how often diseases show it
  on average
- a reported causal gene, and a prevalence prior when common diseases are loaded, add their own terms

Before a Phenopacket Store case is scored, we remove every annotation supported only by that case's own paper.
This is our leakage control. Without it, the ranker recognises the paper instead of diagnosing the patient.

Two options change the weights. `--reliability` says how often findings from a source are real: a finding with
reliability `p` counts as `p*LR + (1-p)`, so unreliable findings move toward no information. `--weights` uses
evidence weights learned by `train` with grouped cross-validation.

Probabilities come from a softmax over the top 20 diseases, at a temperature fitted on held-out diseases, scaled by
how often the true disease is in the top 20 at all.

Rare diseases come from the HPO annotations (8,465 OMIM diseases; Orphanet is optional). Common diseases are lab
profiles learned from MIMIC-IV: for each diagnosis, the share of admissions with each abnormal lab, smoothed as
`(abnormal + 1) / (measured + 2)`.

**Stage 3: grounding and review.** Each disease gets an evidence pack: the patient's findings with their quotes,
and knowledge-base facts, each re-checked. The LLM writes sentences that must end with the IDs they use, like
`[P2, K2]`. Six checks then remove a sentence if it cites nothing, cites the wrong disease's evidence, misstates a
finding's status, does not name what it cites, gets a number wrong, or claims more than the evidence shows. When a
finding matches through a broader HPO term, the sentence must keep that broader term, since the frequency belongs to
it. The review screen (`review_app.py`) shows the result, with a *Why?* button on every sentence. Details are in
[PART4.md](PART4.md).

**Tests and follow-ups.** For every disease on screen, `workup` suggests up to three tests to confirm it and three
to rule it out, its gene test, and exome or genome sequencing. `followup` re-scores the case when results come in.

## Files

```
hoofbeats/model.py      ontology, disease profiles, ranker, explanations, checks
hoofbeats/data.py       loaders (Phenopacket Store, MIMIC, MedPix, Orphanet) and the rules reader
hoofbeats/llm.py        LLM client, re-ranking, reply cache, privacy guard
hoofbeats/llm_reader.py, linker.py, numbers.py, fusion.py, intake.py
                        Stage 1: LLM reader, SapBERT linker, labs and vitals, merging, case files
hoofbeats/context.py    Stage A: image context; multimodal and medpix-context
hoofbeats/train.py      learned evidence weights
hoofbeats/workup.py     test suggestions (MAxO, LOINC2HPO, GTR) and calibrated probabilities
hoofbeats/followup.py   follow-up: before and after, confirmation status
hoofbeats/grounding.py  Stage 3: evidence packs and sentence checks
hoofbeats/pubmed.py     PubMed evidence
hoofbeats/report.py     figures, tables and notes from saved runs
hoofbeats/__main__.py   command line
review_app.py           review screen (Streamlit)
grounding.py, *_part4.py   Stage 3 scripts: demo, model explanations, adversarial set, re-checks (PART4.md)
eval_part5.py           report tables with 95% intervals from saved runs (PART5.md)
scripts/run_part5.sh    re-runs every Part 5 number
docs/index.html         local results viewer
tests.py                147 offline tests
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

The SapBERT linker needs `pip install -r requirements-llm.txt`. A CPU is enough.

LLM backends are global flags, so they go before the command:

```bash
# Qwen3.5-9B on a local Ollama; thinking off, or it never finishes an answer
QWEN=(--llm --llm-endpoint http://localhost:11434/v1/chat/completions --llm-model qwen3.5:9b \
      --llm-extra '{"reasoning_effort": "none"}')
# MedGemma on a local Ollama (set its context length to 16k)
MEDGEMMA=(--llm --llm-endpoint http://localhost:11434/v1/chat/completions \
          --llm-model hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M)
# Gemini, free tier: public data only
GEM=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
FLASH=(--llm --llm-endpoint $GEM --llm-model gemini-3.8-flash --llm-key-env GEMINI_API_KEY \
       --llm-max-tokens 8192 --llm-extra '{"reasoning_effort": "low"}')
```

## Running

### One case

```bash
python -m hoofbeats --common none diagnose examples/marfan_note.json --out results
python -m hoofbeats --common none workup examples/marfan_note.json
python -m hoofbeats --common none followup examples/marfan_note.json examples/followup/marfan_followup.json --out results
streamlit run review_app.py
```

Without `--llm-endpoint`, `diagnose` runs Stage 1 with rules and numbers only, and says so.

### Rare-disease ranking (Phenopacket Store)

```bash
python -m hoofbeats --common none benchmark  --sample 0 --out results/main     # all 3,006 cases
python -m hoofbeats --common none sequential --sample 0 --out results/main
python -m hoofbeats --common none train --sample 0 --weights-out results/train/weights.json
python -m hoofbeats --common none --weights results/train/weights.json benchmark --sample 0 --out results/weighted
python -m hoofbeats --common none workup-eval --sample 300 --out results/workup_eval
```

- `--sample 0` means every case, with at most five per disease.
- `benchmark` also runs the ablations: absent findings, leakage control, gene, onset and sex.
- `train --kb-size` adds features for how many annotations and papers a disease has. They are off by default
  because they learn the footprint of leakage control rather than anything about the patient.

### Language models

```bash
python -m hoofbeats --common none "${QWEN[@]}" llm-bench --llm-cases 100 --out results/compare
python -m hoofbeats --common none "${QWEN[@]}" llm-bench --llm-cases 100 --rerank-shuffle --out results/compare
```

`llm-bench` runs the same cases three ways: the ranker alone, the ranker with the LLM re-ranking its top 10, and
the LLM alone, on the first 100 evaluable cases of the default 300-case sample. Each comparison gets a paired confidence interval and a count of the cases that moved up or down.
`--rerank-shuffle` shows the LLM the shortlist in a random order, to test whether it just keeps the order it was
given. Run it on the same cases as the unshuffled run and compare the two.

### Images (MedPix)

```bash
python -m hoofbeats --common none "${QWEN[@]}" medpix --medpix data/medpix-src/MedPix-2-0 \
  --orphadata-xml data/kb/ORPHA_ICD10_mapping_en.xml --only-labeled --max-images 2 --images-per-call 2 \
  --out results/medpix_all
python -m hoofbeats --common none "${QWEN[@]}" medpix-context --medpix data/medpix-src/MedPix-2-0 \
  --split all --limit 40 --max-images 2 --conditions text,context,verified,reference \
  --reliability-out results/medpix_context/rel.json --out results/medpix_context
python -m hoofbeats --common none "${QWEN[@]}" medpix-context --medpix data/medpix-src/MedPix-2-0 \
  --split all --limit 40 --max-images 2 --conditions context --context-no-history --out results/medpix_context
```

- `medpix` runs all 192 cases whose diagnosis is in the knowledge base. It compares the history and exam text,
  the text with the LLM's image findings, the image findings alone, and the text with the radiologist's findings.
- `medpix-context` runs Stage A on a smaller set. `text` is the history and exam alone, `context` adds the image description, `verified` keeps only findings the
  vision model confirms on a second look, and `reference` adds the radiologist's findings as an upper bound.
- `--context-no-history` shows the vision model only the images, age and sex. If image context still helps, the
  gain comes from the images and not from the history.
- `--reliability-out` writes how often the described findings match the radiologist's. Measure it on one set of
  cases and pass it with `--reliability` on others (`--exclude-cases` keeps the two sets apart).

### Rare and common diseases (MIMIC-IV demo)

```bash
python -m hoofbeats learn-common --mimic data/mimic-iv-demo/2.2 --min-adm 3 \
  --orphadata-xml data/kb/ORPHA_ICD10_mapping_en.xml
python -m hoofbeats mimic --mimic data/mimic-iv-demo/2.2 \
  --orphadata-xml data/kb/ORPHA_ICD10_mapping_en.xml --out results/mimic
```

`learn-common` holds out patients, not admissions. `mimic` scores the held-out admissions three ways (prior only,
labs only, labs and prior), against the primary diagnosis and against any diagnosis.

### Stage 1 on ASU Sol

The LLM reader runs in a vLLM server on the same GPU node, bound to 127.0.0.1. hoofbeats refuses any other
endpoint for patient text. The one-time setup is in [docs/SOL.md](docs/SOL.md). From the repo root on Sol:

```bash
sbatch scripts/sol_diagnose.sbatch                                       # all example cases
bash scripts/sol_run.sh --common none diagnose examples/marfan_note.json --out results   # inside an allocation
```

`scripts/sol_run.sh` starts vLLM, runs the command against it and stops it. Replies are cached in
`results/llm_cache/`. After you copy that folder back, the same command runs on a laptop with
`--llm-offline --llm-cache results/llm_cache`.

### Reproducing the report numbers

```bash
bash scripts/run_part5.sh               # benchmark (500 and all cases), sequential (300), eval_part5.py, report
NOTE="final" bash scripts/run_part5.sh  # the same, with the runs labelled
python eval_part5.py tables             # only the tables, from runs already saved
```

This takes about five minutes on a laptop without a GPU. `eval_part5.py` adds what the commands do not print:
Top-5, a 95% interval on every metric, and the dataset counts. It writes `results/part5/`. [PART5.md](PART5.md)
names the saved file behind every number.

### Output

Every run writes `<out>/runs/<id>.txt`, a readable log with the settings, output, metrics and any error, and
`runs/latest.txt` is always the newest. `report --results results --latex` makes a new numbered folder of figures,
CSV and LaTeX tables each time. LLM replies are cached in `<out>/llm_cache/`, so a run that stopped on a quota only
sends the requests it has not sent yet. `--dry-run` counts the requests a command would send, and `--note "..."`
labels a run.

To browse results, run `python -m http.server 8000 --bind 127.0.0.1` and open `localhost:8000/docs/`.

## Current results

Phenopacket Store 0.1.27, HPO 2026-09-01, seed 13. Intervals are 95% and resample diseases, not cases.
"Evaluable" means the true disease still has annotations after leakage control (1,969 of 3,006 cases).

| Ranking, 3,006 cases | Top-1 | Top-10 |
|---|---|---|
| all cases | 18.8 (16.4–21.2) | 32.9 |
| evaluable cases | 28.7 (25.4–32.0) | 50.3 |
| no leakage control | 70.3 | 85.1 |

- **Leakage control matters most.** Without it, Top-1 is 3.7 times higher.
- **A causal gene** adds 22.8 points of Top-1. Counting every absent finding at full weight costs 3.8 points. Age
  of onset and sex change nothing.
- **Learned weights** raise evaluable Top-1 to 30.6% on held-out diseases and papers (+1.9, +0.2 to +3.5).
- **Probabilities:** the expected calibration error is 0.064, against 0.192 for a softmax over all diseases.
- **Sequential questions** (1,914 cases): asking about the finding with the most information gain adds 1.2 points
  of Top-1 after five questions (+0.4 to +2.0). Only 13–15% of the questions can be answered from a published
  record.
- **Test suggestions** (300 cases): the first suggested test reveals a hidden finding in 82% of cases, against 12%
  for a random test.
- **Qwen3.5-9B** (100 cases): re-ranking adds 1.0 point of Top-1 (0.0 to +3.3), moving one case up and none
  down. With the shortlist shuffled, Top-1 does not change (0.0, −2.7 to +3.2; 4 cases up, 5 down). Alone, the
  model is 16.0 points below the ranker.
- **Images, 40 MedPix cases** (Stage A): image context raises Top-1 from 2.5% to 10.0%, and to 10.0% without the
  history too. But only 4 of the 70 described findings match the radiologist's.
- **Images, all 192 MedPix cases**: the LLM's image findings add 1.6 points of Top-1 (0.0 to +4.2) and 0.5 points
  of Top-10 (−1.8 to +2.6) to the text. The radiologist's findings add 6.2 points of Top-10 (+2.1 to +10.7). The
  LLM's findings match the radiologist's 19.4% of the time.
- **MIMIC-IV demo** (59 primary diagnoses): labs and prior put 13.6% in the top three, the prior alone none. No
  rare diagnosis reaches the top ten when ranked with common ones.
- **Stage 1** (172 PMC test cases): all readers together find 74.7% of the gold terms stated in the text, at about
  76% estimated precision.
- **Grounding** (314 model-written sentences on 30 held-out cases): wrong sentences among those shown fell from
  6.4% to 0.3%. On an adversarial set of 1,168 sentences, the new checks remove every wrong sentence and keep 98.6%
  of the correct ones.

## Rules we keep

- The model never sees the answer. The case's own paper is removed from the knowledge base, diagnoses are removed
  from notes, and MedPix captions and radiologist findings are only used for scoring.
- Patient text and images stay on our own machines. The readers refuse remote endpoints, and this is enforced in
  code. Credentialed MIMIC data stays under `/secure` and is never committed or pasted anywhere.
- Gemini's free tier may use prompts for training, so we only send it public data.
- Ties count against the true disease, and we report evaluable cases next to all cases.
- LLM output is checked before anyone sees it.

## Known gaps

- The true disease is outside the top 50 for about a third of evaluable cases. Tests and re-ranking cannot fix
  that, which is why exome or genome sequencing is always suggested.
- Common-disease profiles only see labs, and the demo has 3 to 12 admissions per disease.
- The local vision model captures little of what a radiologist reports.
- The LLM and image runs are small, so their intervals are wide. Published cases may be in the models' training
  data, so their gains are upper bounds.
- The grounding labels were made with AI help and checked by one of us. They are not independent yet, and a second
  annotator is next.
- External baselines (LIRICAL, RareBench, DeepRare) are not set up yet.
- Phenopacket Store cases arrive already coded in HPO, so the ranking results leave out reading errors. The
  end-to-end evaluation on PMC cases is next.

## Data and credits

HPO and annotations (HPO license), Phenopacket Store (BSD-3), MAxO, LOINC2HPO, NCBI GTR, Orphanet ICD-10 mapping
(CC BY 4.0), MIMIC-IV demo (ODbL), MIMIC-IV (PhysioNet credentialed), MedPix 2.0 (no license stated; ask before
redistributing), MedGemma (HAI-DEF terms). Scoring follows LIRICAL (Robinson et al., AJHG 2020).

This is a research tool. It is not for clinical decisions.
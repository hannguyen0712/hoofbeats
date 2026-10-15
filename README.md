# hoofbeats

*When you hear hoofbeats, think horses, not zebras.* hoofbeats ranks both.

Evidence-grounded diagnosis support for rare and common diseases. Given a
patient's findings (structured terms, lab values, clinical text, or medical images), hoofbeats ranks
candidate diseases in one list, explains every ranking with claims that are checked against their
sources, and suggests which finding to check next. It also evaluates itself without leakage.

The core runs offline on a laptop CPU. MedGemma (Google's medical LLM) is optional: it can
re-rank the shortlist and read clinical notes or images, and every output it produces is verified.

- [Architecture](#architecture)
- [Pipeline in detail](#pipeline-in-detail)
- [Setup](#setup)
- [Run guide](#run-guide)
- [Results viewer](#results-viewer)
- [Pilot results](#pilot-results)
- [Safeguards](#safeguards)
- [Limitations](#limitations)
- [Data sources and credits](#data-sources-and-credits)

---

## Architecture

```
            INPUTS                          CORE (model.py)                       OUTPUTS
 ┌───────────────────────┐        ┌──────────────────────────────┐
 │ Phenopacket Store     │──┐     │ Knowledge base               │      ranked diseases + posterior
 │  (HPO terms, +/-)     │  │     │  HPO ontology  (hp.obo)      │      evidence claims (verified)
 ├───────────────────────┤  │     │  rare profiles (hpoa)        │      next checks (info gain)
 │ MIMIC-IV labs/notes   │──┼──►  │  common profiles (MIMIC)     │──►   MedGemma re-rank (optional)
 ├───────────────────────┤  │Case │                              │      CSVs → results viewer
 │ MedPix text + images  │──┘     │ LRRanker → explain → verify  │
 └───────────────────────┘        │ suggest_tests                │
        ▲   ▲                     └──────────────────────────────┘
        │   └── MedGemma reads images / notes → findings (llm.py)
        └────── offline extractor: text → HPO with negation (data.py)
```

Every source is converted into one shape, a **Case**, so the ranker never needs to know where a
finding came from.

| File | Role |
|---|---|
| `hoofbeats/model.py` | Data types, HPO ontology, disease profiles, likelihood-ratio ranker, explanations and claim verification, next-test selection |
| `hoofbeats/data.py` | Downloads; loaders for Phenopacket Store, MIMIC-IV, Orphadata and MedPix; offline text extractor; de-labelling |
| `hoofbeats/llm.py` | MedGemma client (in-process or local server), shortlist re-ranking, note extraction, image reading, reply repair, privacy guard |
| `hoofbeats/__main__.py` | Command line: `download`, `diagnose`, `benchmark`, `sequential`, `learn-common`, `mimic`, `medpix` |
| `docs/index.html` | Local results viewer (reads CSVs from `results/`) |
| `tests.py` | 31 offline tests on a tiny synthetic dataset; MedGemma is replaced by canned replies |

---

## Pipeline in detail

### 1. The common data shape

A **Finding** is one HPO term with a status:

| Status | Meaning | Used in scoring |
|---|---|---|
| `present` | the patient has it | yes |
| `absent` | looked for and not found | yes |
| `family` | a relative has it | no |
| `uncertain` | hedged ("possible", "rule out") | no |

Each finding also keeps its **source** (`phenopacket`, `lab:Ammonia`, `note`, `image:MPX1009_…`)
and the **evidence** text or value behind it.

A **Case** holds the findings, the true diagnosis ids (used only for scoring), causal genes, the
PubMed ids of the paper the case came from (for leakage control), and `meta`: the privacy policy
and anything the system must *not* use as input.

### 2. Getting findings from each source

**Phenopacket Store** (rare diseases, structured). Observed HPO terms become `present`, excluded
terms become `absent`. The true disease, causal gene and source paper come from the file.

**MIMIC-IV** (real hospital admissions).
- *Labs.* 31 common tests are mapped to HPO terms (for example low alkaline phosphatase →
  `HP:0003282`). A term is `present` if any value in the admission was out of range in that
  direction and `absent` if it was measured and always in range. That gives real negative findings.
- *Labels.* ICD-10 codes among an admission's first three diagnoses are mapped to Orphanet rare
  diseases. Only exact and "Orphanet broader than ICD" mappings are used, because a broad ICD code
  would mislabel many cases. With common-disease profiles loaded, ICD categories (e.g. `ICD10:A41`)
  are labels too.
- *Notes* (credentialed release only). Notes are de-labelled first: diagnosis and instruction
  sections are dropped; disease names, distinctive name words and gene symbols are masked. Phenotype
  words are never masked ("fever" survives in "familial Mediterranean fever"). Leftover diagnosis
  words are reported per admission. The note then goes through the text extractor.

**MedPix 2.0** (teaching cases with CT/MR images).
- *Text input:* History and Exam only, de-labelled.
- *Image input* (with `--llm`): MedGemma reads up to `--max-images` images per case, given the
  modality, plane, body region, age and sex, and is told not to name a diagnosis.
- *Never input:* title, diagnosis, differential, the radiologist's findings, image captions. The
  radiologist's findings are used only for an upper-bound run and to score MedGemma's image reading.
- *Labels:* diagnoses are free text and are matched to disease names in the knowledge base. A name
  matches only if all of its distinctive words appear; one-word names need 8+ characters; the most
  specific match wins. Matches are written to `medpix_labels.csv` for review; corrected files are
  passed back with `--labels`.

**Text → HPO.** Two extractors share one output format.
- *Offline extractor* (default; deterministic; safe for credentialed notes): matches HPO labels and
  exact synonyms, longest match first. Rules set the status: negation before or after a term ("no",
  "denies", "was ruled out") → `absent`; hedges → `uncertain`; family words or a Family History
  section → `family`. Medication and instruction sections are skipped.
- *MedGemma extractor:* proposes finding, status and a verbatim quote. A finding is kept only if the
  quote appears in the note and the phrase maps to an HPO term.

### 3. Knowledge base

- **HPO ontology** (`hp.obo`): the term hierarchy. Alternative and obsolete ids are resolved; only
  terms under "Phenotypic abnormality" are used.
- **Rare-disease profiles** (`phenotype.hpoa`): which HPO terms each disease shows, how often, and
  which papers support each entry. Frequencies like `1/1` are shrunk to 2/3 so a single reported
  patient doesn't make a feature look certain. `NOT` entries mark features a disease explicitly
  lacks. An annotation implies all its ancestors ("focal seizure" implies "seizure").
- **Genes** (`genes_to_disease.txt`): used only in the gene ablation.
- **Common-disease profiles** (from `learn-common`): for each ICD category in MIMIC training
  patients, the share of admissions where each lab finding was abnormal, smoothed as
  (abnormal + 1) / (measured + 2). Two extras:
  - *Scope:* lab-only profiles cannot see faces or seizures, so findings outside the lab terms are
    neutral for common diseases instead of counting against them.
  - *Prior:* each common disease's share of admissions. Rare diseases share the observed
    rare-admission rate, split evenly.

### 4. Ranker (`LRRanker`)

Each disease's score is the sum of log likelihood ratios (LR) of the patient's findings, plus an
optional log prior.

- **Present finding *t*:** LR = f(u, disease) / bg(u) for the ancestor *u* of *t* that fits the
  disease best, where f is how often the disease has *u* and bg is the average over all diseases.
  A patient term more specific than the disease's annotation still matches through the shared
  ancestor (with a smaller LR, since ancestors are more common). No match → floor 0.05. A `NOT`
  annotation → floor.
- **Absent finding *t*:** LR = (1 − f) / (1 − bg) when the disease is annotated with *t*; LR = 1
  (neutral) when it isn't, because the knowledge base is silent. `absent_weight` scales this term.
- **Gene** (optional): ×100 when a case gene is linked to the disease.
- **Prevalence prior** (when common diseases are loaded): log share of admissions;
  `--prior-weight 0` turns it off.
- **Leakage control:** for each test case, annotations supported *only* by the case's own paper are
  removed and the affected diseases are rescored.
- **Output:** a posterior (softmax with a temperature fitted on held-out diseases) and the rank of
  the true diagnosis. Ties count against the true diagnosis, so ranks are never flattering.

Scoring is vectorised (sparse disease × term matrix): about 10 ms for all 8,465 OMIM diseases. A
test checks that the vectorised score equals the sum of the per-finding contributions.

### 5. Explanations and verification

`explain()` turns each disease's contributions into claims, for example:

> Has 'Focal motor seizure'; Disease X shows this in ~67% of cases. LR 190 [PMID:29090338]

`verify_claim()` re-derives each claim independently: the finding is in the record with that
status; the disease has that annotation at that frequency; the cited papers support it; none is the
case's own paper; the patient's term is the cited term or a subtype of it.

### 6. Next-test suggestions (`suggest_tests`)

For every HPO term not yet known, hoofbeats computes the **expected information gain**: how much
checking it would reduce uncertainty over the top 20 diseases plus an "everything else" bucket.
Terms already implied are skipped (having a term implies its ancestors; lacking a term implies
lacking its subtypes).

### 7. MedGemma (`llm.py`)

- **Backends:** in-process with Hugging Face transformers (GPU), or any local OpenAI-compatible
  server (Ollama, vLLM, llama.cpp). Images go as base64 to servers, as image objects in-process.
- **Privacy guard:** credentialed MIMIC cases, and knowledge bases containing profiles learned from
  credentialed MIMIC, can only go to an in-process model or a `localhost` server. Anything else
  raises `PolicyError` before a prompt is built.

| Role | What MedGemma may do | How it is checked |
|---|---|---|
| Re-rank | Reorder the top 10 using numbered evidence `[C1]…[Cn]` | Cannot add or remove diseases. Each explanation sentence must cite claims that exist, belong to that disease, and share at least half its content words; otherwise it is dropped. **Faithfulness** = kept ÷ all sentences. |
| Note extraction | Propose finding, status, quote | Quote must appear verbatim in the note; phrase must map to HPO. |
| Image reading | Free-text report + findings list | Mapped to HPO like notes; precision/recall vs the radiologist's findings (ancestor matches count). Replies cached per image. |

**Reply repair.** Small models often return messy JSON, so replies are cleaned before parsing:
markdown fences, comments and trailing commas are removed; cut-off replies are closed; ids like
`618890` or a disease name are mapped back to the shortlist. If a reply still can't be read,
`diagnose` prints its beginning.

### 8. Evaluation commands

| Command | What it measures |
|---|---|
| `benchmark` | Phenopacket ablations (present only; + absent; no leakage control; + gene; ± prevalence), an absent-weight sweep, calibration (ECE, accuracy on the most confident half), and with `--llm` ranker vs MedGemma plus faithfulness. Max 5 cases per disease. "Evaluable" = the true disease still has annotations after leakage control. |
| `sequential` | Hide half the observed findings; ask one term per round; answer from the hidden record. Compares information gain vs "most frequent feature of the top disease" vs random, and reports how many questions could be answered. |
| `learn-common` | Common-disease profiles from 80% of MIMIC patients (fixed hash split; a patient's admissions stay together). |
| `mimic` | Ranks held-out admissions in the mixed list; reports rare and common diagnoses separately. |
| `medpix` | Text only vs text + radiologist findings (upper bound) vs MedGemma images only vs text + MedGemma images; image precision/recall. |
| `diagnose` | One case: ranking, verified claims, next checks, and MedGemma re-ranking with `--llm`. |

---

## Setup

Requirements: Python 3.10+, about 2 GB of disk for data (plus ~3 GB for an optional local
MedGemma).

```bash
git clone <your repo url> hoofbeats
cd hoofbeats
python3 -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt  # numpy, scipy, pandas
python -m unittest tests                   # expect: Ran 31 tests ... OK
python -m hoofbeats --help
```

If you see `ModuleNotFoundError`, the venv isn't active: run the `activate` line again and install
with `python -m pip`, not `pip`.

### Data

```bash
# HPO ontology + annotations + Phenopacket Store -> data/kb, data/phenopacket-store
python -m hoofbeats download

# Orphanet ICD-10 mapping (CC BY 4.0)
curl -L -o data/kb/ORPHA_ICD10_mapping_en.xml \
  https://raw.githubusercontent.com/orphanet-rare-diseases-issues/RD-CODE/master/ORPHA_ICD10_mapping/ORPHA_ICD10_mapping_en.xml

# MIMIC-IV demo (open, no login)
wget -r -N -c -np -nH --cut-dirs=1 -P data https://physionet.org/files/mimic-iv-demo/2.2/

# MedPix 2.0 (~320 MB, 2,050 CT/MR images)
git clone --depth 1 https://github.com/CHILab1/MedPix-2.0.git data/medpix-src
```

The Orphanet file also ships in `Orphanet_Nomenclature_Pack_EN.zip` from
https://www.orphacode.org/pack-nomenclature/. Without `wget` (Windows), download the MIMIC demo
ZIP from https://physionet.org/content/mimic-iv-demo/2.2/ and unzip it to `data/mimic-iv-demo/2.2/`.

Check:

```bash
ls data/kb                                 # hp.obo, phenotype.hpoa, genes_to_disease.txt, ORPHA_ICD10_mapping_en.xml
ls data/phenopacket-store                  # a version folder, e.g. 0.1.27
ls data/mimic-iv-demo/2.2/hosp             # diagnoses_icd.csv.gz, labevents.csv.gz, ...
ls data/medpix-src/MedPix-2-0              # Case_topic.json, Descriptions.json, images/
```

### MedGemma (optional)

Accept the Health AI Developer Foundations terms on the Hugging Face page of
`google/medgemma-1.5-4b-it` first (each teammate, own account). Pick one backend:

**A. Ollama on a laptop** (4-bit model, no GPU needed; good for development)

```bash
# install from https://ollama.com, then:
ollama pull hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M
```

Raise Ollama's context window or long prompts get cut off: set **Context length** to 16k in the
app's Settings, or run `launchctl setenv OLLAMA_CONTEXT_LENGTH 16384` (macOS) and restart the app,
or quit the app and run `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`. Check with
`curl http://localhost:11434/api/tags`.

**B. vLLM on a GPU** (full model; use for final numbers)

```bash
pip install vllm
vllm serve google/medgemma-1.5-4b-it --max-model-len 16384
# on a remote GPU, tunnel it so it is "localhost" on your laptop:
ssh -L 8000:localhost:8000 you@gpu-machine
```

**C. In-process on a GPU**

```bash
python -m pip install -U "transformers>=4.50" accelerate torch pillow
hf auth login
```

Endpoints and model names used below:

| Backend | `--llm-endpoint` | `--llm-model` |
|---|---|---|
| Ollama | `http://localhost:11434/v1/chat/completions` | `hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M` |
| vLLM | `http://localhost:8000/v1/chat/completions` | `google/medgemma-1.5-4b-it` (default) |
| In-process | *(omit)* | `google/medgemma-1.5-4b-it` (default) |

---

## Run guide

Run from the project root with the venv active. Global options (`--common`, `--prior-weight`,
`--llm`, `--llm-endpoint`, `--llm-model`) go **before** the command; command options go after.
Keep each command on one line, or end continued lines with `\` and nothing after it.

### 1. One case

```bash
python -m hoofbeats --common none diagnose data/phenopacket-store/0.1.27/KCNT1/PMID_23086397_Patient_4.json
```

### 2. Rare-disease benchmarks (Phenopacket Store)

```bash
python -m hoofbeats --common none benchmark  --sample 0   --out results/rare   # all cases
python -m hoofbeats --common none sequential --sample 300 --out results/rare
```

`--common none` keeps these runs rare-only even after common profiles exist.

### 3. Multimodal (MedPix)

```bash
# text only + radiologist upper bound (no GPU)
python -m hoofbeats --common none medpix --medpix data/medpix-src/MedPix-2-0 \
    --orphadata-xml data/kb/ORPHA_ICD10_mapping_en.xml --out results/medpix
```

Review `results/medpix/medpix_labels.csv`, fix wrong matches, and add
`--labels results/medpix/medpix_labels.csv` on later runs.

### 4. Rare + common diseases (MIMIC-IV demo)

```bash
python -m hoofbeats learn-common --mimic data/mimic-iv-demo/2.2 \
    --orphadata-xml data/kb/ORPHA_ICD10_mapping_en.xml --min-adm 3
python -m hoofbeats mimic --mimic data/mimic-iv-demo/2.2 \
    --orphadata-xml data/kb/ORPHA_ICD10_mapping_en.xml --out results/mimic_demo
```

`learn-common` writes `data/kb/common_profiles.json`; every later command uses it unless you pass
`--common none`. The demo (~100 patients) only shows the pipeline works.

### 5. With MedGemma

Set the backend once (Ollama shown; see the table above for others):

```bash
LLM=(--llm --llm-endpoint http://localhost:11434/v1/chat/completions --llm-model hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M)
```

This works in bash and zsh. On Windows PowerShell use
`$LLM = @("--llm", "--llm-endpoint", "http://localhost:11434/v1/chat/completions", "--llm-model", "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M")`
and write `@LLM` where the commands below say `"${LLM[@]}"`.

```bash
# one case first: check for a numbered list with + / x lines and a faithfulness %
python -m hoofbeats --common none "${LLM[@]}" diagnose data/phenopacket-store/0.1.27/KCNT1/PMID_23086397_Patient_4.json

# ranker vs MedGemma + faithfulness
python -m hoofbeats --common none "${LLM[@]}" benchmark --sample 200 --llm-cases 20 --out results/medgemma

# MedGemma reads images: small test first, then everything
python -m hoofbeats --common none "${LLM[@]}" medpix --medpix data/medpix-src/MedPix-2-0 \
    --orphadata-xml data/kb/ORPHA_ICD10_mapping_en.xml --split test --max-images 1 --out results/medpix_test
python -m hoofbeats --common none "${LLM[@]}" medpix --medpix data/medpix-src/MedPix-2-0 \
    --orphadata-xml data/kb/ORPHA_ICD10_mapping_en.xml --out results/medpix
```

Image replies are cached in `<out>/medpix_image_reads.jsonl`, so reruns don't call the model again.
On a laptop CPU expect roughly 30 s to 2 min per case.

### 6. Credentialed MIMIC-IV (secure machine only)

Requires PhysioNet credentialing and signed agreements for MIMIC-IV and MIMIC-IV-Note.

```bash
python -m hoofbeats learn-common --mimic /secure/mimiciv/3.1 \
    --orphadata-xml data/kb/ORPHA_ICD10_mapping_en.xml --min-adm 50 --out /secure/common_profiles.json
python -m hoofbeats --common /secure/common_profiles.json mimic --mimic /secure/mimiciv/3.1 \
    --notes /secure/mimic-iv-note/2.2/note --orphadata-xml data/kb/ORPHA_ICD10_mapping_en.xml --out /secure/results
# MedGemma note extraction: local backend only (the guard refuses anything else)
python -m hoofbeats --common /secure/common_profiles.json "${LLM[@]}" mimic --mimic /secure/mimiciv/3.1 \
    --notes /secure/mimic-iv-note/2.2/note --orphadata-xml data/kb/ORPHA_ICD10_mapping_en.xml --out /secure/results_llm
```

Never commit, upload or paste into any online chatbot anything under `/secure`, including the
profiles file, results and extracted sentences.

### Output files

| File | Written by | Contents |
|---|---|---|
| `benchmark.csv` | `benchmark` | condition, case, truth, rank, evaluable |
| `medgemma_rerank.csv` | `benchmark --llm` | case, ranker rank, MedGemma rank, faithfulness |
| `sequential.csv` | `sequential` | policy, case, truth, round, rank, answered |
| `mimic.csv` | `mimic` | admission, truth, rank, rare/common, top 3, finding counts, de-label leftovers |
| `medpix.csv` | `medpix` | case, diagnosis, mapped truth, rank per input, finding counts, image precision/recall, MedGemma report |
| `medpix_labels.csv` | `medpix` | diagnosis mapping to review and correct |
| `medpix_image_reads.jsonl` | `medpix --llm` | raw MedGemma replies per image (cache) |
| `data/kb/common_profiles.json` | `learn-common` | common-disease profiles and held-out patients |

---

## Results viewer

`docs/index.html` finds every result CSV under `results/` (up to three folders deep) and shows
tabs for the benchmark, next-test, MIMIC, MedGemma and MedPix results.

```bash
python -m http.server 8000 --bind 127.0.0.1    # from the project root, not from docs/
```

Open http://localhost:8000/docs/. The newest file of each kind is shown first; use the **Source**
dropdown under each tab for older runs and **Reload results** after a new run. You can also drag
CSVs onto the page. Don't open `index.html` by double-clicking; the browser then blocks it from
reading `results/`. `--bind 127.0.0.1` keeps the server visible only to your computer.

---

## Pilot results

HPO 2026-09-01, annotations 2026-09-02, Phenopacket Store 0.1.27, seed 13, at most 5 cases per
disease, 8,465 OMIM diseases. Small samples: rerun on everything before reporting.

**Ranking, 400 cases / 243 diseases (295 evaluable)**

| Condition | Top-1 | Top-10 | Evaluable Top-1 |
|---|---|---|---|
| Present findings only | 0.170 | 0.330 | 0.231 |
| Present + absent | 0.138 | 0.270 | 0.186 |
| Present + absent, **no leakage control** | **0.485** | 0.640 | 0.427 |
| Present + absent + causal gene | 0.320 | 0.480 | 0.434 |

- Without leakage control Top-1 is about 3.5× higher. For 105 of 400 cases the knowledge base
  knows the disease only from the test case's own paper.
- Absent findings lower accuracy (absent weight 0 / 0.5 / 1 → Top-1 0.231 / 0.220 / 0.186 on
  evaluable cases), likely because case reports list negatives that are typical of the disease.
- Calibration: temperature 12.5, ECE 0.117.

**Next-test suggestions, 300 evaluable cases, 5 questions**

| Policy | Top-1 start → end | Top-10 start → end | Answered |
|---|---|---|---|
| Information gain | 0.160 → 0.180 | 0.373 → 0.390 | 14% |
| Most frequent feature | 0.160 → 0.167 | 0.373 → 0.373 | 13% |
| Random | 0.160 → 0.163 | 0.373 → 0.367 | 3% |

**MedPix, 163 of 671 cases with a mapped diagnosis (12,820 OMIM + ORPHA diseases)**

| Input | Top-10 |
|---|---|
| History + exam text | 0.074 |
| Text + radiologist image findings (upper bound) | 0.129 |

---

## Safeguards

- **Leakage:** the case's own paper is excluded per case; diagnosis text is stripped from notes
  and MedPix histories; radiologist findings and captions are never input; MIMIC is evaluated on
  held-out patients; `split_by_pmid` separates cases published after a model's training data.
- **Privacy:** credentialed data is tagged `local_only` and the LLM guard enforces it; the default
  extractor runs offline; `.gitignore` excludes `data/`, `results/` and MIMIC folders; the viewer
  refuses credentialed results unless it is served from your own machine.
- **Honest numbers:** ties count against the true disease; evaluable subsets are reported; upper
  bounds sit next to gains; MedGemma outputs are verified, never trusted directly.

## Limitations

- Absent findings hurt a plain likelihood model on Phenopacket Store; how to weight pertinent
  negatives is an open question.
- The next-test benchmark is capped by how often the record can answer (about 14%).
- The dictionary extractor misses much radiology wording, which keeps the MedPix upper bound low.
- Common-disease profiles see labs only until credentialed notes are added; ICD categories are
  coarse; the MIMIC demo is tiny and adult-only.
- MedGemma may have seen published cases and MedPix during training: report its gains as upper
  bounds or on post-cutoff cases. A 4-bit Ollama model can behave differently from the full model.

## Data sources and credits

| Source | License / terms | Cite |
|---|---|---|
| Human Phenotype Ontology + annotations | HPO license (use files unmodified) | the exact release used |
| Phenopacket Store | BSD-3 | Danis et al. 2025, with version |
| Orphanet ICD-10 mapping | CC BY 4.0 | Orphanet, with release date |
| MIMIC-IV demo | ODbL | doi:10.13026/dp1f-ex47 |
| MIMIC-IV / MIMIC-IV-Note | PhysioNet Credentialed Health Data License | per PhysioNet |
| MedPix 2.0 | no license stated in the repository; ask before redistributing | Siragusa et al., *Data Science and Engineering* |
| MedGemma | Health AI Developer Foundations terms | Google model card |

MedGemma and hoofbeats are research tools, not for clinical decisions.
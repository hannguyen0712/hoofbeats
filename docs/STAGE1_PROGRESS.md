# Stage 1 progress: text intake, the rule-based text reader, and the numbers reader

Branch `stage1-intake`. Snapshot of 2026-10-26.

**Data:** HPO 2026-09-01, `phenotype.hpoa` 2026-09-02, MedPix 2.0 (`d3bc431`).

**Tests:** all 91 pass except the 4 PubMed tests. Those fail because `hoofbeats/pubmed.py` is missing, as they did before this branch.

Read the caveats (section 5) before quoting any number from this file.

---

## 1. What was built

### Intake

**`hoofbeats/intake.py`: the case-file format and loader**
- The case-file schema is validated with pydantic, field by field. A bad entry is skipped and recorded in `meta["intake_warnings"]`.
- Readers for HPO terms, note text, image context, image findings, labs and vitals.
- The default `policy` is `local_only`.

**`hoofbeats/fusion.py`: merging into one Case**
- One finding per HPO term. Status priority is present > absent > uncertain.
- Family findings are kept separately and never override the patient's own status.
- With `drop_ancestors` (on by default), redundant ancestors are dropped, and an absent ancestor of a present term is recorded as a conflict.
- Every contribution stays in `meta["provenance"]`.

**`diagnose`** also reads case files and prints an intake summary with grounding counts and disease mentions.

### Text reader (`Extractor` in `data.py`, used by every text path)

| Area | What it does |
|---|---|
| Structure | Mentions keep character offsets into the original text. Patient and family statuses of a term are kept apart. Sentences are not split after common abbreviations. "1st/2nd/3rd/4th" match as "first/second/third/fourth". |
| Sections | Clinical section headings (family, past history, social, HPI, CC, ROS, exam, labs, imaging, assessment). Two-letter headings count only if they are known (CC, FH, PE). An unknown heading must start its line, have at most 4 words, and contain no family word or "pt". |
| Family words | A family word applies only to its own clause; an inline `LABEL:` also ends a clause. |
| Negation (NegEx/ConText style) | Cues, pseudo-cues, contrast words and termination words are in `resources/negation_terms.tsv`. A cue's scope ends at a contrast or termination word, or at a new inline `LABEL:` ("no JVD    LUNGS: wheezes"). Words like "afebrile" give an absent finding (`resources/negated_words.tsv`). |
| Vocabulary | Adds RELATED and layperson HPO synonyms. Lemma matching with spaCy `en_core_web_sm` applies only to text words tagged NOUN or ADJ. |
| Abbreviations | About 230 entries in `resources/abbreviations.tsv`, including AV, with context words and other meanings. A best-guess reading of an ambiguous abbreviation is graded weak: either several expansions' contexts match, or it sits alone in a comma list with no context word in its own list item. |
| Filters | A blocklist, and a test-name filter that is skipped when a result word is near. |
| Disease mentions | Knowledge-base disease names are recorded with their spans but never turned into findings, unless the name is also an HPO term. One-word core names that are family, relationship or common English words are excluded (`resources/disease_name_stoplist.tsv`). |

### Numbers reader (`hoofbeats/numbers.py`)

One core for structured labs, structured vitals, and free text (note text, image context, MedPix history).

**Labs**
- `resources/lab_reference.tsv` lists 43 analytes: typical adult intervals with aliases, units, HPO terms for low and high, and plausible ranges.
- A reference range in the input wins: structured, or "(ref a–b)" in text.
- H/L flags are read. Unlabelled lists such as "ALB 2.6 ALK PHOS 202 AST 572" are read.
- Values in CSF, urine or other non-blood specimens are skipped.

**Vitals, ECG and growth measures** (`resources/vital_thresholds.tsv`, each threshold with a source)

| Measure | Rule |
|---|---|
| Temperature | °C or °F; fever ≥ 38.0 °C, hypothermia < 35 °C |
| Heart rate | tachycardia > 100, bradycardia < 60 |
| Respiratory rate | tachypnea > 20, bradypnea < 12 |
| Blood pressure | hypertension at SBP ≥ 140 **or** DBP ≥ 90 (2018 ESC/ESH); hypotension at SBP < 90 |
| SpO2 | hypoxemia < 90% |
| PR interval | > 200 ms → first-degree AV block |
| QTc | ≥ 460 ms → prolonged QTc |
| z-scores and centiles | aortic root Z ≥ 2; height and OFC ±2 SD, or the 3rd/97th centile |
| Arm span | "arm span > height", or a ratio > 1.05 |

Implausible values (typos such as "O2 sat 995") are skipped with a warning.

**Measure lists.** "Height, weight and head circumference were 131.7 cm (<3rd percentile), 48.6 kg and 55.5 cm (95-97th percentile), respectively" is paired by order; if the counts differ, the sentence is skipped with a warning. "Below the 3rd centile for height, weight and head circumference" gives the centile to each measure. A centile range ("95-97th") counts only if it lies wholly outside the 3rd–97th band; "P90" is read inside lists.

**Normal vital signs (`emit_normal_vitals`, default off).** A normal temperature, heart rate, respiratory rate, SpO2 or blood pressure gives no absent finding; abnormal values still give present findings. The value stays in `meta["vitals"]` (structured) and in `meta["provenance"]["normal_vitals"]` (span-checked). With the option on, the earlier behaviour returns. Labs (input or default ranges), PR/QTc, z-scores and centiles are unchanged.

**Units**
- Only these conversions are supported: glucose, BUN/urea, calcium (mmol/L ↔ mg/dL); creatinine, bilirubin (µmol/L ↔ mg/dL); g/L ↔ g/dL; cell counts per µL or mm³ ↔ K/µL; and identities such as mEq/L = mmol/L for monovalent ions.
- Any other unit is skipped with a warning.
- A missing unit is accepted only if the value is plausible, and is then graded weak.

**Children.** When the age is known and below 18, no finding comes from the adult default ranges or the adult thresholds, unless the input gives a range. Z-scores and centiles are used. Age bands are future work.

**Status**
- Out of range in a direction → that direction's term is present. In range → absent (for vital signs only with `emit_normal_vitals`).
- A lab's own H/L flag decides when it is given.
- If the text and a number disagree ("known hypertension" vs "BP 118/76"), fusion records a conflict and present wins.

**Grounding**

| Level | When |
|---|---|
| strong | the reference range comes from the input |
| supported | a default range with an explicit or converted unit, or an explicit threshold measure |
| weak | the unit was inferred, or the label is ambiguous ("K", "Ca", "T", "P") |

Provenance records value, unit, converted value, the range or threshold and its source, and the flag.

### Grounding (`hoofbeats/grounding.py`)

- Every finding carries its quote, field, offsets, status cue, link method, reader and grounding level.
- `verify_provenance` checks every span.

| Level | Meaning |
|---|---|
| strong | Exact label or EXACT synonym (including EXACT layperson), an HPO id given as input, or a lab with an input range |
| supported | A non-EXACT layperson synonym, an abbreviation, a lemma match, a negated word, or a default range or threshold |
| weak | A RELATED synonym, a best-guess abbreviation, or an inferred unit or ambiguous label |

### Evaluation

- `extract-eval` scores reader output against gold findings that carry exact quotes. Quotes are located by gold offsets or by whole word, so "CP" is not found inside "PCP".
- It reports per case and reader, strict and lenient P/R/F1, span overlap, and a per-method breakdown (rules, numbers, labs, vitals, input).
- Gold sets: `examples/` (dev) and `examples/heldout/` (now "seen").
- `TestGoldFiles` fixes the accepted alternatives.
- **PMC test set** (`hoofbeats/pmc.py`, `hoofbeats/pmc_checks.py`, `pmc_eval/`): real case-report text with Phenopacket Store gold; see section 3.

### LLM text reader (in progress; results pending the first Sol run)

Aim: recover paraphrased findings and normal statements (on PMC dev the rules find 39 of 42 hand-checked terms
written with an HPO label or synonym, but 4 of 27 paraphrased ones) without losing precision or grounding.
- **Model:** Qwen2.5-7B-Instruct through vLLM on Sol, bound to 127.0.0.1; extraction and linking refuse any other
  endpoint. `--llm-model` selects the model; replies are cached per model, and `--llm-offline` replays them.
- **Extraction** (`llm_extract.py`): `{phrase, status, quote}` per text chunk under a JSON schema at temperature 0;
  every quote must be found in the text and gets its offsets. Prompt v2 (default) follows the 5-case smoke run;
  v1 stays available (`--prompt-version v1`).
- **Linking** (`linker.py`): dictionary (the rule vocabulary, whole phrase), else SapBERT nearest labels/synonyms
  (accept at >= 0.82 with a 0.05 margin), else the LLM picks one of the top 10 ids or none (0.62-0.82). Thresholds
  from leave-one-synonym-out on HPO itself: 54% accepted at 90.0% exact; true term in the top 10 for 93%.
- **Cross-check:** the rules on the LLM's quote. Same term: grounding at least supported. Nothing found: kept.
  Unrelated terms that explain every content word: dropped (logged with examples). Otherwise weak. A low SapBERT
  quote-label similarity only marks a finding weak.
- **Merge:** rules and LLM agreeing on the same words give one entry with readers `["rules", "llm"]`, grounding
  strong. A status disagreement on the same words is settled by `status_policy` (llm or rules); both are evaluated.
- **Evaluation** (`llm-eval`): rules, LLM, rules+LLM per status policy, per prompt version; hand-checked recall split
  label vs paraphrase, all-gold recall, precision with extras not judged, absent findings on their own (with their
  effect on the example ranks), and the 15-row regression set.
- **Caveat: dev LLM numbers will be optimistic.** The v2 prompt was written after reading the smoke-run output on
  5 dev cases, and any later prompt change will also be judged on dev. Every prompt version is logged with its dev
  scores in `pmc_eval/runs.jsonl`; the test split stays unrun.

## 2. How to run

```bash
python -m hoofbeats extract-eval --cases examples --gold examples/intended_findings.json [--details] --out results/extract_eval
python -m hoofbeats extract-eval --cases examples/heldout --gold examples/heldout/intended_findings.json --out results/extract_eval_heldout
python -m hoofbeats --common none diagnose examples/marfan_note.json      # case file or phenopacket
python -m unittest tests

# PMC test set
python -m hoofbeats pmc-build                      # fetch (cached), locate, freeze pmc_eval/manifest_v<N>.json
python -m hoofbeats extract-eval --pmc pmc_eval/manifest_v1.json --split dev --out results/pmc_dev
python -m hoofbeats pmc-checks                     # task A and B sheets -> data/pmc/checks/
python -m hoofbeats pmc-checks --items results/pmc_dev/pmc_dev_items.csv     # task C sheet
python -m hoofbeats pmc-score --items results/pmc_dev/pmc_dev_items.csv --summary results/pmc_dev/pmc_dev_summary.json
```

The test split needs `--allow-test`. Every PMC run is appended to `pmc_eval/runs.jsonl` with its commit.

`extract-eval` writes `extract_eval.csv` (one row per gold finding or extra) and a run log in `<out>/runs/`.
- **Strict scoring:** same HPO id (or an accepted alternative) and same status.
- **Lenient scoring:** also counts a parent or child term.
- **Span overlap:** the extracted span overlaps the gold quote.

The first `Extractor` build on a new HPO version lemmatises the vocabulary (about 9 s); the result is cached in `data/kb/.cache/`.

## 3. Results

### Extraction, text readers (note, image context, image-finding text)

| Set | Stage | Strict P | Strict R | Strict F1 | Lenient F1 | Extras | Span overlap |
|---|---|---|---|---|---|---|---|
| Dev | baseline (before Step 1) | 0.67 | 0.14 | 0.24 | 0.29 | 2 | 9/9 (sentence spans) |
| Dev | after Steps 1–3 | 0.84 | 0.29 | 0.43 | 0.50 | 1 | 17/17 |
| Dev | after the audit fixes | 0.79 | 0.27 | 0.40 | 0.47 | 2 | 16/16 |
| Dev | **current** (+ A fixes, numbers reader) | 0.81 | 0.30 | **0.44** | 0.51 | 2 | 18/18 |
| Held-out | baseline | 0.57 | 0.15 | 0.24 | 0.30 | 4 | 8/10 |
| Held-out | after Steps 1–3 (first and only blind run) | 0.65 | 0.28 | **0.39** | 0.47 | 5 | 13/16 |
| Held-out, "seen" | after the A fixes | 0.67 | 0.30 | 0.42 | 0.49 | 5 | 15/17 |
| Held-out, "seen" | + numbers reader | 0.62 | 0.30 | 0.41 | 0.48 | 7 | 15/17 |
| Held-out, "seen" | **current** (normal vitals off) | 0.64 | 0.30 | 0.41 | 0.49 | 6 | 15/17 |

- **Why held-out dropped from 0.42 to 0.41:** Fabry's "BP 142/88" now gives Hypertension present (correct, but not in the gold) plus Hypotension absent.
- **Normal vitals off (current default).** Dev text scores are unchanged. Held-out text loses one extra ("BP 142/88" → Hypotension absent). All readers, dev: strict P 0.69 → 0.86, F1 0.54 → 0.59, extras 12 → 3. Held-out: strict P 0.50 → 0.64, F1 0.43 → 0.48, extras 20 → 10. Recall is unchanged, since the gold has no vitals.

**Per-method breakdown, current dev** (gold findings matched, then extras)

| Reader / method | Exact | Parent/child | Extras |
|---|---|---|---|
| note / rules | 12 | 3 | 1 |
| note / numbers | 2 (aortic root Z, arm span) | 0 | 0 |
| labs / labs | 9 | 0 | 1 ("Polycythemia absent", not in gold) |
| vitals / vitals | 0 | 0 | 0 (9 absent extras before normal vitals were turned off) |
| image context / rules | 1 | 0 | 1 |

### Overfitting audit and ablations

| Variant | Dev strict F1 | MedPix findings | MedPix Top-1 / Top-10 |
|---|---|---|---|
| Full, at the time of the audit | 0.427 | 1430 | 0.026 / 0.098 |
| Without the 46 abbreviation rows that occur in the six notes | 0.290 | 1424 | 0.026 / 0.098 |
| Without the one-word lemma rule | 0.416 | 1441 | 0.026 / 0.093 |
| Context words ignored | 0.427 | 1450 | 0.026 / 0.098 |
| Ambiguous abbreviations removed | 0.361 | 1425 | 0.026 / 0.098 |
| **Current:** without the in-note abbreviation rows | **0.310** (full: 0.442) | — | — |

Of the 63 MedPix findings that come through an abbreviation, 54 come from rows that are not in any note.

### MedPix text condition (194 cases whose diagnosis is in the knowledge base; check only, no tuning)

| Stage | Present/absent findings | Top-1 | Top-10 |
|---|---|---|---|
| Baseline | 1,300 | 0.026 | 0.088 |
| After Steps 1–3 | 1,430 | 0.026 | 0.098 |
| After the audit fixes and abbreviation/label/stoplist items | 1,423 | 0.026 | 0.103 |
| + numbers reader | 1,715 | 0.026 | 0.103 |
| **Current** (normal vitals off) | **1,621** | 0.026 | 0.103 |

The numbers reader adds 292 MedPix findings in 64 cases:
- **Status:** 85 present, 207 absent.
- **Grounding:** strong 2, supported 93, weak 197 (most MedPix lab values have no unit).
- **Kind:** 178 lab-range findings, 114 threshold findings.

Ranking is unchanged.

### Precision samples on MedPix (hand-judged)

| Sample | Result |
|---|---|
| Text reader, 30 Step-2 findings (seed 31337) | 27 correct, 2 debatable, 1 wrong. Abbreviation 17/18; layperson synonym 9/11 strict; lemma 12/14 and RELATED 8/9 (all findings of those types). |
| **Numbers reader, 20 random findings (seed 4040)** | **20/20 correct** (18 absent, 2 present) |
| **Numbers reader, 10 random present findings (seed 4041)** | **9/10 correct.** Wrong: "WBC 133" in a body-fluid cell count; the specimen ("fluid culture") is in the previous sentence. |

The 20-item sample also exposed "O2 sat 995", which led to the plausibility check.

### `diagnose` truth rank (case files, `--common none`)

| Case | Before Step 1 | After Steps 1–3 | After the A fixes | + numbers | **Current** (normal vitals off) |
|---|---|---|---|---|---|
| Marfan | 14 | 1 | 1 | 1 | 1 |
| Wilson | 1 | 1 | 1 | 1 | 1 |
| TSC | 1 | 1 | 1 | 1 | 1 |
| Fabry | 131 | 32 | 32 | **18** (BP 142/88 → hypertension) | 19 |
| DM1 | 932 | 1047 | **107** ("1st degree AV block" now found) | 103 | 107 |
| HHT1 | 64 | 12 | 12 | **82** | 82 |

Turning normal vitals off removes absent findings such as Tachycardia, Bradycardia and Hypotension absent. Fabry and DM1 lose 4 such findings each and move down slightly: these absent findings counted slightly against competing diseases. HHT1 has no vitals.

**Why HHT1 fell to 82.** Two correct new lab findings:
- Ferritin 6 (range 15–150) gives "Decreased circulating ferritin concentration" present. HHT1 is not annotated with it, so it takes the floor penalty (−3.0).
- Hemoglobin's new high direction gives "Polycythemia" absent. HHT1 is annotated with polycythemia, so it costs −0.69.

Diseases with low ferritin or anemia profiles then move above HHT1. This is a knowledge-base and ranker effect, not an extraction error.

### PMC test set (independent, real text)

**Source and gold.** Phenopacket Store patients whose paper is in the PMC Open Access subset with a Creative Commons licence (author manuscripts, "NO-CC CODE" and "TDM" licences excluded). Only PMIDs and PMCIDs are sent (NCBI ID converter, BioC API, `tool=hoofbeats`). Gold = the phenopacket's features; excluded features are absent.

| Step | Papers / patients |
|---|---|
| Phenopacket Store | 1,733 papers, 10,376 phenopackets |
| With a PMCID | 1,024 papers |
| BioC full text | 735 papers |
| Creative Commons | 623 papers |
| Located (all rules, before caps) | 491 patients |
| **Manifest v1** | **236 cases from 203 papers**: 136 single-patient, 100 multi-patient; **dev 64, test 172**; 4,400 gold features (median 14 per case); median 350 words of text per case |

**Locator rules, v1** (`hoofbeats/pmc.py`, committed in `979ecff` before any extractor run on PMC text):
- Never: introduction, discussion, abstract, figures, tables, references, supplementary material, back matter, or headings like "Patients and methods", ethics or consent.
- Single-patient papers: paragraphs in a BioC CASE section (101 cases), or under a case-description heading in Results, Methods or an unlabelled section (36).
- Multi-patient papers: the subject id is parsed into the ways a paper names a patient ("Individual 3" → Patient 3 / P3 / Case 3 ...; pedigree II-3; codes like KBG3). A paragraph is kept if it starts with this patient's id and its first sentence names no other patient (45), names this patient and no other (67), or sits under a heading that names only this patient (11). A shared id (two families with a "Patient 1") never counts.
- Cases under 40 words or with no observed gold term are dropped; at most 3 cases per paper, 2 per disease; selection in md5 order; split by `md5(PMID) % 10 < 3` → dev.
- Excluded with reasons (`pmc_eval/excluded_v1.csv`): 2,610 patients with no located paragraph and 935 whose subject id could not be parsed, mostly in multi-patient papers whose patients are described only in tables. The multi-patient target (100) was reached; single-patient papers ran out at 136 of 200.
- The manifest stores BioC passage indexes and a text hash, never the text; a rebuild after a rule fix must bump `LOCATOR_VERSION`, use dev cases only, and re-freeze. One reporting bug (exclusion reason labels) was fixed before the manifest was first used; the selected cases were identical.

**Dev run (64 cases, first and only extractor runs on PMC text; test not run)**

| Measure | n | Strict | Lenient |
|---|---|---|---|
| Recall, all gold | 1,225 | 22.0% | 27.5% |
| Recall, vocabulary-visible gold | 293 | 90.4% | 93.2% |
| Recall, hand-checked (task B, dev terms expressed in the text) | 69 | 55.1% | 62.3% |
| Precision (extras not adjudicated) | 706 | 38.1% | 46.3% |
| Recall, present gold | 678 | 36.9% | 45.1% |
| Recall, absent gold | 547 | 3.5% | 5.7% |
| Recall, single-patient cases | 617 | 36.1% | 45.1% |
| Recall, multi-patient cases | 608 | 7.6% | 9.7% |

| Method | Gold found exact | Parent/child | Predictions | Extras |
|---|---|---|---|---|
| rules | 265 | 66 | 663 | 342 |
| numbers | 7 | 2 | 48 | 39 |

- Only 24% of gold terms have their label or a synonym in the located text; on those the reader finds 90%. The gap to all-gold recall is paraphrase, and gold that is not in the located text at all (tables, figures, other sections, curators' inferences). Task B separates the two.
- Absent gold is mostly not stated in the text (32 of 547 are vocabulary-visible); curators record excluded features from tables.
- Multi-patient recall is low: located text per patient is short and their features are mostly in tables.
- Extras: 379 (274 present, 62 absent, 33 family, 10 uncertain). Many are correct findings the curators did not record; task C adjudicates 100 of them. Frequent extras include Tall stature (10), Microcephaly (9), Macrocephaly (7) and Asthenia (7).

**Hand checks** (pre-filled CSVs in `data/pmc/checks/`, read back by `pmc-score`)

| Sheet | Rows | Fill in |
|---|---|---|
| `task_a_locator.csv` | 40 cases (20 single, 20 multi; 10 dev, 30 test) | `is_this_patient` yes / no / partly |
| `task_b_gold_in_text.csv` | 554 gold terms in 30 cases (166 dev, 388 test) | `expressed` yes / no / unsure, optional `quote`, `status_matches` yes / no |
| `task_b_second_annotator.csv` | 183 rows (10 of those cases) | same, by a second annotator → Cohen's kappa |
| `task_c_extras.csv` | 100 random dev extras | `verdict` correct_not_in_gold / wrong_term / wrong_status / not_this_patient / unsure |

Answers are copied to `pmc_eval/checks_v1_task_a.csv`, `checks_v1_task_b.csv` and `checks_v1_task_c.csv` (ids, answers, notes, and task B quotes; no case text).

**Task B, gold terms in the text (554 terms in 30 cases: 10 dev, 20 test; single annotator).** The second-annotator sheet was not filled, so there is no kappa; this is a limitation.
- Expressed in the located text: **223 of 554 (40%)**; 309 not, 22 unsure. Present gold 201 of 337 (60%); absent gold 22 of 217 (10%). Dev 69 of 166, test 154 of 388.
- Where expressed, the text's status matches the gold in 219 of 223. The 4 that differ are gold errors against the text (e.g. Enlarged cisterna magna and Seizure recorded absent but stated present); they stay in the denominators.
- Hand-checked recall uses the 69 expressed dev terms; the 154 test terms are scored only after a test run.

| Hand-checked recall (dev) | n | Strict | Lenient |
|---|---|---|---|
| All | 69 | **55.1%** (95% CI 43–66) | **62.3%** (51–73) |
| Present | 59 | 61.0% | 69.5% |
| Absent | 10 | 20.0% | 20.0% |
| Single-patient | 55 | 56.4% | 63.6% |
| Multi-patient | 14 | 50.0% | 57.1% |

- By reader: the text rules find 43 of the 69 (exact or parent/child); the numbers reader 1, also found by the rules.
- Of the 69, 42 are vocabulary-visible and the reader finds 39 of them; of the 27 that are not, it finds 4 (as parent/child). The misses are mostly paraphrase ("testes were not palpable", "dilatation of the ventricles", "an elongated face", "walked after two and a half years") and statements needing inference ("blood pressure was normal", "cognitive development is normal", "head circumference p50").
- Multi-patient recall on expressed terms (50%) is close to single-patient (56%): the low all-gold multi-patient recall (7.6%) comes from gold that is not in the located text, not from the reader.

**Precision with task C verdicts** on the current dev run (717 predictions): **79.9% strict, 88.0% lenient** (37.5% / 45.6% unadjudicated). The 78% correct share comes from the 706-prediction run's sample.

**Task A, locator (40 cases).** 27 yes, 10 partly, 3 no. Single-patient: 17 / 2 / 1. Multi-patient: 10 / 8 / 2. By rule: case_section 13 / 2 / 0; case_heading 4 / 0 / 1; id_start 5 / 2 / 0; id_only with id_start 2 / 2 / 0; id_only alone 3 / 4 / 2. "Partly" is mostly literature reviews, lab or animal experiments, or findings stated for the whole cohort; id_only keeps any paragraph that mentions the id. The 3 dev non-yes rows are all "partly", so the rules stay at v1: a fix would be informed by test rows.

**Task C, 100 dev extras (from the 706-prediction run).** 78 correct but not in gold, 12 wrong status, 6 not this patient, 3 wrong term, 1 unsure. Precision with adjudicated extras: **80.0% strict, 88.2% lenient** (38.1% / 46.3% before adjudication; ±8 points on the 78%). Numbers reader 12 of 13 correct (1 unsure); all 21 errors are from the text rules:
- status (12): hedge cue on plainly stated findings (4), family attribution in both directions (5), negation (3);
- not this patient (6): literature cases, cohort statements, the mother;
- wrong term (3): "MI" in a Michigan address, "weakness in limbs" → Asthenia, "abnormal liver function" → raised transaminases (an HPO synonym).

The 12 status and 3 term errors are kept as a regression set for the LLM step (`pmc_eval/regression_v1_status_term.csv`: expected status and term, quote and sentence offsets in the located text). The rule status logic is not changed for them.

**Dev re-run after the measure-list fix** (task C notes C022 and C037; developed on dev sentences and unit tests):

| | Before | After |
|---|---|---|
| Recall, all gold (strict / lenient) | 22.0% / 27.5% | 22.0% / 27.5% |
| Precision, unadjudicated (strict / lenient) | 38.1% / 46.3% | 37.5% / 45.6% |
| Predictions / extras | 706 / 379 | 717 / 390 |
| numbers: exact / parent-child / pred / extra | 7 / 2 / 48 / 39 | 6 / 2 / 58 / 50 |

- Removed: a false Microcephaly present (a height "<3rd percentile" read as head circumference). It had matched the gold by chance; the text rules still find that gold term, so recall is unchanged.
- Added, all from the numbers reader: Short stature present twice (the paired height and a centile before the list), Microcephaly present (PMID30509212, "below ... 3rd centiles at height, weight and head circumference"), and normal-range absents for paired or newly read measures. Read against the text, all 11 look correct but are not in the gold, so unadjudicated precision falls slightly.
- C037 was not a miss: Microcephaly present was extracted from that sentence; only its Macrocephaly-absent row was in the task C sample.

### LLM text reader on PMC dev (first Sol run, Qwen2.5-7B-Instruct, commit `2be3084`)

All numbers are dev and **optimistic** (prompt v2 was written after reading dev output). Status policy: llm (the
rules policy gives the same recall and precision but fixes none of the regression rows). Recall cells are
strict / lenient.

| | rules | rules + LLM v1 | rules + LLM v2 | LLM alone v1 | LLM alone v2 |
|---|---|---|---|---|---|
| Hand-checked recall, 69 terms | 55.1 / 62.3 | **69.6 / 78.3** | **66.7 / 79.7** | 62.3 / 68.1 | 56.5 / 68.1 |
| … written as a label/synonym, 42 | 90.5 / 92.9 | 90.5 / 92.9 | 90.5 / 92.9 | 78.6 / 78.6 | 73.8 / 76.2 |
| … paraphrased, 27 | 0.0 / 14.8 | **37.0 / 55.6** | **29.6 / 59.3** | 37.0 / 51.9 | 29.6 / 55.6 |
| All-gold recall, 1,225 | 22.0 / 27.5 | 27.9 / 34.9 | 28.2 / 35.3 | 22.3 / 28.3 | 22.9 / 29.1 |
| Present / absent gold recall (strict) | 36.9 / 3.5 | 47.1 / 4.2 | 48.1 / 3.7 | 39.2 / 1.3 | 40.7 / 0.7 |
| Precision, extras not judged | 37.5 / 45.6 | 29.8 / 36.2 | 29.7 / 36.6 | 31.2 / 38.9 | 31.0 / 39.5 |
| Predictions / extras | 717 / 390 | 1,148 / 732 | 1,165 / 739 | 875 / 535 | 902 / 546 |
| Absent predictions / extras | 95 / 70 | 115 / 85 | 130 / 103 | 74 / 66 | 87 / 80 |
| Regression rows fixed / partly / not (15) | 0 / 0 / 15 | 7 / 0 / 8 | 5 / 1 / 9 | 9 / 6 / 0 | 7 / 7 / 1 |
| Examples strict F1, dev / held-out | 58.5 / 47.9 | 67.2 / 53.0 | 68.7 / 57.6 | 62.8 / 42.0 | 64.5 / 49.1 |
| Ranks: Marfan, Wilson, TSC, Fabry, DM1, HHT1 | 1, 1, 1, 19, 107, 82 | 4, 1, 2, 4, 71, 17 | 1, 1, 2, 34, 146, 3 | 4, 1, 2, 4, 35, 3 | 1, 1, 2, 50, 62, 1 |

- **Recall gain is on paraphrase.** The LLM keeps everything the rules find with a label (90.5%) and adds 8-10 of the
  27 paraphrased terms strictly (15-16 leniently). v1 vs v2 differ by 2 terms of 69 overall: within noise.
- **Precision before judging falls** (37.5 to about 30) because the LLM adds ~355 extras per version that only it
  found; task C judged 78% of rule extras correct, so this number says little until the LLM extras are judged.
  A blind sheet of 50 LLM-only extras per version (plus 10 absent ones each) is in `data/pmc/checks/task_c_llm.csv`.
- **Absent findings:** the LLM adds 20 (v1) / 35 (v2) absent predictions; few match an absent gold term (absent gold is
  mostly not in the text). They do not move any example rank: ranks are identical without the LLM-only absent
  findings, for both versions.
- **Extraction:** 128 chunks per version, no unreadable reply; quotes not found 75 (v1) / 43 (v2) of ~1,250 proposed.
  Linking of kept findings: dictionary ~440, SapBERT ~170, LLM choice ~325; unlinked 241 (v1) / 177 (v2, fewer
  molecular phrases). The cross-check dropped one finding per version, both correctly ("foamy urine" -> Hematuria;
  "no contractures" -> Aplasia of the musculature).
- **Why rules+LLM fixes fewer regression rows than the LLM alone (7 vs 9 for v1, 5 vs 7 for v2):** the status policy
  only acts where both readers give the same term (or a parent/child) on the same words. Two rows are fixed in
  LLM-alone only because the LLM says nothing: "MI" in a Michigan address (C067) and a negated family ataxia (C081)
  expect no finding; with the rules also running their wrong finding stays, since silence cannot overrule it. (v2
  also misses C024 and C052 in the merge for the same reason: the LLM gave no Fever or Hearing impairment there.)
- **Why Marfan drops to rank 4 with v1:** one correct finding, "Seasonal allergies" -> Seasonal allergy (SapBERT,
  0.99). Removing it alone restores rank 1. The ranker scores a present term through its best-annotated ancestor:
  Loeys-Dietz 1 and 3 match through "Abnormality of immune system physiology" (log LR +1.6), Marfan has nothing on
  that branch and gets the floor (-3.0), and Stolerman syndrome, annotated with Allergy, gains +5.6. v2 did not list
  the allergy, hence rank 1.
- **Why DM1 drops to 146 with v2 (rules 107):** wrong terms from paraphrase, all grounded weak: "dozes off at work" ->
  Fatigue (the LLM's own term; hypersomnolence is meant), extensor weakness -> Limited wrist extension, percussion
  myotonia -> Abnormality of the thenar eminence, "long thin face with hollow temples" -> Triangular face. Removing
  Fatigue alone gives rank 83. Correct paraphrases help at the same time (without Myotonia, from "hands lock up",
  the rank would be 319). Dropping all weak LLM-only findings is not a fix: it also undoes HHT1 82 -> 3 and Fabry
  19 -> 4 (v1).
- **Ranks on six cases are noisy:** single findings move them by tens of places, in both directions.

**Task C for LLM-only extras** (one blind sheet, 116 rows; answers in `pmc_eval/checks_v1_task_c_llm.csv`):

| | v1 | v2 |
|---|---|---|
| LLM-only extras judged correct (random 50) | **64%** (32/50; 95% CI 50-76) | **60%** (30/50; 46-72) |
| Their errors | wrong term 10, wrong status 4, not this patient 1, unsure 3 | wrong term 13, wrong status 3, not this patient 1, unsure 3 |
| Absent LLM extras correct | 8/12 | 7/13 |
| Precision with every extra judged, strict / lenient | **75.2 / 81.7** | **73.6 / 80.5** |

- Rules alone, judged the same way, are at 79.9 / 88.0: the LLM costs about 5 points of judged precision for
  +14.5 (v1) / +11.6 (v2) points of strict hand-checked recall.
- v1 and v2 do not differ beyond noise on precision either (64% vs 60% on 50 each); v1 is ahead on every point
  estimate (judged precision, absent precision, strict paraphrase recall).
- By link method (both versions pooled, 116 rows): SapBERT accepted directly 21/26 correct, dictionary 12/21 (its
  errors are the LLM's own phrase or status), **LLM choice among candidates 35/69 (51%)**. The chooser is pushed
  into over-specific terms when nothing fits (Post-capillary pulmonary hypertension for "secondary pulmonary
  hypertension", Renal interstitial neutrophil infiltration for a skin infiltrate, Decreased DLCO for diffusion
  restriction on brain MRI).
- Other error types: normal values still called present despite the v2 rule (OFC at the 10th centile, glucose in
  range, head control at 3 months), relatives' or the mother's findings, and a procedure read as a finding
  ("intermittent photic stimulation").

## 4. Remaining misses and known issues

- **Paraphrase with no matching label or synonym (the largest group):**
  - eye and chest: "lens displaced upward", "sunken chest", "collapsed R lung";
  - movement and speech: "fine shaking of both hands", "speech thick and mushy", "hands 'lock up' after gripping", "eyelids droop";
  - other symptoms: "dozes off at work", "food 'sticks'", "cloudy lenses", "receding hairline at the front", "seeing double", "heart fluttering";
  - imaging: "wedge-shaped FLAIR lesions".
- **Words inserted inside a known phrase or a different word order:** "long *skinny* fingers", "high-arched *roof of mouth*", "reflexes reduced at ankles".
- **Exam signs that need inference:** "liver edge 3 cm below costal margin", "spleen tip palpable", "no organomegaly", "ventricles normal in size", "met milestones on time".
- **Wrong concept:** "extensor weakness 4/5" maps to Asthenia (through the generic EXACT synonym "Weakness") instead of muscle weakness.
- **Coarser term than the gold:** "N/V" → "Nausea and vomiting"; "passing out" → Loss of consciousness. **Derivation, out of scope:** "blurry".
- **Numbers reader:**
  - The specimen is only seen within the sentence or section, not in the previous sentence.
  - Values without a separator ("Cr1.35", "GLU-109WBC-13.4") and "fishbone" shorthand ("140/4.4/99/29") are not read.
  - Age bands for children are not implemented.
- **Other:** spaCy tags "wheezes" as a verb in contexts such as "No wheezes or crackles", so those miss lemma matching. "Denies CP, N/V" marks CP weak, because CP also means cerebral palsy.

## 5. Caveats

- **Dev numbers are inflated by shared vocabulary.** The abbreviation table and the example notes share abbreviations. Without the in-note rows, dev F1 is 0.310, not 0.442.
- **The held-out set is "seen".** It was evaluated blind once, after Steps 1–3 (strict F1 0.39). It has since been opened for the DM1 analysis, and the A fixes (ordinals, AV) were prompted by that analysis. All later held-out numbers are not blind.
- **"son"** was added to the family-word list knowing the DM1 held-out note uses it.
- **Gold edits.** One example alternative (Pustule) was added after seeing output and has since been removed. Gold files are no longer edited without an explicit request, and `TestGoldFiles` enforces this. The gold contains no vitals and no numeric findings beyond the dev labs, so correct numeric findings outside the gold count as extras.
- **Tuning came from dev mistakes.** The one-word lemma rule, the "N/V." sentence-split fix and the plausibility check were each motivated by a specific observed error.
- **The six cases are synthetic, written by the developer.** The PMC test set (section 3) is the independent check; the test split has not been run. The measure-list fix was prompted by task C notes on dev cases.
- **PMC gold is the curators' record, not the text.** All-gold recall understates the reader; vocabulary-visible recall uses the reader's own vocabulary and overstates it. Hand-checked recall (task B) is the number to quote: 55.1% strict on dev, from 69 terms in 10 cases (wide interval). Task B has a single annotator; no agreement was measured.
- **PMC dev and test differ in mix:** dev has 40 single- and 24 multi-patient cases, test 96 and 76.

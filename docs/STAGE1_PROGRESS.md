# Stage 1 progress: text intake, the rule-based text reader, and the numbers reader

Branch `stage1-intake`. Snapshot of 2026-10-26.

**Data:** HPO 2026-09-01, `phenotype.hpoa` 2026-09-02, MedPix 2.0 (`d3bc431`).

**Tests:** all 81 pass except the 4 PubMed tests. Those fail because `hoofbeats/pubmed.py` is missing, as they did before this branch.

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

**Units**
- Only these conversions are supported: glucose, BUN/urea, calcium (mmol/L ↔ mg/dL); creatinine, bilirubin (µmol/L ↔ mg/dL); g/L ↔ g/dL; cell counts per µL or mm³ ↔ K/µL; and identities such as mEq/L = mmol/L for monovalent ions.
- Any other unit is skipped with a warning.
- A missing unit is accepted only if the value is plausible, and is then graded weak.

**Children.** When the age is known and below 18, no finding comes from the adult default ranges or the adult thresholds, unless the input gives a range. Z-scores and centiles are used. Age bands are future work.

**Status**
- Out of range in a direction → that direction's term is present. In range → absent.
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

## 2. How to run

```bash
python -m hoofbeats extract-eval --cases examples --gold examples/intended_findings.json [--details] --out results/extract_eval
python -m hoofbeats extract-eval --cases examples/heldout --gold examples/heldout/intended_findings.json --out results/extract_eval_heldout
python -m hoofbeats --common none diagnose examples/marfan_note.json      # case file or phenopacket
python -m unittest tests
```

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
| Held-out, "seen" | **current** (+ numbers reader) | 0.62 | 0.30 | 0.41 | 0.48 | 7 | 15/17 |

- **Why held-out dropped from 0.42 to 0.41:** Fabry's "BP 142/88" now gives Hypertension present (correct, but not in the gold) plus Hypotension absent.
- **All-reader scores fall as well:** structured vitals now produce findings, almost all absent (normal values), and the gold contains no vitals. Each one counts as an extra.

**Per-method breakdown, current dev** (gold findings matched, then extras)

| Reader / method | Exact | Parent/child | Extras |
|---|---|---|---|
| note / rules | 12 | 3 | 1 |
| note / numbers | 2 (aortic root Z, arm span) | 0 | 0 |
| labs / labs | 9 | 0 | 1 ("Polycythemia absent", not in gold) |
| vitals / vitals | 0 | 0 | 9 (all absent, no gold) |
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
| **Current** (+ numbers reader) | **1,715** | 0.026 | 0.103 |

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

| Case | Before Step 1 | After Steps 1–3 | After the A fixes | **Current** (+ numbers) |
|---|---|---|---|---|
| Marfan | 14 | 1 | 1 | 1 |
| Wilson | 1 | 1 | 1 | 1 |
| TSC | 1 | 1 | 1 | 1 |
| Fabry | 131 | 32 | 32 | **18** (BP 142/88 → hypertension) |
| DM1 | 932 | 1047 | **107** ("1st degree AV block" now found) | 103 |
| HHT1 | 64 | 12 | 12 | **82** |

**Why HHT1 fell to 82.** Two correct new lab findings:
- Ferritin 6 (range 15–150) gives "Decreased circulating ferritin concentration" present. HHT1 is not annotated with it, so it takes the floor penalty (−3.0).
- Hemoglobin's new high direction gives "Polycythemia" absent. HHT1 is annotated with polycythemia, so it costs −0.69.

Diseases with low ferritin or anemia profiles then move above HHT1. This is a knowledge-base and ranker effect, not an extraction error.

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
  - In-range structured vitals produce many absent findings.
  - Age bands for children are not implemented.
- **Other:** spaCy tags "wheezes" as a verb in contexts such as "No wheezes or crackles", so those miss lemma matching. "Denies CP, N/V" marks CP weak, because CP also means cerebral palsy.

## 5. Caveats

- **Dev numbers are inflated by shared vocabulary.** The abbreviation table and the example notes share abbreviations. Without the in-note rows, dev F1 is 0.310, not 0.442.
- **The held-out set is "seen".** It was evaluated blind once, after Steps 1–3 (strict F1 0.39). It has since been opened for the DM1 analysis, and the A fixes (ordinals, AV) were prompted by that analysis. All later held-out numbers are not blind.
- **"son"** was added to the family-word list knowing the DM1 held-out note uses it.
- **Gold edits.** One example alternative (Pustule) was added after seeing output and has since been removed. Gold files are no longer edited without an explicit request, and `TestGoldFiles` enforces this. The gold contains no vitals and no numeric findings beyond the dev labs, so correct numeric findings outside the gold count as extras.
- **Tuning came from dev mistakes.** The one-word lemma rule, the "N/V." sentence-split fix and the plausibility check were each motivated by a specific observed error.
- **The six cases are synthetic, written by the developer.** An independent test set from PubMed Central case reports with Phenopacket Store gold is planned; the availability pilot is done. Of 100 papers, 66 have a PMCID, 47 have BioC full text, and 41 carry a Creative Commons licence.

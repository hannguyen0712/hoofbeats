# Test suggestions: per-disease workup

For each leading disease, `workup` lists the tests that would confirm it or rule it out, with the probability of
the disease after a positive and after a negative result. It runs after Stage 1 and the rankers and uses everything
they produce. It never suggests treatment.

```bash
python -m hoofbeats download-tests                     # MAxO + LOINC2HPO + GTR -> data/kb (once; GTR weekly)
python -m hoofbeats --common none workup examples/marfan_note.json
python -m hoofbeats --common none diagnose examples/marfan_note.json   # tests for every displayed disease
python -m hoofbeats --common none workup-eval --sample 300 --out results/workup_eval
```

`diagnose` works up every disease it displays (the ranker's top `-k`, and the re-ranker's list when `--llm` runs)
and saves the tests by disease id under `"tests"` in `results/diagnose/<case>.json`, for a UI to show when a disease
is clicked. The order of the diseases is never changed. `workup` gives the full workup of the ranker's top `-k`.

Example (`examples/marfan_note.json`, rules + numbers reader, top disease):

```
1. Marfan syndrome (OMIM:154700): pre-test 13% (ranker 13%)
   gene test: FBN1 gene testing [burden 4]: positive -> 88%, negative -> 0.3% (LR+ 49.0, LR- 0.02)
   GTR: 175 current clinical tests from 49 labs (125 in the US): 75 targeted, 92 panel, 5 exome/genome, 2 screening, 1 other
      e.g. FBN1 Comprehensive - Sequence & Deletion/Duplication Analysis - Baylor Genetics (US)
           https://www.ncbi.nlm.nih.gov/gtr/tests/508684/
   - <rule out> clinical assessment [burden 1]: positive -> 24%, negative -> <0.1% (LR+ 2.1, LR- 0.00); looks for
     Disproportionate tall stature (~71%); Arachnodactyly (~63%); Narrow foot (~62%); ...
   - <confirm> slit-lamp examination [burden 2]: positive -> 51%, negative -> 3.1% (LR+ 6.9, LR- 0.21); looks for
     Cataract (~59%); Ectopia lentis (~47%); ...
   review existing images: echocardiography: ... looks for Tricuspid valve prolapse (~53%); ...
```

The note reports an outside echo, so echocardiography is "review", not "order". Below the diseases, the workup
names registered panels that cover several of them, for example "Marfan syndrome and Loeys-Dietz syndrome
Comprehensive panel" (Connective Tissue Gene Tests, 6 genes), which covers diseases 1 and 3 of that list.

GTR is optional: without its two files in `data/kb`, `workup` runs as before and says so. If `download-tests`
cannot reach NCBI, download them by hand (section 2) into `data/kb`.

## 1. Inputs used

| Input (Stage 1 and after) | Use |
|---|---|
| Findings (present, absent) | A finding already known is not looked for again. Present terms count with their ancestors; absent terms count with their subtypes. |
| Uncertain findings, weakly grounded findings | Listed under "Findings to verify" with the cheapest tests that show them. |
| Patient details | Sex removes findings of the other sex's genitalia. Prenatal and birth findings (HP:0001197) are history, so no test is suggested for them. An age under 18 together with developmental delay or intellectual disability adds the ACMG note (section 3.5). |
| Images, image context, image findings | They name tests already done ("MRI BRAIN", "RENAL US", an image id `CT-head-3`, a note line `Echo (outside, 3/2026): ...`; `resources/tests_done.tsv`). Those tests move to "review existing images" at burden 1. |
| Gene findings | A gene already given is not tested again. |
| Ranker output (scores, evidence) | Pre-test probabilities (section 3.1). |
| Displayed diseases (ranker's list, re-ranker's list) | Which diseases get tests; their order is kept. |

## 2. Which test shows which finding

| Source | What it gives | Size (versions used) | Licence |
|---|---|---|---|
| MAxO diagnostic annotations (Carmody et al., *Med* 2023) | HPO term → diagnostic action that can show it (`is_observable_through`): exams, imaging, specialist evaluations, biopsies | 19,152 rows for 17,647 terms; 313 actions kept (2025-05-31; commit `42edd96`) | CC BY 4.0 |
| LOINC2HPO (Zhang et al., *npj Digit Med* 2019) | LOINC lab test → HPO term per result (low, normal, high, positive, ...) | 3,118 LOINC codes in 429 analytes (commit `c1068d6`) | LOINC licence; academic use, cite |
| `genes_to_disease.txt` (HPO) | disease → genes (Mendelian; otherwise association type unknown) | as in `data/kb` | HPO |
| NCBI Genetic Testing Registry, GTR (Rubinstein et al., *Nucleic Acids Res* 2013) | registered clinical tests: lab, country, methods, genes, conditions (OMIM) | [`test_version.gz`](https://ftp.ncbi.nlm.nih.gov/pub/GTR/data/test_version.gz) + [`test_condition_gene.txt`](https://ftp.ncbi.nlm.nih.gov/pub/GTR/data/test_condition_gene.txt), about 60 MB, weekly; the version is the latest change in the file (2026-10-28 here): 32,852 current clinical tests for 6,605 OMIM conditions, covering 6,257 of the 8,478 OMIM diseases in the knowledge base | public domain; cite GTR |
| `resources/test_burden.tsv` (hand-made) | burden tier 1-5 and kind for every test | 1 bedside, 2 blood/urine test or specialist visit, 3 X-ray/ultrasound/EEG, 4 CT/MRI/endoscopy/gene test, 5 invasive | this repo |

- LOINC codes are grouped per analyte by their "normal" HPO term. "Abnormal blood potassium concentration" becomes
  the test "blood potassium concentration (LOINC test)", which shows that term and its subtypes.
- MAxO lists *a* test able to show a term, not every one. Cortical tubers, for example, only has "medical oncologist
  evaluation", while its parent (abnormal cerebral cortex morphology) has "MRI of the brain". So a finding gets the
  tests of its own term and of its parents. If neither has any, it gets the nearest annotated ancestor's tests, up
  to 3 levels up.
- For a term with a LOINC lab test, MAxO's generic lab actions ("blood chemistry measurement") are dropped.
- Prenatal ascertainment (`is_prenatally_observable_through`) and fetal procedures are left out.

## 3. Formulas

### 3.1 Pre-test probability

- **Ranker:** the top 20 diseases share the probability that the diagnosis is among them, `CALIB_IN`. Within the
  top 20 they split it by softmax(score / `CALIB_T`); everything else shares the remainder.
  - Both numbers are fitted by maximum likelihood on 400 Phenopacket Store cases (leakage control, seed 13,
    HPO 2026-09-01): T = 2.18, `CALIB_IN` = 0.475.
  - On 400 other cases the top disease's probability averaged 0.174 against a Top-1 accuracy of 0.255
    (ECE 0.081). The softmax over all diseases at the `benchmark` temperature (11.6) gave the top disease about
    0.2% even when it was right.
  - `workup-eval` refits both numbers and saves them to `calibration.json` (`workup --calibration`).
  - Case files are easier than leakage-controlled phenopackets, so for them these probabilities are conservative.

### 3.2 One test, one disease

Notation:
- *U* = the disease's annotated findings that are still unknown and that test *T* can show
- *f(u,d)* = their frequency in `phenotype.hpoa`
- leak λ = 0.02

Formulas:
- Chance the test is positive if the disease is present (noisy-OR, as in QMR-DT; Shwe et al., *Methods Inf Med*
  1991; "positive" means any finding of *U* is there):
  `P(T+|d) = 1 - (1 - λ) * prod_u (1 - f(u,d))`
- Chance it is positive otherwise: `P(T+|not d)` is the same expression for each other disease of the top 20
  (frequencies propagated to ancestors), weighted by their probabilities, plus the remaining mass at background
  frequencies.
- `LR+ = P(T+|d) / P(T+|not d)`, `LR- = P(T-|d) / P(T-|not d)`.
- Post-test probability: post-test odds = pre-test odds × LR (Fagan, *NEJM* 1975).

Tags:
- `<confirm>` for LR+ ≥ 5.
- `<rule out>` for LR− ≤ 0.2.
- These are the "moderate" thresholds of the Users' Guides (Jaeschke, Guyatt & Sackett, *JAMA* 1994); LR > 10 or
  < 0.1 counts as large.

### 3.3 Order of the tests

Tests are ordered by **expected weight of evidence per burden tier**.
- Weight of evidence is Good's log likelihood ratio (Good & Card, *Methods Inf Med* 1971).
- Each finding the test shows is its own result.
- With `f_o` the finding's frequency over the other diseases (as in `P(T+|not d)`):

  `value(T) = sum_u [ p * KL(f(u,d) || f_o(u)) + (1 - p) * KL(f_o(u) || f(u,d)) ]`,   `score = value / burden`

  - The first term is how strongly *T* is expected to support *d* if *d* is the diagnosis.
  - The second is how strongly it is expected to count against *d* if it is not.
- The best confirming test and the best ruling-out test are always shown, even when they are not in the top N.

Why this order, and not the binary information gain:
- First version: tests ranked by information gained about "is it d?" from the binary positive/negative result,
  per burden.
- That treats a broad test (clinical assessment with 30 findings) as one coin flip. Its LR+ is near 1, so it was
  undervalued.
- On the 400 development cases (section 4.2), the weight-of-evidence order raised the true disease's probability
  more than the binary information gain in both simulation worlds: +0.038 vs +0.030 (open), +0.293 vs +0.255
  (imputed).
- It was also better than these alternatives: Brier-score reduction, KL of the binary result, and sensitivity
  per burden.
- It was chosen there and then checked on the 300 held-out cases below.

### 3.4 Gene tests

- Per disease: its Mendelian genes, sequencing plus deletion/duplication analysis, burden 4.
- `P(T+|d)` is the gene-test sensitivity, `--gene-sensitivity` (default 0.98, an upper bound; GeneReviews gives
  per-disease detection rates).
- `P(T+|not d)` counts the other diseases of the same gene (an allelic disorder gives the same positive result),
  plus the leak.
- Panel for the diseases worked up: union of their genes. Expected yield = Σ p(d) × sensitivity, the chance the
  panel finds the cause if the diagnosis is among them.

GTR adds what labs have actually registered for each disease (current, clinical tests whose conditions include the
disease's OMIM number):
- a count of tests, labs and US labs, by kind:
  - **targeted:** at most 3 genes
  - **panel:** 4-500 genes
  - **exome/genome:** more than 500 genes
  - **biochemical:** an enzyme assay, analyte or protein method, for at most 2 conditions (e.g. "Acid
    Alpha-Glucosidase, Leukocytes" for Pompe disease, "Beta-Glucosidase Enzyme Analysis" for Gaucher disease).
    These are listed as confirming tests of their own; for metabolic diseases they usually come before the gene
    test.
  - **screening:** carrier, prenatal, newborn-screening or pharmacogenetic tests (by name). These are counted but
    never suggested.
- two example targeted tests: US labs first, then tests that both sequence and look for deletions/duplications, one
  per lab and name, with the GTR link
- the gene test's genes when HPO has none for the disease: the genes in at least a quarter of its GTR targeted tests
- registered panels for the leading diseases: panels covering at least two of them, ordered by the pre-test
  probability they cover together, then US labs, then fewer genes

GTR has no sensitivity per test, so the gene test's probabilities still use `--gene-sensitivity`.

### 3.5 Exome or genome

With developmental delay or intellectual disability (HP:0012758, HP:0001249) and age under 18 (or unknown), the
workup adds the ACMG recommendation of exome or genome sequencing as a first- or second-tier test (Manickam et al.,
*Genet Med* 2021).

### 3.6 Tests that tell the leading diseases apart, and findings the note may already state

- **Tell apart:** for the top 3 displayed diseases, each candidate test's information (bits) about which of them it
  is, from a positive or negative result, per burden tier. `P(T+|d)` is the noisy-OR over the findings `T` looks for in
  any of the three. The best 3 (at least 0.05 bits) are listed under `"separate"`, with each disease's chance of a
  positive result.
- **May already be answered:** a finding a test looks for that Stage 1 did not record, but whose HPO name or EXACT
  synonym is roughly in one sentence of the note or image description (at least 2 words and 60% of them, compared on
  their first 4 letters, including the first and last word). The sentence is shown as a hint; nothing is removed.

### 3.7 Checks

Every suggested test is re-derived before it is printed, as for the explanation lines:
- each finding it looks for is annotated to the disease with that frequency and reference
- the test is listed (MAxO, LOINC2HPO) as showing it
- a gene test's genes belong to the disease (HPO, or GTR's targeted tests)
- a GTR test shown is current, clinical, and lists the disease among its conditions

The run prints "n/m suggested tests verified".

## 4. Evaluation

Sources:
- SDBench (Nori et al., arXiv 2506.22405, 2025) scores sequential diagnosis by accuracy *and* test cost.
- MIMIC-CDM (Hager et al., *Nat Med* 2024) and CliBench (Ma et al., arXiv 2406.09923) score test orders against
  guidelines and against what clinicians ordered (LOINC-level F1).
- Phenomizer (Köhler et al., *AJHG* 2009) evaluated with simulated patients.

Without a clinician panel or order data yet, the evaluation simulates patients from Phenopacket Store:
- Cases are the first 400 evaluable cases (seed 13) for calibration and development, and the next 300 for the
  numbers below.
- Leakage control is on.
- Each case keeps half of its observed findings; the other half and all its excluded findings are hidden.
- "Doing" a test reveals the hidden findings it shows.

Two worlds:
- **Open:** a test shows only what the record says. This favours tests for frequent findings, because those are
  the ones authors record.
- **Imputed:** each finding the test looks for gets a result. It comes from the record if the record says;
  otherwise the finding is drawn as present with its frequency in the *true* disease (full profile, without
  leakage control). This is closer to a real test, which always returns a result, but it uses the knowledge base
  to make the patient.

Command: `workup-eval --sample 300` (about 1.5 minutes; GTR and MAxO from 2026-10-28 and 2025-05-31). Files: `workup_eval.csv`, `workup_diseases.csv`,
`workup_gene_yield.csv`, `calibration.json`.

### 4.1 Sequential testing: accuracy against burden (open world)

Three tests in a row. Each policy picks the next test:
- **workup:** best test for the top disease
- **workup-top3:** best summed over the top 3, weighted by probability
- **feature-eig:** the existing "next checks" feature, done with its cheapest test
- **most-frequent:** the cheapest test for the top disease's most frequent unknown finding
- **random:** a random test relevant to the top 5

| policy | Top-1 after 3 tests | Top-10 | mean P(truth) | mean burden |
|---|---|---|---|---|
| start (no test) | 0.207 | 0.437 | 0.054 | 0 |
| workup | 0.243 | 0.443 | 0.072 | 4.6 |
| workup-top3 | 0.247 | 0.427 | 0.070 | 4.4 |
| feature-eig | 0.247 | 0.430 | 0.070 | 5.9 |
| most-frequent | 0.240 | 0.447 | 0.070 | 6.2 |
| random | 0.227 | 0.430 | 0.057 | 7.6 |

Paired differences, workup minus each other policy, with 95% bootstrap intervals over cases:

| compared with | Top-1 | P(truth) | burden |
|---|---|---|---|
| feature-eig | −0.003 [−0.027, +0.020] | +0.002 [−0.002, +0.007] | −1.32 [−1.55, −1.09] |
| most-frequent | +0.003 [−0.027, +0.033] | +0.001 [−0.003, +0.006] | −1.55 [−1.76, −1.35] |
| random | +0.017 [−0.027, +0.057] | +0.015 [+0.008, +0.022] | −2.98 [−3.22, −2.75] |

The workup reaches the same accuracy as the existing next checks and the most-frequent heuristic at about a quarter
less burden. It is clearly better than random tests.

### 4.2 Per-disease workups (the feature itself)

Each of the 5 diseases worked up gets its own tests, up to a burden budget of 6. Then the disease's probability is
re-measured:
- If the disease is the diagnosis (108 cases), its probability should rise.
- If it is not, it should fall.

| world | tests chosen by | true disease: change | other diseases: change |
|---|---|---|---|
| open | workup | +0.047 [+0.032, +0.063] | −0.007 [−0.009, −0.005] |
| open | most-frequent | +0.057 [+0.042, +0.072] | −0.007 [−0.008, −0.004] |
| open | random | +0.016 [+0.006, +0.027] | −0.003 [−0.004, −0.002] |
| imputed | workup | +0.236 [+0.206, +0.265] | −0.039 [−0.041, −0.037] |
| imputed | most-frequent | +0.224 [+0.194, +0.254] | −0.038 [−0.040, −0.036] |
| imputed | random | +0.126 [+0.097, +0.155] | −0.031 [−0.033, −0.029] |

Paired, per case (true disease's rise minus the others' mean change), workup minus:

| world | most-frequent | random |
|---|---|---|
| open | −0.009 [−0.018, −0.002] | +0.036 [+0.021, +0.051] |
| imputed | +0.014 [+0.001, +0.026] | +0.119 [+0.091, +0.149] |

- The workup's tests clearly beat random tests in both worlds.
- Against "the most frequent findings first" it is slightly better when every looked-for finding gets a result.
  It is slightly worse when only recorded findings can be found, which is the bias of the open world.

### 4.3 Gene yield

Is the case's causal gene among the gene tests of the top-k diseases? (298 cases, all findings known.)

| k | gene found | median genes |
|---|---|---|
| 1 | 0.272 | 1 |
| 3 | 0.396 | 3 |
| 5 | 0.426 | 5 |
| 10 | 0.487 | 10 |
| best registered GTR panel for the top 5 | 0.493 | 116 (a panel exists for 251 of the 298 cases) |

- The top-k gene tests mostly follow the ranker's Top-k, plus allelic diseases that share a gene.
- The registered panels are much larger (median 116 genes), so they also find genes when the ranker missed the
  disease. When the true disease is among the top 5 (117 cases), the top-5 gene tests find its gene in 98% and the
  best registered panel in 75%: a panel chosen to cover two or more leading diseases does not always include the top
  one's gene.

### 4.4 Not done yet

- LOINC-level comparison with the labs actually ordered (CliBench-style), on credentialed MIMIC-IV, locally.
- A clinician review of workups for a sample of cases, with a rubric (as SDBench's judge), and a comparison with
  GeneReviews "establishing the diagnosis" sections.

## 5. Known gaps

- **MAxO is incomplete per term.** "Aortic root aneurysm" lists MRI of the chest but not echocardiography. The
  parent rule helps; adding missing pairs to MAxO (or a local supplement) would help more.
- **"Positive" on the test level means "any finding it looks for".** LR+ and LR− summarise that; the order uses each
  finding separately. Findings are treated as independent given the disease.
- **Burden is a 1-5 tier**, not a price or a risk estimate.
- **No age bands:** lab findings come from LOINC2HPO names, not from age-specific ranges.
- **Gene-test sensitivity is one number for every disease.** GTR does not give per-test sensitivity.
- **GTR lists what labs register, not how good a test is.** Conditions given only by NCBI MedGen ID, without an OMIM
  number, are not matched (about a third of GTR's condition rows). "Screening" is recognised by test name only.
- **Both simulation worlds are biased** (section 4), and phenopackets are not consecutive patients.
- **Probabilities are calibrated on leakage-controlled phenopackets.** For a new kind of input (MIMIC notes, MedPix),
  refit.

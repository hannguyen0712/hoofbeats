# Part 5: Data and evaluation

Every number here comes from a saved file in the repo; the file is named next to each table. To produce them again
(same seed, same commands, about 5 minutes on a laptop, no GPU):

```bash
bash scripts/run_part5.sh        # benchmark (500 and all cases), next-test loop, eval_part5.py, report
```

`eval_part5.py` adds what the built-in commands do not print: Top-5, a 95% interval on every metric, and the data
section. Intervals are bootstrap intervals that resample whole diseases (cases of one disease are not independent),
via `boot_ci` in `hoofbeats/report.py`; differences between conditions are paired per case. Ties count against the
true disease. Status: run on Nov 11 with `main` at `ee59354`; to be re-run at the code freeze.

## Data

Full counts: `results/part5/data.md` (`python eval_part5.py data`).

| Source | What it is | Size | Used for |
|---|---|---|---|
| HPO + `phenotype.hpoa` (2026-09-01) | disease-finding annotations with the papers behind them | 20,482 HPO terms; 8,465 OMIM diseases; 152,141 annotations from 14,063 references; 6,704 diseases with a known gene | the ranker's knowledge base |
| Phenopacket Store 0.1.27 | published patients already coded in HPO (present and excluded findings, diagnosis, gene, source paper) | 10,377 phenopackets; 10,345 usable (a present finding, disease in the knowledge base); 778 diseases; 697 genes; 1,751 papers; median 7 present and 7 absent findings per case | main ranking benchmark, next-test loop, Part 4 cases |
| MIMIC-IV demo 2.2 (open, ODbL) | de-identified hospital records | 100 patients; 275 admissions; 4,506 diagnosis rows (1,472 ICD codes); 107,727 lab events (498 lab items) | common-disease lab profiles (`learn-common`) |
| MedPix 2.0 | radiology teaching cases with images, history and the radiologist's findings | 671 cases; 2,050 images; 192 with a diagnosis mapped to the knowledge base; 340 with radiologist findings | text and image reading (Stage 1, Stage A) |
| PMC case reports (Part 1) | full text of Phenopacket Store papers under a Creative Commons licence | 236 cases from 203 papers (dev 64, test 172) | Stage 1 extraction on real text |
| Example case files (`examples/`) | hand-written notes with labs, vitals or image findings | 6 (5 with a note; 3 held out) | end-to-end demo cases |

**Leakage control.** Phenopacket Store patients come from the same papers the HPO annotations were built from. Before
scoring a case, every annotation backed *only* by that case's own paper is removed. For 2,023 of the 10,345 usable
cases (20%) this removes every annotation of the true disease: the knowledge base knew the disease only from the
case itself. Those cases cannot be ranked correctly without leakage, so results are reported on all cases and on the
**evaluable** cases (the true disease keeps annotations after the removal).

**Benchmark samples** (seed 13, at most 5 cases per disease so a few diseases cannot dominate): 500 cases (290
diseases, 357 evaluable) and all cases (3,006 cases, 778 diseases, 1,969 evaluable).

## Main result: rare-disease ranking (Phenopacket Store)

The shipped ranker (present and absent findings, leakage control on). Percentages, 95% intervals in brackets.
Source: `results/main/benchmark.csv`, `results/full/benchmark.csv` → `results/part5/main_results.md`.

| Cases | n | Top-1 | Top-3 | Top-5 | Top-10 | MRR |
|---|---|---|---|---|---|---|
| 500 sample, all | 500 | 19.2 (14.2–24.2) | 26.4 (20.7–32.1) | 29.8 (23.9–35.5) | 34.0 (27.7–40.0) | 24.4 (19.4–29.5) |
| 500 sample, evaluable | 357 | 26.9 (20.7–33.4) | 37.0 (30.1–44.0) | 41.7 (34.6–49.0) | 47.6 (40.4–55.5) | 34.2 (28.0–40.6) |
| **all cases** | 3,006 | **18.8** (16.4–21.2) | **25.3** (22.6–28.1) | **28.5** (25.6–31.4) | **32.9** (29.8–36.0) | 23.5 (21.0–26.0) |
| **all cases, evaluable** | 1,969 | **28.7** (25.4–32.0) | **38.6** (35.0–42.4) | **43.5** (39.8–47.3) | **50.3** (46.4–54.0) | 35.9 (32.6–39.3) |

The all-case run is the one to quote: same method, six times the cases, intervals about half as wide. The 500
sample is the one the team asked for and lies inside the all-case intervals.

## Ablations

**Ranker inputs** (all cases, evaluable, n = 1,969; Δ = paired change in Top-1 against the shipped ranker).
Source: `results/full/benchmark.csv`.

| Condition | Top-1 | Top-10 | Δ Top-1, pts |
|---|---|---|---|
| shipped ranker (present + absent) | 28.7 | 50.3 | – |
| **no leakage control** | 66.0 | 81.4 | **+37.3** (+33.7 to +41.1) |
| + causal gene reported | 51.5 | 70.4 | +22.8 (+20.4 to +25.4) |
| present findings only | 28.1 | 50.1 | −0.6 (−1.3 to +0.0) |
| every absent finding at full weight (old rule) | 24.9 | 44.7 | −3.8 (−5.1 to −2.4) |
| + onset | 28.0 | 50.1 | −0.7 (−1.6 to +0.3) |
| + sex | 28.8 | 50.3 | +0.1 (+0.0 to +0.3) |

- Without leakage control, Top-1 more than doubles (all cases: 18.8% → 70.3%). Benchmarks that skip this step
  overstate accuracy; this is one of the project's main findings.
- Counting every absent finding at full weight hurts; the current rule (weight 0.5, only for features the disease
  shows in ≥ 80% of cases) is about the same as ignoring absent findings (−0.6, interval touching 0). The rule was
  tuned on these same cases, so it is not a held-out result.
- Onset and sex do not change accuracy.

**Without the LLM reader** (Stage 1, PMC dev, 69 hand-checked terms). Source: `docs/STAGE1_PROGRESS.md`.
Hand-checked recall strict / lenient: rules alone 55.1% / 62.3%, rules + LLM reader 69.6% / 78.3%; precision with
every extra judged: rules alone 79.9% / 88.0%, with the LLM reader 75.2% / 81.7%. The PMC test-set run is still to do.

**Without images** (MedPix, 40 cases with images and a knowledge-base diagnosis, Qwen3.5-9B describing the images,
rules + numbers reader). Source: `results/compare/runs/20261110-095729_medpix-context_qwen3.5-9b.txt` and, for the
intervals, `results/compare/medpix_context_ctx-qwen3.5-9b.csv` (the later run without the history, which overwrote
the CSV).

| Stage 1 reads | Top-1 | Top-3 | Top-10 |
|---|---|---|---|
| history and exam text only | 2.5 | 7.5 | 12.5 |
| + image context | 10.0 | 12.5 | 17.5 |
| + image context, findings checked by the vision model | 7.5 | 10.0 | 15.0 |
| + radiologist's findings (upper bound) | 7.5 | 12.5 | 20.0 |

On the saved per-case run, the image context changes Top-1 by +7.5 points (+0.0 to +18.2): a direction, not a
result, with 40 cases. MedPix is hard for this ranker in general: on all 192 mapped cases, Top-10 is 9.9% from the
history and exam text and 16.1% with the radiologist's findings added (`results/medpix/runs/20261109-182258_medpix.txt`).

**LLM re-ranking** (Qwen3.5-9B, 100 evaluable cases, the same cases for all three). Source:
`results/compare/runs/20261110-115141_llm-bench_qwen3.5-9b.txt`.

| Ranking | Top-1 | Top-10 | Δ Top-1 vs ranker, pts |
|---|---|---|---|
| ranker alone | 23.0 | 45.0 | – |
| ranker + LLM re-rank of the top 10 | 24.0 | 45.0 | +1.0 (+0.0 to +3.3) |
| LLM alone | 7.0 | 12.0 | −16.0 (−28.7 to −3.9) |

Re-ranking moved 1 case up and none down; the LLM alone is far below the ranker.

**Rare vs common:** not run yet. The common list stays empty until common-disease profiles are in `data/kb`
(`learn-common`, Part 2).

## Next-test loop (Phenopacket Store)

300 evaluable cases, 5 questions each; a simulator answers from the record (yes, no, or unknown). Source:
`results/main/sequential.csv`.

| Question policy | Answered | Top-1 start → end | Top-10 start → end | Δ Top-1, pts | vs random, pts |
|---|---|---|---|---|---|
| information gain | 14.3% | 16.0 → 18.0 | 37.3 → 39.3 | +2.0 (+0.0 to +4.1) | +1.7 (−0.3 to +3.8) |
| most frequent feature | 14.0% | 16.0 → 16.7 | 37.3 → 37.7 | +0.7 (+0.0 to +1.7) | +0.3 (−0.7 to +1.6) |
| random | 3.4% | 16.0 → 16.3 | 37.3 → 37.7 | +0.3 (+0.0 to +1.1) | – |

Information-gain questions help a little, but the difference from random questions is not significant: only 14% of
questions can be answered from a published record. Part 3's `workup-eval` (real tests from MAxO, LOINC2HPO and GTR)
is the fuller evaluation of test suggestions.

## Numbers from the other parts (already done; sources)

| Part | Result | Source |
|---|---|---|
| 1 | Text extraction, strict F1: dev 0.44, held-out 0.39 (first blind run; 0.41 now) | `docs/STAGE1_PROGRESS.md` §3 |
| 1 | PMC dev, hand-checked recall 55.1% (rules) → 69.6% (+ LLM reader); vocabulary-visible recall 90.4% | `docs/STAGE1_PROGRESS.md` |
| 4 | Model explanations kept by the checks: 325 of 326 sentences (99.7%), prompt v3, 30 held-out cases | `PART4.md` |
| 4 | Adversarial set: 100% of 671 wrong sentences removed, 98.4% of 377 correct ones kept (built from the error kinds the checks target) | `PART4.md` |
| 4 | Hand labels: pilot, one annotator; the second annotator and Cohen's kappa are pending | `PART4.md` |

## Not done yet

- **Rare vs common** lists: needs the common-disease profiles (Part 2).
- **Stage 1 on the PMC test set**: ready, waiting on the re-freeze decision (Part 1).
- **Second labeller** for Part 4 (Cohen's kappa).
- **External baselines** (LIRICAL, RareBench, DeepRare): not set up.
- **Final re-run at the code freeze**: `bash scripts/run_part5.sh`.

## Caveats for the report

- Leakage control removes an annotation only when the case's own paper is its *only* source. An annotation that
  another paper also supports is kept, and its frequency may still count the test case's patients. The leakage
  numbers above are therefore a lower bound on the leakage effect.
- The absent-finding rule and the calibration were chosen on Phenopacket Store cases; there is no separate frozen
  test split for the ranker.
- `report` lists a Qwen3.6-27B re-rank row from a run that stopped after 1 case; leave it out of the report.
- LLMs may have read these published cases in training, so LLM gains are upper bounds.

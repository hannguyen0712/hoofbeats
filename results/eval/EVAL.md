# Evaluation

2026-11-24 11:00, HPO hp/releases/2026-09-01. Dev 842 cases / 223 diseases (534 evaluable); test 2164 cases / 555 diseases (1435 evaluable). No disease or paper is in both.

| result | value (95% CI) |
|---|---|
| Ranking (test, evaluable) | Top-1 29.1 (25.1 to 33.0), Top-10 50.2 (45.8 to 54.6) |
| Without leakage control | Top-1 65.5 (61.2 to 69.5) |
| Resnik baseline | Top-1 25.3 (21.6 to 29.3) |
| Learned weights (fitted on dev) | +1.6 (+0.5 to +2.7) pts Top-1 |
| Calibration (test) | ECE 0.077 |
| AUROC, probability / score gap | 0.754 / 0.768 |
| Truth outside the top 50 | 35.3% of evaluable test cases |
| Robustness | Top-1 20.6 (17.5 to 23.8) with half the findings, 14.6 (11.8 to 17.7) with 3 unrelated added |
| Grounding, held-out run | unsupported shown 6.7% → 0.3% |
| Grounding, fresh held-out run | unsupported shown 4.9% → 0.0% |

## Tuning on dev

Dev 842 cases / 223 diseases (534 evaluable); test 2164 cases / 555 diseases (1435 evaluable). No disease or paper is in both.

Absent findings (dev, evaluable, %):

| weight | min disease frequency | Top-1 | MRR |
|---|---|---|---|
| 0.0 | any | 27.2 | 35.5 |
| 0.25 | 0.0 | 26.8 | 34.7 |
| 0.25 | 0.5 | 27.5 | 35.7 |
| 0.25 | 0.8 | 27.2 | 35.6 |
| 0.25 | 0.9 | 27.5 | 35.8 |
| 0.5 | 0.0 | 26.2 | 33.7 |
| 0.5 | 0.5 | 27.5 | 35.2 |
| 0.5 | 0.8 | 27.5 | 35.6 |
| 0.5 | 0.9 | 27.2 | 35.6 |
| 1.0 | 0.0 | 24.5 | 31.9 |
| 1.0 | 0.5 | 27.3 | 34.3 |
| 1.0 | 0.8 | 27.9 | 35.6 |
| 1.0 | 0.9 | 27.0 | 35.4 |

Best: (1.0, 0.8) (one SE = 1.9 pts). Rule: keep the shipped (0.5, 0.8) unless another setting beats it by more than one SE. **Used: weight 0.5, minimum frequency 0.8.** Calibration: temperature 2.31, truth in the top 20 34.7%. Score-gap bins: 0.35, 1.17, 2.98. Learned weights (l2 0.01): present_exact 0.17, present_partial 0.13, absent_hallmark 0.13, absent_excluded 0.08

## Disease ranking

Ties count against the true disease. Evaluable: the true disease keeps annotations after leakage control.

| cases | n | Top-1 | Top-3 | Top-5 | Top-10 | Top-50 | Top-100 | MRR |
|---|---|---|---|---|---|---|---|---|
| test, all | 2164 | 19.3 (16.5 to 22.3) | 25.5 (22.3 to 28.9) | 28.3 (25.0 to 31.9) | 33.3 (29.7 to 37.0) | 42.9 (38.8 to 46.8) | 46.9 (42.8 to 50.9) | 23.9 (21.0 to 27.0) |
| test, evaluable | 1435 | 29.1 (25.1 to 33.0) | 38.4 (33.9 to 42.7) | 42.7 (38.3 to 47.1) | 50.2 (45.8 to 54.6) | 64.7 (60.6 to 69.1) | 70.7 (66.8 to 74.9) | 36.0 (32.0 to 39.9) |
| dev, evaluable (tuned here) | 534 | 27.5 (21.3 to 33.9) | 39.3 (32.2 to 46.5) | 45.7 (38.6 to 53.0) | 50.6 (43.3 to 58.1) | 65.0 (58.2 to 71.5) | 72.3 (65.7 to 78.5) | 35.6 (29.3 to 41.8) |

Ablations, options and baselines (test, evaluable, paired on the same cases; baselines use present findings only):

| condition | Top-1 | Top-10 | MRR | Δ Top-1 vs main, pts |
|---|---|---|---|---|
| main | 29.1 (25.1 to 33.0) | 50.2 (45.8 to 54.6) | 36.0 (32.0 to 39.9) | – |
| present findings only | 28.4 (24.5 to 32.3) | 49.7 (45.2 to 54.1) | 35.7 (31.7 to 39.7) | -0.7 (-1.5 to +0.1) |
| every absent finding at weight 1 (old rule) | 25.0 (21.5 to 28.6) | 44.6 (40.5 to 48.9) | 31.8 (28.1 to 35.5) | -4.1 (-5.6 to -2.6) |
| no leakage control | 65.5 (61.2 to 69.5) | 80.3 (76.8 to 83.7) | 70.9 (66.9 to 74.6) | +36.4 (+32.1 to +40.6) |
| + causal gene | 51.7 (47.6 to 55.9) | 69.8 (65.9 to 73.9) | 58.0 (54.1 to 61.9) | +22.6 (+19.7 to +25.7) |
| + age of onset | 28.4 (24.5 to 32.3) | 49.9 (45.4 to 54.4) | 35.7 (31.8 to 39.5) | -0.8 (-1.9 to +0.3) |
| + sex | 29.3 (25.2 to 33.1) | 50.2 (45.8 to 54.6) | 36.2 (32.1 to 40.0) | +0.1 (+0.0 to +0.4) |
| learned weights (dev) | 30.7 (26.7 to 34.9) | 52.2 (47.7 to 56.8) | 37.9 (33.8 to 41.9) | +1.6 (+0.5 to +2.7) |
| baseline: random order | 0.0 (0.0 to 0.0) | 0.1 (0.0 to 0.4) | 0.1 (0.1 to 0.1) | -29.1 (-33.0 to -25.1) |
| baseline: Resnik similarity (Phenomizer-style) | 25.3 (21.6 to 29.3) | 46.2 (41.6 to 50.7) | 32.9 (29.0 to 36.8) | -3.8 (-5.8 to -1.9) |

## Uncertainty

All 2164 test cases. Top disease's calibrated probability: mean 11.6% vs Top-1 19.3%, ECE 0.077.

| predicted | cases | mean predicted | observed Top-1 |
|---|---|---|---|
| 0–10% | 1220 | 5.3 | 10.0 (7.9 to 12.3) |
| 10–20% | 537 | 14.0 | 20.3 (15.8 to 25.2) |
| 20–30% | 267 | 24.6 | 31.8 (25.2 to 39.1) |
| 30–40% | 140 | 33.3 | 72.9 (63.0 to 81.1) |

Top-1 by the score gap between the top two diseases (bins: dev quartiles):

| gap | cases | Top-1 |
|---|---|---|
| quartile 1 | 567 | 6.2 (3.9 to 9.0) |
| quartile 2 | 543 | 10.5 (7.6 to 13.8) |
| quartile 3 | 544 | 18.6 (14.5 to 23.0) |
| quartile 4 | 510 | 44.1 (37.4 to 50.6) |

AUROC for a correct Top-1: probability 0.754, score gap 0.768. Top-1 when answering only the most confident cases:

| coverage | by probability | by score gap |
|---|---|---|
| 25% | 40.7 (34.0 to 47.1) | 42.7 (36.0 to 48.9) |
| 50% | 29.4 (24.9 to 34.1) | 30.8 (26.1 to 35.5) |
| 75% | 24.1 (20.6 to 27.5) | 23.6 (20.0 to 27.3) |
| 100% | 19.3 (16.5 to 22.3) | 19.3 (16.5 to 22.3) |

## Error analysis

Test, evaluable cases.

By present findings:

|  | cases | Top-1 | Top-10 |
|---|---|---|---|
| ≤ 3 | 203 | 26.1 (17.9 to 34.8) | 50.7 (41.0 to 60.8) |
| 3–7 | 449 | 31.0 (25.4 to 36.6) | 56.3 (49.7 to 62.6) |
| 7–15 | 571 | 28.5 (23.2 to 34.1) | 48.9 (42.4 to 54.6) |
| > 15 | 212 | 29.7 (20.7 to 38.3) | 40.1 (30.5 to 49.2) |

By true disease's annotations after leakage control:

|  | cases | Top-1 | Top-10 |
|---|---|---|---|
| ≤ 5 | 80 | 13.8 (0.0 to 31.3) | 25.0 (7.8 to 44.6) |
| 5–20 | 526 | 16.2 (11.4 to 21.5) | 35.4 (28.8 to 42.5) |
| 20–50 | 572 | 36.0 (29.7 to 42.3) | 59.3 (52.9 to 65.9) |
| > 50 | 257 | 45.1 (36.1 to 53.9) | 68.1 (59.0 to 76.2) |

By share of findings annotated to the true disease itself:

|  | cases | Top-1 | Top-10 |
|---|---|---|---|
| ≤ 0 | 127 | 0.0 (0.0 to 0.0) | 0.8 (0.0 to 2.6) |
| 0–0.25 | 209 | 1.0 (0.0 to 2.5) | 7.7 (4.1 to 12.1) |
| 0.25–0.5 | 410 | 12.4 (8.7 to 16.4) | 36.6 (30.4 to 43.1) |
| > 0.5 | 689 | 53.0 (47.4 to 58.6) | 80.3 (75.7 to 84.3) |

By inheritance of the true disease (groups of 20+ cases):

|  | cases | Top-1 | Top-10 |
|---|---|---|---|
| autosomal recessive | 492 | 32.3 (26.3 to 38.5) | 53.5 (46.5 to 60.4) |
| not annotated | 462 | 25.5 (19.6 to 32.4) | 42.9 (35.9 to 50.7) |
| autosomal dominant | 390 | 26.4 (20.3 to 33.1) | 51.0 (43.6 to 59.1) |
| autosomal dominant/autosomal recessive | 30 | 26.7 (7.1 to 50.0) | 53.3 (24.1 to 81.0) |
| X-linked recessive | 26 | 57.7 (25.9 to 92.3) | 69.2 (37.5 to 94.1) |
| X-linked dominant | 22 | 40.9 (10.5 to 66.7) | 59.1 (22.2 to 85.2) |

Diseases never ranked first (3+ test cases, worst median rank):

| disease | cases | median rank |
|---|---|---|
| Encephalopathy, neonatal severe, with lactic acidosis and brain abnormalities (OMIM:617668) | 3 | 5719 |
| Muscular dystrophy, limb-girdle, autosomal recessive 21 (OMIM:617232) | 4 | 5360 |
| Intellectual developmental disorder with dysmorphic facies and ptosis (OMIM:617333) | 5 | 5142 |
| Skraban-Deardorff syndrome (OMIM:617616) | 5 | 4944 |
| Cutis laxa, autosomal recessive, type IID (OMIM:617403) | 3 | 4481 |
| Retinitis pigmentosa with or without skeletal anomalies (OMIM:250410) | 5 | 4378 |
| Short stature, optic nerve atrophy, and pelger-huet anomaly (OMIM:614800) | 5 | 4136 |
| Joubert syndrome 10 (OMIM:300804) | 5 | 3562 |
| Al Kaissi syndrome (OMIM:617694) | 5 | 3421 |
| Amyloidosis, Finnish type (OMIM:105120) | 5 | 2878 |

Truth outside the top 50 in 507 of 1435 cases (35.3%); there a median 21% of findings are annotated to the true disease itself, vs 75% when it is in the top 10. When the top disease is wrong, it shares a gene with the truth in 3.2% of cases.

## Robustness to noisy patient data

Test, evaluable cases. Unrelated findings are random HPO terms that are neither ancestors nor descendants of the case's findings; broader means a direct parent term.

| condition | Top-1 | Top-10 | MRR | Δ Top-1 vs clean, pts |
|---|---|---|---|---|
| clean | 29.1 (25.1 to 33.0) | 50.2 (45.8 to 54.6) | 36.0 (32.0 to 39.9) | – |
| keep 50% of present findings | 20.6 (17.5 to 23.8) | 40.8 (36.8 to 45.0) | 27.5 (24.4 to 31.0) | -8.6 (-10.7 to -6.4) |
| keep 25% | 13.5 (11.2 to 16.0) | 29.4 (26.2 to 33.0) | 18.9 (16.4 to 21.5) | -15.6 (-18.4 to -12.6) |
| +1 unrelated finding | 22.2 (18.5 to 25.9) | 40.4 (36.0 to 44.7) | 28.7 (24.9 to 32.6) | -7.0 (-8.7 to -5.3) |
| +3 unrelated findings | 14.6 (11.8 to 17.7) | 28.6 (24.5 to 32.6) | 19.7 (16.5 to 23.0) | -14.5 (-17.1 to -12.0) |
| every finding one level broader | 20.7 (17.2 to 24.1) | 41.3 (36.8 to 45.5) | 27.7 (24.0 to 31.4) | -8.4 (-10.6 to -6.5) |
| keep 50% + 3 unrelated + half broader | 6.2 (4.5 to 8.2) | 14.1 (11.4 to 17.0) | 9.1 (7.1 to 11.3) | -22.9 (-26.1 to -19.6) |

## Next-question loop

300 test, evaluable cases. Half of each case's present findings are hidden; each round a policy asks one HPO term, answered from the hidden record. Questions answered: eig 14.3%, frequent 13.5%, random 3.9%.

| condition | Top-1 | Top-10 | MRR | Δ Top-1 vs after 5 questions, random, pts |
|---|---|---|---|---|
| start (half the findings) | 17.3 (12.0 to 23.1) | 36.3 (29.3 to 43.5) | 24.4 (18.8 to 30.3) | -0.3 (-1.0 to +0.0) |
| after 5 questions, eig | 18.3 (12.7 to 24.6) | 38.0 (30.9 to 45.0) | 24.9 (19.3 to 31.2) | +0.7 (-1.5 to +3.0) |
| after 5 questions, frequent | 17.7 (12.3 to 23.2) | 36.0 (28.9 to 43.2) | 24.7 (19.2 to 30.5) | +0.0 (-1.3 to +1.4) |
| after 5 questions, random | 17.7 (12.3 to 23.4) | 36.7 (29.6 to 43.8) | 24.5 (18.9 to 30.4) | – |

## End to end

Not run: none of the PMC dev texts is cached in data/pmc/ (`python -m hoofbeats pmc-build` fetches them).

## Evidence grounding

**Held-out run** (314 sentences)

|  | sentences shown | unsupported among shown, % | supported removed |
|---|---|---|---|
| no checks | 314 | 6.7 | – |
| original checks (V1–V4) | 313 | 6.4 (3.9 to 9.7) | 0 |
| current checks (V1–V6) | 293 | 0.3 (0.0 to 1.9) | 1 |

Two annotators, 314 sentences: agreement 99.4%, Cohen's kappa 0.95.

**Fresh held-out run** (325 sentences)

|  | sentences shown | unsupported among shown, % | supported removed |
|---|---|---|---|
| no checks | 325 | 4.9 | – |
| original checks (V1–V4) | 324 | 4.6 (2.6 to 7.5) | 0 |
| current checks (V1–V6) | 308 | 0.0 (0.0 to 1.2) | 1 |

# Part 4: Evidence grounding and clinician review

Every sentence shown to the clinician cites numbered evidence and is checked before it is shown.

## How it works

1. **Evidence pack** (`grounding.build_pack`), per disease:
   - **P items**: the patient's findings, with the words of the note they were read from.
   - **K items**: knowledge-base facts that support the disease (each re-checked with `verify_claim`), frequent
     features of the disease that the record does not mention, and the findings that separate it from the runner-up.
   - The case's own source paper is removed from every citation list (leakage control); a fact is kept only if
     other papers still support it.
2. **Explanation**: a local model (Qwen2.5-7B-Instruct on Sol) writes sentences that must end with the IDs they use,
   e.g. `[P2, K2]`. Without a model, a template writes them straight from the pack.
3. **Checks** on every sentence (`grounding.check_sentence`; the original V1-V4 stay available as
   `check_sentence_basic` / `strict=False` for comparison):

   | Check | Removes a sentence when |
   | --- | --- |
   | V1 | it cites nothing, or cites an ID that is not in this disease's evidence |
   | V5 | a cited knowledge item belongs to another section (e.g. evidence against the disease under "What fits") |
   | V2 | a cited finding is no longer in the record; an absent finding is not stated as absent *for that finding* (the negation must apply to its name); a present finding is negated; a family-history or uncertain finding is stated as the patient's; a finding the record does not mention is stated as absent or present instead of "not recorded" |
   | V3 | it does not name (whole words) the finding of an item it cites |
   | V4 | a number does not match by kind: a percentage must be a frequency in the cited items, a number after LR / likelihood ratio / against / vs must be one of their LRs, an "A against B" pair must be in that order, any other number must appear in them as a plain number |
   | V6 | it claims more than the evidence: certainty or causal wording (proves, confirms, diagnostic, causes, due to, because, ...), a disease other than this one and its runner-up, or a finding no cited item covers |

   Kept sentences are shown as **Evidence-linked**, tagged with where their evidence comes from: patient record,
   HPO annotation, ranker calculation (or PubMed passage, once those are added). "Evidence-linked" says the
   sentence cites evidence that names the same finding and numbers; it does not prove the reasoning is right.
4. **Review screen** (`review_app.py`): top 3 rare and top 3 common diseases; each sentence has a *Why?* button that
   shows the cited evidence, with the exact words highlighted in the note and links to the PubMed sources; the full
   note with all evidence highlighted; the suggested tests from Part 3; and a button to remove a wrongly read finding,
   which re-scores and re-explains the case.

## Run it

```bash
python demo_part4.py examples/marfan_note.json      # Stage 1 + ranking + Part 4; writes results/part4_demo.json
streamlit run review_app.py                          # the review screen
python eval_part4.py 200                             # checks against planted errors (no model needed; --basic: old checks)
python adversarial_part4.py 200                      # adversarial test set: build part4_adversarial.jsonl and score it
python recheck_part4.py results/part4_llm_sentences_v3_test.jsonl   # saved model sentences, old vs new checks
PROMPT_VERSION=v3 python llm_part4.py part4_cases_test.txt    # model explanations (needs the model server)
```

On Sol, `run_demo.sbatch` and `run_s6.sbatch` start the model server and run these; the replies are saved in
`results/llm_cache/`, and `LLM_OFFLINE=1` replays them on a laptop.

## Results

**Support or opposition fix (Nov 11).** Every claim has a patient term, so the old rule ("for" when the claim has a
patient term) put evidence *against* the disease under "What fits" and never filled "What does not fit". On 200
random cases this affected 11 of 1,208 knowledge items in 9 cases (e.g. "Lacks 'Joint hypermobility', which ... shows
in ~80% of cases. LR 0.45" listed as supporting). The role now comes from the sign of the finding's contribution to
the score, and V5 checks each sentence's section.

**Checks against planted errors** (200 random Phenopacket Store cases, leakage control on, main on Nov 11):

| Planted error | Original checks (V1-V4) | New checks (V1-V6) |
| --- | --- | --- |
| No citation | 100% | 100% (2,051/2,051) |
| Unknown ID | 100% | 100% (2,051/2,051) |
| Wrong number | 99.7% | 99.9% (2,049/2,051) |
| Wrong citation | 100.0% | 100% (2,049/2,049) |
| Present finding stated as absent | 100% | 100% (1,439/1,439) |
| *Correct (template) sentences kept* | 2,051/2,051 | 2,051/2,051 |

**Adversarial test set** (`part4_adversarial.jsonl`, 1,048 sentences built from the evidence packs of 200 cases plus
6 hand-written paraphrases of the Marfan note; at most 60 per category; each judged against its cited evidence):

| Should be | Category | n | Original checks | New checks |
| --- | --- | --- | --- | --- |
| kept | paraphrase: supports / synonym / against / not recorded / vs runner-up | 251 | 100% | 100% |
| kept | paraphrase: family-history / uncertain finding stated as such | 120 | 100% | 100% |
| kept | paraphrase in other words, no HPO name (hand-written) | 6 | 0% | 0% |
| removed | overclaim: certainty ("proves", "confirms", ...) | 60 | 0% | 100% |
| removed | overclaim: causal ("causes", "due to", "because of") | 60 | 0% | 100% |
| removed | wrong disease | 60 | 53% | 100% |
| removed | extra unsupported clause | 60 | 5% | 100% |
| removed | negation not tied to the absent finding | 11 | 0% | 100% |
| removed | not recorded stated as absent / as present | 120 | 0% | 100% |
| removed | family-history / uncertain finding stated as the patient's | 120 | 0% | 100% |
| removed | number: LR given as a frequency | 60 | 0% | 100% |
| removed | number: LRs swapped in a comparison | 60 | 0% | 100% |
| removed | wrong section | 60 | 0% | 100% |
| | **correct sentences kept** | 377 | 98.4% | 98.4% |
| | **wrong sentences removed** | 671 | 5.2% | 100% |

This set was built from the error kinds the new checks target, so it shows that each check does what it is meant
to; it is not an estimate of accuracy on real model output. The hand-written paraphrases show the main cost of
name matching: a correct sentence in other words ("his feet are flat") is still removed.

**Model-written explanations** (Qwen2.5-7B-Instruct; prompts tuned on 31 dev cases, then run once on 30 test cases;
scored with the original V1-V4, before the Nov 9 scoring change; "kept" is an acceptance rate, not accuracy;
`recheck_part4.py` re-scores the saved sentences with the new checks):

| Explanation | Cases | Written | Kept by the checks | Removed by |
| --- | --- | --- | --- | --- |
| Template (no model) | 31 dev | 324 | 324 (100%) | |
| Prompt v1 | 31 dev | 264 | 35 (13.3%) | V3 205, V1 16, V2 8 |
| Prompt v2 | 31 dev | 308 | 294 (95.5%) | V3 9, V4 3, V1 2 |
| Prompt v3 | 31 dev | 325 | 318 (97.8%) | V4 6, V1 1 |
| **Prompt v3** | **30 test** | **326** | **325 (99.7%)** | V3 1 |

- v1 failed on format: the model wrote only the IDs.
- v2 added a worked example and marked which section each item belongs to.
- v3 also tells the model which patient item to cite with each knowledge item.

Examples of errors the checks removed:
- a sentence about hyperreflexia citing an item about axial hypotonia (V3);
- six comparisons with the runner-up whose likelihood ratio of 1.00 was made up (V4);
- a number copied from the prompt's own example (V4).

**Hand labels, pilot** (150 sentences, one annotator, blind to the check results): of the 8 sentences the
original checks removed, 7 were unsupported; 0 of the 142 kept sentences were unsupported. **Provenance (checked Nov 11):** the pilot sheet's sentences come from 32 cases: 30 held-out test cases and 2 development cases, so the pilot is mostly, not purely, held-out. The final evaluation below uses the held-out run only (`part4_llm_sentences_v3_test.jsonl`), and `make_label_sheet.py` now needs the file named and prints where its sentences come from.

**Human evaluation (next):** every sentence of the held-out run, labelled by two annotators independently, then
adjudicated:
1. `python recheck_part4.py results/part4_llm_sentences_v3_test.jsonl`
2. `python make_label_sheet.py results/part4_llm_sentences_v3_test.recheck.jsonl` (prints where the sentences come
   from; the key holds both checkers' decisions)
3. each annotator fills their own copy: supported (1/0) and, if 0, the error type
4. `python score_labels.py sheet_A.csv sheet_B.csv results/labels/label_key_v3_test.recheck.csv [final.csv]` reports
   agreement and Cohen's kappa; the unsupported rate before checking and after each checker; unsupported sentences
   accepted; supported sentences rejected; and results by error type.

**Broader terms (Nov 15).** When a finding is matched through a broader annotation, the knowledge item now names the broader term the frequency belongs to, and V3 removes a sentence that drops it (21 of the 181 labelled test sentences did). "Not recorded" items now skip any finding the record has in a broader or narrower form. Adversarial set rebuilt with a "broader term dropped" category (1,168 sentences): correct sentences kept 431/437 (98.6%), wrong sentences removed 731/731 (100%; original checks 4.1%). Planted-error results unchanged. The one-annotator labels above were made on the earlier version.

## Limitations

- "Evidence-linked" means the sentence names the cited finding and its numbers and adds nothing the checks recognise
  as extra; meaning beyond that (e.g. a subtle misreading in words the checks don't look for) is not verified. A local
  entailment (NLI) model is the next step: `adversarial_part4.py --score --nli <model>` evaluates one on the same set
  (needs `transformers`; run on Sol or a laptop).
- V3 matches finding names and their HPO synonyms, so a correct paraphrase in other words is removed.
- V6's word lists (certainty, causal, other diseases, other findings) are fixed lists: wording outside them passes.
- The model results above were measured with the original checks and the scoring before Nov 9.
- The hand labels so far come from one annotator, and their source file is to be confirmed.
- Evidence comes from the patient record, the HPO annotations and the ranker's own calculation; PubMed passages
  (`pubmed.py`) are not in the evidence pack yet.
- The common-disease list stays empty until common-disease profiles are loaded (Part 2).

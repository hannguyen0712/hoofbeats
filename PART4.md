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
3. **Checks** on every sentence (`grounding.check_sentence`):

   | Check | Removes a sentence when |
   | --- | --- |
   | V1 | it cites nothing, or cites an ID that is not in this disease's evidence |
   | V2 | a cited finding is no longer in the record, an absent finding is stated as present, or a present one as absent |
   | V3 | it does not name the finding of an item it cites |
   | V4 | a number in it is not in the cited evidence |

   Sentences that pass are graded **Verified** (patient and knowledge base), **KB only** or **Patient only**.
4. **Review screen** (`review_app.py`): top 3 rare and top 3 common diseases; each sentence has a *Why?* button that
   shows the cited evidence, with the exact words highlighted in the note and links to the PubMed sources; the full
   note with all evidence highlighted; the suggested tests from Part 3; and a button to remove a wrongly read finding,
   which re-scores and re-explains the case.

## Run it

```bash
python demo_part4.py examples/marfan_note.json      # Stage 1 + ranking + Part 4; writes results/part4_demo.json
streamlit run review_app.py                          # the review screen
python eval_part4.py 200                             # checks against planted errors (no model needed)
PROMPT_VERSION=v3 python llm_part4.py part4_cases_test.txt    # model explanations (needs the model server)
```

On Sol, `run_demo.sbatch` and `run_s6.sbatch` start the model server and run these; the replies are saved in
`results/llm_cache/`, and `LLM_OFFLINE=1` replays them on a laptop.

## Results

**Checks against planted errors** (200 random Phenopacket Store cases, leakage control on, 2,121 sentences):

| Planted error | Caught |
| --- | --- |
| No citation | 100% |
| Unknown ID | 100% |
| Wrong number | 99.6% |
| Wrong citation | 99.9% |
| Present finding stated as absent | 100% |

All 2,121 template sentences pass. Removing the case's own paper from citation lists kept all 1,260
knowledge-base claims (174 had been dropped before because their citations included that paper).

**Model-written explanations** (Qwen2.5-7B-Instruct; prompts tuned on 31 dev cases, then run once on 30 test cases):

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

## Limitations

- "Kept" means the sentence passed the checks, not that it is correct; hand labels of 150 sentences (two annotators,
  Cohen's kappa) measure this.
- V4 checks which numbers appear, not where: a changed number that happens to appear elsewhere in the cited
  evidence is not caught.
- V3 matches finding names and their HPO synonyms; a correct paraphrase with other words is removed.
- The common-disease list stays empty until common-disease profiles are loaded (Part 2).
- Literature (PubMed) evidence items will be added when `pubmed.py` is available.

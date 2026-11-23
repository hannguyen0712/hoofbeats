# Part 4: Evidence grounding and clinician review

Every sentence shown to the clinician cites numbered evidence and is checked before it is shown.

## How it works

1. **Evidence pack** (`grounding.build_pack`), per disease:
   - **P items**: the patient's findings, with the words of the note they were read from.
   - **K items**: knowledge-base facts that support the disease (each re-checked with `verify_claim`), frequent
     features of the disease that the record does not mention, and the findings that separate it from the runner-up.
     When a finding is matched through a broader HPO term, the item names that broader term, since the frequency
     belongs to it. A feature counts as "not recorded" only if the record has nothing broader or narrower than it.
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
   | V3 | it does not name (whole words) the finding of an item it cites, or it drops the broader term a cited frequency belongs to |
   | V4 | a number does not match by kind: a percentage must be a frequency in the cited items, a number after LR / likelihood ratio / against / vs must be one of their LRs, an "A against B" pair must be in that order, any other number must appear in them as a plain number |
   | V6 | it claims more than the evidence: certainty or causal wording (proves, confirms, diagnostic, causes, due to, because, ...), a disease other than this one and its runner-up, or a finding no cited item covers |

   Kept sentences are shown as **Evidence-linked**, tagged with where their evidence comes from: patient record,
   HPO annotation, ranker calculation (or PubMed passage, once those are added). "Evidence-linked" says the
   sentence cites evidence that names the same finding and numbers; it does not prove the reasoning is right.
4. **Review screen** (`review_app.py`): top 3 rare and top 3 common diseases; clicking a finding, or the Evidence button on a
   sentence, shows the cited evidence, with the exact words highlighted in the note and links to the PubMed sources; the full
   note with all evidence highlighted; the suggested tests from Part 3; and a button to remove a wrongly read finding,
   which re-scores and re-explains the case.

## Run it

```bash
python demo_part4.py examples/marfan_note.json      # Stage 1 + ranking + Part 4; writes results/part4_demo.json
streamlit run review_app.py                          # the review screen
LLM_OFFLINE=1 streamlit run review_app.py            # replay saved model replies (laptop)
COMMON_PROFILES=data/kb/common_profiles.json LLM_OFFLINE=1 streamlit run review_app.py   # + top 3 common diseases
python walkthrough_part4.py examples/marfan_note.json   # one case, stage by stage (Enter for the next)
python eval_part4.py 200                             # checks against planted errors (no model needed; --basic: old checks)
python adversarial_part4.py 200                      # adversarial test set: build part4_adversarial.jsonl and score it
python recheck_part4.py results/part4_llm_sentences_v3_test.jsonl   # saved model sentences, old vs new checks
PROMPT_VERSION=v3 python llm_part4.py part4_cases_test.txt    # model explanations (needs the model server)
```

Common diseases appear only with `COMMON_PROFILES` set (profiles learned from MIMIC-IV demo lab values, so a record
without lab results has no common match). Images uploaded in the review screen are described live by a local vision
model (`VISION_MODEL`, else the main model if it reads images, else MedGemma on a local Ollama); without one running,
the screen says the images could not be described.

On Sol, `run_demo.sbatch` and `run_s6.sbatch` start the model server and run these; the replies are saved in
`results/llm_cache/`, and `LLM_OFFLINE=1` replays them on a laptop. The held-out run is
`sbatch --export=ALL,PROMPT_VERSION=v3,CASES=part4_cases_test.txt run_s6.sbatch` (the script's defaults are prompt v1
and the dev cases). `llm_part4.py`, `demo_part4.py` and `walkthrough_part4.py` use the same reply limit
(`LLM_MAX_TOKENS`, default 1,500), because it is part of the reply cache key.

## Results

**Support or opposition fix (Nov 11).** Every claim has a patient term, so the old rule ("for" when the claim has a
patient term) put evidence *against* the disease under "What fits" and never filled "What does not fit". On 200
random cases this affected 11 of 1,208 knowledge items in 9 cases (e.g. "Lacks 'Joint hypermobility', which ... shows
in ~80% of cases. LR 0.45" listed as supporting). The role now comes from the sign of the finding's contribution to
the score, and V5 checks each sentence's section.

**Checks against planted errors** (200 random Phenopacket Store cases, leakage control on; unchanged after the Nov 15
changes):

| Planted error | Original checks (V1-V4) | New checks (V1-V6) |
| --- | --- | --- |
| No citation | 100% | 100% (2,051/2,051) |
| Unknown ID | 100% | 100% (2,051/2,051) |
| Wrong number | 99.7% | 99.9% (2,049/2,051) |
| Wrong citation | 100.0% | 100% (2,049/2,049) |
| Present finding stated as absent | 100% | 100% (1,439/1,439) |
| *Correct (template) sentences kept* | 2,051/2,051 | 2,051/2,051 |

The two missed wrong numbers change the subtype number inside a disease's name ("autosomal dominant, 1" to "14").

**Adversarial test set** (`part4_adversarial.jsonl`, Nov 15 version: 1,168 sentences built from the evidence packs of
200 cases plus 6 hand-written paraphrases of the Marfan note; at most 60 per category; each judged against its cited
evidence):

| Should be | Category | n | Original checks | New checks |
| --- | --- | --- | --- | --- |
| kept | paraphrase: supports / synonym / against / not recorded / vs runner-up | 251 | 100% | 100% |
| kept | paraphrase that keeps the broader term | 60 | 100% | 100% |
| kept | paraphrase: family-history / uncertain finding stated as such | 120 | 100% | 100% |
| kept | paraphrase in other words, no HPO name (hand-written) | 6 | 0% | 0% |
| removed | broader term dropped | 60 | 0% | 100% |
| removed | overclaim: certainty ("proves", "confirms", ...) | 60 | 0% | 100% |
| removed | overclaim: causal ("causes", "due to", "because of") | 60 | 0% | 100% |
| removed | wrong disease | 60 | 48% | 100% |
| removed | extra unsupported clause | 60 | 2% | 100% |
| removed | negation not tied to the absent finding | 11 | 0% | 100% |
| removed | not recorded stated as absent / as present | 120 | 0% | 100% |
| removed | family-history / uncertain finding stated as the patient's | 120 | 0% | 100% |
| removed | number: LR given as a frequency | 60 | 0% | 100% |
| removed | number: LRs swapped in a comparison | 60 | 0% | 100% |
| removed | wrong section | 60 | 0% | 100% |
| | **correct sentences kept** | 437 | 98.6% | 98.6% |
| | **wrong sentences removed** | 731 | 4.1% | 100% |

This set was built from the error kinds the new checks target, so it shows that each check does what it is meant
to; it is not an estimate of accuracy on real model output. The hand-written paraphrases show the main cost of
name matching: a correct sentence in other words ("his feet are flat") is still removed. (The Nov 11 version, 1,048
sentences without the broader-term categories: correct kept 98.4% with both checkers, wrong removed 5.2% vs 100%.)

**NLI comparison (Nov 22).** A general-domain entailment model (cross-encoder/nli-deberta-v3-base; cited evidence
as premise, sentence without citations as hypothesis, entailment >= 0.5) on the same set
(`results/adversarial_nli.txt`):

| | Correct kept (437) | Wrong removed (731) |
| --- | --- | --- |
| New checks | 98.6% | 100% |
| NLI alone | 72.3% | 35.0% |
| New checks and NLI (both must pass) | 71.9% | 100% |

- NLI missed errors that depend on structure rather than wording: broader term dropped, family or uncertain finding
  stated as the patient's, wrong section (0% each), not recorded stated as present (2%), certainty (7%), swapped LRs (10%).
- It removed all 60 correct family-finding sentences and about half the correct synonym and uncertain-finding ones.
- It kept 2 of the 6 hand-written paraphrases V3 removes, but it also kept all 60 broader-term errors V3 catches,
  so it cannot safely overrule V3.

**Prompt development** (Qwen2.5-7B-Instruct; prompts tuned on 31 dev cases, then run once on 30 test cases; scored
with the original V1-V4, before the Nov 9 scoring change; "kept" is an acceptance rate, not accuracy):

| Explanation | Cases | Written | Kept by the checks | Removed by |
| --- | --- | --- | --- | --- |
| Template (no model) | 31 dev | 324 | 324 (100%) | |
| Prompt v1 | 31 dev | 264 | 35 (13.3%) | V3 205, V1 16, V2 8 |
| Prompt v2 | 31 dev | 308 | 294 (95.5%) | V3 9, V4 3, V1 2 |
| Prompt v3 | 31 dev | 325 | 318 (97.8%) | V4 6, V1 1 |
| Prompt v3 | 30 test | 326 | 325 (99.7%) | V3 1 |

- v1 failed on format: the model wrote only the IDs.
- v2 added a worked example and marked which section each item belongs to.
- v3 also tells the model which patient item to cite with each knowledge item.

Examples of errors the checks removed:
- a sentence about hyperreflexia citing an item about axial hypotonia (V3);
- six comparisons with the runner-up whose likelihood ratio of 1.00 was made up (V4);
- a number copied from the prompt's own example (V4).

**Broader terms (Nov 15).** When a finding is matched through a broader annotation, the knowledge item now names the
broader term the frequency belongs to, and V3 removes a sentence that drops it (21 of 181 sentences in an earlier
labelled run did). "Not recorded" items now skip any finding the record has in a broader or narrower form. The
prompt asks the model to keep the broader term, passes it the calibrated probability shown on screen (it used to
receive the ranker's raw share, e.g. 70.7% instead of 13.1% for the Marfan example), and allows 1,500 tokens per reply.

**Final run and labels (Nov 15; second annotation Nov 22).** Prompt v3 with the changes above, run once on the 30
held-out test cases: 314 sentences. The original checks kept 313 (99.7%), the new checks 293 (93.3%). Two label sets:
sheet A was labelled with AI assistance and reviewed by one team member who had seen some of the checks' decisions;
sheet B was labelled by hand by two other team members working together, without seeing the key or the checks'
output. Agreement: 312 of 314 (99.4%), Cohen's kappa 0.95. Both disagreements (ids 128 and 155) are comparisons with
the runner-up placed under "supports"; sheet B judged only the content. They were adjudicated by the rule that a
sentence in the wrong section is unsupported (`results/labels/label_sheet_v3_test_final.csv`). The table uses the
adjudicated labels; on the 312 sentences both sheets agree on, 19 were unsupported (6.1%) and the figures are 18 of
311 kept (5.8%) for the original checks and 1 of 293 (0.3%) for the new checks.

| | Original checks (V1-V4) | New checks (V1-V6) |
| --- | --- | --- |
| Unsupported sentences accepted | 20 of 21 | 1 of 21 |
| Unsupported among kept sentences | 20 of 313 (6.4%) | 1 of 293 (0.3%; 95% upper bound about 1.6%) |
| Supported sentences removed | 0 of 293 | 1 of 293 |

The 21 unsupported sentences (6.7% of 314): 14 gave a narrower finding the frequency of its broader term; 4 applied
broader-term wording to an item that has none (a side effect of the new prompt instruction; the new checks caught 3);
2 cited comparison evidence as support; 1 had a malformed citation. The one supported sentence removed misspelled the
disease name ("autosmal"), so its subtype number was read as an unsupported number.

## Limitations

- "Evidence-linked" means the sentence names the cited finding and its numbers and adds nothing the checks recognise
  as extra; meaning beyond that (e.g. a subtle misreading in words the checks don't look for) is not verified.
  A general-domain NLI model did worse than the checks on the adversarial set (see Results); a medical NLI
  model is untested.
- V3 matches finding names and their HPO synonyms, so a correct paraphrase in other words is removed.
- V6's word lists (certainty, causal, other diseases, other findings) are fixed lists: wording outside them passes.
- Asking the model to keep broader terms made it add broader-term wording where there is none (4 of 314 sentences).
- The prompt-development table was measured with the original checks and the scoring before Nov 9; the final run
  uses the current scoring and checks.
- Sheet A was AI-assisted and reviewed by a team member who had seen some of the checks' decisions; sheet B is a hand
  annotation blind to the checks (kappa 0.95). Both cover one run of 30 test cases, so the error estimates are small-sample.
- Where the HPO annotations give no frequency, the ranker uses 50%, and sentences then say "~50% of cases".
- Evidence comes from the patient record, the HPO annotations and the ranker's own calculation; PubMed passages
  (`pubmed.py`) are not in the evidence pack yet.
- The common-disease list stays empty until common-disease profiles are loaded (Part 2).

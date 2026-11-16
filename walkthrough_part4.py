"""Walk one case through the whole system, stage by stage, printing what each file produces.

    python walkthrough_part4.py data/phenopacket-store/0.1.27/SAMD9L/PMID_27259050_III_3.json
    python walkthrough_part4.py examples/marfan_note.json
    options: --no-pause (print everything at once), --disease OMIM:xxxxxx (which disease to explain)

Press Enter to go to the next stage. Uses saved model replies when they exist (LLM_OFFLINE), else the template.
"""
import json, os, sys, textwrap
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("LLM_OFFLINE", "1")
from hoofbeats.__main__ import load_kb
from hoofbeats.data import Extractor, load_phenopackets
from hoofbeats.intake import load_case
from hoofbeats.llm import MedGemma
from hoofbeats.model import explain
import grounding as g

args = [a for a in sys.argv[1:] if not a.startswith("--")]
PAUSE = "--no-pause" not in sys.argv
path = args[0] if args else "examples/marfan_note.json"
want = sys.argv[sys.argv.index("--disease") + 1] if "--disease" in sys.argv else None


def stage(title, where):
    if PAUSE:
        input("\n(press Enter for the next stage) ")
    print("\n" + "=" * 100 + f"\n{title}\n   code: {where}\n" + "=" * 100)


def wrap(text, indent="   "):
    return textwrap.fill(text, 110, initial_indent=indent, subsequent_indent=indent)


raw = json.loads(Path(path).read_text())
is_case_file = "case_id" in raw

# ---------------------------------------------------------------------------------------------- stage 0
stage("STAGE 0 · THE DATA: what goes in", path)
if is_case_file:
    print(f"   A case file written by a doctor (or by the review screen): {raw['case_id']}")
    if raw.get("note_text"):
        print("   Clinical note (first lines):")
        for line in raw["note_text"].splitlines()[:8]:
            print("     | " + line[:110])
    for k in ("image_context", "labs", "hpo_terms", "genes", "images"):
        if raw.get(k):
            print(f"   {k}: {json.dumps(raw[k])[:200]}")
    print(f"   True diagnosis (kept only for scoring, never shown to the system): {raw.get('truth')}")
else:
    print("   A Phenopacket Store file: a published patient, already written as HPO terms.")
    for f in raw.get("phenotypicFeatures", []):
        print(f"     {f['type']['id']}  {f['type']['label']}" + ("  (excluded = absent)" if f.get("excluded") else ""))
    print(f"   True diagnosis (hidden from the system, used only to score): "
          f"{[d['term']['label'] + ' ' + d['term']['id'] for d in raw.get('diseases', [])]}")
    print(f"   Source paper: {[x['id'] for x in raw.get('metaData', {}).get('externalReferences', [])]}")

# ---------------------------------------------------------------------------------------------- stage 1
stage("STAGE 1 · READ THE RECORD: words -> standard findings (HPO terms)",
      "hoofbeats/intake.py load_case()  (notes, labs, images)  ·  hoofbeats/data.py load_phenopackets()")
r = load_kb(SimpleNamespace(kb="data/kb", common="none", prior_weight=1.0, absent_weight=0.5, absent_min_freq=0.8,
                            use_onset=False, use_sex=False))
ext = Extractor(r.ont)
case = load_case(path, r.ont, ext, disease_names=r.db.names) if is_case_file else load_phenopackets(path)[0]
prov = case.meta.get("provenance", {})
print(f"   {len(case.findings)} findings:")
for f in case.findings:
    t = r.ont.resolve(f.hpo_id)
    where = ""
    for e in prov.get(t, [])[:1]:
        if e.get("quote"):
            pos = f" at characters {e['start']}-{e['end']}" if e.get("start") is not None else ""
            where = f'  <- "{e["quote"]}" ({e.get("field") or "entered value"}{pos})'
    print(f"     {f.status.value:<8} {r.ont.name(t)}{where}")
ex = frozenset(case.source_pmids)
print(f"   Leakage control: the case's own paper(s) {sorted(ex) or '(none)'} are removed from the knowledge base.")

# ---------------------------------------------------------------------------------------------- stage 2
stage("STAGE 2 · RANK THE DISEASES: add up likelihood ratios for all 8,465 diseases",
      "hoofbeats/model.py LRRanker.score() and contributions()  ·  % on screen: hoofbeats/workup.py ranker_probs()")
ranking = r.score(case, ex)
try:
    from hoofbeats.workup import done_texts, load_workupper, ranker_probs, workup_json
    wk = load_workupper(r, "data/kb")
    cal = ranker_probs(ranking)
except Exception:
    wk, cal = None, None
order = [d for d, _, _ in ranking.top(400)]
truth = set(case.truth)
for i, (d, _, p) in enumerate(ranking.top(5), 1):
    shown = float(cal[r.d_index[d]]) if cal is not None else p
    print(f"   {i}. {r.db.names[d]} ({d})  {shown:.1%}" + ("   <- true diagnosis" if d in truth else ""))
rank = next((order.index(d) + 1 for d in truth if d in order), None)
print(f"   True diagnosis ranked: {rank}")
d = want or next((x for x in order[:3] if x in truth), order[0])
runner = order[order.index(d) + 1]
print(f"\n   Why {r.db.names[d]} scores what it does (each finding's likelihood ratio, > 1 pushes it up, < 1 down):")
import math
for c in sorted(r.contributions(case, d, ex), key=lambda c: -abs(c.log_lr))[:8]:
    freq = f"in ~{c.kb_freq:.0%} of patients" if c.kb_freq is not None else "not described for it"
    print(f"     LR {math.exp(c.log_lr):>8.2f}   {c.status.value:<8} {r.ont.name(c.hpo_id)}  ({freq})")

# ---------------------------------------------------------------------------------------------- stage 3
stage("STAGE 3 · SUGGEST NEXT TESTS", "hoofbeats/workup.py load_workupper(), build(), workup_json()")
if wk is not None:
    case.meta["done_texts"] = done_texts(path) if is_case_file else []
    w = workup_json(wk.build(case, ranking, ex, 3, 3, shown=[d]))
    t = next((x for x in w["diseases"] if x["id"] == d), {})
    for x in ((([t["gene_test"]] if t.get("gene_test") else []) + t.get("tests", []))[:4]):
        print(f"   {x['test']}: if positive -> {x['post_test_if_positive']:.0%}, if negative -> "
              f"{x['post_test_if_negative']:.1%} (effort {x['burden']}/5)")
else:
    print("   (test data not downloaded on this machine)")

# ---------------------------------------------------------------------------------------------- stage 4a
stage("STAGE 4a · MY PART: BUILD THE FOOTNOTES (the evidence pack)",
      "grounding.py build_pack() -> patient_items(), knowledge_items(), missing_items(), runner_up_items(), clean_claim()")
claims = explain(r, case, ranking, order.index(d) + 1, ex, max_claims=8)[order.index(d)][2]
pack = g.build_pack(r, case, d, 0.0, ex, runner, claims)
for i in pack.values():
    kind = "patient" if i.kind == "patient" else i.role
    print(f"   [{i.id:>3}] {kind:<8} {i.text}")

# ---------------------------------------------------------------------------------------------- stage 4b
stage("STAGE 4b · MY PART: THE AI WRITES REASONS WITH FOOTNOTES",
      "grounding.py make_prompt_v3() -> Qwen2.5-7B on Sol (saved replies offline) -> parse_trace(); "
      "fallback: template_trace()")
p = float(cal[r.d_index[d]]) if cal is not None else {dd: pp for dd, _, pp in ranking.top(400)}[d]
prompt = g.make_prompt_v3(r, d, p, pack, runner)
print("   The prompt (first lines; the evidence list above is pasted at the end):")
for line in prompt.splitlines()[:8]:
    print("     | " + line)
llm = MedGemma(os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
               os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"), max_new_tokens=int(os.environ.get("LLM_MAX_TOKENS", "1500")),
               timeout=120, cache_dir="results/llm_cache", extra={"response_format": {"type": "json_object"}},
               offline=os.environ.get("LLM_OFFLINE") == "1")
try:
    trace = g.parse_trace(llm.chat(prompt))
except Exception:
    trace = {}
source = "the local AI model (saved reply)" if sum(len(v) for v in trace.values()) else "the template (no saved AI reply for this prompt)"
if source.startswith("the template"):
    trace = g.template_trace(pack)
print(f"\n   Written by {source}:")
for sec, sents in trace.items():
    for s in sents:
        print(wrap(f"[{sec}] {s}"))

# ---------------------------------------------------------------------------------------------- stage 4c
stage("STAGE 4c · MY PART: SIX CHECKS DELETE WHAT THE FOOTNOTES DON'T SUPPORT",
      "grounding.py check_trace() -> check_sentence() (V1 footnotes, V5 right list, V2 record, V3 same finding, "
      "V4 numbers, V6 nothing extra)")
names = [r.db.names[d], r.db.names[runner]]
kept, removed = g.check_trace(trace, pack, r, case, names)
print(f"   kept {sum(len(v) for v in kept.values())}, removed {len(removed)}")
for x in removed:
    print(wrap(f"REMOVED: {x['text']}  ->  {x['reason']}"))
first = next((s for s in trace.get("supports", [])), None)
if first:
    body, cite = g.strip_cites(first).rstrip(" ."), first[len(g.strip_cites(first).rstrip()):]
    for label, broken in (("adds 'proves'", f"{body}, which proves {r.db.names[d]} {cite.strip()}"),
                          ("changes the number", body.replace("~", "~1", 1) + " " + cite.strip())):
        res = g.check_sentence(broken, pack, r, case, names, section="supports")
        print(wrap(f"Live test, {label}: {'kept' if res['ok'] else 'REMOVED -> ' + res['reason']}"))

# ---------------------------------------------------------------------------------------------- stage 5
stage("STAGE 5 · THE CONCLUSION THE DOCTOR SEES", "demo_part4.py -> results/part4_demo.json -> review_app.py")
top = order[0]
print(f"   Most likely: {r.db.names[top]}" + (f" ({float(cal[r.d_index[top]]):.0%})" if cal is not None else ""))
print(f"   Explained here: {r.db.names[d]}" + ("  (the true diagnosis)" if d in truth else ""))
for sec, title in (("supports", "What fits"), ("against", "What does not fit"),
                   ("expected_but_missing", "Usually present, not recorded"), ("vs_runner_up", "Why it ranks above the next")):
    if kept.get(sec):
        print(f"   {title}:")
        for x in kept[sec][:3]:
            print(wrap(f"- {x['text']}   [Evidence-linked: {', '.join(x['sources'])}]", "     "))
print("\n   On the screen each of these can be clicked back to the patient's words or the source paper.")

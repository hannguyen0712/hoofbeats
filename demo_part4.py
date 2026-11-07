"""Part 4 pipeline for one case: Stage 1 -> ranking -> evidence pack -> explanation (model, else template)
-> checks -> suggested tests (Part 3). Saves results/part4_demo.json for review_app.py.

    python demo_part4.py examples/marfan_note.json [--drop HP:0002616 ...]
    model: LLM_ENDPOINT, LLM_MODEL, PROMPT_VERSION (default v3); LLM_OFFLINE=1 uses saved replies only
"""
import argparse, json, os
from pathlib import Path
from types import SimpleNamespace
from hoofbeats.__main__ import load_kb
from hoofbeats.data import Extractor, load_phenopackets
from hoofbeats.intake import load_case
from hoofbeats.llm import MedGemma
from hoofbeats.model import Case, explain
import grounding as g

ap = argparse.ArgumentParser()
ap.add_argument("case", nargs="?", default="examples/marfan_note.json")
ap.add_argument("--drop", nargs="*", default=[], help="HPO ids the clinician removed")
ap.add_argument("--k", type=int, default=3)
a = ap.parse_args()

r = load_kb(SimpleNamespace(kb="data/kb", common="none", prior_weight=1.0))
if a.case.startswith("examples/"):
    case = load_case(a.case, r.ont, Extractor(r.ont), disease_names=r.db.names)   # note -> findings (Stage 1)
else:
    case = load_phenopackets(a.case)[0]                                            # phenopacket
dropped = {r.ont.resolve(h) for h in a.drop}
if dropped:                                                                        # clinician removed findings
    case = Case(case.case_id, [f for f in case.findings if r.ont.resolve(f.hpo_id) not in dropped],
                case.truth, case.genes, case.source_pmids, case.meta)
ex = frozenset(case.source_pmids)            # leakage control
ranking = r.score(case, ex)

# the model; with no server (or offline with no saved reply) the template explanation is used instead
VERSION = os.environ.get("PROMPT_VERSION", "v3")
make_prompt = {"v1": g.make_prompt, "v2": g.make_prompt_v2, "v3": g.make_prompt_v3}[VERSION]
llm = MedGemma(os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
               os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"),
               max_new_tokens=800, timeout=120, cache_dir="results/llm_cache",
               extra={"response_format": {"type": "json_object"}}, offline=os.environ.get("LLM_OFFLINE") == "1")


def explain_with_model(pack, d, p, runner):
    try:
        trace = g.parse_trace(llm.chat(make_prompt(r, d, p, pack, runner)))
    except Exception as e:                                                         # server not running, timeout
        print(f"   model not reachable ({type(e).__name__}); using the template explanation")
        trace = {}
    if sum(len(v) for v in trace.values()):
        return trace, f"model ({llm.model.split('/')[-1]}, prompt {VERSION})"
    return g.template_trace(pack), "template (no model reply)"


# the original document text, and where in it each finding was read (Stage 1 keeps the character offsets)
raw = json.loads(Path(a.case).read_text())
texts = {k: v for k, v in raw.items() if k in ("note_text", "image_context") and isinstance(v, str) and v}
prov = case.meta.get("provenance", {})


def spans_for(hpo_id):
    return [{"field": e["field"], "start": e["start"], "end": e["end"], "quote": e["quote"],
             "cue": e.get("status_cue"), "grounding": e.get("grounding")}
            for e in prov.get(hpo_id, []) if e.get("field") in texts and e.get("start") is not None]


def item_json(i):
    d = vars(i) | {"refs": list(i.refs)}
    if i.kind == "patient":
        d["spans"] = spans_for(i.hpo_id)
    return d


# two lists: rare and common diseases (common ones are those with a prevalence prior)
rare, common = [], []
for d, _, p in ranking.top(400):
    (common if d in r.db.prior else rare).append((d, p))
lists = {"rare": rare[:a.k], "common": common[:a.k]}

# Part 3: suggested tests for every displayed disease (None if the test data is not downloaded)
tests = {}
try:
    from hoofbeats.workup import done_texts, load_workupper, workup_json
    wk = load_workupper(r, "data/kb")
    if wk is not None:
        shown = [d for lst in lists.values() for d, _ in lst]
        case.meta["done_texts"] = done_texts(a.case)
        tests = {x["id"]: x for x in workup_json(wk.build(case, ranking, ex, len(shown), 3, shown=shown))["diseases"]}
except Exception as e:
    print(f"[part4] test suggestions unavailable: {e}")

out = {"case": case.case_id, "case_path": a.case, "truth": case.truth, "dropped": sorted(dropped),
       "source": {"kind": "clinical note" if texts else "phenopacket (structured)", "texts": texts},
       "findings": [{"hpo_id": r.ont.resolve(f.hpo_id), "name": r.ont.name(r.ont.resolve(f.hpo_id)),
                     "status": f.status.value, "words": f.evidence or ""} for f in case.findings
                    if r.ont.resolve(f.hpo_id)],
       "lists": {}}
order = [d for d, _, _ in ranking.top(400)]
for label, lst in lists.items():
    out["lists"][label] = []
    for n, (d, p) in enumerate(lst, 1):
        runner = order[order.index(d) + 1]
        claims = explain(r, case, ranking, order.index(d) + 1, ex, max_claims=8)[order.index(d)][2]
        pack = g.build_pack(r, case, d, p, ex, runner, claims)
        trace, source = explain_with_model(pack, d, p, runner)
        kept, removed = g.check_trace(trace, pack, r, case, [r.db.names[d], r.db.names[runner]])
        print(f"{label} {n}. {r.db.names[d]} ({d}) {p:.1%} | {source} | kept {sum(len(v) for v in kept.values())}, "
              f"removed {len(removed)}")
        for x in removed:
            print(f"     removed: {x['text']}  ->  {x['reason']}")
        out["lists"][label].append({"id": d, "name": r.db.names[d], "probability": p, "runner_up": runner,
                                    "runner_up_name": r.db.names[runner], "explanation_source": source,
                                    "pack": [item_json(i) for i in pack.values()], "explanation": kept,
                                    "removed": removed, "tests": tests.get(d)})

Path("results").mkdir(exist_ok=True)
Path("results/part4_demo.json").write_text(json.dumps(out, indent=1))
print("saved results/part4_demo.json")

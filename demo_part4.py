import json, sys
from pathlib import Path
from types import SimpleNamespace
from hoofbeats.__main__ import load_kb
from hoofbeats.data import Extractor, load_phenopackets
from hoofbeats.intake import load_case
from hoofbeats.model import explain
import grounding as g

CASE = sys.argv[1] if len(sys.argv) > 1 else "examples/marfan_note.json"
r = load_kb(SimpleNamespace(kb="data/kb", common="none", prior_weight=1.0))
if CASE.startswith("examples/"):
    case = load_case(CASE, r.ont, Extractor(r.ont), disease_names=r.db.names)   # note -> findings (Stage 1)
else:
    case = load_phenopackets(CASE)[0]                                            # phenopacket
ex = frozenset(case.source_pmids)            # leakage control
ranking = r.score(case, ex)
top = ranking.top(4)

# the original document text, and where in it each finding was read (Stage 1 keeps the character offsets)
raw = json.loads(Path(CASE).read_text())
texts = {k: v for k, v in raw.items() if k in ("note_text", "image_context") and isinstance(v, str) and v}
prov = case.meta.get("provenance", {})


def spans_for(hpo_id):
    out = []
    for e in prov.get(hpo_id, []):
        if e.get("field") in texts and e.get("start") is not None:
            out.append({"field": e["field"], "start": e["start"], "end": e["end"], "quote": e["quote"],
                        "cue": e.get("status_cue"), "grounding": e.get("grounding")})
    return out


def item_json(i):
    d = vars(i) | {"refs": list(i.refs)}
    if i.kind == "patient":
        d["spans"] = spans_for(i.hpo_id)
    return d


out = {"case": case.case_id, "source": {"path": CASE, "kind": "clinical note" if texts else "phenopacket (structured)",
                                        "texts": texts}, "diseases": []}
for n, (d, score, p) in enumerate(top[:3], 1):
    runner = top[n][0]
    claims = explain(r, case, ranking, n, ex, max_claims=8)[n - 1][2]
    pack = g.build_pack(r, case, d, p, ex, runner, claims)
    trace = g.template_trace(pack)
    kept, removed = g.check_trace(trace, pack, r, case, [r.db.names[d], r.db.names[runner]])
    print(f"\n{n}. {r.db.names[d]} ({d})  {p:.1%}   runner-up: {r.db.names[runner]}")
    for sec, sents in kept.items():
        for x in sents:
            print(f"   [{x['grade']:<12}] {x['text']}")
    for x in removed:
        print(f"   [REMOVED     ] {x['text']}  ->  {x['reason']}")
    out["diseases"].append({"id": d, "name": r.db.names[d], "probability": p, "runner_up": runner,
                            "pack": [item_json(i) for i in pack.values()],
                            "explanation": kept, "removed": removed})

Path("results").mkdir(exist_ok=True)
Path("results/part4_demo.json").write_text(json.dumps(out, indent=1))
print("\nsaved results/part4_demo.json")

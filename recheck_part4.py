"""Re-check saved model sentences with the current checks, without the model.

    python recheck_part4.py results/part4_llm_sentences_v3_test.jsonl
    -> results/part4_llm_sentences_v3_test.recheck.jsonl  (each sentence with ok_basic / ok_strict and the reasons)

Each case's evidence pack is rebuilt and compared with the evidence saved next to every sentence; a case whose pack
has changed since the model run (e.g. after a scoring change) is reported and skipped, so no sentence is judged
against evidence the model never saw.
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from hoofbeats.__main__ import load_kb
from hoofbeats.data import Extractor, load_phenopackets
from hoofbeats.intake import load_case
from hoofbeats.model import explain
import grounding as g

src = Path(sys.argv[1])
rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
r = load_kb(SimpleNamespace(kb="data/kb", common="none", prior_weight=1.0, absent_weight=0.5, absent_min_freq=0.8,
                            use_onset=False, use_sex=False))
ext = Extractor(r.ont)
by_case = defaultdict(list)
for x in rows:
    by_case[x["case"]].append(x)

out, stats, reasons, skipped = [], Counter(), Counter(), []
for path, xs in by_case.items():
    case = load_case(path, r.ont, ext, disease_names=r.db.names) if path.startswith("examples/") \
        else load_phenopackets(path)[0]
    ex = frozenset(case.source_pmids)
    ranking = r.score(case, ex)
    order = [d for d, _, _ in ranking.top(400)]
    d = xs[0]["disease"]
    if d not in order:
        skipped.append((path, "disease no longer ranked")); continue
    runner = order[order.index(d) + 1]
    claims = explain(r, case, ranking, order.index(d) + 1, ex, max_claims=8)[order.index(d)][2]
    pack = g.build_pack(r, case, d, {dd: pp for dd, _, pp in ranking.top(400)}[d], ex, runner, claims)
    saved = {c: t for x in xs for c, t in (x.get("evidence") or {}).items() if not t.startswith("(this ID")}
    changed = [c for c, t in saved.items() if c not in pack or pack[c].text != t]
    if changed:
        skipped.append((path, f"evidence changed since the model run ({', '.join(changed[:3])})")); continue
    names = [r.db.names[d], r.db.names[runner]]
    for x in xs:
        b = g.check_sentence(x["text"], pack, r, case, names, section=x["section"], strict=False)
        s = g.check_sentence(x["text"], pack, r, case, names, section=x["section"])
        out.append(x | {"ok_basic": b["ok"], "reason_basic": b.get("reason", ""), "ok_strict": s["ok"],
                        "reason_strict": s.get("reason", ""), "sources": s.get("sources", [])})
        stats["sentences"] += 1
        stats["kept basic"] += b["ok"]
        stats["kept strict"] += s["ok"]
        if b["ok"] and not s["ok"]:
            reasons[s["reason"].split(":")[0]] += 1

dst = src.with_suffix(".recheck.jsonl")
dst.write_text("".join(json.dumps(x) + "\n" for x in out))
n = stats["sentences"] or 1
print(f"{src}: {len(rows)} sentences from {len(by_case)} cases; re-checked {stats['sentences']}")
print(f"kept by the original checks (V1-V4): {stats['kept basic']} ({stats['kept basic'] / n:.1%})")
print(f"kept by the new checks (V1-V6):      {stats['kept strict']} ({stats['kept strict'] / n:.1%})")
print("newly removed, by check:", dict(reasons))
for x in [x for x in out if x["ok_basic"] and not x["ok_strict"]][:15]:
    print(f"  {x['reason_strict']}\n     {x['text']}")
for p, why in skipped:
    print(f"skipped {p}: {why}")
print(f"saved {dst}")

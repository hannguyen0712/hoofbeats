import json, os, sys, time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from hoofbeats.__main__ import load_kb
from hoofbeats.data import Extractor, load_phenopackets
from hoofbeats.intake import load_case
from hoofbeats.llm import MedGemma
from hoofbeats.model import explain
import grounding as g

MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
ENDPOINT = os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
OFFLINE = os.environ.get("LLM_OFFLINE") == "1"
VERSION = os.environ.get("PROMPT_VERSION", "v1")
make_prompt = {"v1": g.make_prompt, "v2": g.make_prompt_v2, "v3": g.make_prompt_v3}[VERSION]
cases = [c for c in Path(sys.argv[1] if len(sys.argv) > 1 else "part4_cases.txt").read_text().split() if c]

r = load_kb(SimpleNamespace(kb="data/kb", common="none", prior_weight=1.0))
ext = Extractor(r.ont)
llm = MedGemma(MODEL, ENDPOINT, max_new_tokens=800, timeout=300, cache_dir="results/llm_cache",
               extra={"response_format": {"type": "json_object"}}, offline=OFFLINE)

stats, grades, by_check, sections = Counter(), Counter(), Counter(), Counter()
rows = []
t0 = time.time()
for path in cases:
    case = load_case(path, r.ont, ext, disease_names=r.db.names) if path.startswith("examples/") \
        else load_phenopackets(path)[0]
    if not case.findings:
        continue
    ex = frozenset(case.source_pmids)
    ranking = r.score(case, ex)
    (d, _, p), (runner, _, _) = ranking.top(2)
    claims = explain(r, case, ranking, 1, ex, max_claims=8)[0][2]
    pack = g.build_pack(r, case, d, p, ex, runner, claims)
    names = [r.db.names[d], r.db.names[runner]]
    def ev(s):
        return {c: (pack[c].text if c in pack else "(this ID is not in the evidence)") for c in g.cites(s)}

    raw = llm.chat(make_prompt(r, d, p, pack, runner))
    trace = g.parse_trace(raw)
    n_sent = sum(len(v) for v in trace.values())
    stats["cases"] += 1
    if n_sent == 0:
        stats["no usable reply"] += 1
    kept, removed = g.check_trace(trace, pack, r, case, names)
    for sec, v in kept.items():
        for x in v:
            grades[x["grade"]] += 1
            sections[sec] += 1
            rows.append({"case": path, "disease": d, "disease_name": r.db.names[d], "evidence": ev(x["text"]), "section": sec, "text": x["text"], "ok": True,
                         "grade": x["grade"], "reason": ""})
    for x in removed:
        by_check[x["reason"][:2]] += 1
        rows.append({"case": path, "disease": d, "disease_name": r.db.names[d], "evidence": ev(x["text"]), "section": x["section"], "text": x["text"], "ok": False,
                     "grade": "", "reason": x["reason"]})
    stats["written"] += n_sent
    stats["kept"] += sum(len(v) for v in kept.values())
    stats["removed"] += len(removed)
    print(f"{Path(path).name}: {r.db.names[d]} | written {n_sent}, kept {n_sent - len(removed)}, removed {len(removed)}")

Path("results").mkdir(exist_ok=True)
with open(f"results/part4_llm_sentences_{VERSION}.jsonl", "w") as f:
    for row in rows:
        f.write(json.dumps(row) + "\n")

w = stats["written"] or 1
print(f"\nmodel: {MODEL}  | prompt {VERSION} | {stats['cases']} cases in {time.time() - t0:.0f}s  |  no usable reply: {stats['no usable reply']}")
print(f"sentences written: {stats['written']}, kept {stats['kept']} ({stats['kept'] / w:.1%}), "
      f"removed {stats['removed']} ({stats['removed'] / w:.1%})")
print("removed by:", dict(by_check))
print("grades of kept:", dict(grades))
print("kept per section:", dict(sections))
print("\nexamples removed:")
for row in [x for x in rows if not x["ok"]][:10]:
    print(f"  {row['reason']}\n     {row['text']}")
print(f"\nsaved results/part4_llm_sentences_{VERSION}.jsonl")

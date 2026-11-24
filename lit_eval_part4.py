"""How often the review screen can show a highlighted paper sentence for an evidence item, and a sheet to check them.

    python lit_eval_part4.py                         # the fresh held-out cases (part4_cases_fresh.txt)
    python lit_eval_part4.py part4_cases_test.txt
    -> prints coverage; writes results/labels/lit_sheet_<set>.csv (a random 30 highlights to label by hand)

Needs internet for PubMed the first time (only paper IDs are sent); later runs use data/kb/pubmed_cache.json.
"""
import csv, random, sys
from pathlib import Path
from types import SimpleNamespace
from hoofbeats.__main__ import load_kb
from hoofbeats.data import Extractor, load_phenopackets
from hoofbeats.model import explain
from hoofbeats.workup import ranker_probs
import grounding as g
from literature_part4 import fetch_papers, highlights

src = Path(sys.argv[1] if len(sys.argv) > 1 else "part4_cases_fresh.txt")
cases = [c for c in src.read_text().split() if c and not c.startswith("examples/")]
r = load_kb(SimpleNamespace(kb="data/kb", common="none", prior_weight=1.0, absent_weight=0.5, absent_min_freq=0.8,
                            use_onset=False, use_sex=False))
ext = Extractor(r.ont)

items = []
for path in cases:
    found = load_phenopackets(path)
    if not found or not found[0].findings:
        continue
    case = found[0]
    ex = frozenset(case.source_pmids)
    ranking = r.score(case, ex)
    (d, _, p), (runner, _, _) = ranking.top(2)
    p = float(ranker_probs(ranking)[r.d_index[d]])
    claims = explain(r, case, ranking, 1, ex, max_claims=8)[0][2]
    pack = g.build_pack(r, case, d, p, ex, runner, claims)
    for i in pack.values():
        if i.kind == "knowledge" and i.hpo_id:
            items.append({"case": path, "disease": r.db.names[d], **vars(i), "refs": list(i.refs)})

papers, note = fetch_papers([x for i in items for x in i["refs"] if str(x).startswith("PMID:")])
if note:
    print(note)
memo, hits = {}, []
n_pmid = n_fetched = n_hit = 0
for i in items:
    pm = [x for x in i["refs"] if str(x).startswith("PMID:")]
    n_pmid += bool(pm)
    n_fetched += any(x.split(":")[-1] in papers for x in pm)
    h = highlights(r.ont, ext, i, papers, memo)
    n_hit += bool(h)
    hits += [{"item": i, **x} for x in h]

pct = lambda a, b: f"{a} of {b} ({a / b:.0%})" if b else "0 of 0"
print(f"{len(cases)} cases, {len(items)} knowledge items")
print(f"  cite at least one PubMed paper:        {pct(n_pmid, len(items))}")
print(f"  at least one abstract fetched:         {pct(n_fetched, len(items))}")
print(f"  at least one highlighted sentence:     {pct(n_hit, len(items))} of all items; "
      f"{pct(n_hit, n_fetched)} of items with an abstract")

rng = random.Random(0)
sample = rng.sample(hits, min(30, len(hits)))
out = Path("results/labels") / f"lit_sheet_{src.stem.replace('part4_cases_', '').replace('part4_cases', 'dev')}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "disease", "finding", "pmid", "sentence (highlighted words in [[ ]])",
                "label (1 = the sentence reports this finding in this disease, 0 = not)", "note"])
    for n, x in enumerate(sample, 1):
        s, marked, pos = x["sentence"], [], 0
        for a, b in sorted(x["marks"]):
            marked += [s[pos:a], "[[", s[a:b], "]]"]
            pos = b
        marked.append(s[pos:])
        w.writerow([n, x["item"]["disease"], r.ont.name(r.ont.resolve(x["item"]["hpo_id"])), x["pmid"], "".join(marked),
                    "", ""])
print(f"saved {out} ({len(sample)} highlights to label)")

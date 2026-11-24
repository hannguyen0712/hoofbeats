"""How often the review screen can show a highlighted paper sentence for an evidence item, and a sheet to check them.

    python lit_eval_part4.py                         # the fresh held-out cases (part4_cases_fresh.txt)
    python lit_eval_part4.py part4_cases_test.txt
    -> prints coverage (abstracts only, and with open-access full text); writes results/labels/lit_sheet_<set>.csv
       (a random 30 highlights to label by hand)

Needs internet the first time (only paper IDs are sent); later runs use the caches in data/kb/. Each case's own
source paper is left out, as in the ranking (leakage control).
"""
import csv, random, sys
from pathlib import Path
from types import SimpleNamespace
from hoofbeats.__main__ import load_kb
from hoofbeats.data import Extractor, load_phenopackets
from hoofbeats.model import explain
from hoofbeats.workup import ranker_probs
import grounding as g
from literature_part4 import fetch_fulltext, fetch_papers, highlights

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
            items.append({"case": path, "disease": r.db.names[d], "exclude": sorted(ex), **vars(i), "refs": list(i.refs)})

pmids = [x for i in items for x in i["refs"] if str(x).startswith("PMID:")]
papers, note = fetch_papers(pmids)
full, note2 = fetch_fulltext(pmids, limit=1000)
for n in (note, note2):
    if n:
        print(n)
memo, hits = {}, []
count = {"pmid": 0, "own paper only": 0, "abstract": 0, "full text": 0, "hit abstract": 0, "hit any": 0}
for i in items:
    own = {x.split(":")[-1] for x in i["exclude"]}
    pm = [x.split(":")[-1] for x in i["refs"] if str(x).startswith("PMID:")]
    other = [x for x in pm if x not in own]
    count["pmid"] += bool(pm)
    count["own paper only"] += bool(pm) and not other
    count["abstract"] += any(x in papers for x in other)
    count["full text"] += any(x in full for x in other)
    count["hit abstract"] += bool(highlights(r.ont, ext, i, papers, memo, None, i["exclude"]))
    h = highlights(r.ont, ext, i, papers, memo, full, i["exclude"])
    count["hit any"] += bool(h)
    hits += [{"item": i, **x} for x in h]

pct = lambda a, b: f"{a} of {b} ({a / b:.0%})" if b else "0 of 0"
n = len(items)
print(f"{len(cases)} cases, {n} knowledge items")
print(f"  cite at least one PubMed paper:              {pct(count['pmid'], n)}")
print(f"  cite only the case's own paper (left out):   {pct(count['own paper only'], n)}")
print(f"  an abstract of another cited paper:          {pct(count['abstract'], n)}")
print(f"  an open-access full text of one:             {pct(count['full text'], n)}")
print(f"  highlighted, abstracts only:                 {pct(count['hit abstract'], n)}")
print(f"  highlighted, abstracts and full text:        {pct(count['hit any'], n)}")
print("  where the highlights come from:", ", ".join(f"{k} {v}" for k, v in
      sorted(__import__("collections").Counter(h["where"] for h in hits).items(), key=lambda kv: -kv[1])))

rng = random.Random(0)
sample = rng.sample(hits, min(30, len(hits)))
out = Path("results/labels") / f"lit_sheet_{src.stem.replace('part4_cases_', '').replace('part4_cases', 'dev')}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "disease", "finding", "pmid", "where", "sentence (highlighted words in [[ ]])",
                "label (1 = the sentence reports this finding in this disease, 0 = not)", "note"])
    for k, x in enumerate(sample, 1):
        s, marked, pos = x["sentence"], [], 0
        for a, b in sorted(x["marks"]):
            marked += [s[pos:a], "[[", s[a:b], "]]"]
            pos = b
        marked.append(s[pos:])
        w.writerow([k, x["item"]["disease"], r.ont.name(r.ont.resolve(x["item"]["hpo_id"])), x["pmid"], x["where"],
                    "".join(marked), "", ""])
print(f"saved {out} ({len(sample)} highlights to label)")

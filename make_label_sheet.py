"""Blind labelling sheet for model-written sentences.

    python recheck_part4.py results/part4_llm_sentences_v3_test.jsonl          # adds both checkers' decisions
    python make_label_sheet.py results/part4_llm_sentences_v3_test.recheck.jsonl      # every sentence (recommended)
    python make_label_sheet.py results/part4_llm_sentences_v3_test.recheck.jsonl 150  # all removed + a random sample

Writes results/labels/label_sheet_<name>.csv (what annotators see: no check results), label_key_<name>.csv (the
checks' decisions, opened only after labelling) and label_meta_<name>.json (where the sentences came from, and the
sampling, so rates can be weighted back to the whole file).
"""
import csv, json, random, sys, time
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit(__doc__)
arg = sys.argv[1]
src = Path(arg) if arg.endswith(".jsonl") else Path(f"results/part4_llm_sentences_{arg}.jsonl")
if not arg.endswith(".jsonl"):
    print(f"note: '{arg}' is read as {src}")
N = sys.argv[2] if len(sys.argv) > 2 else "all"
rows = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]

# provenance: which case list do these sentences come from?
cases = sorted({x["case"] for x in rows})
lists = {p: set(Path(p).read_text().split()) for p in ("part4_cases.txt", "part4_cases_test.txt") if Path(p).exists()}
print(f"source: {src}  (modified {time.strftime('%Y-%m-%d %H:%M', time.localtime(src.stat().st_mtime))})")
print(f"{len(rows)} sentences from {len(cases)} cases")
for p, s in lists.items():
    print(f"  cases also in {p}: {len(set(cases) & s)} of {len(cases)}")

ok_b = lambda x: x.get("ok_basic", x.get("ok"))
ok_s = lambda x: x.get("ok_strict")
removed = [x for x in rows if not ok_b(x) or ok_s(x) is False]
kept = [x for x in rows if x not in removed]
rng = random.Random(0)
if N == "all":
    sample, strata = rows[:], {"all": len(rows)}
else:
    sample = removed + rng.sample(kept, min(len(kept), max(0, int(N) - len(removed))))
    strata = {"removed": len(removed), "kept": len(kept)}
rng.shuffle(sample)

stem = src.name.replace("part4_llm_sentences_", "").replace(".jsonl", "")
out = Path("results/labels"); out.mkdir(parents=True, exist_ok=True)
sheet, key, meta = out / f"label_sheet_{stem}.csv", out / f"label_key_{stem}.csv", out / f"label_meta_{stem}.json"
with open(sheet, "w", newline="") as f, open(key, "w", newline="") as k:
    ws, wk = csv.writer(f), csv.writer(k)
    ws.writerow(["id", "disease", "section", "sentence", "evidence", "label (1 = supported, 0 = not)",
                 "error type if 0 (overclaim / wrong finding / wrong number / wrong status / not in evidence / wrong section / other)",
                 "note"])
    wk.writerow(["id", "case", "stratum", "ok_basic", "reason_basic", "ok_strict", "reason_strict"])
    for n, x in enumerate(sample, 1):
        ev = "\n".join(f"[{cid}] {txt}" for cid, txt in (x.get("evidence") or {}).items()) or "(no evidence cited)"
        ws.writerow([n, x["disease_name"], x["section"], x["text"], ev, "", "", ""])
        stratum = "all" if N == "all" else ("removed" if x in removed else "kept")
        wk.writerow([n, x["case"], stratum, int(bool(ok_b(x))), x.get("reason_basic", x.get("reason", "")),
                     "" if ok_s(x) is None else int(ok_s(x)), x.get("reason_strict", "")])
meta.write_text(json.dumps({"source": str(src), "modified": time.ctime(src.stat().st_mtime), "sentences": len(rows),
                            "cases": cases, "strata_population": strata,
                            "strata_sample": {s: sum(1 for x in sample if (s == "all" or (s == "removed") == (x in removed)))
                                              for s in strata}}, indent=1))
print(f"{len(sample)} sentences -> {sheet}  (give each annotator their own copy)")
print(f"answer key (open only after both annotators finish) -> {key}")
print(f"provenance and sampling -> {meta}")

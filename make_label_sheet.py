import csv, json, random, sys
from pathlib import Path

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v3"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
rows = [json.loads(l) for l in open(f"results/part4_llm_sentences_{VERSION}.jsonl")]
rng = random.Random(0)
removed = [x for x in rows if not x["ok"]]
kept = [x for x in rows if x["ok"]]
sample = removed + rng.sample(kept, min(len(kept), max(0, N - len(removed))))   # every removed one + random kept
rng.shuffle(sample)

Path("results/labels").mkdir(parents=True, exist_ok=True)
sheet = Path(f"results/labels/label_sheet_{VERSION}.csv")       # what the annotators see: no check results
key = Path(f"results/labels/label_key_{VERSION}.csv")           # the check results, opened only after labelling
with open(sheet, "w", newline="") as f, open(key, "w", newline="") as k:
    ws, wk = csv.writer(f), csv.writer(k)
    ws.writerow(["id", "disease", "section", "sentence", "evidence", "label (1 = supported, 0 = not)", "note"])
    wk.writerow(["id", "case", "check_ok", "grade", "reason"])
    for n, x in enumerate(sample, 1):
        ev = "\n".join(f"[{cid}] {txt}" for cid, txt in x.get("evidence", {}).items()) or "(no evidence cited)"
        ws.writerow([n, x["disease_name"], x["section"], x["text"], ev, "", ""])
        wk.writerow([n, x["case"], int(x["ok"]), x["grade"], x["reason"]])
print(f"{len(sample)} sentences ({len(removed)} removed by the checks, {len(sample) - len(removed)} kept) -> {sheet}")
print(f"answer key (do not open until both annotators finish) -> {key}")

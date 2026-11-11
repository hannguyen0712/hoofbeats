"""Score two annotators' labels and the checks against them.

    python score_labels.py sheet_A.csv sheet_B.csv results/labels/label_key_<name>.csv [final_sheet.csv]

final_sheet: the adjudicated labels after the annotators discussed their disagreements (same format as a sheet);
without it, only sentences both annotators agree on are used.
"""
import csv, json, sys
from collections import Counter
from pathlib import Path

LABEL = "label (1 = supported, 0 = not)"


def read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {int(r["id"]): r for r in csv.DictReader(f)}


def labels(rows):
    return {i: int(r[LABEL].strip()) for i, r in rows.items() if r[LABEL].strip() in ("0", "1")}


def etype(row):
    col = next((c for c in row if c.startswith("error type")), None)
    return (row.get(col) or "").strip().lower() or "not given"


def kappa(a, b):
    ids = sorted(set(a) & set(b))
    n = len(ids)
    po = sum(a[i] == b[i] for i in ids) / n
    pa1, pb1 = sum(a[i] for i in ids) / n, sum(b[i] for i in ids) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return n, po, (po - pe) / (1 - pe) if pe < 1 else 1.0


if len(sys.argv) < 4:
    sys.exit(__doc__)
A, B = read(sys.argv[1]), read(sys.argv[2])
key_path = Path(sys.argv[3])
F = read(sys.argv[4]) if len(sys.argv) > 4 else None
a, b = labels(A), labels(B)
key = {int(r["id"]): r for r in csv.DictReader(open(key_path, newline="", encoding="utf-8-sig"))}
meta_path = key_path.with_name(key_path.name.replace("label_key_", "label_meta_").replace(".csv", ".json"))
meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
if meta:
    print(f"sentences from {meta['source']} ({meta['sentences']} sentences, {len(meta['cases'])} cases)")
pop, smp = meta.get("strata_population", {}), meta.get("strata_sample", {})
weight = lambda i: (pop[key[i]["stratum"]] / smp[key[i]["stratum"]]) if key[i].get("stratum") in pop and smp.get(key[i]["stratum"]) else 1.0

n, po, k = kappa(a, b)
print(f"\nannotators: {n} sentences labelled by both; agreement {po:.1%}; Cohen's kappa {k:.2f}")
dis = [i for i in sorted(set(a) & set(b)) if a[i] != b[i]]
if dis:
    print(f"{len(dis)} disagreements (discuss, then fill a final sheet):")
    for i in dis:
        print(f"  id {i}: A={a[i]} B={b[i]}  {A[i]['sentence'][:110]}")

gold = labels(F) if F else {i: a[i] for i in a if i in b and a[i] == b[i]}
types = {i: etype((F or A)[i]) for i in gold if gold[i] == 0}
print(f"\ngold labels: {len(gold)} ({'adjudicated final sheet' if F else 'sentences both annotators agree on'})"
      + (" -- weighted back to the whole file (sampled)" if any(weight(i) != 1 for i in gold) else ""))

W = lambda ids: sum(weight(i) for i in ids)
unsup = [i for i in gold if gold[i] == 0]
print(f"unsupported before any checking: {W(unsup):.0f} of {W(gold):.0f} = {W(unsup) / W(gold):.1%}")
for name in ("basic", "strict"):
    col = f"ok_{name}"
    if not all(key[i].get(col, "") in ("0", "1") for i in gold):
        continue
    kept = [i for i in gold if key[i][col] == "1"]
    fn = [i for i in kept if gold[i] == 0]                                    # unsupported, accepted
    fp = [i for i in gold if key[i][col] == "0" and gold[i] == 1]           # supported, rejected
    label = "original checks (V1-V4)" if name == "basic" else "new checks (V1-V6)"
    print(f"\n{label}:")
    print(f"  unsupported after checking: {W(fn):.0f} of {W(kept):.0f} kept = {W(fn) / max(W(kept), 1e-9):.1%}")
    print(f"  unsupported sentences accepted: {W(fn):.0f} of {W(unsup):.0f} = {W(fn) / max(W(unsup), 1e-9):.1%}")
    sup = [i for i in gold if gold[i] == 1]
    print(f"  supported sentences rejected:  {W(fp):.0f} of {W(sup):.0f} = {W(fp) / max(W(sup), 1e-9):.1%}")
    by = Counter(types[i] for i in unsup)
    caught = Counter(types[i] for i in unsup if key[i][col] == "0")
    print("  by error type (caught / unsupported): " + ", ".join(f"{t} {caught[t]}/{by[t]}" for t in sorted(by)))
    for i in fn[:10]:
        print(f"    accepted but unsupported, id {i}: {A[i]['sentence'][:110]}")
    for i in fp[:10]:
        print(f"    rejected but supported, id {i}: {A[i]['sentence'][:90]} -> {key[i].get('reason_' + name, '')}")

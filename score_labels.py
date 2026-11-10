import csv, sys

LABEL = "label (1 = supported, 0 = not)"


def read(path):
    with open(path, newline="", encoding = "utf-8-sig") as f:
        return {int(r["id"]): r for r in csv.DictReader(f)}


def labels(rows):
    out = {}
    for i, r in rows.items():
        v = r[LABEL].strip()
        if v in ("0", "1"):
            out[i] = int(v)
    return out


def kappa(a, b):
    ids = sorted(set(a) & set(b))
    n = len(ids)
    po = sum(a[i] == b[i] for i in ids) / n
    pa1, pb1 = sum(a[i] for i in ids) / n, sum(b[i] for i in ids) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return n, po, (po - pe) / (1 - pe) if pe < 1 else 1.0


sheet_a, sheet_b, key_path = sys.argv[1], sys.argv[2], sys.argv[3]
final_path = sys.argv[4] if len(sys.argv) > 4 else None
A, B = read(sheet_a), read(sheet_b)
a, b = labels(A), labels(B)
key = {int(r["id"]): r for r in csv.DictReader(open(key_path, newline=""))}

n, po, k = kappa(a, b)
print(f"annotators: {n} sentences labelled by both, agreement {po:.1%}, Cohen's kappa {k:.2f}")
dis = [i for i in sorted(set(a) & set(b)) if a[i] != b[i]]
if dis:
    print(f"\n{len(dis)} disagreements (discuss these, then fill a final sheet):")
    for i in dis:
        print(f"  id {i}: A={a[i]} B={b[i]}  {A[i]['sentence'][:110]}")

gold = labels(read(final_path)) if final_path else {i: a[i] for i in a if i in b and a[i] == b[i]}
print(f"\ngold labels used: {len(gold)} ({'final sheet' if final_path else 'only sentences both agree on'})")

tp = sum(1 for i in gold if gold[i] == 0 and key[i]["check_ok"] == "0")
fp = sum(1 for i in gold if gold[i] == 1 and key[i]["check_ok"] == "0")
fn = sum(1 for i in gold if gold[i] == 0 and key[i]["check_ok"] == "1")
tn = sum(1 for i in gold if gold[i] == 1 and key[i]["check_ok"] == "1")
kept = tn + fn
print(f"\n                      people: supported   not supported")
print(f"checks kept               {tn:>10}   {fn:>13}")
print(f"checks removed            {fp:>10}   {tp:>13}")
print(f"\nprecision of removals (removed and really unsupported): {tp}/{tp + fp}" + (f" = {tp / (tp + fp):.1%}" if tp + fp else ""))
print(f"recall (unsupported sentences the checks caught):       {tp}/{tp + fn}" + (f" = {tp / (tp + fn):.1%}" if tp + fn else ""))
print(f"unsupported sentences that got through the checks:      {fn}/{kept}" + (f" = {fn / kept:.1%} of kept" if kept else ""))
if fn:
    print("\nmissed by the checks (labelled not supported, but kept):")
    for i in gold:
        if gold[i] == 0 and key[i]["check_ok"] == "1":
            print(f"  id {i}: {A[i]['sentence'][:120]}  | note: {A[i].get('note', '')}")

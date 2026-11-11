import random, sys, time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from hoofbeats.__main__ import load_kb
from hoofbeats.data import load_phenopackets
from hoofbeats.model import explain
import grounding as g

N = next((int(x) for x in sys.argv[1:] if x.isdigit()), 100)
STRICT = "--basic" not in sys.argv          # --basic: the original V1-V4 checker, for comparison
random.seed(0)
files = random.sample(sorted(Path("data/phenopacket-store/0.1.27").glob("*/*.json")), N)
r = load_kb(SimpleNamespace(kb="data/kb", common="none", prior_weight=1.0, absent_weight=0.5, absent_min_freq=0.8, use_onset=False, use_sex=False))


def name_of(it):
    return r.ont.name(r.ont.resolve(it.hpo_id)) if it.hpo_id else None


def perturb(s, pack, rng):
    """Planted errors: (kind, broken sentence) for one correct sentence."""
    out = []
    ids = g.cites(s)
    body = g.strip_cites(s).rstrip()
    cite = s[len(body):]
    out.append(("no citation", body))
    out.append(("unknown ID", f"{body} [K99]"))
    nums = g.NUMBER.findall(body)
    if nums:
        n = rng.choice(nums)
        out.append(("wrong number", body.replace(n, str(int(float(n)) + 13), 1) + cite))
    others = [i for i in pack.values() if i.kind == "knowledge" and i.id not in ids and i.hpo_id
              and name_of(i) and name_of(i).lower() not in body.lower()]
    if others and any(pack[i].hpo_id for i in ids):
        out.append(("wrong citation", s.replace(ids[-1], rng.choice(others).id)))
    for i in ids:
        it = pack[i]
        if it.kind == "patient" and it.status == "present" and name_of(it):
            out.append(("negated present finding", f"The patient does not have {name_of(it)} {cite.strip()}"))
            break
    return out


rng = random.Random(0)
stats, grades, by_check, caught, planted, missed = Counter(), Counter(), Counter(), Counter(), Counter(), []
t0 = time.time()
for f in files:
    case = load_phenopackets(f)[0]
    if not case.findings:
        continue
    ex = frozenset(case.source_pmids)
    ranking = r.score(case, ex)
    (d, _, p), (runner, _, _) = ranking.top(2)
    claims = explain(r, case, ranking, 1, ex, max_claims=8)[0][2]
    pack = g.build_pack(r, case, d, p, ex, runner, claims)
    names = [r.db.names[d], r.db.names[runner]]
    kept, removed = g.check_trace(g.template_trace(pack), pack, r, case, names, strict=STRICT)
    stats["cases"] += 1
    stats["claims"] += len(claims)
    stats["claims_kept"] += sum(1 for i in pack.values() if i.role in ("for", "against"))
    stats["kept"] += sum(len(v) for v in kept.values())
    stats["removed"] += len(removed)
    for x in removed:
        print("  template removed:", f.name, x["text"], "->", x["reason"])
    for sec, v in kept.items():
        for x in v:
            grades[" + ".join(x.get("sources") or [x["grade"]])] += 1
            for kind, bad in perturb(x["text"], pack, rng):
                planted[kind] += 1
                res = g.check_sentence(bad, pack, r, case, names, section=sec, strict=STRICT)
                if not res["ok"]:
                    caught[kind] += 1
                    by_check[res["reason"][:2]] += 1
                elif len(missed) < 5:
                    missed.append((kind, bad))

tot = stats["kept"] + stats["removed"]
print(f"\n{stats['cases']} cases in {time.time() - t0:.0f}s (leakage control on)")
print(f"knowledge-base claims: {stats['claims']}, re-verified {stats['claims_kept']}, "
      f"dropped by verify_claim {stats['claims'] - stats['claims_kept']}")
print(f"template sentences: {tot}, kept {stats['kept']} ({stats['kept'] / tot:.1%}), removed {stats['removed']}")
print("checker:", "strict (V1-V6)" if STRICT else "basic (V1-V4)")
print("evidence types of kept sentences:", dict(grades))
print("\nplanted errors caught:")
for k in planted:
    print(f"  {k:<25} {caught[k]}/{planted[k]}  ({caught[k] / planted[k]:.1%})")
print("caught by:", dict(by_check))
print("\nexamples not caught:")
for m in missed:
    print("  ", m)

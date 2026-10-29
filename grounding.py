from dataclasses import dataclass
from hoofbeats.model import verify_claim
import math

@dataclass
class Item:
    id: str
    kind: str
    text: str
    hpo_id:str = None
    status: str = None
    words: str = ""
    source: str = ""
    refs: tuple = ()
    role: str = ""

def patient_items(r, case, claims):
    """Make one P item for each patient finding that claim uses."""
    items = {}
    by_term = {r.ont.resolve(f.hpo_id): f for f in case.findings}
    for cl in claims:
        f = by_term.get(cl.patient_term)
        if f is None:
            continue
        if any(i.hpo_id == cl.patient_term for i in items.values()):
            continue
        pid = f"P{len(items)+1}"
        name = r.ont.name(cl.patient_term)
        status = f.status.value
        text = f"Patient: '{name}' is {status} (source: {f.source})"
        words = f.evidence if f.evidence and f.evidence.lower() != name.lower() else ""
        items[pid] = Item(
            pid,
            "patient",
            text,
            cl.patient_term,
            status,
            words,
            f.source)
    return items

def knowledge_items(r, case, d, claims, ex, start=1):
    """Make one K item for each knowledge claim."""
    items = {}
    lr_of = {c.hpo_id: c.log_lr for c in r.contributions(case, d, ex)}
    n = start
    for cl in claims:
        ok, reason = verify_claim(r, case, cl, ex)
        if not ok:
            print(" dropped", cl.text, "->", reason)
            continue
        role = "for" if cl.patient_term else "against"
        kid = f"K{n}"
        text = cl.text.split(" [")[0]
        items[kid] = Item(
            kid,
            "knowledge",
            text,
            cl.patient_term,
            cl.status.value,
            source = "HPO Annotations",
            refs = tuple(sorted(cl.refs)),
            role = role)
        n += 1
    return items

def known_terms(r, case):
    """Return every HPO term the record already covers: present findings and their ancestors, absent findings and their descendants."""
    known = set()
    for f in case.findings:
        t = r.ont.resolve(f.hpo_id)
        if t is None:
            continue
        if f.status.value == "present":
            known |= r.ont.ancestors(t)
        elif f.status.value == "absent":
            known |= r.ont.descendants(t)
    return known

def missing_items(r, case, d, ex, start, max_missing = 3):
    """Make one M item for each missing finding in the case."""
    items = {}
    name = r.db.names[d]
    known = known_terms(r, case)

    candidates = []
    for a in r.db.annotations(d, ex):
        t = r.ont.resolve(a.hpo_id)
        if a.negated or a.freq < 0.5 or t is None or t in known:
            continue
        candidates.append((a.freq, t, a.refs))

    candidates.sort(key=lambda x: (-x[0], r.ont.name(x[1])))

    seen = set()
    n = start
    for freq, t, refs in candidates:
        if t in seen:
            continue
        seen.add(t)
        kid = f"K{n}"
        text = f"{name} shows '{r.ont.name(t)}' in ~{freq:.0%} of cases; the patient record does not mention this finding."
        items[kid] = Item(
            kid,
            "knowledge",
            text,
            t,
            source="HPO Annotations",
            refs=tuple(sorted(refs)),
            role = "missing")
        n += 1
        if len(seen) >= max_missing:
            break
    return items

def runner_up_items(r, case, d, runner, ex, start, max_items = 2):
    """Make one R item for each runner-up disease."""
    items = {}
    mine = {c.hpo_id: c.log_lr for c in r.contributions(case, d, ex)}
    theirs = {c.hpo_id: c.log_lr for c in r.contributions(case, runner, ex)}

    diffs = []
    for t in mine:
        if t in theirs:
            diffs.append((mine[t] - theirs[t], t))
    diffs.sort(reverse=True)

    n = start
    for gap, t in diffs[:max_items]:
        if gap <0.5:
            break
        kid = f"K{n}"
        text = f"'{r.ont.name(t)}' fits {r.db.names[d]} better than {r.db.names[runner]}: " f"likelihood ratio {math.exp(mine[t]):.2f} against {math.exp(theirs[t]):.2f}"
        items[kid] = Item(
            kid,
            "knowledge",
            text,
            t,
            source="ranker contributions",
            role = "compare")
        n += 1
    return items

def build_pack(r, case, d, p, ex, runner, claims):
    pack = patient_items(r, case, claims)

    k = knowledge_items(r, case, d, claims, ex, start=1)
    pack.update(k)

    m = missing_items(r, case, d, ex, start=len(k)+1)
    pack.update(m)

    c = runner_up_items(r, case, d, runner, ex, start=len(k)+len(m)+1)
    pack.update(c)
    return pack
# fusion: merge findings from every reader into one finding per HPO term, with provenance
from __future__ import annotations

from dataclasses import replace

from .grounding import RANK as GROUNDING_RANK, GroundedFinding, level_of, provenance_entry
from .model import Finding, Ontology, Status

RANK = {Status.PRESENT: 3, Status.ABSENT: 2, Status.UNCERTAIN: 1}


def _merge(h: str, status: Status, fs: list[Finding]) -> Finding:
    # sources that reported this status, in input order; the best-grounded contributor gives quote and grounding
    source = ";".join(dict.fromkeys(f.source for f in fs))
    best = max(fs, key=lambda f: GROUNDING_RANK[level_of(f)])      # first one wins ties
    if isinstance(best, GroundedFinding):
        return replace(best, hpo_id=h, status=status, source=source)
    return Finding(h, status, source, fs[0].evidence)


def _add_entry(entries: list[dict], f: Finding):
    # one entry per contribution; the same span found again (e.g. by another reader) only adds its reader
    e = provenance_entry(f)
    if e.get("start") is not None:
        same = next((x for x in entries if all(x.get(k) == e.get(k) for k in ("source", "field", "start", "end", "status"))), None)
        if same:
            same["readers"] += [r for r in e["readers"] if r not in same["readers"]]
            return
    entries.append(e)


def fuse(finding_lists, ont: Ontology, drop_ancestors: bool = True) -> tuple[list[Finding], dict]:
    # one patient finding per HPO id (present > absent > uncertain); family findings are kept separately
    # provenance: hpo id -> [entry per contribution], plus "subsumed_by", "conflicts" and "unresolved"
    prov: dict = {"subsumed_by": {}, "conflicts": [], "unresolved": []}
    groups: dict[tuple[str, bool], list[Finding]] = {}
    for f in (f for fl in finding_lists for f in fl):
        h = ont.resolve(f.hpo_id)
        if h is None:
            prov["unresolved"].append({"hpo_id": f.hpo_id, "source": f.source, "evidence": f.evidence})
            continue
        _add_entry(prov.setdefault(h, []), f)
        groups.setdefault((h, f.status == Status.FAMILY), []).append(f)

    patient, family = {}, {}
    for (h, fam), fs in groups.items():
        if fam:
            family[h] = _merge(h, Status.FAMILY, fs)
            continue
        best = max((f.status for f in fs), key=RANK.get)
        if best == Status.PRESENT and any(f.status == Status.ABSENT for f in fs):
            prov["conflicts"].append({"hpo_id": h, "status": "absent", "contradicted_by": [h], "action": "dropped",
                                      "reason": "same term reported present and absent"})
        patient[h] = _merge(h, best, [f for f in fs if f.status == best])

    present = [h for h, f in patient.items() if f.status == Status.PRESENT]
    absent = [h for h, f in patient.items() if f.status == Status.ABSENT]
    drop = set()
    for a in absent:            # an absent term cannot be the ancestor of a present one
        below = [p for p in present if p != a and a in ont.ancestors(p)]
        if below:
            prov["conflicts"].append({"hpo_id": a, "status": "absent", "contradicted_by": below,
                                      "action": "dropped" if drop_ancestors else "kept",
                                      "reason": "absent term is an ancestor of a present term"})
            if drop_ancestors:
                drop.add(a)
    if drop_ancestors:
        for p in present:       # present: keep the most specific term
            below = [q for q in present if q != p and p in ont.ancestors(q)]
            if below:
                prov["subsumed_by"][p] = below
                drop.add(p)
        kept_absent = [a for a in absent if a not in drop]
        for a in kept_absent:   # absent: keep the most general term (lacking it implies lacking its subtypes)
            above = [b for b in kept_absent if b != a and b in ont.ancestors(a)]
            if above:
                prov["subsumed_by"][a] = above
                drop.add(a)
    return [f for h, f in patient.items() if h not in drop] + list(family.values()), prov

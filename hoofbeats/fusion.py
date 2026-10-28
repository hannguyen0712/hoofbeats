# fusion: merge findings from every reader into one finding per HPO term, with provenance
from __future__ import annotations

from dataclasses import replace

from .grounding import RANK as GROUNDING_RANK, GroundedFinding, level_of, provenance_entry
from .model import Finding, Ontology, Status

RANK = {Status.PRESENT: 3, Status.ABSENT: 2, Status.UNCERTAIN: 1}
STATUS_POLICIES = ("llm", "rules")


def _overlap(a: Finding, b: Finding) -> bool:
    return (isinstance(a, GroundedFinding) and isinstance(b, GroundedFinding) and a.field is not None and a.field == b.field
            and None not in (a.start, a.end, b.start, b.end) and a.start < b.end and b.start < a.end)


def _agreed(fs: list[Finding]) -> bool:
    # the rule reader and the LLM reader found this term with this status in overlapping words
    rules = [f for f in fs if getattr(f, "reader", "") == "rules"]
    return any(_overlap(r, g) for r in rules for g in fs if getattr(g, "reader", "") == "llm")


def _merge(h: str, status: Status, fs: list[Finding]) -> Finding:
    # sources that reported this status, in input order; the best-grounded contributor gives quote and grounding;
    # rules and LLM agreeing on the same words makes it strong
    source = ";".join(dict.fromkeys(f.source for f in fs))
    best = max(fs, key=lambda f: GROUNDING_RANK[level_of(f)])      # first one wins ties
    if isinstance(best, GroundedFinding):
        return replace(best, hpo_id=h, status=status, source=source, grounding="strong" if _agreed(fs) else best.grounding)
    return Finding(h, status, source, fs[0].evidence)


def _add_entry(entries: list[dict], f: Finding):
    # one entry per contribution; the same span found again (e.g. by another reader) only adds its reader. Another
    # reader's overlapping span with the same status also joins the entry (its words kept in other_spans);
    # rules + llm together make the entry strong
    e = provenance_entry(f)
    if e.get("start") is not None:
        same = next((x for x in entries if all(x.get(k) == e.get(k) for k in ("source", "field", "start", "end", "status"))), None)
        if same is None:
            same = next((x for x in entries if x.get("start") is not None and x.get("field") == e["field"]
                         and x["status"] == e["status"] and x["start"] < e["end"] and e["start"] < x["end"]
                         and not set(e["readers"]) & set(x["readers"])), None)
            if same is not None:
                same.setdefault("other_spans", []).append({k: e[k] for k in ("readers", "quote", "start", "end", "link")})
        if same:
            same["readers"] += [r for r in e["readers"] if r not in same["readers"]]
            if {"rules", "llm"} <= set(same["readers"]):
                same["grounding"] = "strong"
            return
    entries.append(e)


def _status_conflicts(fs: list[tuple[str, Finding]], ont: Ontology, policy: str, prov: dict) -> list[tuple[str, Finding]]:
    # the rules and the LLM give the same term (or a parent/child) different statuses on overlapping words: keep the
    # status of the reader the policy names, drop the other finding and record it
    if policy not in STATUS_POLICIES:
        raise ValueError(f"status policy must be one of {STATUS_POLICIES}")
    rel = lambda a, b: a == b or a in ont.ancestors(b) or b in ont.ancestors(a)
    by = {r: [(h, f) for h, f in fs if getattr(f, "reader", "") == r] for r in STATUS_POLICIES}
    drop = set()
    for hr, r in by["rules"]:
        for hl, g in by["llm"]:
            if r.status != g.status and rel(hr, hl) and _overlap(r, g):
                (lh, lose), (wh, win) = ((hr, r), (hl, g)) if policy == "llm" else ((hl, g), (hr, r))
                if id(lose) not in drop:
                    drop.add(id(lose))
                    prov["conflicts"].append({"hpo_id": lh, "status": lose.status.value, "contradicted_by": [wh],
                                              "action": "dropped", "reason": f"{getattr(lose, 'reader', '')} status "
                                              f"{lose.status.value} vs {getattr(win, 'reader', '')} {win.status.value} on "
                                              f"the same words; status policy {policy}",
                                              "quote": lose.quote, "start": lose.start, "end": lose.end})
    return [(h, f) for h, f in fs if id(f) not in drop]


def fuse(finding_lists, ont: Ontology, drop_ancestors: bool = True, status_policy: str = "llm") -> tuple[list[Finding], dict]:
    # one patient finding per HPO id (present > absent > uncertain); family findings are kept separately
    # provenance: hpo id -> [entry per contribution], plus "subsumed_by", "conflicts" and "unresolved"
    # status_policy: whose status wins when the rules and the LLM disagree on the same words (llm | rules)
    prov: dict = {"subsumed_by": {}, "conflicts": [], "unresolved": []}
    groups: dict[tuple[str, bool], list[Finding]] = {}
    resolved = []
    for f in (f for fl in finding_lists for f in fl):
        h = ont.resolve(f.hpo_id)
        if h is None:
            prov["unresolved"].append({"hpo_id": f.hpo_id, "source": f.source, "evidence": f.evidence})
            continue
        resolved.append((h, f))
    for h, f in _status_conflicts(resolved, ont, status_policy, prov):
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

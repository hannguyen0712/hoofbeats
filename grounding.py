from dataclasses import dataclass

@dataclass
class Item:
    id: str
    kind: str
    text: str
    hpo_id:str = None
    status: str = None
    words: str = ""
    source: str = ""

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
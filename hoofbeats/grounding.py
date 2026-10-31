# evidence grounding: findings that carry the exact words (and offsets) they came from, and checks on them
from __future__ import annotations

from dataclasses import dataclass

from .model import Case, Finding

RANK = {"strong": 3, "supported": 2, "weak": 1}


@dataclass(frozen=True)
class GroundedFinding(Finding):
    # a Finding plus where it came from; field names the input text, and text[start:end] == quote
    quote: str = ""
    field: str | None = None
    start: int | None = None
    end: int | None = None
    status_cue: tuple | None = None      # (word, start, end) in the same field
    link: tuple | None = None            # (method, matched label/synonym/value)
    grounding: str = "weak"              # strong | supported | weak
    reader: str = ""
    section: str | None = None
    lab: tuple | None = None             # numeric details as (key, value) pairs: value, unit, ref_low, ref_high, range_source, ...
    disease_mention: bool = False        # the words are also a knowledge-base disease name


def level_of(f: Finding) -> str:
    return f.grounding if isinstance(f, GroundedFinding) else "weak"


def provenance_entry(f: Finding) -> dict:
    e = {"source": f.source, "status": f.status.value, "evidence": f.evidence}
    if isinstance(f, GroundedFinding):
        e.update({"quote": f.quote, "field": f.field, "start": f.start, "end": f.end,
                  "status_cue": dict(zip(("word", "start", "end"), f.status_cue)) if f.status_cue else None,
                  "link": dict(zip(("method", "matched_string"), f.link)) if f.link else None,
                  "readers": [f.reader] if f.reader else [], "grounding": f.grounding, "section": f.section})
        if f.lab:
            e.update(dict(f.lab))
        if f.disease_mention:
            e["disease_mention"] = True
    return e


def verify_provenance(case: Case, texts: dict[str, str]) -> list[str]:
    # every recorded span must reproduce its quote (and cue word) in the original input text
    bad = []

    def check(what, field, start, end, quote):
        t = texts.get(field)
        if t is None:
            bad.append(f"{what}: no original text for field {field}")
        elif not (0 <= start <= end <= len(t)) or t[start:end] != quote:
            bad.append(f"{what}: {field}[{start}:{end}] is {t[start:end]!r}, not {quote!r}")

    for h, entries in case.meta.get("provenance", {}).items():
        if not h.startswith("HP:"):
            continue
        for e in entries:
            if e.get("field") is None or e.get("start") is None:
                continue
            check(f"{h} quote", e["field"], e["start"], e["end"], e["quote"])
            for o in e.get("other_spans", []):
                check(f"{h} quote ({'+'.join(o['readers'])})", e["field"], o["start"], o["end"], o["quote"])
            if e.get("status_cue"):
                c = e["status_cue"]
                check(f"{h} status cue", e["field"], c["start"], c["end"], c["word"])
    for e in case.meta.get("provenance", {}).get("normal_vitals", []):
        if e.get("field") is not None and e.get("start") is not None:
            check(f"normal {e['measure']} quote", e["field"], e["start"], e["end"], e["quote"])
    for d in case.meta.get("disease_mentions", []):
        check(f"disease mention {d['name']}", d["field"], d["start"], d["end"], d["quote"])
    return bad

# intake: the hoofbeats case format (notes, labs, vitals, images, genes, HPO terms) -> one fused Case
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .data import LABS, Extractor
from .fusion import fuse
from .model import Case, Finding, Ontology, Status

FORMAT = "hoofbeats-intake/1"
StatusName = Literal["present", "absent", "uncertain", "family"]


class IntakeError(ValueError):
    # the file itself cannot be read (not JSON, not an object); field problems are warnings instead
    pass


# schema
class _Item(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Lab(_Item):
    name: str
    value: float
    unit: Optional[str] = None
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None


class Vitals(_Item):
    bp: Optional[str] = Field(None, pattern=r"^\s*\d{2,3}\s*/\s*\d{2,3}\s*$")
    temp_c: Optional[float] = None
    hr: Optional[float] = None
    rr: Optional[float] = None
    spo2: Optional[float] = None


class HpoTerm(_Item):
    id: str = Field(pattern=r"^HP:\d{7}$")
    status: StatusName = "present"


class ImageFinding(_Item):
    hpo_id: Optional[str] = Field(None, pattern=r"^HP:\d{7}$")
    finding: Optional[str] = None
    status: Literal["present", "absent", "uncertain"] = "present"
    image_id: Optional[str] = None
    evidence: Optional[str] = None

    @model_validator(mode="after")
    def _needs_term(self):
        if not self.hpo_id and not (self.finding or "").strip():
            raise ValueError("needs hpo_id or finding")
        return self


class IntakeCase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    case_id: str = Field(min_length=1)
    note_text: Optional[str] = None
    image_context: Optional[str] = None
    labs: list[Lab] = []
    vitals: Optional[Vitals] = None
    age: Optional[Union[int, float, str]] = None
    sex: Optional[Literal["male", "female", "other", "unknown"]] = None
    genes: list[str] = []
    hpo_terms: list[HpoTerm] = []
    image_findings: list[ImageFinding] = []
    truth: list[str] = []      # gold disease ids, for evaluation only
    policy: Literal["open", "local_only"] = "local_only"

    @field_validator("sex", mode="before")
    @classmethod
    def _lower(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


LIST_FIELDS = ("labs", "genes", "hpo_terms", "image_findings", "truth")


def is_intake(obj) -> bool:
    return isinstance(obj, dict) and "case_id" in obj and not {"phenotypicFeatures", "interpretations"} & obj.keys()


def _why(e: ValidationError, depth: int) -> str:
    # depth: path parts to hide (field name, plus the item index for list fields)
    return "; ".join(f"{'.'.join(map(str, x['loc'][depth:])) or 'value'}: {x['msg']}" for x in e.errors())


def validate(obj: dict, warn, default_id: str = "case") -> IntakeCase:
    # field by field (list fields item by item), so one bad entry never discards the rest
    clean = {}
    for k, v in obj.items():
        if k not in IntakeCase.model_fields:
            warn(f"unknown field '{k}' ignored")
        elif k in LIST_FIELDS and isinstance(v, list):
            items = []
            for i, x in enumerate(v):
                try:
                    items += getattr(IntakeCase.model_validate({"case_id": "x", k: [x]}), k)
                except ValidationError as e:
                    warn(f"{k}[{i}] skipped ({_why(e, 2)})")
            clean[k] = items
        elif k == "vitals" and isinstance(v, dict):
            ok = {}
            for kk, vv in v.items():
                try:
                    ok.update(IntakeCase.model_validate({"case_id": "x", k: {kk: vv}}).vitals.model_dump(exclude_none=True))
                except ValidationError as e:
                    warn(f"vitals.{kk} skipped ({_why(e, 2)})")
            clean[k] = Vitals(**ok)
        else:
            try:
                clean[k] = getattr(IntakeCase.model_validate({"case_id": "x", k: v}), k)
            except ValidationError as e:
                warn(f"{k} skipped ({_why(e, 1)})")
    if "case_id" not in clean:
        warn(f"case_id missing or invalid; using '{default_id}'")
        clean["case_id"] = default_id
    return IntakeCase(**clean)


# readers: each returns findings tagged with their source and the words or value behind them
def read_hpo_terms(items: list[HpoTerm], ont: Ontology, warn) -> list[Finding]:
    out = []
    for t in items:
        h = ont.resolve(t.id)
        if h is None:
            warn(f"hpo_terms: {t.id} is not a current HPO phenotype term; skipped")
            continue
        out.append(Finding(h, Status(t.status), "input:hpo", t.id))
    return out


def read_text(text: str | None, ext: Extractor, source: str) -> list[Finding]:
    return ext.findings(text, source) if text and text.strip() else []


def read_image_findings(items: list[ImageFinding], ont: Ontology, ext: Extractor, warn) -> list[Finding]:
    out = []
    for x in items:
        src, st = f"image:{x.image_id or 'unknown'}", Status(x.status)
        if x.hpo_id:
            h = ont.resolve(x.hpo_id)
            if h is None:
                warn(f"image_findings: {x.hpo_id} is not a current HPO phenotype term; skipped")
                continue
            out.append(Finding(h, st, src, x.evidence or x.hpo_id))
            continue
        hits = ext.findings(x.finding)
        if not hits:
            warn(f"image_findings: no HPO term found for '{x.finding}'; skipped")
        out += [Finding(f.hpo_id, st, src, x.evidence or x.finding) for f in hits]   # the stated status wins
    return out


def _lab_index() -> dict:
    # full LABS label or the abbreviation in its parentheses, case-insensitive
    idx = {}
    for (label, _), hpos in LABS.items():
        for name in (label, *re.findall(r"\(([^)]+)\)", label)):
            idx.setdefault(name.lower(), (label, hpos))
    return idx


LAB_INDEX = _lab_index()


def read_labs(items: list[Lab], warn) -> list[Finding]:
    # value vs reference range for labs in the LABS map; present if out of range that way, else absent
    out = []
    for lab in items:
        hit = LAB_INDEX.get(lab.name.strip().lower())
        if hit is None:
            warn(f"labs: '{lab.name}' is not in the lab map; skipped")
            continue
        if lab.ref_low is None and lab.ref_high is None:
            warn(f"labs: '{lab.name}' has no reference range; skipped")
            continue
        label, (low, high, _) = hit
        rng = f"{'' if lab.ref_low is None else f'{lab.ref_low:g}'}-{'' if lab.ref_high is None else f'{lab.ref_high:g}'}"
        ev = f"{lab.name}={lab.value:g}{' ' + lab.unit if lab.unit else ''} (ref {rng})"
        found = [Finding(hpo, Status.PRESENT if abnormal else Status.ABSENT, f"lab:{label}", ev)
                 for hpo, bound, abnormal in ((low, lab.ref_low, lab.ref_low is not None and lab.value < lab.ref_low),
                                              (high, lab.ref_high, lab.ref_high is not None and lab.value > lab.ref_high))
                 if hpo and bound is not None]
        if not found:
            warn(f"labs: '{lab.name}' has no HPO term for the given reference range; skipped")
        out += found
    return out


# case
def parse_case(obj, ont: Ontology, ext: Extractor | None = None, *, drop_ancestors: bool = True, path=None) -> Case:
    if not isinstance(obj, dict):
        raise IntakeError("a case must be a JSON object")
    warnings: list[str] = []
    d = validate(obj, warnings.append, Path(path).stem if path else "case")
    ext = ext or Extractor(ont)
    findings, prov = fuse([read_hpo_terms(d.hpo_terms, ont, warnings.append),
                           read_text(d.note_text, ext, "note"),
                           read_text(d.image_context, ext, "image_context"),
                           read_image_findings(d.image_findings, ont, ext, warnings.append),
                           read_labs(d.labs, warnings.append)], ont, drop_ancestors)
    warnings += [f"{c['hpo_id']} {ont.name(c['hpo_id'])}: {c['reason']} "
                 f"({', '.join(c['contradicted_by'])}); absent {c['action']}" for c in prov["conflicts"]]
    meta = {"input_format": FORMAT, "policy": d.policy, "images": [], "age": d.age, "sex": d.sex,
            "vitals": d.vitals.model_dump(exclude_none=True) if d.vitals else {},
            "provenance": prov, "intake_warnings": warnings}
    if path:
        meta["path"] = str(path)
    return Case(d.case_id, findings, list(dict.fromkeys(d.truth)), list(dict.fromkeys(g.strip() for g in d.genes if g.strip())),
                meta=meta)


def load_case(path, ont: Ontology, ext: Extractor | None = None, *, drop_ancestors: bool = True) -> Case:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise IntakeError(f"{path}: cannot read case file ({e})") from None
    return parse_case(obj, ont, ext, drop_ancestors=drop_ancestors, path=path)


def intake_summary(c: Case, ont: Ontology) -> str:
    prov, m = c.meta.get("provenance", {}), c.meta
    st = Counter(f.status.value for f in c.findings)
    src = Counter(e["source"] for h, es in prov.items() if h.startswith("HP:") for e in es)
    src.update(e["source"] for e in prov.get("unresolved", []))
    ctx = [f"age {m['age']}" if m.get("age") is not None else "", f"sex {m['sex']}" if m.get("sex") else "",
           " ".join(f"{k} {v:g}" if isinstance(v, float) else f"{k} {v}" for k, v in m.get("vitals", {}).items()),
           f"genes {', '.join(c.genes)}" if c.genes else ""]
    lines = [f"[intake] {c.case_id}: {len(c.findings)} findings ("
             + ", ".join(f"{n} {s}" for s, n in st.most_common()) + f"); policy {m.get('policy')}",
             "  from: " + (", ".join(f"{s} {n}" for s, n in src.most_common()) or "nothing")]
    if any(ctx):
        lines.append("  context: " + "; ".join(x for x in ctx if x))
    for h, by in prov.get("subsumed_by", {}).items():
        lines.append(f"  subsumed: {ont.name(h)} ({h}) <- " + ", ".join(f"{ont.name(b)} ({b})" for b in by))
    lines += [f"  warning: {w}" for w in m.get("intake_warnings", [])]
    return "\n".join(lines)

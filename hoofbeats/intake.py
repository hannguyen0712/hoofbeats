# intake: the hoofbeats case format (notes, labs, vitals, images, genes, HPO terms) -> one fused Case
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .data import DiseaseNames, Extractor, Mention
from .fusion import fuse
from .grounding import GroundedFinding, level_of, verify_provenance
from .numbers import age_from_text, age_years, read_numbers, read_structured_labs, read_structured_vitals
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


def validate(obj: dict, warn, default_id: str = "case", positions: dict | None = None) -> IntakeCase:
    # field by field (list fields item by item), so one bad entry never discards the rest;
    # positions (if given) receives each list field's original item indices
    clean = {}
    for k, v in obj.items():
        if k not in IntakeCase.model_fields:
            warn(f"unknown field '{k}' ignored")
        elif k in LIST_FIELDS and isinstance(v, list):
            items, kept = [], []
            for i, x in enumerate(v):
                try:
                    items += getattr(IntakeCase.model_validate({"case_id": "x", k: [x]}), k)
                    kept.append(i)
                except ValidationError as e:
                    warn(f"{k}[{i}] skipped ({_why(e, 2)})")
            clean[k] = items
            if positions is not None:
                positions[k] = kept
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


# readers: each returns findings tagged with their source and the exact words or value behind them
def _grounded(m: Mention, source: str, field: str, status: Status | None = None, evidence: str | None = None):
    # a text mention as a finding; with status given (stated elsewhere), the text's own cue is not the reason
    return GroundedFinding(m.hpo_id, status or m.status, source, evidence or m.quote, quote=m.quote, field=field,
                           start=m.start, end=m.end, status_cue=None if status else m.cue, link=m.link,
                           grounding=m.grounding, reader="rules", section=m.section or None,
                           disease_mention=m.disease_mention)


def read_hpo_terms(items: list[HpoTerm], ont: Ontology, warn, pos=None) -> list[Finding]:
    out = []
    for i, t in zip(pos or range(len(items)), items):
        h = ont.resolve(t.id)
        if h is None:
            warn(f"hpo_terms: {t.id} is not a current HPO phenotype term; skipped")
            continue
        out.append(GroundedFinding(h, Status(t.status), "input:hpo", t.id, quote=t.id, field=f"hpo_terms[{i}].id",
                                   start=0, end=len(t.id), link=("hpo_id", t.id), grounding="strong", reader="input"))
    return out


def read_text(text: str | None, ext: Extractor, source: str, field: str | None = None, diseases=None,
              disease_mentions: list | None = None) -> list[Finding]:
    # disease_mentions (if given) receives the disease names found in the text
    if not text or not text.strip():
        return []
    found, dm = ext.analyze(text, diseases)
    if disease_mentions is not None:
        disease_mentions += [{**x, "field": field or source, "source": source} for x in dm]
    return [_grounded(m, source, field or source) for m in found]


def read_image_findings(items: list[ImageFinding], ont: Ontology, ext: Extractor, warn, pos=None) -> list[Finding]:
    out = []
    for i, x in zip(pos or range(len(items)), items):
        src, st = f"image:{x.image_id or 'unknown'}", Status(x.status)
        if x.hpo_id:
            h = ont.resolve(x.hpo_id)
            if h is None:
                warn(f"image_findings: {x.hpo_id} is not a current HPO phenotype term; skipped")
                continue
            out.append(GroundedFinding(h, st, src, x.evidence or x.hpo_id, quote=x.hpo_id, field=f"image_findings[{i}].hpo_id",
                                       start=0, end=len(x.hpo_id), link=("hpo_id", x.hpo_id), grounding="strong",
                                       reader="input"))
            continue
        hits = ext.mentions(x.finding)
        if not hits:
            warn(f"image_findings: no HPO term found for '{x.finding}'; skipped")
        out += [_grounded(m, src, f"image_findings[{i}].finding", st, x.evidence or x.finding) for m in hits]
    return out


def read_labs(items: list[Lab], warn, age: float | None = None) -> list[Finding]:
    # labs from the case file, through the numbers reader: the input reference range wins, else the default table
    return read_structured_labs(items, warn, age)


def original_texts(d: IntakeCase, pos: dict) -> dict[str, str]:
    # every input string a provenance span can point into, by field name
    texts = {"note_text": d.note_text or "", "image_context": d.image_context or ""}
    if d.vitals and d.vitals.bp:
        texts["vitals.bp"] = d.vitals.bp
    texts.update({f"hpo_terms[{i}].id": t.id for i, t in zip(pos.get("hpo_terms", []), d.hpo_terms)})
    for i, x in zip(pos.get("image_findings", []), d.image_findings):
        texts.update({f"image_findings[{i}].hpo_id": x.hpo_id or "", f"image_findings[{i}].finding": x.finding or ""})
    return texts


def parse_case(obj, ont: Ontology, ext: Extractor | None = None, *, drop_ancestors: bool = True, path=None,
               disease_names: dict[str, str] | None = None, emit_normal_vitals: bool = False) -> Case:
    # disease_names: knowledge-base names (id -> name); their mentions are recorded, never turned into findings.
    # emit_normal_vitals: normal vital signs also give absent findings; when off they are only recorded in
    # meta["provenance"]["normal_vitals"]
    if not isinstance(obj, dict):
        raise IntakeError("a case must be a JSON object")
    warnings: list[str] = []
    pos: dict = {}
    d = validate(obj, warnings.append, Path(path).stem if path else "case", pos)
    ext = ext or Extractor(ont)
    diseases = DiseaseNames(disease_names, ext) if disease_names else None
    mentions: list[dict] = []
    age = age_years(d.age) if d.age is not None else age_from_text(d.note_text)
    w = warnings.append
    normals: list[dict] = []
    nv = {"emit_normal_vitals": emit_normal_vitals, "normals": normals}
    findings, prov = fuse([read_hpo_terms(d.hpo_terms, ont, w, pos.get("hpo_terms")),
                           read_text(d.note_text, ext, "note", "note_text", diseases, mentions),
                           read_numbers(d.note_text, "note", "note_text", age, w, **nv),
                           read_text(d.image_context, ext, "image_context", "image_context", diseases, mentions),
                           read_numbers(d.image_context, "image_context", "image_context", age, w, **nv),
                           read_image_findings(d.image_findings, ont, ext, w, pos.get("image_findings")),
                           read_labs(d.labs, w, age), read_structured_vitals(d.vitals, w, age, **nv)], ont, drop_ancestors)
    prov["normal_vitals"] = normals
    warnings += [f"{c['hpo_id']} {ont.name(c['hpo_id'])}: {c['reason']} "
                 f"({', '.join(c['contradicted_by'])}); absent {c['action']}" for c in prov["conflicts"]]
    meta = {"input_format": FORMAT, "policy": d.policy, "images": [], "age": d.age, "sex": d.sex,
            "vitals": d.vitals.model_dump(exclude_none=True) if d.vitals else {},
            "provenance": prov, "disease_mentions": mentions, "intake_warnings": warnings}
    if path:
        meta["path"] = str(path)
    case = Case(d.case_id, findings, list(dict.fromkeys(d.truth)),
                list(dict.fromkeys(g.strip() for g in d.genes if g.strip())), meta=meta)
    warnings += [f"provenance: {p}" for p in verify_provenance(case, original_texts(d, pos))]
    return case


def load_case(path, ont: Ontology, ext: Extractor | None = None, *, drop_ancestors: bool = True,
              disease_names: dict[str, str] | None = None, emit_normal_vitals: bool = False) -> Case:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise IntakeError(f"{path}: cannot read case file ({e})") from None
    return parse_case(obj, ont, ext, drop_ancestors=drop_ancestors, path=path, disease_names=disease_names,
                      emit_normal_vitals=emit_normal_vitals)


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
    g = Counter(level_of(f) for f in c.findings)
    lines.append("  grounding: " + ", ".join(f"{k} {g[k]}" for k in ("strong", "supported", "weak")))
    for d in m.get("disease_mentions", []):
        lines.append(f"  disease mention: '{d['quote']}' = {d['name']} ({', '.join(d['disease_ids'][:3])}"
                     f"{', ...' if len(d['disease_ids']) > 3 else ''}), {d['status']}"
                     + ("; also an HPO term, kept as a finding" if d["also_hpo"] else "; not used as a finding"))
    for h, by in prov.get("subsumed_by", {}).items():
        lines.append(f"  subsumed: {ont.name(h)} ({h}) <- " + ", ".join(f"{ont.name(b)} ({b})" for b in by))
    lines += [f"  warning: {w}" for w in m.get("intake_warnings", [])]
    return "\n".join(lines)

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

from .model import Annotation, AnnotationDB, parse_frequency

DEFAULT_PATHS = ("data/kb/common_hpo.tsv", "data/kb/common_hpo", "data/kb/common-diseases")
DEFAULT_FREQ = 0.3          # used when a file gives no frequency (text-mined annotations); HPO's "frequent" starts at 30%
DEFAULT_MASS = 0.5          # prior share of all symptom-profile common diseases together, when no prevalence is given
HP = re.compile(r"HP:\d{7}")
PMID = re.compile(r"PMID:\s*\d+")
DISEASE_ID = re.compile(r"^(?:(MESH|DOID|ICD10|ICD10CM|ICD9|UMLS|OMIM|ORPHA|MONDO|SNOMEDCT)[:_ ]?(\S+)|([DC]\d{6,7}))$", re.I)
FILE_TYPES = {".tsv", ".tab", ".txt", ".hpoa", ".csv"}
HEADERS = {  # normalised header name -> field
    "disease": "databaseid diseaseid dbobjectid disease meshid mesh doid id diseasecode",
    "name": "diseasename dbname name diseaselabel label diseaseterm",
    "hpo": "hpoid phenotypeid hpo hpotermid hpterm phenotypehpoid hpotermidentifier",
    "freq": "frequency freq frequencymodifier frequencyhpo", "qualifier": "qualifier negation not", "aspect": "aspect",
    "ref": "reference references pub pmid pmids source", "prevalence": "prevalence prior"}
FIELD_OF = {h: f for f, names in HEADERS.items() for h in names.split()}


def norm_id(raw: str) -> str | None:
    # "D003924", "MESH_D003924", "MeSH:D003924" -> "MESH:D003924"; other prefixes keep their name
    m = DISEASE_ID.match((raw or "").strip())
    return m and (f"MESH:{m[3]}" if m[3] else f"{m[1].upper()}:{m[2]}")


def parse_freq(raw: str) -> float | None:
    # share 0-1, HPO frequency term, n/m or x%; None if empty or unreadable
    try:
        x = float(raw)
        return x if 0 <= x <= 1 else None
    except ValueError:
        x = parse_frequency(raw.strip(), default=-1.0)
        return None if x < 0 else min(x, 1.0)


def parse_prevalence(raw: str) -> float | None:
    # 0.001, 0.1%, 1/1000 or 1:1000
    raw = raw.strip().replace(",", "")
    try:
        n, _, m = re.split(r"([/:])", raw, maxsplit=1) if re.search(r"[/:]", raw) else (raw.rstrip("%"), "", "")
        x = float(n) / float(m) if m else float(n) / (100 if raw.endswith("%") else 1)
    except (ValueError, ZeroDivisionError):
        return None
    return x if 0 < x <= 1 else None


def read_rows(path: Path) -> list[dict]:
    # one file -> rows {disease, name, hpo, freq, qualifier, aspect, ref, prevalence}; header or none, tabs or commas
    delim = "," if path.suffix == ".csv" else "\t"
    header, body = None, []
    for line in filter(str.strip, path.read_text(encoding="utf-8", errors="replace").splitlines()):
        cells = next(csv.reader([line.lstrip("#")], delimiter=delim))
        if header is None and not body and not HP.search(line) and len(cells) > 1:
            header = cells                            # first tabular line without an HP id, even after '#'
        elif not line.startswith("#"):                # skips #version:, #description: and other comments
            body.append(cells)
    if not body:
        return []
    cols = {}
    for i, h in enumerate(header or []):
        cols.setdefault(FIELD_OF.get(re.sub(r"[^a-z0-9]", "", h.lower())), i)
    cols.pop(None, None)
    if "hpo" not in cols:                             # no usable header: guess the columns from their content
        share = lambda i, ok: sum(i < len(r) and bool(ok(r[i].strip())) for r in body) / len(body)
        width = range(max(map(len, body)))
        for f, ok in (("hpo", HP.fullmatch), ("disease", norm_id)):
            best = max(width, key=lambda i: share(i, ok))
            if share(best, ok) > 0.5:
                cols.setdefault(f, best)
        cols.setdefault("name", next((i for i in width if i not in cols.values() and i > cols.get("disease", -1)
                                      and share(i, lambda v: re.search(r"[A-Za-z]{3}", v) and not HP.search(v)) > 0.5), None))
    if cols.get("hpo") is None:
        raise ValueError(f"{path}: no column of HPO ids found")
    file_id = norm_id(path.stem.split(".")[0])
    rows = []
    for cells in body:
        row = {f: cells[cols[f]].strip() if cols.get(f) is not None and cols[f] < len(cells) else "" for f in HEADERS}
        hpo, row["disease"] = HP.search(row["hpo"]), norm_id(row["disease"]) or file_id
        if hpo and row["disease"]:
            rows.append(row | {"hpo": hpo[0]})
    return rows


def find_default() -> str | None:
    return next((p for p in DEFAULT_PATHS if Path(p).exists()), None)


def load_common_hpo(db: AnnotationDB, path, default_freq: float = DEFAULT_FREQ, mass: float = DEFAULT_MASS,
                    min_terms: int = 3) -> dict:
    # add every disease in a file (or every file in a folder) to db as a common disease; returns a summary
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in FILE_TYPES) if root.is_dir() else [root]
    terms, refs, names, prev = defaultdict(dict), defaultdict(set), {}, {}
    estimated, existing, bad = set(), set(), 0
    for f in files:
        try:
            rows = read_rows(f)
        except ValueError:
            bad += 1
            continue
        for row in rows:
            did, hpo = row["disease"], db.ont.resolve(row["hpo"])
            if did in db.names:                       # already a rare disease or lab profile: never overwrite it
                existing.add(did)
            elif hpo and row["aspect"].upper() in ("", "P"):
                neg = row["qualifier"].upper() == "NOT"
                freq = 0.0 if neg else parse_freq(row["freq"])
                if freq is None:
                    freq = default_freq
                    estimated.add(did)
                names.setdefault(did, row["name"] or did)
                terms[did][hpo, neg] = max(terms[did].get((hpo, neg), 0.0), freq)
                refs[did, hpo, neg].update(m.replace(" ", "") for m in PMID.findall(row["ref"]))
                prev[did] = parse_prevalence(row["prevalence"]) or prev.get(did)
    added = [d for d, tt in terms.items() if sum(not neg for _, neg in tt) >= min_terms]
    without = sum(not prev[d] for d in added)
    for did in added:
        db.names[did], db.prior[did] = names[did], prev[did] or mass / without
        db.annos[did] = [Annotation(hpo, freq, frozenset(refs[did, hpo, neg]), neg) for (hpo, neg), freq in terms[did].items()]
        for r in set().union(*(a.refs for a in db.annos[did])):
            db.ref_index[r].add(did)
    # read by grounding.py: diseases whose frequencies are the default, not measured
    db.common_hpo = {"source": str(root), "diseases": set(added), "estimated": estimated & set(added),
                     "default_freq": default_freq}
    return {"path": str(root), "files": len(files), "diseases": len(added), "estimated": len(db.common_hpo["estimated"]),
            "with_prevalence": len(added) - without, "skipped_existing": len(existing),
            "skipped_few_terms": len(terms) - len(added), "unreadable_files": bad}


def is_estimated(db, d: str) -> bool:
    return d in getattr(db, "common_hpo", {}).get("estimated", ())


def is_symptom_profile(db, d: str) -> bool:
    return d in getattr(db, "common_hpo", {}).get("diseases", ())
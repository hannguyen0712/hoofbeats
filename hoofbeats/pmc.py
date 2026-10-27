# pmc: a text-reader test set from PubMed Central case descriptions, with Phenopacket Store features as gold.
# Only PMIDs and PMCIDs are sent (NCBI ID converter and BioC API, tool=hoofbeats); responses are cached under data/pmc/.
# The locator rules below are fixed before any extractor output is seen; a change bumps LOCATOR_VERSION and the
# manifest is rebuilt and re-frozen.
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

LOCATOR_VERSION = 1
SEED = 576
CACHE = Path("data/pmc")
UA = {"User-Agent": "hoofbeats/0.1 (research; tool=hoofbeats)"}
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=hoofbeats&format=json&idtype=pmid&ids="
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{}/unicode"

MAX_PER_PAPER, MAX_PER_DISEASE = 3, 2
TARGET = {"single": 200, "multi": 100}
DEV_SHARE = 3                  # md5(PMID) % 10 < 3 -> dev
MIN_WORDS = 40


def get(url: str, path: Path) -> bytes | None:
    # cached GET; a 400/404 is cached as empty, other failures are retried and not cached
    if path.exists():
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    time.sleep(0.4)                                  # under 3 requests per second without an API key
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
            path.write_bytes(data)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                path.write_bytes(b"")
                return b""
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 * (attempt + 1))
    return None


def md5_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def split_of(pmid: str) -> str:
    return "dev" if md5_int(pmid) % 10 < DEV_SHARE else "test"


# Phenopacket Store
def store_patients(store) -> list[dict]:
    # one row per phenopacket that cites exactly one PMID: subject id, disease, gold features (excluded = absent)
    out = []
    for p in sorted(Path(store).rglob("*.json")):
        try:
            o = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(o, dict) or "phenotypicFeatures" not in o:
            continue
        refs = o.get("metaData", {}).get("externalReferences", [])
        pmids = sorted({r["id"][5:] for r in refs if r.get("id", "").startswith("PMID:")}
                       or set(re.findall(r"PMID_(\d+)", o.get("id", ""))))
        gold = {}
        for f in o["phenotypicFeatures"]:
            t = f.get("type", {})
            st = "absent" if f.get("excluded") else "present"
            if t.get("id"):
                gold.setdefault(t["id"], {"hpo_id": t["id"], "label": t.get("label", ""), "status": set()})["status"].add(st)
        gold = [{**g, "status": g["status"].pop()} for g in gold.values() if len(g["status"]) == 1]
        dz = (o.get("diseases") or [{}])[0].get("term", {})
        out.append({"file": str(p.relative_to(store)), "phenopacket_id": o.get("id", ""), "pmids": pmids,
                    "subject_id": o.get("subject", {}).get("id", ""), "disease_id": dz.get("id", ""),
                    "disease_label": dz.get("label", ""), "gold": gold})
    return out


# fetching
def fetch(pmids: list[str], log=print) -> dict[str, dict]:
    # PMID -> {pmcid, license, has_text}; BioC JSON cached as data/pmc/bioc/<PMCID>.json
    recs = {}
    for i in range(0, len(pmids), 200):
        batch = pmids[i:i + 200]
        key = hashlib.md5(",".join(batch).encode()).hexdigest()[:12]
        raw = get(IDCONV + ",".join(batch), CACHE / "idconv" / f"{key}.json")
        for r in json.loads(raw or b'{"records": []}').get("records", []):
            if r.get("pmcid"):
                recs[str(r.get("pmid"))] = r["pmcid"]
    out = {}
    for n, pmid in enumerate(pmids, 1):
        pmcid = recs.get(pmid)
        row = {"pmcid": pmcid, "license": None, "has_text": False}
        if pmcid:
            doc = bioc_document(pmcid, fetch=True)
            if doc:
                row["license"] = doc.get("infons", {}).get("license")
                row["has_text"] = bool(doc.get("passages"))
        out[pmid] = row
        if n % 50 == 0:
            log(f"[pmc] {n}/{len(pmids)} papers checked")
    return out


def bioc_document(pmcid: str, fetch: bool = False) -> dict | None:
    path = CACHE / "bioc" / f"{pmcid}.json"
    raw = get(BIOC.format(pmcid), path) if fetch else (path.read_bytes() if path.exists() else None)
    if not raw:
        return None
    try:
        doc = json.loads(raw)
    except ValueError:
        return None
    doc = doc[0] if isinstance(doc, list) and doc else doc
    docs = (doc or {}).get("documents", [])
    return docs[0] if docs else None


def open_licence(lic: str | None) -> bool:
    # Creative Commons only: author manuscripts and "NO-CC CODE" are excluded
    return bool(lic) and lic.upper().startswith("CC")


# locator rules, version 1
NEVER = {"INTRO", "DISCUSS", "REF", "FIG", "TABLE", "SUPPL", "ABSTRACT", "TITLE", "ACK_FUND", "COMP_INT", "AUTH_CONT",
         "CONCL", "APPENDIX", "REVIEW_INFO", "KEYWORD", "ABBR"}
ALLOWED = {"CASE", "RESULTS", "METHODS", ""}
CASE_HEADING = re.compile(
    r"\b(?:case\s+(?:reports?|presentations?|descriptions?|histor(?:y|ies)|summar(?:y|ies)|details?)"
    r"|clinical\s+(?:reports?|descriptions?|summar(?:y|ies)|findings|presentations?|features|histor(?:y|ies)|data)"
    r"|patients?\s+(?:descriptions?|presentations?|reports?)|subjects?)\b"
    r"|^\s*(?:case|patient|proband)(?:\s+\d+)?\s*$", re.I)
EXCLUDED_HEADING = re.compile(r"\b(?:and|&)\s+methods?\b|\bmethods?\s+(?:and|&)\b|\bmaterials\b|\bethic|\bconsent\b|"
                              r"\bdeclaration|\bfunding\b|\backnowledg|\babbreviation|\bavailability\b", re.I)


def passages(doc: dict) -> list[dict]:
    # paragraphs with their section type and the stack of headings above them
    out, stack, sec = [], [], None
    for i, p in enumerate(doc.get("passages", [])):
        inf = p.get("infons", {})
        st, typ, text = inf.get("section_type", "") or "", inf.get("type", "") or "", p.get("text", "") or ""
        if st != sec:
            stack, sec = [], st
        if typ.startswith("title"):
            m = re.search(r"(\d+)$", typ)
            level = int(m.group(1)) if m else 1
            stack = stack[:level - 1] + [text]
            continue
        if typ != "paragraph" or not text.strip():
            continue
        out.append({"index": i, "offset": p.get("offset"), "section_type": st, "headings": list(stack), "text": text})
    return out


def _usable(p: dict) -> bool:
    return p["section_type"] in ALLOWED and p["section_type"] not in NEVER and \
        not any(EXCLUDED_HEADING.search(h) for h in p["headings"])


def single_patient_paragraphs(doc: dict) -> list[tuple[dict, str]]:
    # a CASE section, or paragraphs under a case-description heading
    out = []
    for p in passages(doc):
        if not _usable(p):
            continue
        if p["section_type"] == "CASE":
            out.append((p, "case_section"))
        elif any(CASE_HEADING.search(h) for h in p["headings"]):
            out.append((p, "case_heading"))
    return out


# patient ids: keys are ("label", "3"), ("ped", "II", "3") or ("code", "kbg3")
LABEL = r"([A-Z]{0,5}\d+(?:[.\-]\d+)?[a-z]?)"
END = r"(?![\w]|[\-.:]\d)"
ROLE_RX = re.compile(r"\b(?:patient|individual|case|subject|proband|pt\.?|ind\.?)\s*[-#]?\s*" + LABEL + END, re.I)
P_RX = re.compile(r"(?<![\w\-])PT?-?\s?(\d+[a-z]?)" + END)
PLURAL_RX = re.compile(r"\b(?:patients|individuals|cases|subjects|probands|pts)\s+(\d+[a-z]?(?:\s*(?:,|and|&|-|–|to)\s*\d+[a-z]?)+)", re.I)
PED_RX = re.compile(r"(?<![\w\-])([IVX]{1,4})\s*[-:.]\s*(\d{1,2})(?!\d)")
CODE_RX = re.compile(r"[A-Za-z]{1,6}[-]?[A-Za-z]?\d+(?:[-.]\d+)?")


def subject_keys(subject_id: str) -> set[tuple]:
    # how the paper may name this patient; empty if the id cannot be parsed
    s = subject_id.replace("_", " ").strip()
    keys = {("ped", a, str(int(b))) for a, b in PED_RX.findall(s)}
    m = ROLE_RX.search(s)
    if m:
        keys.add(("label", m.group(1).lower()))
        if re.search(r"[A-Za-z]", m.group(1)):
            keys.add(("code", m.group(1).lower()))
    elif not keys:
        m = P_RX.search(s)
        if m:
            keys.add(("label", m.group(1).lower()))
        elif re.fullmatch(r"\d+", s):
            keys.add(("label", s))
        elif CODE_RX.fullmatch(s):
            keys.add(("code", s.lower()))
    return keys


def text_keys(text: str, codes: set[str], plural: bool = True) -> list[tuple[int, tuple]]:
    # (position, key) for every patient named in text; plural=False leaves out lists ("patients 1 and 3")
    out = [(m.start(), ("label", m.group(1).lower())) for m in ROLE_RX.finditer(text)]
    out += [(m.start(), ("label", m.group(1).lower())) for m in P_RX.finditer(text)]
    for m in PLURAL_RX.finditer(text if plural else ""):
        nums = re.findall(r"\d+[a-z]?|-|–|to", m.group(1))
        for i, x in enumerate(nums):
            if x in ("-", "–", "to") and 0 < i < len(nums) - 1 and nums[i - 1].isdigit() and nums[i + 1].isdigit():
                out += [(m.start(), ("label", str(k))) for k in range(int(nums[i - 1]) + 1, int(nums[i + 1]))]
            elif x not in ("-", "–", "to"):
                out.append((m.start(), ("label", x.lower())))
    out += [(m.start(), ("ped", m.group(1), str(int(m.group(2))))) for m in PED_RX.finditer(text)]
    for c in codes:
        out += [(m.start(), ("code", c)) for m in re.finditer(r"(?<![\w\-])" + re.escape(c) + r"(?![\w]|-\d)", text, re.I)]
    return out


def multi_patient_paragraphs(doc: dict, keys: dict[str, set]) -> dict[str, list[tuple[dict, str]]]:
    # subject id -> paragraphs that name this patient and no other patient (in the paragraph or its headings),
    # or that start with this patient's id (no other patient shares it or is named in the first sentence)
    codes = {k[1] for ks in keys.values() for k in ks if k[0] == "code"}
    owners = defaultdict(set)
    for sid, ks in keys.items():
        for k in ks:
            owners[k].add(sid)
    out = defaultdict(list)
    for p in passages(doc):
        if not _usable(p):
            continue
        named = text_keys(p["text"], codes)
        found = {k for _, k in named}
        head = {k for h in p["headings"] for _, k in text_keys(h, codes)}
        lead = p["text"].lstrip(" \t\"'(•-–")
        shift = len(p["text"]) - len(lead)
        first = min(text_keys(p["text"], codes, plural=False), default=None)
        m = re.search(r"[.!?](?:\s+|$)", p["text"])
        in_first = {k for _, k in text_keys(p["text"][:m.end() if m else len(p["text"])], codes)}
        for sid, ks in keys.items():
            if not ks:
                continue
            if first and first[0] <= shift + 3 and first[1] in ks and owners[first[1]] == {sid} and in_first <= ks:
                out[sid].append((p, "id_start"))
            elif found and found <= ks and all(owners[k] == {sid} for k in found) and (not head or head <= ks):
                out[sid].append((p, "id_only"))
            elif not found and head and head <= ks and all(owners[k] == {sid} for k in head):
                out[sid].append((p, "id_heading"))
    return out


def case_text(doc: dict, indexes: list[int]) -> str:
    ps = doc.get("passages", [])
    return "\n\n".join(ps[i]["text"].strip() for i in indexes)


def locator_commit(paths=("hoofbeats/pmc.py",)) -> str:
    # the commit the locator rules come from; refuses uncommitted changes to them
    dirty = subprocess.run(["git", "status", "--porcelain", *paths], capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"locator code has uncommitted changes; commit before freezing a manifest:\n{dirty}")
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def build_manifest(store, ont=None, log=print, do_fetch: bool = True) -> tuple[dict, list[dict]]:
    # every patient is located (or excluded with a reason) before selection; selection is by hash order with caps
    commit = locator_commit()
    pts = store_patients(store)
    by_pmid = defaultdict(list)
    excluded = []
    for p in pts:
        if len(p["pmids"]) != 1:
            excluded.append({**_ex(p), "reason": "not exactly one PMID"})
        else:
            by_pmid[p["pmids"][0]].append(p)
    papers = sorted(by_pmid, key=lambda x: md5_int(f"{SEED}:{x}"))
    info = fetch(papers, log) if do_fetch else {}
    located = []
    for pmid in papers:
        group = sorted(by_pmid[pmid], key=lambda p: md5_int(f"{SEED}:{pmid}:{p['phenopacket_id']}"))
        kind = "single" if len(group) == 1 else "multi"
        row = info.get(pmid, {})
        why = ("no PMCID" if not row.get("pmcid") else "no BioC full text" if not row.get("has_text")
               else f"licence {row.get('license')}" if not open_licence(row.get("license")) else None)
        if why:
            excluded += [{**_ex(p, pmid, kind), "reason": why} for p in group]
            continue
        doc = bioc_document(row["pmcid"])
        if kind == "single":
            found = {group[0]["phenopacket_id"]: single_patient_paragraphs(doc)}
        else:
            keys = {p["phenopacket_id"]: subject_keys(p["subject_id"]) for p in group}
            found = dict(multi_patient_paragraphs(doc, keys))
            for p in group:
                if not keys[p["phenopacket_id"]]:
                    found[p["phenopacket_id"]] = None
        for p in group:
            paras = found.get(p["phenopacket_id"], [])
            if paras is None:
                excluded.append({**_ex(p, pmid, kind), "reason": "subject id not parsed"})
                continue
            if not paras:
                excluded.append({**_ex(p, pmid, kind), "reason": "no located paragraph"})
                continue
            idx = [q["index"] for q, _ in paras]
            text = case_text(doc, idx)
            gold = p["gold"]
            if ont is not None:
                gold = [{**g, "hpo_id": ont.resolve(g["hpo_id"]) or g["hpo_id"]} for g in gold]
            if len(text.split()) < MIN_WORDS:
                excluded.append({**_ex(p, pmid, kind), "reason": f"under {MIN_WORDS} words"})
                continue
            if not any(g["status"] == "present" for g in gold):
                excluded.append({**_ex(p, pmid, kind), "reason": "no observed gold term"})
                continue
            located.append({"case_id": f"PMID{pmid}_{_slug(p['subject_id'] or p['phenopacket_id'])}", "pmid": pmid,
                            "pmcid": row["pmcid"], "license": row["license"], "split": split_of(pmid), "paper_kind": kind,
                            "paper_patients": len(group), "phenopacket_id": p["phenopacket_id"], "phenopacket_file": p["file"],
                            "subject_id": p["subject_id"], "disease_id": p["disease_id"], "disease_label": p["disease_label"],
                            "rules": sorted({r for _, r in paras}),
                            "passages": [{"index": q["index"], "offset": q["offset"], "section_type": q["section_type"],
                                          "headings": q["headings"], "rule": r} for q, r in paras],
                            "n_words": len(text.split()), "text_sha1": hashlib.sha1(text.encode()).hexdigest(), "gold": gold})
    cases, per_paper, per_disease, n = [], Counter(), Counter(), Counter()
    for c in located:
        reason = ("target reached" if n[c["paper_kind"]] >= TARGET[c["paper_kind"]] else
                  f"over {MAX_PER_PAPER} per paper" if per_paper[c["pmid"]] >= MAX_PER_PAPER else
                  f"over {MAX_PER_DISEASE} per disease" if per_disease[c["disease_id"]] >= MAX_PER_DISEASE else None)
        if reason:
            excluded.append({"phenopacket_id": c["phenopacket_id"], "pmid": c["pmid"], "paper_kind": c["paper_kind"],
                             "subject_id": c["subject_id"], "disease_id": c["disease_id"], "reason": reason})
            continue
        per_paper[c["pmid"]] += 1
        per_disease[c["disease_id"]] += 1
        n[c["paper_kind"]] += 1
        cases.append(c)
    seen = Counter(c["case_id"] for c in cases)
    for c in cases:
        if seen[c["case_id"]] > 1:
            c["case_id"] += "_" + c["phenopacket_id"][-8:]
    manifest = {"locator_version": LOCATOR_VERSION, "locator_commit": commit, "seed": SEED,
                "rules": {"licence": "Creative Commons only", "max_per_paper": MAX_PER_PAPER, "max_per_disease": MAX_PER_DISEASE,
                          "target": TARGET, "split": f"md5(PMID) % 10 < {DEV_SHARE} -> dev", "min_words": MIN_WORDS},
                "counts": {"store_patients": len(pts), "papers": len(papers), "located": len(located), "cases": len(cases),
                           "by_kind": dict(Counter(c["paper_kind"] for c in cases)),
                           "by_split": dict(Counter(c["split"] for c in cases)),
                           "papers_used": len({c["pmid"] for c in cases}),
                           "excluded": dict(Counter(e["reason"] for e in excluded))},
                "cases": cases}
    return manifest, excluded


def _ex(p, pmid=None, kind=None) -> dict:
    return {"phenopacket_id": p["phenopacket_id"], "pmid": pmid or ";".join(p["pmids"]), "paper_kind": kind,
            "subject_id": p["subject_id"], "disease_id": p["disease_id"]}


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:40]


def load_case_text(c: dict) -> str:
    # the located text of a manifest case, checked against its frozen hash
    doc = bioc_document(c["pmcid"])
    if doc is None:
        raise SystemExit(f"{c['case_id']}: {c['pmcid']} is not cached; run pmc-build first")
    text = case_text(doc, [p["index"] for p in c["passages"]])
    if hashlib.sha1(text.encode()).hexdigest() != c["text_sha1"]:
        raise SystemExit(f"{c['case_id']}: cached text differs from the frozen manifest")
    return text

# data loading: Phenopacket Store, MIMIC-IV, Orphadata, MedPix, note extraction
from __future__ import annotations

import csv
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from .model import Case, Finding, Ontology, Status

# download (open data only)
HPO_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/"
PPS_URL = "https://github.com/monarch-initiative/phenopacket-store/releases/latest/download/all_phenopackets.zip"


def download(dest="data"):
    kb, pps = Path(dest, "kb"), Path(dest, "phenopacket-store")
    kb.mkdir(parents=True, exist_ok=True)
    pps.mkdir(parents=True, exist_ok=True)
    for f in ("hp.obo", "phenotype.hpoa", "genes_to_disease.txt"):
        print("downloading", f)
        urllib.request.urlretrieve(HPO_URL + f, kb / f)
    print("downloading Phenopacket Store")
    urllib.request.urlretrieve(PPS_URL, pps / "all.zip")
    with zipfile.ZipFile(pps / "all.zip") as z:
        z.extractall(pps)
    print("versions:", [p.name for p in pps.iterdir() if p.is_dir()],
          next(l.strip() for l in open(kb / "hp.obo") if l.startswith("data-version")))


# Phenopacket Store
def load_phenopackets(root) -> list[Case]:
    # root is a folder (searched recursively) or one phenopacket file
    root, cases = Path(root), []
    for p in [root] if root.is_file() else sorted(root.rglob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(obj, dict) or "interpretations" not in obj:
            continue
        findings = [Finding(pf["type"]["id"], Status.ABSENT if pf.get("excluded") else Status.PRESENT,
                            "phenopacket", pf["type"].get("label", ""))
                    for pf in obj.get("phenotypicFeatures", []) if pf.get("type", {}).get("id")]
        truth, genes = [], []
        for it in obj["interpretations"]:
            diag = it.get("diagnosis", {})
            truth.append(diag.get("disease", {}).get("id"))
            for gi in diag.get("genomicInterpretations", []):
                genes.append(gi.get("variantInterpretation", {}).get("variationDescriptor", {})
                             .get("geneContext", {}).get("symbol"))
        pmids = [r["id"] for r in obj.get("metaData", {}).get("externalReferences", [])
                 if r.get("id", "").startswith("PMID:")]
        pmids += [f"PMID:{m}" for m in re.findall(r"PMID_(\d+)", obj.get("id", ""))]
        cases.append(Case(obj.get("id", p.stem), findings, [t for t in dict.fromkeys(truth) if t],
                          [g for g in dict.fromkeys(genes) if g], list(dict.fromkeys(pmids)),
                          {"path": str(p)}))
    return cases


def split_by_pmid(cases, min_pmid: int):
    # split cases by PMID as a time proxy, for LLM memorization checks
    after = [c for c in cases if c.source_pmids and min(int(p[5:]) for p in c.source_pmids) >= min_pmid]
    ids = {c.case_id for c in after}
    return [c for c in cases if c.case_id not in ids], after


# Orphadata (ICD-10 -> ORPHA labels)
def orphadata_icd10(xml_path, relations=("E", "BTNT")):
    # ICD-10 -> ORPHA map from Orphadata en_product1.xml (exact and broader mappings only)
    icd, names = defaultdict(set), {}
    for _, el in ET.iterparse(str(xml_path)):
        if el.tag != "Disorder" or not el.findtext("OrphaCode"):
            continue
        oid = "ORPHA:" + el.findtext("OrphaCode")
        names[oid] = [el.findtext("Name") or ""] + [s.text or "" for s in el.findall("SynonymList/Synonym")]
        for ref in el.findall("ExternalReferenceList/ExternalReference"):
            rel = (ref.findtext("DisorderMappingRelation/Name") or "").split(" ")[0]
            if ref.findtext("Source") == "ICD-10" and rel in relations:
                icd[ref.findtext("Reference").replace(".", "").upper()].add(oid)
        el.clear()
    return dict(icd), names


def match_icd(code: str, icd_map) -> set[str]:
    # match the longest Orphadata ICD-10 prefix of a MIMIC ICD-10-CM code
    c = code.replace(".", "").upper()
    return next((icd_map[c[:k]] for k in range(len(c), 2, -1) if c[:k] in icd_map), set())


# note extraction
NEG = ["no", "not", "denies", "denied", "without", "negative for", "absence of", "free of",
       "no evidence of", "no history of", "never", "ruled out"]
NEG_AFTER = ["was ruled out", "is ruled out", "not seen", "is absent", "was absent", "resolved"]
HEDGE = ["possible", "possibly", "probable", "suspected", "rule out", "r o", "concern for", "cannot exclude"]
STOP_SCOPE = {"but", "however", "although", "except"}
FAMILY = {"mother", "father", "brother", "sister", "sibling", "aunt", "uncle", "grandmother",
          "grandfather", "cousin", "maternal", "paternal", "family"}
SKIP_SECTIONS = ("allergies", "medications", "discharge medications", "medications on admission",
                 "discharge instructions", "followup instructions", "discharge disposition", "facility")
DX_SECTIONS = ("discharge diagnos", "final diagnos", "principal diagnos", "primary diagnos",
               "secondary diagnos", "diagnos", "discharge condition", "discharge instructions",
               "followup instructions")
HEADING = re.compile(r"^\s*([A-Za-z][A-Za-z /&\-]{2,40}):", re.M)


def tokens(text: str) -> list[str]:
    return [t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "is", "us")) else t
            for t in re.findall(r"[a-z0-9]+", text.lower())]


def _has(window, phrases):
    return any(tuple(window[i:i + len(p)]) == p for p in phrases for i in range(len(window)))


class Extractor:
    # offline HPO matcher with negation, family and uncertainty rules

    def __init__(self, ont: Ontology):
        self.index, self.vocab = defaultdict(list), set()
        for t in ont.terms.values():
            if t.obsolete or not ont.is_phenotype(t.id):
                continue
            for label in (t.name, *t.synonyms):
                toks = tuple(tokens(label))
                if toks and len(" ".join(toks)) >= 4:
                    self.vocab.update(toks)
                    self.index[toks[0]].append((toks, t.id))
        for v in self.index.values():
            v.sort(key=lambda x: -len(x[0]))
        self.neg, self.neg_after, self.hedge = ([tuple(tokens(p)) for p in ps] for ps in (NEG, NEG_AFTER, HEDGE))

    def findings(self, text: str, source="note") -> list[Finding]:
        best, order = {}, {Status.PRESENT: 3, Status.ABSENT: 2, Status.UNCERTAIN: 1, Status.FAMILY: 0}
        heads = [(m.start(), m.end(), m.group(1).lower()) for m in HEADING.finditer(text)]
        bounds = [(0, 0, "")] + heads + [(len(text), len(text), "")]
        for (_, start, sec), (nxt, _, _) in zip(bounds, bounds[1:]):
            if sec.startswith(SKIP_SECTIONS):
                continue
            for sent in re.split(r"(?<=[.;!?])\s+|\n\s*\n", text[start:nxt]):
                toks, i = tokens(sent), 0
                while i < len(toks):
                    hit = next(((c, h) for c, h in self.index.get(toks[i], ()) if tuple(toks[i:i + len(c)]) == c), None)
                    if not hit:
                        i += 1
                        continue
                    pre = toks[max(0, i - 6):i]
                    for k in range(len(pre) - 1, -1, -1):
                        if pre[k] in STOP_SCOPE:
                            pre = pre[k + 1:]
                            break
                    if sec.startswith("family") or FAMILY & set(toks):
                        st = Status.FAMILY
                    elif _has(pre, self.hedge):
                        st = Status.UNCERTAIN
                    elif _has(pre, self.neg) or _has(toks[i + len(hit[0]):i + len(hit[0]) + 4], self.neg_after):
                        st = Status.ABSENT
                    else:
                        st = Status.PRESENT
                    if hit[1] not in best or order[st] > order[best[hit[1]].status]:
                        best[hit[1]] = Finding(hit[1], st, source, sent.strip()[:300])
                    i += len(hit[0])
        return list(best.values())


def delabel(text: str, names, genes, vocab) -> tuple[str, dict]:
    # remove the diagnosis from a note; never masks phenotype words
    keep, skip, dropped = [], False, []
    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            skip = m.group(1).strip().lower().startswith(DX_SECTIONS)
            dropped += [m.group(1).strip().lower()] if skip else []
        if not skip:
            keep.append(line)
    text, masked = "\n".join(keep), defaultdict(int)
    generic = {"disease", "syndrome", "type", "deficiency", "disorder", "familial", "congenital", "hereditary"}
    pats = []
    for n in sorted({n for n in names if n}, key=len, reverse=True):
        words = re.findall(r"[A-Za-z0-9]+", n)
        pats.append((n, re.compile(r"\b" + r"\W+".join(map(re.escape, words)) + r"(?:'s)?\b", re.I)))
        pats += [(w, re.compile(r"\b" + re.escape(w) + r"(?:'s)?\b", re.I)) for w in words
                 if len(w) >= 5 and w.lower() not in generic and tokens(w)[0] not in vocab]
    pats += [(g, re.compile(r"\b" + re.escape(g) + r"\b")) for g in genes if len(g) >= 3]
    for label, pat in pats:
        text, k = pat.subn("[MASKED]", text)
        masked[label] += k
    residual = [w for n in names for w in re.findall(r"[A-Za-z]{5,}", n)
                if w.lower() not in generic and tokens(w)[0] not in vocab and w.lower() in text.lower()]
    return text, {"dropped_sections": dropped, "masked": {k: v for k, v in masked.items() if v}, "residual": residual}


# MIMIC-IV
# (label, fluid) -> HPO if low, high, flagged
LABS = {
    ("Alkaline Phosphatase", "Blood"): ("HP:0003282", "HP:0003155", ""),
    ("Ammonia", "Blood"): ("", "HP:0001987", ""),
    ("Alanine Aminotransferase (ALT)", "Blood"): ("", "HP:0002910", ""),
    ("Asparate Aminotransferase (AST)", "Blood"): ("", "HP:0002910", ""),
    ("Creatine Kinase (CK)", "Blood"): ("", "HP:0003236", ""),
    ("Lactate", "Blood"): ("", "HP:0002151", ""),
    ("Glucose", "Blood"): ("HP:0001943", "", ""),
    ("Sodium", "Blood"): ("HP:0002902", "HP:0003228", ""),
    ("Potassium", "Blood"): ("HP:0002900", "HP:0002153", ""),
    ("Calcium, Total", "Blood"): ("HP:0002901", "HP:0003072", ""),
    ("Phosphate", "Blood"): ("HP:0002148", "HP:0002905", ""),
    ("Magnesium", "Blood"): ("HP:0002917", "", ""),
    ("Hemoglobin", "Blood"): ("HP:0001903", "", ""),
    ("Platelet Count", "Blood"): ("HP:0001873", "HP:0001894", ""),
    ("White Blood Cells", "Blood"): ("HP:0001882", "HP:0001974", ""),
    ("Creatinine", "Blood"): ("", "HP:0003259", ""),
    ("Urea Nitrogen", "Blood"): ("", "HP:0003138", ""),
    ("Bilirubin, Total", "Blood"): ("", "HP:0002904", ""),
    ("Albumin", "Blood"): ("HP:0003073", "", ""),
    ("Uric Acid", "Blood"): ("", "HP:0002149", ""),
    ("Cholesterol, Total", "Blood"): ("", "HP:0003124", ""),
    ("Triglycerides", "Blood"): ("", "HP:0002155", ""),
    ("Lactate Dehydrogenase (LD)", "Blood"): ("", "HP:0025435", ""),
    ("Ferritin", "Blood"): ("", "HP:0003281", ""),
    ("C-Reactive Protein", "Blood"): ("", "HP:0011227", ""),
    ("pH", "Blood"): ("HP:0001941", "HP:0001948", ""),
    ("Bicarbonate", "Blood"): ("HP:0032066", "HP:0032067", ""),
    ("Ceruloplasmin", "Blood"): ("HP:0010837", "", ""),
    ("Homocysteine", "Blood"): ("", "HP:0002160", ""),
    ("Protein", "Urine"): ("", "", "HP:0000093"),
    ("Blood", "Urine"): ("", "", "HP:0000790"),
}


def _csv(folder: Path, name: str, **kw):
    path = next((p for p in (folder / name, folder / (name + ".gz")) if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"{name}(.gz) not found in {folder}")
    return pd.read_csv(path, **kw)


def lab_findings(labs: pd.DataFrame) -> dict[int, list[Finding]]:
    # present if ever out of range, absent if measured and always in range
    state = {}   # (hadm, hpo) -> [label, abnormal evidence or None]
    for r in labs.itertuples(index=False):
        hpos = LABS.get((r.label, r.fluid))
        if not hpos or pd.isna(r.hadm_id):
            continue
        flagged = str(r.flag).lower() == "abnormal"
        ranged = pd.notna(r.valuenum) and pd.notna(r.ref_range_lower) and pd.notna(r.ref_range_upper)
        for hpo, kind in zip(hpos, ("low", "high", "flag")):
            if not hpo:
                continue
            if kind == "flag":
                abn = flagged
            elif ranged:
                abn = r.valuenum < r.ref_range_lower if kind == "low" else r.valuenum > r.ref_range_upper
            else:
                continue
            st = state.setdefault((int(r.hadm_id), hpo), [r.label, None])
            if abn and st[1] is None:
                st[1] = f"{r.label}={r.value} (ref {r.ref_range_lower}-{r.ref_range_upper})"
    out = defaultdict(list)
    for (h, hpo), (label, ev) in state.items():
        out[h].append(Finding(hpo, Status.PRESENT if ev else Status.ABSENT, f"lab:{label}", ev or "in range"))
    return out


def icd_category(code: str, version: int) -> str:
    # ICD category: first 3 characters (4 for ICD-9 E-codes)
    code = str(code).replace(".", "").upper()
    return f"ICD{version}:" + (code[:4] if version == 9 and code.startswith("E") else code[:3])


def split_subjects(subject_ids, test_frac=0.2, seed=13) -> set[int]:
    # fixed held-out patients; all admissions of a patient stay together
    import hashlib
    return {int(s) for s in subject_ids
            if int(hashlib.md5(f"{seed}:{int(s)}".encode()).hexdigest(), 16) % 1000 < test_frac * 1000}


def _mimic_tables(root: Path):
    hosp = root / "hosp"
    dx = _csv(hosp, "diagnoses_icd.csv", dtype={"icd_code": str})
    dx = dx[dx.hadm_id.notna()].copy()
    dx["hadm_id"], dx["subject_id"] = dx.hadm_id.astype(int), dx.subject_id.astype(int)
    d_icd = _csv(hosp, "d_icd_diagnoses.csv", dtype={"icd_code": str})
    return hosp, dx, dict(zip(zip(d_icd.icd_code, d_icd.icd_version), d_icd.long_title))


def _mimic_labs(hosp: Path, hadms: set[int]) -> dict[int, list[Finding]]:
    items = _csv(hosp, "d_labitems.csv")
    items["label"], items["fluid"] = items.label.astype(str).str.strip(), items.fluid.astype(str).str.strip()
    items = items[[(l, f) in LABS for l, f in zip(items.label, items.fluid)]]
    cols = ["hadm_id", "itemid", "value", "valuenum", "ref_range_lower", "ref_range_upper", "flag"]
    chunks = [c[c.itemid.isin(items.itemid) & c.hadm_id.isin(hadms)]
              for c in _csv(hosp, "labevents.csv", usecols=cols, chunksize=2_000_000)]
    return lab_findings(pd.concat(chunks).merge(items[["itemid", "label", "fluid"]], on="itemid"))


def learn_common(mimic_dir, icd_map=None, max_seq=3, min_adm=5, test_frac=0.2, seed=13) -> dict:
    # common-disease lab profiles from training patients only; keep credentialed output private
    root = Path(mimic_dir)
    hosp, dx, titles = _mimic_tables(root)
    test = split_subjects(dx.subject_id.unique(), test_frac, seed)
    train = dx[~dx.subject_id.isin(test)]
    labs = _mimic_labs(hosp, set(train.hadm_id))
    top = train[train.seq_num <= max_seq]
    rare_adm = {h for h, c, v in zip(top.hadm_id, top.icd_code, top.icd_version)
                if v == 10 and match_icd(c, icd_map or {})}
    cats = defaultdict(set)
    for h, c, v in zip(top.hadm_id, top.icd_code, top.icd_version):
        if not (v == 10 and match_icd(c, icd_map or {})):
            cats[icd_category(c, v)].add(h)
    with_labs = set(labs)
    diseases = {}
    for cat, hs in cats.items():
        hs = hs & with_labs
        if len(hs) < min_adm:
            continue
        counts = defaultdict(lambda: [0, 0])
        for h in hs:
            for f in labs[h]:
                counts[f.hpo_id][1] += 1
                counts[f.hpo_id][0] += f.status == Status.PRESENT
        version = int(cat[3:cat.index(":")])
        names = sorted((len(c), t) for (c, v), t in titles.items()
                       if v == version and icd_category(c, v) == cat)
        diseases[cat] = {"name": names[0][1] if names else cat, "n_adm": len(hs), "counts": dict(counts)}
    n = len(set(train.hadm_id) & with_labs)
    return {"source": "MIMIC-IV-demo" if "demo" in str(root).lower() else "MIMIC-IV",
            "policy": "open" if "demo" in str(root).lower() else "local_only",
            "n_admissions": n, "n_rare_admissions": len(rare_adm & with_labs),
            "scope": sorted({h for v in LABS.values() for h in v if h}),
            "test_subjects": sorted(test), "diseases": diseases}


def load_mimic(mimic_dir, icd_map=None, orpha_names=None, disease_genes=None, note_dir=None,
               extractor: Extractor | None = None, max_seq=3, labeled_only=True, extract_fn=None,
               common: dict | None = None, subjects: set[int] | None = None) -> list[Case]:
    # MIMIC cases from labs (+ de-labelled notes); truth = rare ORPHA or known common category
    root = Path(mimic_dir)
    policy = "open" if "demo" in str(root).lower() else "local_only"
    hosp, dx, titles = _mimic_tables(root)
    if subjects is not None:
        dx = dx[dx.subject_id.isin(subjects)]
    known = set((common or {}).get("diseases", {}))
    labels, codes = defaultdict(set), defaultdict(list)
    for h, c, v in dx[dx.seq_num <= max_seq][["hadm_id", "icd_code", "icd_version"]].itertuples(index=False):
        rare = match_icd(c, icd_map or {}) if v == 10 else set()
        cat = icd_category(c, v)
        if rare or cat in known:
            labels[h] |= rare or {cat}
            codes[h].append((c, v))
    hadms = set(labels) if labeled_only and (icd_map or common) else set(dx.hadm_id)
    lab_f = _mimic_labs(hosp, hadms)

    notes = defaultdict(str)
    if note_dir and extractor:
        for c in _csv(Path(note_dir), "discharge.csv", usecols=["hadm_id", "text"], chunksize=200_000):
            for h, t in zip(c.hadm_id, c.text):
                if h in hadms:
                    notes[int(h)] += "\n\n" + str(t)

    cases = []
    for h in sorted(hadms):
        findings = list(lab_f.get(h, []))
        meta = {"hadm_id": h, "policy": policy, "label_icd": [c for c, _ in codes.get(h, [])]}
        if h in notes:
            names = [titles.get(cv, "") for cv in codes.get(h, [])]
            names += [n for o in labels[h] for n in (orpha_names or {}).get(o, [])]
            genes = [g for o in labels[h] for g in (disease_genes or {}).get(o, ())]
            text, meta["delabel"] = delabel(notes[h], names, genes, extractor.vocab)
            have = {f.hpo_id for f in findings}
            found = extract_fn(text, policy) if extract_fn else extractor.findings(text)
            findings += [f for f in found if f.hpo_id not in have]
        cases.append(Case(f"mimic-{h}", findings, sorted(labels.get(h, ())), meta=meta))
    return cases


# MedPix 2.0 (github.com/CHILab1/MedPix-2.0)
NAME_STOP = {"disease", "syndrome", "type", "of", "the", "and", "with", "due", "to", "form", "primary", "secondary",
             "acute", "chronic", "left", "right", "isolated", "familial", "in", "a", "an", "or", "by", "on", "at", "for"}
ROMAN = {"i", "ii", "iii", "iv", "v", "vi"}


def _name_tokens(name: str) -> frozenset[str]:
    return frozenset(t for t in tokens(name) if t not in NAME_STOP and (len(t) >= 3 or t.isdigit() or t in ROMAN))


class DiseaseMatcher:
    # map free-text diagnoses to disease ids; most specific name wins

    def __init__(self, names: dict[str, str], synonyms: dict[str, list[str]] | None = None):
        self.index = defaultdict(list)
        for d, n in names.items():
            for label in [n, *((synonyms or {}).get(d, []))]:
                t = _name_tokens(label)
                if t and (len(t) >= 2 or len(next(iter(t))) >= 8):
                    self.index[min(t)].append((t, d))

    def match(self, text: str) -> list[str]:
        words, found = set(tokens(text)), {}
        for w in words:
            for t, d in self.index.get(w, ()):
                if t <= words:
                    found[d] = max(found.get(d, 0), len(t))
        best = max(found.values(), default=0)
        return sorted(d for d, v in found.items() if v == best)


def load_medpix(root, matcher: DiseaseMatcher, ext: Extractor, split: str = "all", labels_csv=None) -> list[Case]:
    # input = history + exam only; findings, captions and diagnosis kept in meta
    root = Path(root)
    cases_json = json.loads((root / "Case_topic.json").read_text(encoding="utf-8"))
    desc = json.loads((root / "Descriptions.json").read_text(encoding="utf-8"))
    keep = None
    if split != "all":
        keep = set(json.loads((root / "splitted_dataset" / "uid_list.json").read_text())[split])
    overrides = {}
    if labels_csv and Path(labels_csv).exists():
        for row in csv.DictReader(Path(labels_csv).read_text(encoding="utf-8").splitlines()):
            overrides[row["U_id"]] = [t for t in row["truth"].split(";") if t.strip()]
    images = defaultdict(list)
    for d in desc:
        dd = d.get("Description") or {}
        images[d["U_id"]].append({
            "id": d["image"], "path": str(root / "images" / f"{d['image']}.png"),
            "modality": dd.get("Modality") or d.get("Type", ""), "plane": dd.get("Plane") or "",
            "location": d.get("Location") or "", "age": dd.get("Age") or "", "sex": dd.get("Sex") or "",
            "caption": dd.get("Caption") or ""})
    out = []
    for c in cases_json:
        uid, info = c["U_id"], c.get("Case") or {}
        if keep is not None and uid not in keep:
            continue
        title = (info.get("Title") or "").strip()
        names = [title, info.get("Case Diagnosis") or "", (c.get("Topic") or {}).get("Title") or ""]
        field = lambda k: "" if (info.get(k) or "").strip() in ("", "N/A") else info[k]
        history, _ = delabel(f"{field('History')}\n{field('Exam')}", names, [], ext.vocab)
        reference, _ = delabel(field("Findings") + "\n" + "\n".join(i["caption"] for i in images[uid]),
                               names, [], ext.vocab)
        truth = overrides.get(uid) or matcher.match(title) or matcher.match(info.get("Case Diagnosis") or "")
        findings = [f for f in ext.findings(history, "medpix:history") if f.status in (Status.PRESENT, Status.ABSENT)]
        out.append(Case(uid, findings, truth, meta={
            "policy": "open", "title": title, "differential": field("Differential Diagnosis"),
            "history": history, "reference_text": reference, "images": images[uid]}))
    return out
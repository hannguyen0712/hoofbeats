# data loading: Phenopacket Store, MIMIC-IV, Orphadata, MedPix, note extraction
from __future__ import annotations

import csv
import json
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
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
RESOURCES = Path(__file__).parent / "resources"


def resource(name: str) -> list[dict]:
    # tab-separated table from hoofbeats/resources; lines starting with "#" are comments
    lines = [l for l in (RESOURCES / name).read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


FAMILY = {"mother", "father", "brother", "sister", "sibling", "aunt", "uncle", "grandmother",
          "grandfather", "cousin", "maternal", "paternal", "family", "son", "daughter", "child", "children",
          "parent", "grandparent", "niece", "nephew", "twin", "relative", "mom", "dad", "sib"}
SECTION_TYPES = {   # heading -> section type (case-insensitive; a heading may continue, e.g. "Family history of")
    "family": ("fhx", "fh", "family hx", "family history", "family"),
    "past": ("pmh", "psh", "hx", "past medical history", "past surgical history", "past history",
             "medical history", "surgical history"),
    "social": ("shx", "sh", "social", "social history", "social hx"),
    "hpi": ("hpi", "history of present illness"),
    "cc": ("cc", "chief complaint"),
    "ros": ("ros", "review of systems"),
    "exam": ("exam", "pe", "physical exam", "physical examination", "examination"),
    "labs": ("labs", "lab", "laboratory", "laboratory data"),
    "imaging": ("imaging", "findings", "radiology"),
    "assessment": ("assessment", "plan", "a/p", "assessment and plan", "impression"),
}
LINE_HEADING = re.compile(r"^[ \t]*([A-Za-z][A-Za-z /&\-]{1,40}):", re.M)
CLAUSE_WORDS = {"but", "while"}
TEST_WORDS = {"panel", "panels", "test", "tests", "testing", "screen", "screening", "level", "levels", "titer", "titers",
              "titre", "titres", "serology", "serologies", "culture", "cultures", "study", "studies", "assay", "assays",
              "workup", "antibody", "antibodies", "pcr", "swab"}   # "hepatitis panel" names a test, not a finding
RESULT_WORDS = {"elevated", "increased", "decreased", "low", "high", "positive", "abnormal", "raised"}   # ... unless a result is given


def section_type(heading: str) -> str:
    h = " ".join(heading.lower().split())
    for typ, names in SECTION_TYPES.items():
        if any(h == n or h.startswith(n + " ") for n in names):
            return typ
    return "other"


def _unknown_heading_ok(name: str) -> bool:
    # an unknown heading: 3+ characters, at most 4 words, no family word or "pt" ("Mother with seizures   Pt:" is text)
    words = [_norm(w) for w in re.findall(r"[A-Za-z]+", name)]
    return len(name) >= 3 and len(words) <= 4 and not any(w in FAMILY or w in ("pt", "patient") for w in words)


def sections(text: str) -> list[tuple[int, int, str, str, tuple[int, int] | None]]:
    # (content start, end, heading, type, heading span); known headings (CC, FH, PE, ...) at any length,
    # unknown ones only if _unknown_heading_ok; a heading always starts its line
    heads = []
    for m in LINE_HEADING.finditer(text):
        name = m.group(1).strip()
        typ = section_type(name)
        if typ != "other" or _unknown_heading_ok(name):
            heads.append((m.start(1), m.end(), name.lower(), typ, (m.start(1), m.start(1) + len(name))))
    out, prev = [], (0, "", "", None)
    for h0, h1, name, typ, span in heads:
        out.append((prev[0], h0, prev[1], prev[2], prev[3]))
        prev = (h1, name, typ, span)
    out.append((prev[0], len(text), prev[1], prev[2], prev[3]))
    return [x for x in out if x[1] > x[0]]
SKIP_SECTIONS = ("allergies", "medications", "discharge medications", "medications on admission",
                 "discharge instructions", "followup instructions", "discharge disposition", "facility")
DX_SECTIONS = ("discharge diagnos", "final diagnos", "principal diagnos", "primary diagnos",
               "secondary diagnos", "diagnos", "discharge condition", "discharge instructions",
               "followup instructions")
HEADING = re.compile(r"^\s*([A-Za-z][A-Za-z /&\-]{2,40}):", re.M)


def tokens(text: str) -> list[str]:
    return [t[:-1] if len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "is", "us")) else t
            for t in re.findall(r"[a-z0-9]+", text.lower())]


def _phrase_index(phrases) -> dict[str, list[tuple[str, ...]]]:
    # first token -> token tuples, longest first
    idx = defaultdict(list)
    for p in phrases:
        t = tuple(tokens(p))
        if t:
            idx[t[0]].append(t)
    for v in idx.values():
        v.sort(key=len, reverse=True)
    return idx


def _find(norms: list[str], idx) -> list[tuple[int, int]]:
    # (start, end) token spans of every phrase in idx, longest match at each position
    out = []
    for k, w in enumerate(norms):
        p = next((p for p in idx.get(w, ()) if tuple(norms[k:k + len(p)]) == p), None)
        if p:
            out.append((k, k + len(p)))
    return out


CUES = defaultdict(list)
for _row in resource("negation_terms.tsv"):
    CUES[_row["type"]].append(_row["phrase"])
CUE_INDEX = {k: _phrase_index(v) for k, v in CUES.items()}


def _norm(word: str) -> str:
    # same normalisation as tokens(): lower case, trailing plural "s" dropped
    w = word.lower()
    return w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "is", "us")) else w


WORD = re.compile(r"[A-Za-z0-9]+")
NO_SPLIT = {"v", "vs", "dr", "pt", "approx", "e.g", "i.e", "etc", "mr", "mrs", "ms", "st", "fig"}
BOUNDARY = re.compile(r"(?<=[.;!?])\s+|\n\s*\n|\n(?=[ \t]*(?:[-*•]|\d{1,2}[.)])\s)")
ORDER = {Status.PRESENT: 3, Status.ABSENT: 2, Status.UNCERTAIN: 1, Status.FAMILY: 0}


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    # sentence offsets in text; no split after common abbreviations ("v.", "Dr.", "e.g.") or "No." before a number,
    # and a new sentence at each list item
    end = len(text) if end is None else end
    cuts, s = [], start
    for m in BOUNDARY.finditer(text, start, end):
        if text[m.start() - 1] == "." and not re.search(r"\n\s*\n", m.group()):
            w = re.search(r"(?:^|[\s(])([A-Za-z.]+)\.$", text[s:m.start()])    # a free-standing word, not "N/V."
            w = w.group(1).lower() if w else ""
            if w in NO_SPLIT or (w == "no" and text[m.end():m.end() + 1].isdigit()):
                continue
        cuts.append((s, m.start()))
        s = m.end()
    cuts.append((s, end))
    out = []
    for a, b in cuts:
        piece = text[a:b]
        if piece.strip():
            out.append((a + len(piece) - len(piece.lstrip()), b - len(piece) + len(piece.rstrip())))
    return out


@dataclass
class Mention:
    # one matched phrase: text[start:end] == quote
    hpo_id: str
    status: Status
    quote: str
    start: int
    end: int
    sentence: tuple[int, int]
    section: str = ""
    cue: tuple[str, int, int] | None = None   # status cue word and its offsets
    link: tuple[str, str] = ("exact_label", "")   # (method, matched label or synonym)
    grounding: str = "strong"
    disease_mention: bool = False   # the words are also a knowledge-base disease name


GROUNDING = {"exact_label": "strong", "synonym:exact": "strong", "synonym:related": "weak",
             "synonym:layperson": "supported", "negated_word": "supported", "lemma": "supported"}
LEVEL = {"strong": 3, "supported": 2, "weak": 1}
SPACY_MODEL = "en_core_web_sm"


@lru_cache(maxsize=1)
def _spacy():
    # spaCy lemmatizer, loaded once; None (lemma matching off) if spaCy or the model is missing
    try:
        import spacy
        return spacy.load(SPACY_MODEL, exclude=["parser", "ner"])
    except (ImportError, OSError) as e:
        print(f"[extract] lemma matching off ({type(e).__name__}: {e})", file=sys.stderr)
        return None


LEMMA_POS = {"NOUN", "ADJ"}    # text words matched by lemma; verbs never ("jumped" is not Jumping)


def _lemma_map(doc, pos: set[str] | None = None) -> dict[int, tuple[int, str]]:
    # character start -> (end, lemma) for every spaCy token (only these parts of speech, if given)
    return {t.idx: (t.idx + len(t.text), t.lemma_.lower()) for t in doc if pos is None or t.pos_ in pos}


def _lemmas(words, lmap) -> tuple[str, ...]:
    # one lemma per WORD token; a token spaCy splits differently keeps its normal form
    return tuple(lmap[a][1] if a in lmap and lmap[a][0] == b else _norm(w) for w, a, b in words)


LIST_DELIM = re.compile(r"[,;]|\b(?:and|or)\b", re.I)
ORDINALS = {"1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth"}   # matched as words ("1st degree AV block")
INLINE_LABEL = re.compile(r"(?<![A-Za-z0-9/])([A-Z][A-Za-z/&\-]*(?: [A-Za-z][A-Za-z/&\-]*){0,2}):(?!\d)")   # "LUNGS:", "Heart:"


def grounding_of(method: str) -> str:
    return GROUNDING.get(method) or ("supported" if method.startswith("abbreviation:") else "weak")


class Abbreviations:
    # resources/abbreviations.tsv: whole-token matches, optional context words in the same sentence
    def __init__(self, rows):
        self.table = {True: defaultdict(list), False: defaultdict(list)}    # case-sensitive? -> abbr -> rows
        self.ambiguous = {True: set(), False: set()}       # several rows, or other common meanings
        for r in rows:
            abbr, cs = r["abbr"].strip(), any(ch.isupper() for ch in r["abbr"])
            key = abbr if cs else abbr.lower()
            ctx = [c for c in (r.get("context") or "").lower().split("|") if c]
            self.table[cs][key].append((r["expansion"].strip(), ctx))
            if (r.get("other_meanings") or "").strip() or len(self.table[cs][key]) > 1:
                self.ambiguous[cs].add(key)
        # capitalised abbreviations may take a plural "s" (AMLs, LFTs)
        self.rx = {cs: re.compile(r"(?<![A-Za-z0-9])(" + "|".join(sorted(map(re.escape, t), key=len, reverse=True))
                                  + (r")s?(?![A-Za-z0-9])" if cs else r")(?![A-Za-z0-9])"), 0 if cs else re.I)
                   for cs, t in self.table.items() if t}

    def find(self, text: str, s: int, e: int, words: set[str]) -> list[tuple[int, int, str, bool]]:
        # (start, end, expansion, weak) in text[s:e], longest first, no overlaps; weak = a best guess:
        # contexts of several expansions match, or an ambiguous abbreviation sits alone in a comma-separated list
        # ("IVH, CP, MR, congenital cardiac defects") with no context word in its own list item
        fits = lambda ctx, ws: any(w == c or (c.endswith("*") and w.startswith(c[:-1])) for c in ctx for w in ws)
        hits = []
        for cs, rx in self.rx.items():
            for m in rx.finditer(text, s, e):
                key = m.group(1) if cs else m.group(1).lower()
                rows = self.table[cs][key]
                matched = [(x, ctx) for x, ctx in rows if ctx and fits(ctx, words)]
                row = (matched or [(x, ctx) for x, ctx in rows if not ctx] or [None])[0]
                if not row:
                    continue
                weak = len(matched) > 1
                if key in self.ambiguous[cs] and not weak:
                    left = max([s] + [d.end() for d in LIST_DELIM.finditer(text, s, m.start())])
                    right = next((d.start() for d in LIST_DELIM.finditer(text, m.end(), e)), e)
                    in_list = any(d.group().strip() in ",;" for d in LIST_DELIM.finditer(text, max(s, left - 1), min(e, right + 1)))
                    own = {w.lower() for w in WORD.findall(text[left:right])}
                    weak = in_list and not (row[1] and fits(row[1], own))
                hits.append((m.start(), m.end(), row[0], weak))
        out, last = [], -1
        for h in sorted(hits, key=lambda h: (h[0], h[0] - h[1])):
            if h[0] >= last:
                out.append(h)
                last = h[1]
        return out


class LemmaIndex:
    # HPO labels by lemma sequence; label lemmas are cached next to hp.obo (.cache/)
    def __init__(self, nlp, entries, ont: Ontology):
        self.nlp, self.index = nlp, defaultdict(list)
        cache = Path(getattr(ont, "path", ".")).parent / ".cache" / re.sub(
            r"[^A-Za-z0-9.]+", "-", f"lemmas-{ont.version}-{nlp.meta['name']}-{nlp.meta['version']}.json")
        known = json.loads(cache.read_text()) if cache.exists() else {}
        todo = sorted({e[3] for e in entries} - known.keys())
        for label, doc in zip(todo, nlp.pipe(todo, batch_size=2000)):
            words = [(m.group(), m.start(), m.end()) for m in WORD.finditer(label)]
            known[label] = list(_lemmas(words, _lemma_map(doc)))
        if todo:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(known))
        for toks, h, method, label in entries:
            lem = tuple(known[label])
            if lem and lem != toks:
                self.index[lem[0]].append((lem, h, method, label))
        for v in self.index.values():
            v.sort(key=lambda x: (-len(x[0]), GROUNDING[x[2]] != "strong"))

    def text_lemmas(self, text: str) -> dict[int, tuple[int, str]]:
        return _lemma_map(self.nlp(text), LEMMA_POS)


class Extractor:
    # offline HPO matcher with negation, family and uncertainty rules; every mention keeps its offsets

    def __init__(self, ont: Ontology, lemmas: bool = True, abbreviations: bool = True):
        # lemmas: also match on spaCy lemmas (wheezes ~ wheezing); off if spaCy is not installed
        # abbreviations: expand resources/abbreviations.tsv before matching (spans stay on the original text)
        self.ont = ont
        self.abbr = Abbreviations(resource("abbreviations.tsv")) if abbreviations else None
        block = {(r["hpo_id"].strip(), r["phrase"].strip().lower()) for r in resource("blocklist.tsv")}
        self.index, self.vocab = defaultdict(list), set()
        for t in ont.terms.values():
            if t.obsolete or not ont.is_phenotype(t.id):
                continue
            for label, method in ((t.name, "exact_label"), *((s, "synonym:exact") for s in t.synonyms),
                                  *((s, f"synonym:{kind}") for s, kind in t.extra_synonyms)):
                toks = tuple(tokens(label))
                if toks and len(" ".join(toks)) >= 4:
                    if method in ("exact_label", "synonym:exact"):   # de-labelling never masks these words
                        self.vocab.update(toks)
                    if not {(t.id, label.lower()), (t.id, "*"), ("*", label.lower())} & block:
                        self.index[toks[0]].append((toks, t.id, method, label))
        for v in self.index.values():   # longest first; for the same words, exact labels and EXACT synonyms first
            v.sort(key=lambda x: (-len(x[0]), GROUNDING[x[2]] != "strong"))
        self.negated = defaultdict(list)    # "afebrile" -> Fever absent
        for row in resource("negated_words.tsv"):
            ids = [h for h in (ont.resolve(x.strip()) for x in row["hpo_ids"].split(";")) if h]
            t = tuple(tokens(row["phrase"]))
            if ids and t:
                self.negated[t[0]].append((t, ids, row["phrase"]))
        nlp = _spacy() if lemmas else None
        self.lemma = LemmaIndex(nlp, [x for v in self.index.values() for x in v], ont) if nlp else None

    def analyze(self, text: str, diseases: DiseaseNames | None = None) -> tuple[list[Mention], list[dict]]:
        # findings and, if diseases is given, disease-name mentions; findings inside a disease name that is
        # not also an HPO term are dropped, otherwise they are kept and flagged
        out, dm = [], []
        lmap = self.lemma.text_lemmas(text) if self.lemma and text.strip() else {}
        for start, end, name, typ, head in sections(text):
            if name.startswith(SKIP_SECTIONS):
                continue
            for s, e in sentence_spans(text, start, end):
                ms, ds = self._sentence(text, s, e, typ, head, lmap, diseases)
                out += ms
                dm += ds
        return out, dm

    def mentions(self, text: str) -> list[Mention]:
        return self.analyze(text)[0]

    def _match(self, norms, lems, raws, i):
        # longest label at token i: surface words first, lemmas only if they match more words;
        # a one-word label needs an inflected text word ("wheezes" ~ Wheezing, but a bare "tear" is not Tearing)
        hit = next((x for x in self.index.get(norms[i], ()) if tuple(norms[i:i + len(x[0])]) == x[0]), None)
        if self.lemma:
            lhit = next((x for x in self.lemma.index.get(lems[i], ()) if tuple(lems[i:i + len(x[0])]) == x[0]
                         and (len(x[0]) > 1 or raws[i] != lems[i])), None)
            if lhit and (not hit or len(lhit[0]) > len(hit[0])):
                return lhit, ("lemma", lhit[3])
        return hit, (hit[2], hit[3]) if hit else None

    def _words(self, text: str, s: int, e: int):
        # (word, start, end, expansion or None, weak guess); an abbreviation becomes its expansion's words, all with its span
        words = [(m.group(), m.start(), m.end()) for m in WORD.finditer(text, s, e)]
        found = self.abbr.find(text, s, e, {w.lower() for w, _, _ in words}) if self.abbr else []
        out, j, done = [], 0, set()
        for w, a, b in words:
            while j < len(found) and found[j][1] <= a:
                j += 1
            if j < len(found) and found[j][0] <= a < found[j][1]:
                if j not in done:
                    done.add(j)
                    out += [(x, found[j][0], found[j][1], found[j][2], found[j][3]) for x in WORD.findall(found[j][2])]
                continue
            out.append((ORDINALS.get(w.lower(), w), a, b, None, False))
        return out

    def _sentence(self, text: str, s: int, e: int, sec: str, head, lmap=None, diseases=None):
        words = self._words(text, s, e)
        toks = [(_norm(w[0]), w[1], w[2]) for w in words]
        lems = tuple(_norm(w[0]) if w[3] or w[0] in ORDINALS.values() else l
                     for w, l in zip(words, _lemmas([w[:3] for w in words], lmap or {})))
        raws, expanded, guessed = [w[0].lower() for w in words], [w[3] for w in words], [w[4] for w in words]
        norms, out, i = [t[0] for t in toks], [], 0
        # a "LABEL:" inside the sentence starts a new segment ("no JVD/HJR    LUNGS: rhales, wheezes")
        labels = {k for m in INLINE_LABEL.finditer(text, s, e) if m.start() > s
                  for k in [next((k for k, t in enumerate(toks) if t[1] >= m.start(1)), None)] if k}
        clause, c = [], 0    # family words only reach their own clause (split at , ; but while "and pt", labels)
        for k, (w, a, _) in enumerate(toks):
            if k and (re.search(r"[,;]", text[toks[k - 1][2]:a]) or w in CLAUSE_WORDS or k in labels
                      or (w == "and" and k + 1 < len(toks) and toks[k + 1][0] in ("pt", "patient"))):
                c += 1
            clause.append(c)
        cues = self._cues(norms)
        cues["stop"] = sorted(set(cues["stop"]) | labels)     # negation and hedge scopes end at a new label
        dspans, dm = [], []         # disease names: (start, end, also an HPO term?)
        for k, n, ids, name, is_hpo in (diseases.find(norms) if diseases else []):
            a, b = toks[k][1], toks[k + n - 1][2]
            dspans.append((a, b, is_hpo))
            st, cue = self._status(text, toks, norms, clause, cues, k, n, sec, head)
            if st is not None:
                dm.append({"name": name, "disease_ids": ids, "quote": text[a:b], "start": a, "end": b, "status": st.value,
                           "status_cue": dict(zip(("word", "start", "end"), cue)) if cue else None, "also_hpo": is_hpo})
        while i < len(toks):
            hit, link = self._match(norms, lems, raws, i)
            neg = None if hit else next((x for x in self.negated.get(norms[i], ()) if tuple(norms[i:i + len(x[0])]) == x[0]), None)
            if not hit and not neg:
                i += 1
                continue
            n = len((hit or neg)[0])
            # names a test ("hepatitis panel"), unless a result word is near: up to 2 words before, inside the term,
            # or up to 2 words after the test word ("elevated D dimer test", "hepatitis panel positive")
            if (hit and i + n < len(raws) and raws[i + n] in TEST_WORDS
                    and not RESULT_WORDS & set(raws[max(0, i - 2):i + n] + raws[i + n + 1:i + n + 3])):
                i += n
                continue
            a, b = toks[i][1], toks[i + n - 1][2]
            exp = next((x for x in expanded[i:i + n] if x), None)
            if hit and exp:
                link = (f"abbreviation:{exp}", link[1])
            st, cue = self._status(text, toks, norms, clause, cues, i, n, sec, head)
            if neg:       # the word itself says absent; a relative's "afebrile" is not about the patient
                if st != Status.FAMILY:
                    out += [Mention(h, Status.ABSENT, text[a:b], a, b, (s, e), sec, (text[a:b], a, b),
                                    ("negated_word", neg[2]), GROUNDING["negated_word"]) for h in neg[1]]
            elif st is not None:
                inside = [x for x in dspans if x[0] <= a and b <= x[1]]
                if not inside or any(x[2] for x in inside):     # dropped inside a disease name unless it is also HPO
                    # the weaker of the label kind (a RELATED synonym is weak) and the way it matched (lemma, abbreviation;
                    # an abbreviation read as a best guess is weak)
                    level = min(GROUNDING[hit[2]], grounding_of(link[0]), "weak" if any(guessed[i:i + n]) else "strong",
                                key=LEVEL.get)
                    out.append(Mention(hit[1], st, text[a:b], a, b, (s, e), sec, cue, link, level, bool(inside)))
            i += n
        return out, dm

    @staticmethod
    def _cues(norms: list[str]) -> dict:
        # NegEx/ConText cues in one sentence; cues inside a pseudo-cue ("no change") do not count
        pseudo = {k for a, b in _find(norms, CUE_INDEX["pseudo"]) for k in range(a, b)}
        found = {t: [(a, b) for a, b in _find(norms, CUE_INDEX[t]) if a not in pseudo] for t in ("pre", "post", "hedge")}
        found["stop"] = sorted(a for t in ("contrast", "termination") for a, _ in _find(norms, CUE_INDEX[t]))
        return found

    def _status(self, text, toks, norms, clause, cues, i, n, sec, head):
        # family (section or a family word in the clause) > nearest hedge or negation cue in scope > present;
        # None = a relative's finding that is negated ("no family history of ..."), which is dropped
        span = lambda a, b: (text[toks[a][1]:toks[b - 1][2]], toks[a][1], toks[b - 1][2])
        blocked = lambda lo, hi: any(lo <= t < hi for t in cues["stop"])
        before = [(b, a, kind) for kind in ("pre", "hedge") for a, b in cues[kind] if b <= i and not blocked(b, i)]
        after = [(a, b) for a, b in cues["post"] if i + n <= a <= i + n + 3 and not blocked(i + n, a)]
        near = max(before, default=None)
        negated = (near and near[2] == "pre") or after
        if sec == "family" or any(w in FAMILY and clause[k] == clause[i] for k, w in enumerate(norms)):
            if negated:
                return None, None
            if sec == "family":
                return Status.FAMILY, (text[head[0]:head[1]], *head)
            k = next(k for k, w in enumerate(norms) if w in FAMILY and clause[k] == clause[i])
            return Status.FAMILY, span(k, k + 1)
        if near:
            return (Status.UNCERTAIN if near[2] == "hedge" else Status.ABSENT), span(near[1], near[0])
        if after:
            return Status.ABSENT, span(*after[0])
        return Status.PRESENT, None

    def findings(self, text: str, source="note") -> list[Finding]:
        # one finding per term for the patient (present > absent > uncertain) and, separately, one for the family;
        # evidence is the sentence (MIMIC, MedPix and the LLM readers use this)
        best = {}
        for m in self.mentions(text):
            key = (m.hpo_id, m.status == Status.FAMILY)
            if key not in best or ORDER[m.status] > ORDER[best[key].status]:
                best[key] = m
        return [Finding(m.hpo_id, m.status, source, text[m.sentence[0]:m.sentence[1]][:300]) for m in best.values()]


NAME_TAIL = {"syndrome", "disease", "disorder", "type"}
ROMAN_TAIL = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
RELATIONSHIP = {"wife", "husband", "spouse", "partner", "friend", "roommate", "boyfriend", "girlfriend", "caregiver",
                "guardian", "parents", "cousins", "kids", "grandchild", "fiance"}
CORE_STOP = {r["word"] for r in resource("disease_name_stoplist.tsv")} | FAMILY | RELATIONSHIP


class DiseaseNames:
    # knowledge-base disease names as token sequences: each full name, plus a one-word core name when the name
    # minus a trailing syndrome/disease/type N is one distinctive word (5+ letters, not an HPO word): "Marfan"
    def __init__(self, names: dict[str, str], ext: Extractor):
        hpo = {x[0] for v in ext.index.values() for x in v if GROUNDING[x[2]] == "strong"}   # labels, EXACT synonyms
        found = {}
        for did, name in sorted(names.items()):
            full = tuple(tokens(name))
            core = list(full)
            while core and (core[-1] in NAME_TAIL or core[-1].isdigit() or core[-1] in ROMAN_TAIL or len(core[-1]) == 1):
                core.pop()
            keys = [full] + ([tuple(core)] if len(core) == 1 and tuple(core) != full and len(core[0]) >= 5
                             and core[0] not in ext.vocab and core[0] not in CORE_STOP else [])
            for k in keys:
                if k:
                    found.setdefault(k, [name, []])[1].append(did)
        self.index = defaultdict(list)
        for k, (name, ids) in found.items():
            self.index[k[0]].append((k, ids, name, k in hpo))
        for v in self.index.values():
            v.sort(key=lambda x: -len(x[0]))

    def find(self, norms: list[str]) -> list[tuple[int, int, list[str], str, bool]]:
        # (token start, length, disease ids, name, also an HPO term) for each name in a sentence, longest first
        out, k = [], 0
        while k < len(norms):
            hit = next((x for x in self.index.get(norms[k], ()) if tuple(norms[k:k + len(x[0])]) == x[0]), None)
            if hit:
                out.append((k, len(hit[0]), hit[1], hit[2], hit[3]))
                k += len(hit[0])
            else:
                k += 1
        return out


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
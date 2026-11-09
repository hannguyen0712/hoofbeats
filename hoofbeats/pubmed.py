from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .data import Extractor, _name_tokens, tokens
from .model import Case, Ontology, Status

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
SKIP_TYPES = {"Retracted Publication", "Retraction of Publication", "Expression of Concern", "Preprint"}


class PubMedPolicyError(RuntimeError):
    pass


@dataclass
class Paper:
    pmid: str
    title: str
    year: int | None
    journal: str
    types: list[str]
    sentences: list[str]


@dataclass
class LitClaim:
    disease: str
    pmid: str
    year: int | None
    journal: str
    sentence: str
    hpo_ids: tuple[str, ...]
    text: str = ""


class PubMed:
    # search + fetch with a JSON cache; NCBI asks for <= 3 requests/s (10 with an API key)
    def __init__(self, cache_path=None, api_key=None, email=None, opener=urllib.request.urlopen):
        self.cache_path = Path(cache_path) if cache_path else None
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.email = email or os.environ.get("NCBI_EMAIL")
        self.opener, self.last, self.calls = opener, 0.0, 0
        self.cache = {"search": {}, "papers": {}}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text())

    def _get(self, endpoint: str, params: dict) -> bytes:
        params = {**params, "tool": "hoofbeats"}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        wait = (0.11 if self.api_key else 0.34) - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        url = EUTILS + endpoint + "?" + urllib.parse.urlencode(params)
        assert url.startswith(EUTILS)                      # PubMed only
        for attempt in range(3):
            try:
                with self.opener(url, timeout=30) as resp:
                    self.last, self.calls = time.time(), self.calls + 1
                    return resp.read()
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)

    def search(self, query: str, retmax: int = 5, maxdate: str | None = None) -> list[str]:
        key = f"{query}|{retmax}|{maxdate}"
        if key not in self.cache["search"]:
            params = {"db": "pubmed", "term": query, "retmax": retmax, "retmode": "json", "sort": "relevance"}
            if maxdate:
                params.update({"datetype": "pdat", "mindate": "1800", "maxdate": maxdate})
            self.cache["search"][key] = json.loads(self._get("esearch.fcgi", params))["esearchresult"]["idlist"]
        return self.cache["search"][key]

    def fetch(self, pmids: list[str]) -> dict[str, Paper]:
        need = [p for p in pmids if p not in self.cache["papers"]]
        if need:
            xml = self._get("efetch.fcgi", {"db": "pubmed", "id": ",".join(need), "retmode": "xml"})
            for p in parse_pubmed_xml(xml):
                self.cache["papers"][p.pmid] = asdict(p)
        return {p: Paper(**self.cache["papers"][p]) for p in pmids if p in self.cache["papers"]}

    def save(self):
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if len(s.strip()) > 20]


def parse_pubmed_xml(xml: bytes) -> list[Paper]:
    out = []
    for art in ET.fromstring(xml).iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        a = cit.find("Article")
        title = "".join(a.find("ArticleTitle").itertext()) if a.find("ArticleTitle") is not None else ""
        abstract = " ".join("".join(t.itertext()) for t in a.findall("Abstract/AbstractText"))
        year = a.findtext("Journal/JournalIssue/PubDate/Year") or \
            (re.findall(r"\d{4}", a.findtext("Journal/JournalIssue/PubDate/MedlineDate") or "") or [None])[0]
        out.append(Paper(cit.findtext("PMID"), title.strip(), int(year) if year else None,
                         a.findtext("Journal/Title") or "",
                         [t.text or "" for t in a.findall("PublicationTypeList/PublicationType")],
                         split_sentences(abstract)))
    return out


def build_query(ont: Ontology, case: Case, disease_name: str, max_terms: int = 4) -> str:
    # built only from ontology labels and the disease name: never from patient text
    terms = [t for t in (ont.resolve(x) for x in case.terms(Status.PRESENT)) if t]
    terms.sort(key=lambda t: -len(ont.ancestors(t)))       # most specific first
    name = re.sub(r"\s*\(.*?\)", "", disease_name).replace('"', "")
    q = f'"{name}"[tiab]'
    if terms:
        q += " AND (" + " OR ".join(f'"{ont.name(t)}"[tiab]' for t in terms[:max_terms]) + ")"
    return q + " AND hasabstract NOT retracted publication[pt]"


def _mentions_disease(name: str, text: str) -> bool:
    need = _name_tokens(re.sub(r"\s*\(.*?\)", "", name))
    return bool(need) and need <= set(tokens(text))


def _patient_matches(ont: Ontology, ext: Extractor, case: Case, sentence: str) -> tuple[str, ...]:
    # patient findings (or their parents/subtypes) stated as present in the sentence
    patient = {r for r in (ont.resolve(t) for t in case.terms(Status.PRESENT)) if r}
    hits = set()
    for f in ext.findings(sentence):
        if f.status != Status.PRESENT:
            continue
        for p in patient:
            if f.hpo_id in ont.ancestors(p) or p in ont.ancestors(f.hpo_id):
                hits.add(p)
    return tuple(sorted(hits))


def literature_claims(pm: PubMed, ont: Ontology, ext: Extractor, case: Case, disease: str, name: str,
                      exclude: frozenset[str] = frozenset(), per_disease: int = 3, papers: int = 5,
                      maxdate: str | None = None) -> list[LitClaim]:
    # sentences from PubMed abstracts that tie this disease to the patient's findings
    if case.meta.get("policy") == "local_only":
        raise PubMedPolicyError("PubMed search is disabled for credentialed cases")
    pmids = [p for p in pm.search(build_query(ont, case, name), papers, maxdate) if f"PMID:{p}" not in exclude]
    found = []
    for pmid, paper in pm.fetch(pmids).items():
        if SKIP_TYPES & set(paper.types):
            continue
        about = _mentions_disease(name, paper.title)
        for s in paper.sentences:
            hits = _patient_matches(ont, ext, case, s)
            if hits and (about or _mentions_disease(name, s)):
                found.append(LitClaim(disease, pmid, paper.year, paper.journal, s, hits))
    found.sort(key=lambda c: (-len(c.hpo_ids), -(c.year or 0)))
    out, seen = [], set()
    for c in found:
        if c.pmid in seen:                                  # one sentence per paper
            continue
        seen.add(c.pmid)
        names = ", ".join(ont.name(h) for h in c.hpo_ids)
        c.text = f'PubMed PMID:{c.pmid} ({c.year or "n.d."}), on {names}: "{c.sentence}"'
        out.append(c)
        if len(out) == per_disease:
            break
    return out


def verify_lit(pm: PubMed, ont: Ontology, ext: Extractor, case: Case, c: LitClaim, name: str,
               exclude: frozenset[str] = frozenset()) -> tuple[bool, str]:
    # re-check a literature claim against the cached PubMed record
    if f"PMID:{c.pmid}" in exclude:
        return False, "cites the case's own paper"
    paper = pm.fetch([c.pmid]).get(c.pmid)
    if paper is None:
        return False, "PMID not found in PubMed"
    if SKIP_TYPES & set(paper.types):
        return False, "retracted or flagged publication"
    if c.sentence not in paper.sentences:
        return False, "sentence not in the abstract"
    if not set(c.hpo_ids) <= set(_patient_matches(ont, ext, case, c.sentence)) or not c.hpo_ids:
        return False, "sentence does not state the patient's findings"
    if not (_mentions_disease(name, paper.title) or _mentions_disease(name, c.sentence)):
        return False, "paper is not about this disease"
    return True, "verified"


def case_year(pm: PubMed, case: Case) -> int | None:
    # publication year of the case's own paper (for --pubmed-before-case)
    pmids = [p.split(":")[1] for p in case.source_pmids]
    years = [p.year for p in pm.fetch(pmids).values() if p.year] if pmids else []
    return min(years) if years else None
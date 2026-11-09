# Per-disease workup: for each leading disease, the tests that would confirm it or rule it out.
# Design, formulas and sources: docs/TEST_SUGGESTIONS.md. In short:
# - which diseases: the ones displayed (the ranker's list, and a re-ranker's when one ran); their order is never
#   changed here
# - pre-test probability: the knowledge-base ranker's calibrated posterior
# - which test shows which finding: MAxO diagnostic annotations (exams, imaging, specialist evaluations) and
#   LOINC2HPO (lab tests), both downloaded to the knowledge-base folder (`download-tests`); burden tiers by hand
#   (resources/test_burden.tsv)
# - per disease d and test T: P(T+|d) = 1 - (1 - leak) * prod(1 - f(u,d)) over d's unknown findings T can show
#   (noisy-OR), P(T+|not d) the same over the other diseases weighted by probability; LR+, LR-, post-test
#   probabilities (odds x LR) for the reader
# - tests are ordered by expected weight of evidence per burden tier: each finding T shows is its own result, and
#   p * KL(f_d || f_other) + (1 - p) * KL(f_other || f_d), summed over the findings, is how strongly T is expected to
#   point the right way (for d if d is the diagnosis, against it if not). Chosen over the binary information gain
#   and over "most frequent finding first" on development cases (docs/TEST_SUGGESTIONS.md)
# - gene tests from genes_to_disease (Mendelian associations); a gene test is also positive for the other diseases
#   of the same gene
# - NCBI's Genetic Testing Registry (GTR, optional): the clinical tests labs have registered for each disease (gene
#   tests, biochemical tests such as enzyme assays), and registered panels that cover several leading diseases
# Functions get `hb`, the hoofbeats command-line module, for its shared helpers (see context.py).
from __future__ import annotations

import csv
import json
import math
import random
import re
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .data import resource
from .model import Case, Finding, LRRanker, Ontology, Ranking, Status, most_frequent_test, suggest_tests

# pinned versions, so a run can be repeated
MAXO_FILE, LOINC_FILE = "maxo_diagnostic_annotations.tsv", "loinc2hpo-annotations.tsv"
MAXO_URL = ("https://raw.githubusercontent.com/monarch-initiative/maxo-annotations/"
            "42edd966faac5229ce7c35fc3848f2e6b3d45bda/data/poet/maxo_diagnostic_annotations.tsv")
LOINC_URL = ("https://raw.githubusercontent.com/TheJacksonLaboratory/loinc2hpoAnnotation/"
             "c1068d6d6b80ce757ff7a26e4c38a5ac8e7c830c/loinc2hpo-annotations.tsv")

GTR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/GTR/data/"     # updated weekly: the download date is the version
GTR_FILES = ("test_condition_gene.txt", "test_version.gz")

LEAK = 0.02            # a test shows a finding no candidate explains, or misses one (noisy-OR leak; also the clip)
LR_CONFIRM = 5.0       # LR+ >= 5: a positive result moves the probability a lot (Jaeschke, Guyatt, Sackett 1994)
LR_RULE_OUT = 0.2      # LR- <= 0.2: a negative result does
# pre-test probabilities from ranker scores: the top DIFFERENTIAL diseases share CALIB_IN (how often the true disease
# is among them) by softmax(score / CALIB_T); the rest share 1 - CALIB_IN. Fitted on 400 Phenopacket Store cases (seed 13,
# leakage control, HPO 2026-09-01); refit with `workup-eval`, which prints and saves both
CALIB_T, CALIB_IN, DIFFERENTIAL = 2.18, 0.475, 20
MAX_UP = 3             # a finding inherits the tests of an annotated ancestor at most this many levels up
DD_ID = ("HP:0012758", "HP:0001249")    # neurodevelopmental delay, intellectual disability (ACMG 2021 note)
# Broad genetic test (ACMG, Manickam et al. 2021; EuroGentest WGS recommendations, Souche et al. 2022): exome or genome
# sequencing finds the cause whether or not it is among the diseases shown. It is always listed, and listed FIRST when
# the shown diseases together are unlikely to contain the diagnosis: on 500 Phenopacket cases, when their calibrated
# probabilities summed to < 0.30 (19% of cases) the diagnosis was missing from the top 10 in 89% of them (64% otherwise).
BROAD_FIRST = 0.30
# Gene tests that need a special method (EuroGentest: standard sequencing misses repeat expansions; imprinting
# disorders need methylation analysis): read from how GTR's registered tests for the disease are made. A disease needs
# repeat-expansion testing when >= 15% of its targeted tests are repeat tests (>= 3 tests), methylation analysis when
# >= 25% use methylation analysis (>= 5 tests).
REPEAT_NAME = re.compile(r"repeat|triplet|trinucleotide|\b(CTG|CAG|CGG|CCTG|GGGGCC|ATTCT)\b", re.I)
REPEAT_SHARE, METHYL_SHARE = 0.15, 0.25
OTHER_SEX = {"male": ("HP:0010460",), "female": ("HP:0010461",)}   # female / male genitalia: not for the other sex
PRENATAL = "HP:0001197"   # abnormality of prenatal development or birth: history, not something a test shows now


def fetch(kb) -> list[Path]:
    # download MAxO diagnostic annotations and LOINC2HPO into the knowledge-base folder
    kb = Path(kb)
    kb.mkdir(parents=True, exist_ok=True)
    out = []
    for name, url in ((MAXO_FILE, MAXO_URL), (LOINC_FILE, LOINC_URL), *((f, GTR_URL + f) for f in GTR_FILES)):
        print("downloading", name)
        urllib.request.urlretrieve(url, kb / name)
        out.append(kb / name)
    return out


# tests and what they show
@dataclass(frozen=True)
class Test:
    id: str                 # MAXO:..., LOINC:<n> (lab group), GENE:<symbols>
    name: str
    kind: str               # exam, lab, imaging, physiology, specialist, invasive, gene, other
    burden: int             # 1-5, resources/test_burden.tsv
    source: str
    codes: tuple[str, ...] = ()     # LOINC codes of a lab group, gene symbols of a gene test


def _burden(name: str) -> tuple[str, int]:
    for row in _BURDEN():
        if re.search(row["pattern"], name, re.I):
            return row["kind"], int(row["burden"])
    return "other", 3


@lru_cache(maxsize=None)
def _BURDEN():
    return resource("test_burden.tsv")


@lru_cache(maxsize=None)
def _DONE():
    return [(re.compile(r["pattern"], re.I), r["test"].split("|")) for r in resource("tests_done.tsv")]


class TestCatalog:
    # tests that can show each HPO term: MAxO is_observable_through (exams, imaging, specialists) + LOINC2HPO (labs)
    def __init__(self, ont: Ontology, kb):
        self.ont = ont
        self.tests: dict[str, Test] = {}
        self.direct: dict[str, set[str]] = defaultdict(set)      # term -> tests annotated at that term
        self.version = {}
        kb = Path(kb)
        maxo, loinc = kb / MAXO_FILE, kb / LOINC_FILE
        missing = [p.name for p in (maxo, loinc) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{', '.join(missing)} not in {kb}: run `python -m hoofbeats download-tests`")
        lines = maxo.read_text(encoding="utf-8").splitlines()
        self.version["maxo"] = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("#version")), "")
        for row in csv.DictReader([l for l in lines if not l.startswith("#")], delimiter="\t"):
            if row["predicate_id"] != "is_observable_through":     # prenatal ascertainment is left out
                continue
            u = ont.resolve(row["hpo_id"])
            kind, burden = _burden(row["maxo_label"])
            if u is None or kind == "skip":
                continue
            tid = row["maxo_id"]
            self.tests.setdefault(tid, Test(tid, row["maxo_label"], kind, burden, "MAxO", (tid,)))
            self.direct[u].add(tid)
        # LOINC2HPO: one lab test per analyte, named by its "normal" term ("Abnormal blood potassium concentration")
        groups = defaultdict(lambda: {"codes": set(), "terms": set(), "normal": None})
        for row in csv.DictReader(loinc.read_text(encoding="utf-8").splitlines(), delimiter="\t"):
            u = ont.resolve(row["hpoTermId"])
            if u is None or len(ont.ancestors(u)) <= 3:     # too general to name or define a test (metabolism/homeostasis)
                continue
            key = row["loincId"]
            groups[key]["codes"].add(key)
            groups[key]["terms"].add(u)
            if row["outcome"] in ("N", "NEG"):
                groups[key]["normal"] = u
        by_normal = defaultdict(lambda: {"codes": set(), "terms": set(), "named": True})
        for g in groups.values():
            if not g["terms"]:
                continue
            # without a usable "normal" term, the group is keyed by its most general result term and named after
            # its results ("lab test: Conjugated hyperbilirubinemia / Increased total bilirubin")
            top = g["normal"] or min(g["terms"], key=lambda t: (len(ont.ancestors(t)), t))
            by_normal[top]["codes"] |= g["codes"]
            by_normal[top]["terms"] |= g["terms"] | {top}
            by_normal[top]["named"] &= g["normal"] is not None
        for top, g in by_normal.items():
            named = g["named"] or ont.name(top).startswith("Abnormal")
            name = (re.sub(r"^(Abnormal(ity of)?|Abnormally)\s+", "", ont.name(top)).strip() if named else
                    "lab test: " + " / ".join(sorted(ont.name(t) for t in g["terms"])[:3]))
            kind, burden = _burden(name)
            kind, burden = (kind, burden) if kind in ("exam", "invasive") else ("lab", 2)
            tid = f"LOINC:{top}"
            self.tests[tid] = Test(tid, f"{name} (LOINC test)", kind, burden, "LOINC2HPO", tuple(sorted(g["codes"])[:5]))
            for u in g["terms"]:
                self.direct[u].add(tid)
        self.version["loinc2hpo"] = "c1068d6"

    @lru_cache(maxsize=None)
    def observing(self, u: str) -> tuple[str, ...]:
        # tests annotated at u or its parents (MAxO lists "a" test able to show a term, not every one: Cortical
        # tubers -> oncologist evaluation, its parent Abnormal cerebral cortex morphology -> MRI of the brain); if
        # none, the nearest annotated ancestors up to MAX_UP levels. A LOINC lab test replaces MAxO's generic lab
        # actions ("blood chemistry measurement") for the same term.
        level, seen, found = [u], {u}, set()
        for depth in range(MAX_UP + 1):
            found |= set().union(*(self.direct.get(t, set()) for t in level))
            if found and depth >= 1:
                break
            nxt = []
            for t in level:
                term = self.ont.terms.get(t)
                for p in term.parents if term else ():
                    if p not in seen and self.ont.is_phenotype(p):
                        seen.add(p)
                        nxt.append(p)
            level = nxt
        if any(t.startswith("LOINC:") for t in found):
            found = {t for t in found if t.startswith("LOINC:") or self.tests[t].kind != "lab"}
        return tuple(sorted(found))

    def done_from_text(self, texts: list[str]) -> set[str]:
        # tests a case already has, from image metadata, image context and image ids
        names = {t.name.lower(): tid for tid, t in self.tests.items()}
        out = set()
        for text in texts:
            for rx, tests in _DONE():
                if text and rx.search(text):
                    out |= {names[n.lower()] for n in tests if n.lower() in names}
        return out


# NCBI Genetic Testing Registry
BIOCHEMICAL = {"Enzyme assay", "Analyte", "Protein analysis", "Protein expression"}
NOT_DIAGNOSTIC = re.compile(r"carrier|prenatal|fetal|newborn screen|\bNBS\b|pharmacogen|drug response", re.I)


@dataclass(frozen=True)
class GtrTest:
    acc: str                        # GTR accession with version, e.g. GTR000566628.1
    name: str
    lab: str
    country: str
    methods: tuple[str, ...]        # GTR method categories ("Sequence analysis of the entire coding region", ...)
    genes: tuple[str, ...]
    conditions: frozenset[str]      # OMIM ids

    @property
    def kind(self) -> str:
        if NOT_DIAGNOSTIC.search(self.name):
            return "screening"          # carrier, prenatal or pharmacogenetic: not a diagnostic test for this patient
        if set(self.methods) & BIOCHEMICAL and len(self.conditions) <= 2:     # for this disease, not a screen
            return "biochemical"
        n = len(self.genes)
        return "other" if n == 0 else "targeted" if n <= 3 else "panel" if n <= 500 else "exome/genome"

    @property
    def url(self) -> str:
        return f"https://www.ncbi.nlm.nih.gov/gtr/tests/{int(self.acc[3:].split('.')[0])}/"

    def line(self) -> str:
        where = "US" if self.country == "United States" else self.country
        return f"{self.name} - {self.lab} ({where}) {self.url}"


class GTR:
    # current clinical tests from GTR's bulk files, by the OMIM ids of the conditions they test
    def __init__(self, kb):
        import gzip
        import io
        kb = Path(kb)
        self.tests: dict[str, GtrTest] = {}
        self.by_disease: dict[str, list[str]] = defaultdict(list)
        meta, self.version = {}, ""
        with io.TextIOWrapper(gzip.open(kb / GTR_FILES[1]), encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                if row.get("now_current") == "1":
                    meta[row["test_accession_ver"]] = row
                    self.version = max(self.version, row.get("last_touch_date") or "")
        conds, genes = defaultdict(set), defaultdict(set)
        with open(kb / GTR_FILES[0], encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                acc = row["#accession_version"]
                if acc not in meta or row["test_type"] != "Clinical":
                    continue
                if row["object"] == "condition" and row["MIM_number"] not in ("", "N/A"):
                    conds[acc].add(f"OMIM:{row['MIM_number']}")
                elif row["object"] == "gene" and row["gene_symbol"] not in ("", "N/A"):
                    genes[acc].add(row["gene_symbol"])
        for acc, cs in conds.items():
            m = meta[acc]
            t = GtrTest(acc, m["lab_test_name"].strip() or m["manufacturer_test_name"].strip() or acc,
                        m["name_of_laboratory"].strip(), m["facility_country"].strip(),
                        tuple(x for x in m["method_categories"].split("|") if x), tuple(sorted(genes[acc])), frozenset(cs))
            self.tests[acc] = t
            for d in cs:
                self.by_disease[d].append(acc)

    @classmethod
    def load(cls, kb):
        return cls(kb) if all((Path(kb) / f).exists() for f in GTR_FILES) else None

    def for_disease(self, d: str) -> list[GtrTest]:
        return [self.tests[a] for a in self.by_disease.get(d, ())]

    def examples(self, d: str, kind: str, n: int = 2) -> list[GtrTest]:
        # US labs first, then tests that both sequence and look for deletions/duplications, then by name
        full = {"Sequence analysis of the entire coding region", "Deletion/duplication analysis"}
        ts = sorted((t for t in self.for_disease(d) if t.kind == kind),
                    key=lambda t: (t.country != "United States", -len(full & set(t.methods)), len(t.genes), t.name, t.acc))
        out, seen = [], set()
        for t in ts:                    # one per lab and name (labs register the same test in several versions)
            if (t.name.lower(), t.lab) not in seen:
                seen.add((t.name.lower(), t.lab))
                out.append(t)
        return out[:n]

    def method_profile(self, d: str) -> dict:
        # share of the disease's targeted tests (one per lab and name) that are repeat-expansion or methylation tests
        ts = {}
        for t in self.for_disease(d):
            if t.kind == "targeted":
                ts.setdefault((t.name.lower(), t.lab), t)
        n = len(ts)
        rep = sum(bool(REPEAT_NAME.search(t.name)) for t in ts.values()) / n if n else 0.0
        met = sum("Methylation analysis" in t.methods for t in ts.values()) / n if n else 0.0
        return {"n": n, "repeat": rep if n >= 3 else 0.0, "methylation": met if n >= 5 else 0.0}

    def method(self, d: str) -> str:
        # "repeat", "methylation" or "sequencing"
        m = self.method_profile(d)
        return "repeat" if m["repeat"] >= REPEAT_SHARE else "methylation" if m["methylation"] >= METHYL_SHARE \
            else "sequencing"

    def gene_counts(self, d: str) -> dict[str, int]:
        c = defaultdict(int)
        for t in self.for_disease(d):
            if t.kind == "targeted":
                for g in t.genes:
                    c[g] += 1
        return dict(c)

    def summary(self, d: str) -> dict:
        ts = self.for_disease(d)
        kinds = defaultdict(int)
        for t in ts:
            kinds[t.kind] += 1
        return {"tests": len(ts), "labs": len({t.lab for t in ts}),
                "us": sum(t.country == "United States" for t in ts), "kinds": dict(kinds)}

    def panels(self, top: list[tuple[str, float]], n: int = 2) -> list[dict]:
        # registered panels that cover the most probability among the leading diseases (at least two of them)
        p = dict(top)
        cands = {a for d in p for a in self.by_disease.get(d, ()) if self.tests[a].kind == "panel"}
        scored = []
        for a in cands:
            t = self.tests[a]
            cov = [d for d in p if d in t.conditions]
            if len(cov) >= 2:
                scored.append((-sum(p[d] for d in cov), -len(cov), t.country != "United States", len(t.genes), a))
        out = []
        for neg_p, neg_n, _, _, a in sorted(scored)[:n]:
            t = self.tests[a]
            out.append({"test": t, "covers": [d for d in p if d in t.conditions], "p": -neg_p})
        return out


# what is known about the patient
def known_sets(ont: Ontology, case: Case) -> tuple[set[str], set[str]]:
    # present terms imply their ancestors; absent terms imply their subtypes are absent
    yes, no = set(), set()
    for f in case.findings:
        t = ont.resolve(f.hpo_id)
        if t and f.status == Status.PRESENT:
            yes |= ont.ancestors(t)
        elif t and f.status == Status.ABSENT:
            no |= ont.descendants(t)
    return yes, no


def unsure_findings(case: Case) -> list[Finding]:
    # hedged findings, and findings read with weak grounding (best-guess abbreviation, inferred unit, ...)
    from .grounding import GroundedFinding
    return [f for f in case.findings if f.status == Status.UNCERTAIN
            or (isinstance(f, GroundedFinding) and f.grounding == "weak" and f.status != Status.FAMILY)]


def case_texts(case: Case) -> list[str]:
    # where tests already done are named: images (multimodal input), image context, image-finding ids
    out = [f"{i.get('modality', '')} {i.get('plane', '')} {i.get('location', '')}" for i in case.meta.get("images", [])]
    prov = case.meta.get("provenance", {})
    for entries in prov.values() if isinstance(prov, dict) else ():
        for e in entries if isinstance(entries, list) else ():
            if isinstance(e, dict) and str(e.get("field", "")).startswith("image"):
                out.append(str(e.get("quote") or e.get("evidence") or ""))
    out.append(str(case.meta.get("image_context") or ""))       # incl. the Stage A image description
    out += [str(x) for x in case.meta.get("done_texts", [])]
    return [t for t in out if t.strip()]


# pre-test probabilities
def ranker_probs(ranking: Ranking, calib=(CALIB_T, CALIB_IN)) -> np.ndarray:
    # calibrated pre-test probabilities (see CALIB_T): softmax at temperature T, the top DIFFERENTIAL rescaled to
    # share `inside`, every other disease to share the rest
    temperature, inside = calib
    z = ranking.log_scores / temperature
    p = np.exp(z - z.max())
    top = np.zeros(len(p), bool)
    top[ranking.order[:DIFFERENTIAL]] = True
    if not (~top).any() or p[~top].sum() <= 0:      # a knowledge base with no disease outside the top
        return p / p.sum()
    out = np.where(top, p / p[top].sum() * inside, 0.0)
    out[~top] = p[~top] / p[~top].sum() * (1 - inside)
    return out


def calibrate(r: LRRanker, cases: list[Case], leak) -> tuple[float, float]:
    # fit (T, inside) on cases with known diagnoses: inside = share whose diagnosis is in the top DIFFERENTIAL,
    # T = the temperature that best predicts which of them it is (maximum likelihood)
    data = []
    for c in cases:
        rk = r.score(c, leak(c))
        top = rk.order[:DIFFERENTIAL]
        gold = [j for j, i in enumerate(top) if r.diseases[i] in c.truth]
        data.append((rk.log_scores[top], gold[0] if gold else None))
    inside = float(np.mean([g is not None for _, g in data])) if data else CALIB_IN

    def nll(t):
        tot = 0.0
        for s, g in data:
            if g is not None:
                z = s / t
                tot += z.max() + math.log(np.exp(z - z.max()).sum()) - z[g]
        return tot
    temps = np.exp(np.linspace(math.log(0.2), math.log(20), 80))
    return float(min(temps, key=nll)) if any(g is not None for _, g in data) else CALIB_T, inside


# per-disease workup
@dataclass
class Shown:
    hpo_id: str
    name: str
    freq: float
    refs: tuple[str, ...]


@dataclass
class TestResult:
    test: Test
    shows: list[Shown]              # the disease's findings this test can show (most frequent first)
    p_pos: float                    # P(T+ | d)
    p_pos_other: float              # P(T+ | not d)
    lr_pos: float
    lr_neg: float
    post_pos: float                 # P(d | T+)
    post_neg: float                 # P(d | T-)
    bits: float                     # information gained about "is it d?"
    score: float                    # what tests are ordered by (ORDER_BY), per burden tier
    done: bool = False              # already taken (images): review it instead of ordering it
    woe_confirm: float = 0.0        # expected weight of evidence for d if d is the diagnosis (nats, summed per finding)
    woe_rule_out: float = 0.0       # expected weight of evidence against d if it is not
    mentioned: list = field(default_factory=list)   # findings it looks for that the note may already state

    @property
    def role(self) -> str:
        r = [x for x, ok in (("confirm", self.lr_pos >= LR_CONFIRM), ("rule out", self.lr_neg <= LR_RULE_OUT)) if ok]
        return " + ".join(r) or "weak"

    def line(self) -> str:
        shows = "; ".join(f"{s.name} (~{s.freq:.0%})" for s in self.shows[:4]) + (" ..." if len(self.shows) > 4 else "")
        what = "review the images already taken" if self.done else f"burden {self.test.burden}"
        return (f"{self.test.name} [{what}]: positive -> {pct(self.post_pos)}, negative -> {pct(self.post_neg)} "
                f"(LR+ {self.lr_pos:.1f}, LR- {self.lr_neg:.2f})" + (f"; looks for {shows}" if shows else ""))


@dataclass
class DiseaseWorkup:
    disease: str
    name: str
    p: float                        # pre-test probability: the ranker's score, calibrated
    gene: TestResult | None
    confirm: list[TestResult]
    rule_out: list[TestResult]
    review: list[TestResult]
    other: list[TestResult]         # best remaining tests when none reaches the LR cut-offs
    genes_known: list[str] = field(default_factory=list)
    ranked: list[TestResult] = field(default_factory=list)   # every test not yet done, best first
    gtr: dict | None = None         # GTR: summary, example targeted tests, biochemical tests


def shortlist(dw: DiseaseWorkup, n: int) -> list[TestResult]:
    # the n best tests, plus the best confirming and the best ruling-out test when those are not among them
    out = list(dw.ranked[:n])
    for pool in (dw.confirm, dw.rule_out):
        if pool and not any(x in out for x in pool):
            out.append(pool[0])
    return out


@dataclass
class Workup:
    case_id: str
    diseases: list[DiseaseWorkup]
    panel: dict
    verify: list[dict]
    done: list[str]
    notes: list[str]
    gtr_panels: list[dict] = field(default_factory=list)
    separate: list[dict] = field(default_factory=list)   # tests that best tell the leading diseases apart
    broad: dict | None = None                             # exome / genome sequencing (Workupper.broad)


# "may already be answered": a finding the test looks for that Stage 1 did not pick up, but whose name or synonym is
# roughly in one sentence of the note or image description (at least 2 words and 60% of them, compared on their
# first 4 letters, including the term's first and last word; a hint for the reader, so it errs towards missing rather than false alarms)
_GENERIC = {"abno", "incr", "decr", "morp", "phen", "abse", "seve", "mild", "prog", "type", "left", "righ", "both",
            "bila", "unil", "flat", "poor"}


def _stems(text: str) -> set[str]:
    return {w[:4] for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 4}


def mentions(ont: Ontology, case: Case, terms) -> dict[str, str]:
    # HPO term -> the sentence that may already state it
    text = "\n".join(str(case.meta.get(k) or "") for k in ("history", "image_context"))
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text) if x.strip()]
    stems = [(x, _stems(x)) for x in sents]
    out = {}
    for u in terms:
        t = ont.terms.get(u)
        if t is None:
            continue
        for label in [t.name, *t.synonyms]:
            words = _stems(label) - _GENERIC
            ends = [w[:4] for w in re.findall(r"[a-z]+", label.lower()) if len(w) >= 4 and w[:4] not in _GENERIC]
            if not words or not ends:
                continue
            hit = next((x for x, st in stems if len(words & st) >= 2 and len(words & st) / len(words) >= 0.6
                        and ends[0] in st and ends[-1] in st), None)       # first and last word (qualifier, body part)
            if hit:
                out[u] = hit[:200]
                break
    return out


def pct(p: float) -> str:
    return f"{p:.0%}" if p >= 0.095 or p == 0 else f"{p:.1%}" if p >= 0.001 else "<0.1%"


ORDER_BY = {     # test value before dividing by the burden tier; see docs/TEST_SUGGESTIONS.md for the comparison
    "bits": lambda x, p: x.bits,
    "woe": lambda x, p: p * x.woe_confirm + (1 - p) * x.woe_rule_out,
    "woe-confirm": lambda x, p: x.woe_confirm,
    "jeffreys": lambda x, p: x.woe_confirm + x.woe_rule_out,
    "sensitivity": lambda x, p: x.p_pos,
}
ORDERING = "woe"


def _post(p: float, lr: float) -> float:
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    o = p / (1 - p) * lr
    return o / (1 + o)


def _h(p: float) -> float:
    return 0.0 if p <= 0 or p >= 1 else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def score_test(p: float, p_pos: float, p_pos_other: float) -> tuple[float, float, float, float, float]:
    # LR+, LR-, P(d|T+), P(d|T-), bits about "is it d?"
    lr_pos = p_pos / max(p_pos_other, 1e-9)
    lr_neg = (1 - p_pos) / max(1 - p_pos_other, 1e-9)
    t_pos = p * p_pos + (1 - p) * p_pos_other
    post_pos, post_neg = _post(p, lr_pos), _post(p, lr_neg)
    bits = _h(p) - t_pos * _h(post_pos) - (1 - t_pos) * _h(post_neg)
    return lr_pos, lr_neg, post_pos, post_neg, max(bits, 0.0)


class Workupper:
    # builds workups for one case; caches disease profiles across the diseases of the case
    def __init__(self, r: LRRanker, cat: TestCatalog, gene_sensitivity: float = 1 - LEAK, genes_path=None,
                 gtr: GTR | None = None):
        self.r, self.cat, self.gene_sens, self.gtr = r, cat, gene_sensitivity, gtr
        self._implied = {}
        self.genes = defaultdict(set)        # disease -> Mendelian genes (else genes of unknown association type)
        self.gene_diseases = defaultdict(set)
        rows = list(csv.DictReader(Path(genes_path).read_text(encoding="utf-8").splitlines(), delimiter="\t")) \
            if genes_path and Path(genes_path).exists() else []
        by_type = defaultdict(lambda: defaultdict(set))
        for row in rows:
            if row["disease_id"] in r.d_index:
                by_type[row["disease_id"]][row["association_type"]].add(row["gene_symbol"])
        for d, types in by_type.items():
            gs = types.get("MENDELIAN") or types.get("UNKNOWN") or set()
            self.genes[d] = gs
            for g in gs:
                self.gene_diseases[g].add(d)

    def implied(self, d: str, exclude: frozenset) -> dict[str, float]:
        key = (d, exclude)
        if key not in self._implied:
            self._implied[key] = self.r.implied(d, exclude)[0]
        return self._implied[key]

    def gene_list(self, d: str) -> tuple[list[str], str]:
        # the disease's genes: HPO's list; without one, the genes of at least a quarter of GTR's targeted tests
        if self.genes.get(d):
            return sorted(self.genes[d]), "genes_to_disease"
        counts = self.gtr.gene_counts(d) if self.gtr else {}
        n = sum(1 for t in self.gtr.for_disease(d) if t.kind == "targeted") if self.gtr else 0
        gs = sorted(g for g, c in counts.items() if n and c / n >= 0.25)
        return gs, "GTR"

    def features(self, d: str, exclude: frozenset, yes: set, no: set, sex: str | None) -> list[Shown]:
        # d's annotated findings not yet known (present or absent) for this patient
        ont, seen = self.r.ont, {}
        banned = set().union(ont.descendants(PRENATAL), *(ont.descendants(t) for t in OTHER_SEX.get(sex or "", ())))
        for a in self.r.db.annotations(d, exclude):
            u = a.hpo_id
            if a.negated or a.freq <= 0 or u in yes or u in no or u in banned:
                continue
            if u not in seen or a.freq > seen[u].freq:
                seen[u] = Shown(u, ont.name(u), a.freq, tuple(sorted(a.refs)))
        return sorted(seen.values(), key=lambda s: (-s.freq, s.name))

    def build(self, case: Case, ranking: Ranking, exclude: frozenset = frozenset(), k: int = 5, per_list: int = 3,
              calib=(CALIB_T, CALIB_IN), phenotype_only: bool = False, shown: list[str] | None = None) -> Workup:
        # shown: the diseases to work up, in the order they are displayed (the ranker's list, or a re-ranker's);
        # default: the ranker's top k. The order is never changed here.
        r = self.r
        p = ranker_probs(ranking, calib)
        order = np.argsort(-p, kind="stable")
        diff = [r.diseases[i] for i in order[:DIFFERENTIAL]]
        targets = list(dict.fromkeys(d for d in shown if d in r.d_index)) if shown else diff[:k]
        yes, no = known_sets(r.ont, case)
        sex = case.meta.get("sex")
        done = self.cat.done_from_text(case_texts(case))
        out = []
        for d in targets:
            out.append(self.disease(case, d, p, diff, exclude, yes, no, sex, done, per_list, phenotype_only))
        recorded = {r.ont.resolve(f.hpo_id) for f in case.findings}      # any status, family included: not a hint
        said = mentions(r.ont, case, {s.hpo_id for dw in out for x in dw.ranked + dw.review for s in x.shows} - recorded)
        for dw in out:
            for x in {id(y): y for y in dw.ranked + dw.review + ([dw.gene] if dw.gene else [])}.values():
                x.mentioned = [{"finding": s.name, "sentence": said[s.hpo_id]} for s in x.shows if s.hpo_id in said]
        separate = self.separate(case, targets[:3], p, exclude, yes, no, sex, done)
        notes = []
        panel = self.panel(case, targets, p)
        if panel["genes"] and self.dd_id(case, yes):
            notes.append("Developmental delay or intellectual disability: ACMG (Manickam et al. 2021) recommends exome "
                         "or genome sequencing as a first- or second-tier test (onset before 18).")
        gtr_panels = self.gtr.panels([(d, float(p[r.d_index[d]])) for d in targets]) if self.gtr else []
        broad = self.broad(case, targets, p, yes)
        return Workup(case.case_id, out, panel, self.verify(case), sorted(self.cat.tests[t].name for t in done),
                      notes, gtr_panels, separate, broad)

    def broad(self, case, targets, p, yes) -> dict | None:
        # exome / genome sequencing: finds the cause even if it is not among the diseases shown (see BROAD_FIRST).
        # Genome rather than exome when a shown disease needs repeat-expansion testing (exome misses repeats);
        # trio (both parents too) for developmental delay / intellectual disability in a child (de novo variants).
        if case.genes:
            return None                     # a gene result is already known
        cover = float(sum(p[self.r.d_index[d]] for d in targets))
        special = [d for d in targets if self.gtr and self.gtr.method(d) == "repeat"]
        trio = self.dd_id(case, yes)
        test = ("genome" if special else "exome") + " sequencing" + (", trio (with both parents)" if trio else "")
        first = cover < BROAD_FIRST
        why = (f"the diseases shown add up to {cover:.0%}: the cause may well be another disease, and "
               f"{test.split(',')[0]} tests (nearly) all genes at once" if first else
               f"if the targeted tests are negative: the diseases shown add up to {cover:.0%}")
        if special:
            why += ("; genome, not exome, because " + ", ".join(self.r.db.names[d] for d in special[:3])
                    + " need repeat-expansion detection")
        return {"test": test, "first": first, "p_in_list": round(cover, 4), "why": why,
                "repeat_diseases": special}

    def separate(self, case, ds: list[str], p, exclude, yes, no, sex, done, n: int = 3) -> list[dict]:
        # the tests that best tell the leading diseases apart: information (bits) about which of them it is, from a
        # positive or negative result, per burden tier. P(T+|d) = noisy-OR over the findings T looks for in any of them.
        r, cat = self.r, self.cat
        if len(ds) < 2:
            return []
        w = np.array([p[r.d_index[d]] for d in ds], float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(ds), 1 / len(ds))
        prof = {d: self.implied(d, exclude) for d in ds}
        by_test = defaultdict(set)
        for d in ds:
            for s in self.features(d, exclude, yes, no, sex):
                for tid in cat.observing(s.hpo_id):
                    by_test[tid].add(s.hpo_id)
        h = lambda q: -sum(x * math.log2(x) for x in q if x > 0)
        out = []
        for tid, terms in by_test.items():
            top = [u for u in terms if not any(v != u and v in r.ont.ancestors(u) for v in terms)]
            pos = np.array([1 - (1 - LEAK) * np.prod([1 - min(prof[d].get(u, 0.0), 1 - LEAK) for u in top]) for d in ds])
            pp = float(w @ pos)
            if pp <= 0 or pp >= 1:
                continue
            bits = h(w) - pp * h(w * pos / pp) - (1 - pp) * h(w * (1 - pos) / (1 - pp))
            burden = 1 if tid in done else cat.tests[tid].burden
            out.append({"test": cat.tests[tid].name, "id": tid, "burden": cat.tests[tid].burden, "done": tid in done,
                        "bits": round(bits, 4), "score": bits / burden,
                        "p_pos": {d: round(float(x), 3) for d, x in zip(ds, pos)}})
        out.sort(key=lambda x: (-x["score"], x["burden"], x["test"]))
        return [x for x in out if x["bits"] >= 0.05][:n]

    def disease(self, case, d, p, diff, exclude, yes, no, sex, done, per_list, phenotype_only):
        r, cat = self.r, self.cat
        i = r.d_index[d]
        pd_ = float(p[i])
        others = [x for x in diff if x != d]
        q = np.array([p[r.d_index[x]] for x in others])
        q_other = max(1.0 - pd_ - q.sum(), 0.0)
        z = q.sum() + q_other
        q, q_other = (q / z, q_other / z) if z > 0 else (q, 1.0)
        prof = [self.implied(x, exclude) for x in others]
        feats = self.features(d, exclude, yes, no, sex)
        by_test = defaultdict(list)
        for s in feats:
            for tid in cat.observing(s.hpo_id):
                by_test[tid].append(s)
        results = []
        for tid, shows in by_test.items():
            test = cat.tests[tid]
            terms = [s.hpo_id for s in shows]
            top = [u for u in terms if not any(v != u and v in r.ont.ancestors(u) for v in terms)]   # most general
            f_d = np.array([min(next(s.freq for s in shows if s.hpo_id == u), 1 - LEAK) for u in top])
            p_pos = 1 - (1 - LEAK) * np.prod(1 - f_d)
            F = np.array([[min(pr.get(u, 0.0), 1 - LEAK) for u in top] for pr in prof]) if prof else np.zeros((0, len(top)))
            p_comp = 1 - (1 - LEAK) * np.prod(1 - F, axis=1) if len(F) else np.zeros(0)
            bg = np.array([min(r.bg_of(u), 1 - LEAK) for u in top])
            p_bg = 1 - (1 - LEAK) * np.prod(1 - bg)
            p_pos_other = float(q @ p_comp + q_other * p_bg)
            lr_pos, lr_neg, post_pos, post_neg, bits = score_test(pd_, float(p_pos), p_pos_other)
            # each finding is its own result: expected weight of evidence (Good & Card 1971), summed over findings
            us = [s.hpo_id for s in shows]
            fd = np.clip([s.freq for s in shows], LEAK, 1 - LEAK)
            Fo = np.array([[pr.get(u, 0.0) for u in us] for pr in prof]) if prof else np.zeros((0, len(us)))
            fo = np.clip((q @ Fo if len(Fo) else 0) + q_other * np.array([r.bg_of(u) for u in us]), LEAK, 1 - LEAK)
            woe_c = float(np.sum(fd * np.log(fd / fo) + (1 - fd) * np.log((1 - fd) / (1 - fo))))
            woe_r = float(np.sum(fo * np.log(fo / fd) + (1 - fo) * np.log((1 - fo) / (1 - fd))))
            is_done = tid in done
            burden = 1 if is_done else test.burden
            res = TestResult(test, shows, float(p_pos), p_pos_other, lr_pos, lr_neg, post_pos, post_neg,
                             bits, 0.0, is_done, woe_c, woe_r)
            res.score = ORDER_BY[ORDERING](res, pd_) / burden
            results.append(res)
        results.sort(key=lambda x: (-x.score, x.test.burden, x.test.name))
        review = [x for x in results if x.done]
        todo = [x for x in results if not x.done]
        confirm = [x for x in todo if x.lr_pos >= LR_CONFIRM][:per_list]
        rule_out = [x for x in todo if x.lr_neg <= LR_RULE_OUT][:per_list]
        other = [] if confirm or rule_out else todo[:per_list]
        gene, known = (None, []) if phenotype_only else self.gene_test(case, d, pd_, others, q, q_other)
        gtr = None
        if self.gtr and not phenotype_only and self.gtr.by_disease.get(d):
            gtr = {**self.gtr.summary(d), "targeted": self.gtr.examples(d, "targeted"),
                   "biochemical": self.gtr.examples(d, "biochemical", 3)}
        return DiseaseWorkup(d, r.db.names[d], pd_, gene, confirm, rule_out,
                             review[:per_list], other, known, todo, gtr)

    def gene_test(self, case, d, pd_, others, q, q_other):
        genes, source = self.gene_list(d)
        if not genes:
            return None, []
        known = [g for g in genes if g in case.genes]
        if known:
            return None, known
        shared = np.array([bool(set(self.gene_list(x)[0]) & set(genes)) for x in others])
        p_pos_other = float((q * shared).sum() * self.gene_sens + (q * ~shared).sum() * LEAK + q_other * LEAK) \
            if len(others) else LEAK
        lr_pos, lr_neg, post_pos, post_neg, bits = score_test(pd_, self.gene_sens, p_pos_other)
        kl = lambda a, b: a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))
        sens, fpos = min(max(self.gene_sens, LEAK), 1 - LEAK), min(max(p_pos_other, LEAK), 1 - LEAK)
        method = self.gtr.method(d) if self.gtr else "sequencing"
        what = {"repeat": "repeat expansion testing (standard sequencing misses repeat expansions)",
                "methylation": "methylation analysis first (imprinting), then sequencing and deletion/duplication "
                               "analysis if normal",
                "sequencing": "gene testing (sequencing and deletion/duplication analysis)"}[method]
        test = Test("GENE:" + ",".join(genes), f"{', '.join(genes[:6])}{' ...' if len(genes) > 6 else ''} {what}",
                    "gene", 4, source, tuple(genes))
        res = TestResult(test, [], self.gene_sens, p_pos_other, lr_pos, lr_neg, post_pos, post_neg, bits, 0.0,
                         False, kl(sens, fpos), kl(fpos, sens))
        res.score = ORDER_BY[ORDERING](res, pd_) / 4
        return res, []

    def panel(self, case, top, p) -> dict:
        # one multi-gene panel for the leading diseases: genes and the chance it finds the cause (if the
        # diagnosis is among them and the test detects the variant)
        genes, yield_ = [], 0.0
        for d in top:
            gs = self.gene_list(d)[0]
            if gs and not set(gs) & set(case.genes):
                genes += [g for g in gs if g not in genes]
                yield_ += float(p[self.r.d_index[d]]) * self.gene_sens
        return {"genes": genes, "expected_yield": min(yield_, 1.0)}

    def dd_id(self, case, yes) -> bool:
        age = case.meta.get("age")
        try:
            young = age is None or float(str(age).split()[0]) < 18
        except ValueError:
            young = True
        return young and any(t in yes for t in DD_ID)

    def verify(self, case) -> list[dict]:
        out = []
        for f in unsure_findings(case):
            t = self.r.ont.resolve(f.hpo_id)
            tests = sorted((self.cat.tests[x] for x in self.cat.observing(t)), key=lambda x: (x.burden, x.name)) \
                if t else []
            if tests:
                out.append({"hpo_id": t, "term": self.r.ont.name(t), "status": f.status.value,
                            "why": "uncertain" if f.status == Status.UNCERTAIN else "weak grounding",
                            "tests": [x.name for x in tests[:2]]})
        return out


def verify_gtr(wk: Workupper, d: str, t: GtrTest) -> bool:
    # a GTR test shown for d is current, clinical, and lists d as a condition it tests
    return bool(wk.gtr) and t.acc in wk.gtr.tests and d in wk.gtr.tests[t.acc].conditions


def verify_result(wk: Workupper, d: str, res: TestResult, exclude: frozenset = frozenset()) -> tuple[bool, str]:
    # re-check a suggested test: each finding it looks for is annotated to d with that frequency and reference,
    # and the test can show it (MAxO / LOINC2HPO)
    if res.test.kind == "gene":
        ok = bool(res.test.codes) and set(res.test.codes) <= set(wk.gene_list(d)[0])
        return ok, "verified" if ok else "gene not associated with the disease"
    annos = {}
    for a in wk.r.db.annotations(d, exclude):
        if not a.negated:
            annos.setdefault(a.hpo_id, []).append(a)
    for s in res.shows:
        if not any(abs(a.freq - s.freq) < 1e-9 and set(s.refs) <= a.refs for a in annos.get(s.hpo_id, [])):
            return False, f"{s.name}: annotation, frequency or reference not in the knowledge base"
        if res.test.id not in wk.cat.observing(s.hpo_id):
            return False, f"{s.name}: {res.test.name} is not listed as showing it"
    return True, "verified"


# command line
def print_workup(w: Workup, wk: Workupper, per_list: int = 3) -> tuple[int, int]:
    n_ok = n_all = 0

    def show(title, res, d):
        nonlocal n_ok, n_all
        ok = verify_result(wk, d, res)[0]
        n_ok, n_all = n_ok + ok, n_all + 1
        role = "" if res.done or res.test.kind == "gene" else f" <{res.role}>"
        print(f"   {title}{role} {res.line()}" + ("" if ok else "  [NOT VERIFIED]"))

    print(f"\nWorkup for {w.case_id}" + (f" (tests already done: {', '.join(w.done)})" if w.done else ""))
    for i, dw in enumerate(w.diseases, 1):
        print(f"\n{i}. {dw.name} ({dw.disease}): pre-test {pct(dw.p)}")
        if dw.genes_known:
            print(f"   gene finding already given: {', '.join(dw.genes_known)}")
        if dw.gene:
            show("gene test:", dw.gene, dw.disease)
        if dw.gtr:
            g, k = dw.gtr, dw.gtr["kinds"]
            kinds = ", ".join(f"{k[x]} {x}" for x in ("targeted", "panel", "exome/genome", "biochemical", "screening",
                                                          "other") if k.get(x))
            print(f"   GTR: {g['tests']} current clinical test{'s' if g['tests'] != 1 else ''} from {g['labs']} "
                  f"lab{'s' if g['labs'] != 1 else ''} ({g['us']} in the US): {kinds}")
            for t in g["targeted"]:
                ok = verify_gtr(wk, dw.disease, t)
                n_ok, n_all = n_ok + ok, n_all + 1
                print(f"      e.g. {t.line()}" + ("" if ok else "  [NOT VERIFIED]"))
            for t in g["biochemical"]:
                ok = verify_gtr(wk, dw.disease, t)
                n_ok, n_all = n_ok + ok, n_all + 1
                print(f"   biochemical test (GTR, {', '.join(m for m in t.methods if m in BIOCHEMICAL)}): {t.line()}"
                      + ("" if ok else "  [NOT VERIFIED]"))
        for res in shortlist(dw, per_list):
            show("-", res, dw.disease)
        for res in dw.review:
            show("review existing images:", res, dw.disease)
        if not dw.ranked and not dw.gene:
            print("   no test found for this disease's unknown findings")
    if w.panel["genes"]:
        g = w.panel["genes"]
        n_d, n_g = len(w.diseases), len(g)
        print(f"\nGene panel for {'these ' + str(n_d) + ' diseases' if n_d > 1 else 'this disease'}: "
              f"{', '.join(g[:15])}{' ...' if n_g > 15 else ''} ({n_g} gene{'s' if n_g > 1 else ''}); "
              f"finds the cause with probability ~{w.panel['expected_yield']:.0%} "
              "(if the diagnosis is among them and the test detects the variant)")
    if w.gtr_panels:
        names = {d.disease: d.name for d in w.diseases}
        print("Registered panels covering several of these diseases (GTR):")
        for x in w.gtr_panels:
            t = x["test"]
            ok = all(verify_gtr(wk, d, t) for d in x["covers"])
            n_ok, n_all = n_ok + ok, n_all + 1
            print(f"   {t.line()}: {len(t.genes)} genes; covers {len(x['covers'])} of {len(w.diseases)} "
                  f"({', '.join(names[d] for d in x['covers'])}; together {pct(x['p'])} pre-test)"
                  + ("" if ok else "  [NOT VERIFIED]"))
    for n in w.notes:
        print(f"Note: {n}")
    if w.verify:
        print("\nFindings to verify:")
        for v in w.verify:
            print(f"   {v['term']} ({v['why']}): {' or '.join(v['tests'])}")
    print("\n<confirm>: LR+ >= 5, a positive result raises the probability a lot; <rule out>: LR- <= 0.2, a negative "
          "result lowers it a lot; <weak>: neither. 'Positive' = any finding it looks for is there. Tests are ordered "
          "by expected weight of evidence per burden tier (1 bedside ... 5 invasive).")
    print(f"{n_ok}/{n_all} suggested tests verified against the knowledge base, MAxO, LOINC2HPO, the gene list"
          + (" and GTR" if wk.gtr else ""))
    return n_ok, n_all


def print_compact(w: Workup):
    # one block per displayed disease: its best confirming and ruling-out test and its gene test (diagnose)
    said = lambda x: ("; note may already answer: " + ", ".join(f"{m['finding']} ('{m['sentence'][:70]}')"
                                                                  for m in x.mentioned[:2])) if x.mentioned else ""
    short = lambda x: f"{x.test.name} [burden {x.test.burden}] (+ -> {pct(x.post_pos)}, - -> {pct(x.post_neg)}){said(x)}"
    print("\nTests per disease (full list in the JSON)" + (f"; already done: {', '.join(w.done)}" if w.done else "") + ":")
    for i, dw in enumerate(w.diseases, 1):
        parts = [f"confirm: {short(dw.confirm[0])}" if dw.confirm else "",
                 f"rule out: {short(dw.rule_out[0])}" if dw.rule_out else "",
                 f"best: {short(dw.other[0])}" if dw.other else "",
                 f"gene: {short(dw.gene)}" if dw.gene else "",
                 f"review images: {dw.review[0].test.name}" if dw.review else ""]
        print(f"{i:>2}. {dw.name}: pre-test {pct(dw.p)}" + "".join(f"\n      {x}" for x in parts if x))
    if w.separate:
        names = {dw.disease: dw.name for dw in w.diseases}
        print(f"   best to tell the top {len(w.separate[0]['p_pos'])} apart:")
        for x in w.separate:
            print(f"      {x['test']} [burden {x['burden']}{', done: review' if x['done'] else ''}], {x['bits']:.2f} bits; "
                  "positive in " + ", ".join(f"{names.get(d, d)[:28]} {pct(v)}" for d, v in x["p_pos"].items()))
    if w.broad:
        print(f"   broad genetic test{' (FIRST)' if w.broad['first'] else ''}: {w.broad['test']} - {w.broad['why']}")
    if w.verify:
        print("   findings to verify: " + "; ".join(f"{v['term']} ({' or '.join(v['tests'])})" for v in w.verify))


def workup_json(w: Workup, per_list: int = 3) -> dict:
    res = lambda x: {"test": x.test.name, "id": x.test.id, "kind": x.test.kind, "burden": x.test.burden,
                     "source": x.test.source, "codes": list(x.test.codes), "role": x.role, "done": x.done,
                     "p_pos_if_disease": round(x.p_pos, 4), "p_pos_if_not": round(x.p_pos_other, 4),
                     "lr_pos": round(x.lr_pos, 3), "lr_neg": round(x.lr_neg, 4),
                     "post_test_if_positive": round(x.post_pos, 4), "post_test_if_negative": round(x.post_neg, 4),
                     "bits": round(x.bits, 4), "woe_confirm": round(x.woe_confirm, 4),
                     "woe_rule_out": round(x.woe_rule_out, 4), "score": round(x.score, 4),
                     "shows": [vars(s) for s in x.shows], "may_already_be_answered": x.mentioned}
    gt = lambda t: {"accession": t.acc, "name": t.name, "lab": t.lab, "country": t.country, "methods": list(t.methods),
                    "genes": list(t.genes[:50]), "n_genes": len(t.genes), "url": t.url}
    gtr = lambda g: None if not g else {**{k: g[k] for k in ("tests", "labs", "us", "kinds")},
                                         "targeted": [gt(t) for t in g["targeted"]],
                                         "biochemical": [gt(t) for t in g["biochemical"]]}
    return {"case_id": w.case_id, "tests_done": w.done, "notes": w.notes, "panel": w.panel, "verify": w.verify,
            "broad_test": w.broad,
            "gtr_panels": [{**gt(x["test"]), "covers": x["covers"], "pre_test_covered": round(x["p"], 4)}
                           for x in w.gtr_panels],
            "separate": [{k: v for k, v in x.items() if k != "score"} for x in w.separate],
            "diseases": [{"id": d.disease, "name": d.name, "pre_test": round(d.p, 4), "genes_known": d.genes_known,
                          "gene_test": res(d.gene) if d.gene else None, "tests": [res(x) for x in shortlist(d, per_list)],
                          "review": [res(x) for x in d.review], "gtr": gtr(d.gtr)} for d in w.diseases]}


def done_texts(path) -> list[str]:
    # image context and image descriptions in a case file or a `multimodal --save` file (tests already done)
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(obj, dict):
        return []
    out = []
    ctx = obj.get("image_context")
    for x in ctx if isinstance(ctx, list) else [ctx]:
        out += [str(x.get("label", "")), str(x.get("text", ""))] if isinstance(x, dict) else [str(x or "")]
    for x in obj.get("image_findings") or obj.get("images") or []:
        if isinstance(x, dict):
            out.append(" ".join(str(x.get(k) or "") for k in ("image_id", "modality", "plane", "location", "finding")))
    # a note line that reports a result under a test's name ("Echo (outside, 3/2026): aortic root 4.1 cm")
    for line in str(obj.get("note_text") or obj.get("text") or "").splitlines():
        head = line.split(":", 1)[0]
        if ":" in line and len(head) <= 40 and any(rx.search(head) for rx, _ in _DONE()):
            out.append(head)
    return [t for t in out if t.strip()]


def read_calibration(path) -> tuple[float, float]:
    if not path:
        return CALIB_T, CALIB_IN
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(obj["temperature"]), float(obj["inside"])


def cmd_download_tests(a, hb):
    for p in fetch(a.kb):
        hb.wrote(p)


def load_gtr(kb, use: bool = True) -> GTR | None:
    if not use:
        return None
    gtr = GTR.load(kb)
    if gtr is None:
        print(f"[workup] GTR files not in {kb} (run download-tests): no registered tests shown")
    else:
        print(f"[workup] GTR (last change {gtr.version}): {len(gtr.tests)} current clinical tests for "
              f"{len(gtr.by_disease)} OMIM conditions")
    return gtr


def load_workupper(r: LRRanker, kb, use_gtr: bool = True, gene_sensitivity: float = 1 - LEAK):
    # the workup's test map (MAxO, LOINC2HPO, burden tiers) and GTR; None, with a message, if not downloaded
    missing = [f for f in (MAXO_FILE, LOINC_FILE) if not (Path(kb) / f).exists()]
    if missing:
        print(f"[workup] test suggestions off: {', '.join(missing)} not in {kb}. Run once: python -m hoofbeats "
              "download-tests")
        return None
    wk = Workupper(r, TestCatalog(r.ont, kb), gene_sensitivity, Path(kb) / "genes_to_disease.txt", load_gtr(kb, use_gtr))
    print(f"[workup] MAxO {wk.cat.version['maxo']}, LOINC2HPO {wk.cat.version['loinc2hpo']}: {len(wk.cat.tests)} tests "
          f"for {len(wk.cat.direct)} HPO terms")
    return wk


def cmd_workup(a, hb):
    # the full workup of the ranker's top k diseases, one case file or phenopacket at a time
    r = hb.load_kb(a)
    wk = load_workupper(r, a.kb, not a.no_gtr, a.gene_sensitivity)
    if wk is None:
        raise SystemExit("run download-tests first")
    calib = read_calibration(a.calibration)
    out = Path(a.out) / "workup"
    out.mkdir(parents=True, exist_ok=True)
    hb.RUN["settings"].update({"calibration": calib, "gene_sensitivity": a.gene_sensitivity, "k": a.k,
                               "maxo": wk.cat.version["maxo"], "gtr": wk.gtr.version if wk.gtr else None})
    for path in a.case:
        c, ex = hb.read_case(path, r)
        c.meta["done_texts"] = done_texts(path)
        w = wk.build(c, r.score(c, ex), ex, a.k, a.per_list, calib)
        n_ok, n_all = print_workup(w, wk, a.per_list)
        hb.note("workup", case=c.case_id, diseases=len(w.diseases), tests_verified=f"{n_ok}/{n_all}")
        (out / f"{c.case_id}.json").write_text(json.dumps(workup_json(w, a.per_list), indent=1), encoding="utf-8")
        print(f"[workup] -> {hb.wrote(out / f'{c.case_id}.json')}")


# evaluation: simulated workups on Phenopacket Store (docs/TEST_SUGGESTIONS.md)
POLICIES = ("workup", "workup-top3", "feature-eig", "most-frequent", "random")


def cheapest(cat: TestCatalog, u: str, asked: set) -> str | None:
    tests = [t for t in cat.observing(u) if t not in asked]
    return min(tests, key=lambda t: (cat.tests[t].burden, cat.tests[t].name)) if tests else None


def choose(policy: str, wk: Workupper, case: Case, ranking: Ranking, ex: frozenset, asked: set, rng: random.Random,
           calib, genes: bool) -> str | None:
    # the next test under each policy
    r, cat = wk.r, wk.cat
    if policy in ("workup", "workup-top3"):
        w = wk.build(case, ranking, ex, 1 if policy == "workup" else 3, 10_000, calib, phenotype_only=not genes)
        score = defaultdict(float)
        for dw in w.diseases:
            for x in dw.ranked + ([dw.gene] if dw.gene else []):
                if x.test.id not in asked:
                    score[x.test.id] += dw.p * x.score if policy == "workup-top3" else x.score
        return max(score, key=lambda t: (score[t], t)) if score else None
    if policy == "feature-eig":       # the existing next check (a feature), done with its cheapest test
        for u, _, _ in suggest_tests(r, ranking, case, ex, n=50):
            t = cheapest(cat, u, asked)
            if t:
                return t
        return None
    if policy == "most-frequent":     # the top disease's most frequent unknown feature, its cheapest test
        known = set(asked)
        while True:
            u = most_frequent_test(r, ranking, case, ex, known)
            if u is None:
                return None
            t = cheapest(cat, u, asked)
            if t:
                return t
            known.add(u)
    yes, no = known_sets(r.ont, case)
    pool = sorted({t for d, _, _ in ranking.top(5) for s in wk.features(d, ex, yes, no, None)
                   for t in cat.observing(s.hpo_id)} - asked)
    return rng.choice(pool) if pool else None


def reveal(wk: Workupper, tid: str, hidden: list[Finding], genes: list[str]) -> tuple[list[Finding], list[str]]:
    # the hidden findings this test shows (open world: what the record does not say stays unknown)
    if tid.startswith("GENE:"):
        return [], [g for g in tid[5:].split(",") if g in genes]
    return [f for f in hidden if wk.r.ont.resolve(f.hpo_id) and tid in wk.cat.observing(wk.r.ont.resolve(f.hpo_id))], []


def simulate_case(wk: Workupper, c: Case, ex: frozenset, policy: str, steps: int, seed: int, calib, genes: bool):
    # hide half of the observed findings (and all excluded ones), then let the policy order `steps` tests
    r = wk.r
    rng = random.Random(f"{seed}:{c.case_id}:{policy}")
    cur, hidden = hide(c, seed)
    asked, burden, rows, use_genes = set(), 0, [], r.use_genes
    try:
        for step in range(steps + 1):
            r.use_genes = bool(cur.genes)
            rk = r.score(cur, ex)
            p = ranker_probs(rk, calib)
            rows.append({"case": c.case_id, "policy": policy, "step": step, "rank": rk.rank_of(c.truth),
                         "p_truth": float(sum(p[r.d_index[t]] for t in c.truth if t in r.d_index)), "burden": burden,
                         "hidden_left": len(hidden)})
            if step == steps:
                break
            tid = choose(policy, wk, cur, rk, ex, asked, rng, calib, genes)
            if not tid:
                continue
            asked.add(tid)
            burden += 4 if tid.startswith("GENE:") else wk.cat.tests[tid].burden
            found, gfound = reveal(wk, tid, hidden, c.genes)
            rows[-1].update({"test": wk.cat.tests[tid].name if tid in wk.cat.tests else tid,
                             "revealed": len(found) + len(gfound)})
            cur.findings += found
            hidden = [f for f in hidden if f not in found]
            cur.genes = sorted(set(cur.genes) | set(gfound))
    finally:
        r.use_genes = use_genes
    return rows


def hide(c: Case, seed: int) -> tuple[Case, list[Finding]]:
    # the simulated patient: half of the observed findings known, the rest and all excluded findings hidden
    present = [f for f in c.findings if f.status == Status.PRESENT]
    random.Random(f"{seed}:{c.case_id}").shuffle(present)
    keep = max(1, math.ceil(len(present) / 2))
    return (Case(c.case_id, list(present[:keep]), c.truth, [], meta={}),
            present[keep:] + [f for f in c.findings if f.status == Status.ABSENT])


DISEASE_POLICIES = ("workup", "most-frequent", "random")
WORLDS = ("open", "imputed")


def budgeted(cat: TestCatalog, order: list[str], budget: int) -> list[str]:
    # tests in the policy's order while the burden budget allows
    out, spent = [], 0
    for t in order:
        b = cat.tests[t].burden
        if t not in out and spent + b <= budget:
            out.append(t)
            spent += b
    return out


def disease_eval(wk: Workupper, c: Case, ex: frozenset, seed: int, calib, budget: int = 6, k: int = 5) -> list[dict]:
    # every worked-up disease d: do d's tests (up to `budget`) and see how P(d) moves. If d is the diagnosis it
    # should rise (confirm), otherwise fall (rule out). Policies: the workup's order; the cheapest test for each of
    # d's most frequent unknown findings; random tests among d's. Worlds: open (a test shows only what the record
    # says) and imputed (a finding the record does not mention is present with the TRUE disease's annotated
    # frequency, drawn once per case and finding; the full profile, without leakage control, generates it).
    r = wk.r
    cur, hidden = hide(c, seed)
    rk = r.score(cur, ex)
    p0 = ranker_probs(rk, calib)
    truth = [t for t in c.truth if t in r.d_index]
    if not truth:
        return []
    w = wk.build(cur, rk, ex, k, 3, calib, phenotype_only=True)
    hid = [(r.ont.resolve(f.hpo_id), f) for f in hidden if r.ont.resolve(f.hpo_id)]
    true_freq = r.implied(truth[0])[0]
    yes, no = known_sets(r.ont, cur)
    rows = []
    for dw in w.diseases:
        if not dw.ranked:
            continue
        shows = {x.test.id: x.shows for x in dw.ranked}
        freq_order = []
        for s in wk.features(dw.disease, ex, yes, no, None):          # most frequent first
            t = cheapest(wk.cat, s.hpo_id, set(freq_order))
            if t and t in shows and t not in freq_order:
                freq_order.append(t)
        orders = {"workup": [x.test.id for x in dw.ranked], "most-frequent": freq_order,
                  "random": [x.test.id for x in random.Random(f"{seed}:{c.case_id}:{dw.disease}").sample(
                      dw.ranked, len(dw.ranked))]}
        for pol, order in orders.items():
            tests = budgeted(wk.cat, order, budget)
            for world in WORLDS:
                if world == "open":
                    found = list({id(f): f for t in tests for f in reveal(wk, t, hidden, [])[0]}.values())
                else:
                    found, seen = [], set()
                    for t in tests:
                        for s in shows.get(t, []):
                            if s.hpo_id in seen:
                                continue
                            seen.add(s.hpo_id)
                            rec = next((f for u, f in hid if u in r.ont.descendants(s.hpo_id)), None)
                            if rec is None:
                                draw = random.Random(f"{seed}:{c.case_id}:{s.hpo_id}").random()
                                rec = Finding(s.hpo_id, Status.PRESENT if draw < true_freq.get(s.hpo_id, 0.0)
                                              else Status.ABSENT)
                            found.append(rec)
                p1 = ranker_probs(r.score(Case(c.case_id, cur.findings + found, c.truth, [], meta={}), ex), calib)
                i = r.d_index[dw.disease]
                rows.append({"case": c.case_id, "disease": dw.disease, "is_truth": dw.disease in truth,
                             "policy": pol, "world": world, "tests": " | ".join(wk.cat.tests[t].name for t in tests),
                             "burden": sum(wk.cat.tests[t].burden for t in tests), "results": len(found),
                             "p_before": float(p0[i]), "p_after": float(p1[i]), "dp": float(p1[i] - p0[i])})
    return rows


def bootstrap(x: np.ndarray, n: int = 2000, seed: int = 13) -> tuple[float, float]:
    # 95% interval of the mean, resampling cases
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def gene_yield(wk: Workupper, c: Case, ex: frozenset, calib, ks=(1, 3, 5, 10)) -> dict:
    # is the case's causal gene among the genes of the top-k diseases' gene tests? (all findings known)
    rk = wk.r.score(c, ex)
    p = ranker_probs(rk, calib)
    top = [wk.r.diseases[i] for i in np.argsort(-p, kind="stable")[:max(ks)]]
    out = {"case": c.case_id, "genes": ";".join(c.genes), "truth_rank": rk.rank_of(c.truth)}
    for k in ks:
        panel = sorted(set().union(*(set(wk.gene_list(d)[0]) for d in top[:k])))
        out[f"found@{k}"] = bool(set(c.genes) & set(panel))
        out[f"panel@{k}"] = len(panel)
    if wk.gtr:      # the best registered GTR panel for the top 5 (section 3.4 of the docs)
        best = wk.gtr.panels([(d, float(p[wk.r.d_index[d]])) for d in top[:5]], 1)
        out["gtr_panel_found"] = bool(best) and bool(set(c.genes) & set(best[0]["test"].genes))
        out["gtr_panel_genes"] = len(best[0]["test"].genes) if best else 0
        out["gtr_panel_exists"] = bool(best)
    return out


def cmd_workup_eval(a, hb):
    import pandas as pd
    r = hb.load_kb(a)
    wk = Workupper(r, TestCatalog(r.ont, a.kb), a.gene_sensitivity, Path(a.kb) / "genes_to_disease.txt",
                   load_gtr(a.kb, not a.no_gtr))
    every = [c for c in hb.pick_cases(r, a.phenopackets, None, seed=a.seed)
             if len(c.present) >= 2 and hb.evaluable(r, c)]
    fit, cases = every[: a.calibration_cases], every[a.calibration_cases: a.calibration_cases + a.sample]
    pols = [x.strip() for x in a.policies.split(",") if x.strip()]
    bad = [x for x in pols if x not in POLICIES]
    if bad:
        raise SystemExit(f"unknown policy {', '.join(bad)}; choose from {', '.join(POLICIES)}")
    if not cases:
        raise SystemExit(f"no evaluation cases left after {len(fit)} calibration cases")
    calib = calibrate(r, fit, hb.leak)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps({"temperature": calib[0], "inside": calib[1], "top": DIFFERENTIAL,
                                                      "cases": len(fit), "hpo": r.ont.version, "seed": a.seed}, indent=1),
                                          encoding="utf-8")
    hb.wrote(out / "calibration.json")
    print(f"[workup-eval] calibration on {len(fit)} cases: T={calib[0]:.2f}, the diagnosis is in the top "
          f"{DIFFERENTIAL} for {calib[1]:.1%} (-> {out / 'calibration.json'}, use with workup --calibration)")
    print(f"[workup-eval] {len(cases)} other evaluable cases; half the observed findings and all excluded ones hidden; "
          f"{a.steps} tests each; gene tests {'on' if a.genes else 'off'}; {len(wk.cat.tests)} tests")
    hb.RUN["settings"].update({"policies": pols, "steps": a.steps, "calibration": calib, "genes": a.genes,
                               "calibration_cases": len(fit)})
    rows = []
    for i, c in enumerate(cases, 1):
        ex = hb.leak(c)
        for pol in pols:
            rows += simulate_case(wk, c, ex, pol, a.steps, a.seed, calib, a.genes)
        print(f"\r[workup-eval] {i}/{len(cases)}", end="", flush=True)
    print()
    df = pd.DataFrame(rows)
    df.to_csv(hb.wrote(out / "workup_eval.csv"), index=False)
    ranks = lambda s: [None if pd.isna(x) else int(x) for x in s]
    print("\n| policy | tests done | Top-1 | Top-3 | Top-10 | MRR | mean P(truth) | mean burden | test showed a hidden finding |"
          "\n|---|---|---|---|---|---|---|---|---|")
    summary = {"task": "workup-eval", "cases": len(cases), "steps": a.steps, "genes": a.genes,
               "calibration_T": round(calib[0], 3), "calibration_inside": round(calib[1], 3)}
    for pol in pols:
        for step in range(a.steps + 1):
            d = df[(df.policy == pol) & (df.step == step)]
            m = hb.topk(ranks(d["rank"]))
            prev = df[(df.policy == pol) & (df.step == step - 1)]
            rev = f"{(prev['revealed'].fillna(0) > 0).mean():.0%}" if step and "revealed" in prev else "–"
            print(f"| {pol} | {step} | {m['top1']:.3f} | {m['top3']:.3f} | {m['top10']:.3f} | {m['mrr']:.3f} | "
                  f"{d.p_truth.mean():.3f} | {d.burden.mean():.1f} | {rev} |")
            summary.update({f"{pol}_step{step}_{k}": round(v, 4) for k, v in m.items()})
            summary[f"{pol}_step{step}_p_truth"] = round(float(d.p_truth.mean()), 4)
            summary[f"{pol}_step{step}_burden"] = round(float(d.burden.mean()), 2)
    dv = pd.DataFrame([x for c in cases for x in disease_eval(wk, c, hb.leak(c), a.seed, calib, a.budget)])
    if len(dv):
        dv.to_csv(hb.wrote(out / "workup_diseases.csv"), index=False)
        n_true = dv[dv.is_truth].case.nunique()
        print(f"\nPer-disease workups: each of the 5 diseases worked up gets its own tests (burden budget {a.budget}); "
              f"change in that disease's probability.\nTrue disease ({n_true} cases where it is among the 5): it should "
              "rise. Other diseases: they should fall. 95% intervals resample cases.")
        print("| world | tests chosen by | true disease: change | other diseases: change | mean burden |\n|---|---|---|---|---|")
        for world in WORLDS:
            for pol in DISEASE_POLICIES:
                d = dv[(dv.world == world) & (dv.policy == pol)]
                t = d[d.is_truth].dp.to_numpy()
                o = d[~d.is_truth].groupby("case").dp.mean().to_numpy()
                (tl, th), (ol, oh) = bootstrap(t), bootstrap(o)
                print(f"| {world} | {pol} | {t.mean():+.4f} [{tl:+.4f}, {th:+.4f}] | {o.mean():+.4f} [{ol:+.4f}, {oh:+.4f}] | "
                      f"{d.burden.mean():.1f} |")
                summary.update({f"disease_{world}_{pol}_true_dp": round(float(t.mean()), 4),
                                f"disease_{world}_{pol}_other_dp": round(float(o.mean()), 4)})
        # paired: the workup against each other policy, per case: rise of the true disease minus mean change of others
        print("\nPaired difference per case (true disease's rise minus the others' mean change), workup minus:")
        for world in WORLDS:
            d = dv[dv.world == world]
            per = {}
            for pol in DISEASE_POLICIES:
                x = d[d.policy == pol]
                t = x[x.is_truth].set_index("case").dp
                o = x[~x.is_truth].groupby("case").dp.mean()
                per[pol] = (t - o.reindex(t.index).fillna(0))
            for pol in DISEASE_POLICIES[1:]:
                diff = (per["workup"] - per[pol]).dropna().to_numpy()
                lo, hi = bootstrap(diff)
                print(f"   {world}: {pol} {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] ({len(diff)} cases)")
                summary[f"disease_{world}_workup_minus_{pol}"] = round(float(diff.mean()), 4)
    gy = pd.DataFrame([gene_yield(wk, c, hb.leak(c), calib) for c in cases if c.genes])
    if len(gy):
        gy.to_csv(hb.wrote(out / "workup_gene_yield.csv"), index=False)
        print(f"\nCausal gene among the gene tests of the top-k diseases ({len(gy)} cases with a gene, all findings known):")
        print("| k | gene found | median genes tested |\n|---|---|---|")
        for k in (1, 3, 5, 10):
            print(f"| {k} | {gy[f'found@{k}'].mean():.3f} | {gy[f'panel@{k}'].median():.0f} |")
            summary[f"gene_found@{k}"] = round(float(gy[f"found@{k}"].mean()), 4)
            summary[f"gene_panel@{k}"] = float(gy[f"panel@{k}"].median())
        if "gtr_panel_found" in gy:
            has = gy[gy.gtr_panel_exists]
            print(f"| best registered GTR panel for the top 5 | {gy.gtr_panel_found.mean():.3f} | "
                  f"{has.gtr_panel_genes.median():.0f} (a panel exists for {len(has)} of {len(gy)} cases) |")
            summary["gtr_panel_found"] = round(float(gy.gtr_panel_found.mean()), 4)
            summary["gtr_panel_exists"] = round(float(gy.gtr_panel_exists.mean()), 4)
    hb.note("workup_eval", **summary)
    hb.log_compare(a, summary)


def add_arguments(p, sub):
    s = sub.add_parser("download-tests", help="MAxO diagnostic annotations, LOINC2HPO and GTR -> --kb folder")
    s.add_argument("--out", default="results", help="run log goes to <out>/runs/")
    s = sub.add_parser("workup", help="per-disease workup: tests that would confirm or rule out each leading disease")
    s.add_argument("case", nargs="+", help="case file (examples/*.json) or phenopacket")
    s.add_argument("-k", type=int, default=5, help="diseases to work up")
    s.add_argument("--per-list", type=int, default=3, help="tests shown per disease")
    s.add_argument("--calibration", help="calibration.json from workup-eval (default: the values in workup.py)")
    s.add_argument("--gene-sensitivity", type=float, default=1 - LEAK,
                   help="chance a gene test finds the variant when the disease is present (default: upper bound)")
    s.add_argument("--no-gtr", action="store_true", help="leave out the Genetic Testing Registry even if downloaded")
    s.add_argument("--out", default="results")
    s = sub.add_parser("workup-eval", help="simulated workups on Phenopacket Store: does the suggested test help?")
    s.add_argument("--phenopackets", default="data/phenopacket-store")
    s.add_argument("--calibration-cases", type=int, default=400, help="first cases: fit the pre-test calibration")
    s.add_argument("--sample", type=int, default=200, help="next cases: evaluate")
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--steps", type=int, default=3, help="tests per simulated case")
    s.add_argument("--policies", default=",".join(POLICIES), help=f"comma-separated, from {', '.join(POLICIES)}")
    s.add_argument("--genes", action="store_true", help="let the workup policies choose gene tests too")
    s.add_argument("--budget", type=int, default=6, help="burden budget per disease in the per-disease evaluation")
    s.add_argument("--gene-sensitivity", type=float, default=1 - LEAK)
    s.add_argument("--no-gtr", action="store_true", help="leave out the Genetic Testing Registry even if downloaded")
    s.add_argument("--out", default="results/workup_eval")


def commands(hb) -> dict:
    return {"download-tests": lambda a: cmd_download_tests(a, hb), "workup": lambda a: cmd_workup(a, hb),
            "workup-eval": lambda a: cmd_workup_eval(a, hb)}

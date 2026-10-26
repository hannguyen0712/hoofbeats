# core model: ontology, annotations, LR ranker, explanations, next-test selection
from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = "HP:0000118"  # Phenotypic abnormality


# data types
class Status(str, Enum):
    PRESENT = "present"      # observed in the patient
    ABSENT = "absent"        # looked for, not found
    FAMILY = "family"        # a relative's finding -- ignored by the ranker
    UNCERTAIN = "uncertain"  # hedged ("rule out") -- ignored by the ranker


@dataclass(frozen=True)
class Finding:
    hpo_id: str
    status: Status
    source: str = ""     # "phenopacket", "lab:Ammonia", "note"
    evidence: str = ""   # patient-side text or value


@dataclass
class Case:
    case_id: str
    findings: list[Finding]
    truth: list[str] = field(default_factory=list)         # gold disease ids
    genes: list[str] = field(default_factory=list)
    source_pmids: list[str] = field(default_factory=list)  # for leakage control
    meta: dict = field(default_factory=dict)

    def terms(self, status: Status) -> list[str]:
        return list(dict.fromkeys(f.hpo_id for f in self.findings if f.status == status))

    @property
    def present(self):
        return self.terms(Status.PRESENT)

    @property
    def absent(self):
        return self.terms(Status.ABSENT)


# ontology
@dataclass
class Term:
    id: str
    name: str = ""
    synonyms: list[str] = field(default_factory=list)  # EXACT only
    parents: list[str] = field(default_factory=list)
    obsolete: bool = False
    replaced_by: str | None = None
    extra_synonyms: list[tuple[str, str]] = field(default_factory=list)   # (text, "related" | "layperson")


SYNONYM = re.compile(r'"((?:[^"\\]|\\.)*)"\s+(EXACT|RELATED|BROAD|NARROW)(?:\s+([A-Za-z_]+))?')


class Ontology:
    def __init__(self, path, extra_synonyms: bool = True):
        # extra_synonyms: also keep RELATED and layperson synonyms (in Term.extra_synonyms) for text matching
        self.path = Path(path)
        self.terms: dict[str, Term] = {}
        self.alt: dict[str, str] = {}
        self.version = ""
        cur, in_term = None, False
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.startswith("data-version:"):
                self.version = line.split(":", 1)[1].strip()
            if line.startswith("["):
                in_term, cur = line == "[Term]", None
                continue
            if not in_term or ":" not in line:
                continue
            key, val = (s.strip() for s in line.split(":", 1))
            if key == "id":
                cur = self.terms.setdefault(val, Term(val))
            elif cur is None:
                continue
            elif key == "name":
                cur.name = val
            elif key == "is_a":
                cur.parents.append(val.split("!")[0].strip())
            elif key == "alt_id":
                self.alt[val] = cur.id
            elif key == "is_obsolete":
                cur.obsolete = val == "true"
            elif key == "replaced_by":
                cur.replaced_by = val
            elif key == "synonym" and " EXACT" in val:
                cur.synonyms.append(val.split('"')[1])
            elif key == "synonym" and extra_synonyms:
                m = SYNONYM.match(val)
                if m and m.group(3) != "obsolete_synonym" and (m.group(2) == "RELATED" or m.group(3) == "layperson"):
                    cur.extra_synonyms.append((m.group(1), "layperson" if m.group(3) == "layperson" else "related"))
        self.children = defaultdict(list)
        for t in self.terms.values():
            for p in t.parents:
                self.children[p].append(t.id)
        self.ancestors = lru_cache(maxsize=None)(self._walk_up)

    def _walk_up(self, tid: str) -> frozenset[str]:
        out, stack = {tid}, [tid]
        while stack:
            t = self.terms.get(stack.pop())
            for p in t.parents if t else ():
                if p not in out:
                    out.add(p)
                    stack.append(p)
        return frozenset(out)

    def descendants(self, tid: str) -> set[str]:
        out, stack = {tid}, [tid]
        while stack:
            for c in self.children.get(stack.pop(), ()):
                if c not in out:
                    out.add(c)
                    stack.append(c)
        return out

    def resolve(self, tid: str | None) -> str | None:
        # current id for alt/obsolete ids; None if not a phenotype
        t = self.terms.get(self.alt.get(tid, tid))
        if t is None:
            return None
        if t.obsolete:
            return self.resolve(t.replaced_by) if t.replaced_by else None
        return t.id if self.is_phenotype(t.id) else None

    def is_phenotype(self, tid: str) -> bool:
        return tid != ROOT and ROOT in self.ancestors(tid)

    def name(self, tid: str) -> str:
        return self.terms[tid].name if tid in self.terms else tid


# annotations
FREQ_TERMS = {"HP:0040280": 1.0, "HP:0040281": 0.895, "HP:0040282": 0.545,
              "HP:0040283": 0.17, "HP:0040284": 0.025, "HP:0040285": 0.0}


def parse_frequency(raw: str, default: float = 0.5) -> float:
    # frequency term, n/m (shrunk so 1/1 is 2/3), or x%
    raw = (raw or "").strip()
    try:
        if raw in FREQ_TERMS:
            return FREQ_TERMS[raw]
        if raw.endswith("%"):
            return float(raw[:-1]) / 100
        if "/" in raw:
            n, m = (int(x) for x in raw.split("/"))
            return (n + 1) / (m + 2)
    except ValueError:
        pass
    return default


@dataclass(frozen=True)
class Annotation:
    hpo_id: str
    freq: float
    refs: frozenset[str]
    negated: bool = False


class AnnotationDB:
    # disease profiles from phenotype.hpoa, with each annotation's references

    def __init__(self, hpoa_path, ont: Ontology, prefixes=("OMIM",), genes_path=None):
        self.ont = ont
        self.names: dict[str, str] = {}
        self.annos: dict[str, list[Annotation]] = defaultdict(list)
        self.ref_index: dict[str, set[str]] = defaultdict(set)
        self.gene_diseases: dict[str, set[str]] = defaultdict(set)
        self.disease_genes: dict[str, set[str]] = defaultdict(set)
        self.scope: dict[str, set[str]] = {}       # common diseases: the terms their source could observe
        self.prior: dict[str, float] = {}          # common diseases: share of admissions with the diagnosis
        self.policy = "open"                       # "local_only" once credentialed-derived profiles are added
        self.version = ""
        lines = []
        for line in Path(hpoa_path).read_text(encoding="utf-8").splitlines(keepends=True):
            if line.startswith("#version:"):
                self.version = line.split(":", 1)[1].strip()
            elif not line.startswith("#"):
                lines.append(line)
        for row in csv.DictReader(lines, delimiter="\t"):
            did, hpo = row["database_id"], ont.resolve(row["hpo_id"])
            if did.split(":")[0] not in prefixes or row["aspect"] != "P" or hpo is None:
                continue
            refs = frozenset(r.strip() for r in row["reference"].split(";") if r.strip())
            neg = row["qualifier"].strip() == "NOT"
            self.names[did] = row["disease_name"]
            self.annos[did].append(Annotation(hpo, 0.0 if neg else parse_frequency(row["frequency"]), refs, neg))
            for r in refs:
                self.ref_index[r].add(did)
        if genes_path:
            for row in csv.DictReader(Path(genes_path).read_text(encoding="utf-8").splitlines(), delimiter="\t"):
                if row["disease_id"] in self.names:
                    self.gene_diseases[row["gene_symbol"]].add(row["disease_id"])
                    self.disease_genes[row["disease_id"]].add(row["gene_symbol"])

    def add_common(self, profiles: dict) -> int:
        # add common-disease profiles from learn_common
        scope = set(profiles["scope"])
        for did, p in profiles["diseases"].items():
            self.names[did] = p["name"]
            self.scope[did] = scope
            self.prior[did] = p["n_adm"] / profiles["n_admissions"]
            self.annos[did] = [Annotation(self.ont.resolve(h), (pos + 1) / (meas + 2), frozenset({profiles["source"]}))
                               for h, (pos, meas) in p["counts"].items() if meas > 0 and self.ont.resolve(h)]
        if profiles["policy"] != "open":
            self.policy = "local_only"
        return len(profiles["diseases"])

    def annotations(self, d: str, exclude: frozenset[str] = frozenset()) -> list[Annotation]:
        # drop annotations supported only by excluded references
        return [a for a in self.annos.get(d, []) if not a.refs or not a.refs <= exclude]


# ranker
@dataclass
class Contribution:
    hpo_id: str
    status: Status
    log_lr: float
    kb_term: str | None = None     # disease-side term that matched
    kb_freq: float | None = None
    refs: tuple[str, ...] = ()


@dataclass
class Ranking:
    diseases: list[str]
    log_scores: np.ndarray
    probs: np.ndarray
    order: np.ndarray

    def top(self, k=10):
        return [(self.diseases[i], float(self.log_scores[i]), float(self.probs[i])) for i in self.order[:k]]

    def rank_of(self, truth: list[str]) -> int | None:
        gold = np.array([d in truth for d in self.diseases])
        if not gold.any():
            return None
        best = self.log_scores[gold].max()
        return int((self.log_scores[~gold] >= best).sum()) + 1   # ties count against the gold disease

    def prob_of(self, truth: list[str]) -> float:
        return float(sum(p for d, p in zip(self.diseases, self.probs) if d in truth))


class LRRanker:
    def __init__(self, ont: Ontology, db: AnnotationDB, lr_floor=0.05, absent_weight=1.0,
                 use_genes=False, gene_lr=100.0, temperature=1.0, prior_weight=1.0, rare_mass=0.04):
        # prior_weight 0 = evidence only; rare_mass = share of rare admissions
        self.ont, self.db = ont, db
        self.prior_weight = prior_weight
        self.lr_floor, self.absent_weight = lr_floor, absent_weight
        self.use_genes, self.gene_lr, self.temperature = use_genes, gene_lr, temperature
        self.diseases = sorted(db.names)
        self.d_index = {d: i for i, d in enumerate(self.diseases)}
        rows, cols, vals, nr, nc, t_index = [], [], [], [], [], {}
        for j, d in enumerate(self.diseases):
            freq, _, neg = self.implied(d)
            for t, f in freq.items():
                rows.append(t_index.setdefault(t, len(t_index))); cols.append(j); vals.append(f)
            for t in neg:
                for x in ont.descendants(t):
                    nr.append(t_index.setdefault(x, len(t_index))); nc.append(j)
        shape = (len(t_index), len(self.diseases))
        self.t_index = t_index
        self.M = sparse.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=shape)
        self.NEG = sparse.csr_matrix((np.ones(len(nr), bool), (nr, nc)), shape=shape)
        self.bg = (np.asarray(self.M.sum(axis=1)).ravel() + 0.5) / (len(self.diseases) + 1)
        # common diseases only "see" their measured terms (plus ancestors/subtypes of them)
        self.scoped = {}
        for j, d in enumerate(self.diseases):
            if d in db.scope:
                key = id(db.scope[d])
                if key not in self.scoped:
                    ext = set()
                    for t in db.scope[d]:
                        ext |= ont.ancestors(t) | ont.descendants(t)
                    self.scoped[key] = (ext - {ROOT}, [])
                self.scoped[key][1].append(j)
        n_rare = sum(d not in db.prior for d in self.diseases)
        self.log_prior = np.array([math.log(db.prior[d]) if d in db.prior else math.log(rare_mass / max(n_rare, 1))
                                   for d in self.diseases]) if db.prior else np.zeros(len(self.diseases))

    def in_scope(self, t: str, d: str) -> bool:
        if d not in self.db.scope:
            return True
        return any(t in ext for ext, idx in self.scoped.values() if self.d_index[d] in idx)

    def _unscope(self, t: str, lr: np.ndarray) -> np.ndarray:
        for ext, idx in self.scoped.values():
            if t not in ext:
                lr[idx] = 1.0      # outside what the source measured: no information
        return lr

    @property
    def policy(self) -> str:
        return self.db.policy

    def implied(self, d: str, exclude: frozenset[str] = frozenset()):
        # term frequencies propagated to ancestors, plus negated terms
        freq, refs, neg = {}, {}, set()
        for a in self.db.annotations(d, exclude):
            if a.negated:
                neg.add(a.hpo_id)
                continue
            for u in self.ont.ancestors(a.hpo_id):
                if not self.ont.is_phenotype(u):
                    continue
                if a.freq > freq.get(u, -1):
                    freq[u], refs[u] = a.freq, set(a.refs)
                elif a.freq == freq[u]:
                    refs[u] |= a.refs
        return freq, refs, neg

    def bg_of(self, t: str) -> float:
        i = self.t_index.get(t)
        return float(self.bg[i]) if i is not None else 0.5 / (len(self.diseases) + 1)

    def _terms(self, case: Case, status: Status) -> list[str]:
        return list(dict.fromkeys(r for r in map(self.ont.resolve, case.terms(status)) if r))

    def score(self, case: Case, exclude: frozenset[str] = frozenset()) -> Ranking:
        # exclude = references to ignore for this case (leakage control)
        n = len(self.diseases)
        s = np.zeros(n)
        for t in self._terms(case, Status.PRESENT):
            anc = [self.t_index[u] for u in self.ont.ancestors(t) if u in self.t_index]
            lr = np.full(n, self.lr_floor)
            if anc:
                lr = np.maximum((self.M[anc].toarray() / self.bg[anc][:, None]).max(axis=0), self.lr_floor)
            if t in self.t_index:
                lr[self.NEG[self.t_index[t]].toarray().ravel()] = self.lr_floor
            s += np.log(self._unscope(t, lr))
        for t in self._terms(case, Status.ABSENT):
            i = self.t_index.get(t)
            if i is None:
                continue
            lr, bg = np.ones(n), min(self.bg[i], 0.99)
            lr[self.NEG[i].toarray().ravel()] = 1 / (1 - bg)
            row = self.M[i]
            lr[row.indices] = (1 - row.data) / (1 - bg)
            s += self.absent_weight * np.log(self._unscope(t, np.maximum(lr, self.lr_floor)))
        if self.use_genes:
            for g in case.genes:
                for d in self.db.gene_diseases.get(g, ()):
                    s[self.d_index[d]] += math.log(self.gene_lr)
        # recompute exactly the diseases whose profile changes once leaked references are removed
        for d in {d for r in exclude for d in self.db.ref_index.get(r, ())}:
            j = self.d_index[d]
            gene = math.log(self.gene_lr) if self.use_genes and set(case.genes) & self.db.disease_genes[d] else 0
            s[j] = sum(c.log_lr for c in self.contributions(case, d, exclude)) + gene
        s = s + self.prior_weight * self.log_prior
        z = s / self.temperature
        p = np.exp(z - z.max())
        return Ranking(self.diseases, s, p / p.sum(), np.lexsort((np.arange(n), -s)))

    def contributions(self, case: Case, d: str, exclude: frozenset[str] = frozenset()) -> list[Contribution]:
        # per-finding log-LR for one disease (matches score())
        freq, refs, neg = self.implied(d, exclude)
        lacks = set().union(*(self.ont.descendants(t) for t in neg)) if neg else set()
        out = []
        for t in self._terms(case, Status.PRESENT):
            if not self.in_scope(t, d):
                out.append(Contribution(t, Status.PRESENT, 0.0))
                continue
            best, u_best = self.lr_floor, None
            for u in sorted(self.ont.ancestors(t)):     # sorted: same tie-break in every run
                if u in freq and freq[u] / self.bg_of(u) > best:
                    best, u_best = freq[u] / self.bg_of(u), u
            if t in lacks:
                best, u_best = self.lr_floor, None
            out.append(Contribution(t, Status.PRESENT, math.log(best), u_best, freq.get(u_best),
                                    tuple(sorted(refs.get(u_best, ())))))
        for t in self._terms(case, Status.ABSENT):
            if t not in self.t_index or not self.in_scope(t, d):
                continue
            bg = min(self.bg_of(t), 0.99)
            lr = (1 - freq[t]) / (1 - bg) if t in freq else (1 / (1 - bg) if t in lacks else 1.0)
            out.append(Contribution(t, Status.ABSENT, self.absent_weight * math.log(max(lr, self.lr_floor)),
                                    t if t in freq else None, freq.get(t), tuple(sorted(refs.get(t, ())))))
        return out

    def calibrate(self, score_sets: list[np.ndarray], gold_idx: list[int]) -> float:
        # fit softmax temperature on a dev split
        def nll(T):
            tot = 0.0
            for s, g in zip(score_sets, gold_idx):
                z = s / T
                tot += z.max() + math.log(np.exp(z - z.max()).sum()) - z[g]
            return tot
        self.temperature = float(min(np.exp(np.linspace(math.log(0.2), math.log(20), 60)), key=nll))
        return self.temperature


# explanations
@dataclass
class Claim:
    disease: str
    text: str
    patient_term: str
    status: Status
    kb_term: str | None
    kb_freq: float | None
    refs: tuple[str, ...]


def explain(r: LRRanker, case: Case, ranking: Ranking, k=5, exclude=frozenset(), max_claims=8):
    # top diseases with posterior and evidence claims
    out = []
    for d, _, p in ranking.top(k):
        name = r.db.names[d]
        claims = []
        for c in sorted(r.contributions(case, d, exclude), key=lambda c: -abs(c.log_lr))[:max_claims]:
            if abs(c.log_lr) < 0.1:
                continue
            pt, lr = r.ont.name(c.hpo_id), math.exp(c.log_lr)
            if c.status == Status.PRESENT and c.kb_term:
                via = "" if c.kb_term == c.hpo_id else f" (via '{r.ont.name(c.kb_term)}')"
                txt = f"Has '{pt}'; {name} shows this in ~{c.kb_freq:.0%} of cases{via}. LR {lr:.2f}"
            elif c.status == Status.PRESENT:
                txt = f"Has '{pt}', not described for {name}. LR {lr:.2f}"
            elif c.kb_term:
                txt = f"Lacks '{pt}', which {name} shows in ~{c.kb_freq:.0%} of cases. LR {lr:.2f}"
            else:
                txt = f"Lacks '{pt}', consistent with {name}. LR {lr:.2f}"
            refs = f" [{', '.join(c.refs)}]" if c.refs else ""
            claims.append(Claim(d, txt + refs, c.hpo_id, c.status, c.kb_term, c.kb_freq, c.refs))
        out.append((d, p, claims))
    return out


def verify_claim(r: LRRanker, case: Case, cl: Claim, exclude=frozenset()) -> tuple[bool, str]:
    # re-check a claim against the patient record and knowledge base
    if not any(r.ont.resolve(f.hpo_id) == cl.patient_term and f.status == cl.status for f in case.findings):
        return False, "finding not in patient record"
    if cl.kb_term is None:
        return True, "no annotation cited"
    freq, refs, _ = r.implied(cl.disease, exclude)
    if cl.kb_term not in freq or cl.kb_freq is None or abs(freq[cl.kb_term] - cl.kb_freq) > 1e-6:
        return False, "annotation or frequency not in knowledge base"
    if not set(cl.refs) <= refs[cl.kb_term] or set(cl.refs) & exclude:
        return False, "reference does not support it, or is the case's own paper"
    if cl.status == Status.PRESENT and cl.kb_term not in r.ont.ancestors(cl.patient_term):
        return False, "patient term is not the cited term or a subtype of it"
    return True, "verified"


# next-test selection
def _entropy(p):
    p = np.clip(p, 1e-12, 1)
    return -(p * np.log2(p)).sum(axis=-1)


def suggest_tests(r: LRRanker, ranking: Ranking, case: Case, exclude=frozenset(), asked=(),
                  top_k=20, n=5, noise=0.02) -> list[tuple[str, float, list]]:
    # rank unasked HPO terms by expected information gain
    top = ranking.top(top_k)
    ds = [d for d, _, _ in top]
    p = np.array([x for _, _, x in top] + [max(0.0, 1 - sum(x for _, _, x in top))])
    p /= p.sum()
    profiles = [r.implied(d, exclude)[0] for d in ds]
    known = set(asked)
    for t in case.present:
        known |= r.ont.ancestors(t)       # having t implies its ancestors
    for t in case.absent:
        known |= r.ont.descendants(t)     # lacking t implies lacking its subtypes
    cands = sorted({t for prof in profiles for t in prof} - known)
    if not cands:
        return []
    F = np.clip([[prof.get(q, 0) for prof in profiles] + [r.bg_of(q)] for q in cands], noise, 1 - noise)
    yes = F @ p
    eig = _entropy(p) - yes * _entropy(F * p / yes[:, None]) - (1 - yes) * _entropy((1 - F) * p / (1 - yes)[:, None])
    best = np.argsort(-eig, kind="stable")[:n]
    return [(cands[i], float(eig[i]), list(zip(ds, F[i, :-1].round(2).tolist()))) for i in best if eig[i] > 1e-3]


def most_frequent_test(r: LRRanker, ranking: Ranking, case: Case, exclude=frozenset(), asked=()):
    # baseline: most frequent unknown feature of the top disease
    known = set(asked)
    for t in case.present:
        known |= r.ont.ancestors(t)
    for t in case.absent:
        known |= r.ont.descendants(t)
    freq = r.implied(ranking.top(1)[0][0], exclude)[0]
    cands = [(f, -r.bg_of(t), t) for t, f in freq.items() if t not in known]
    return max(cands)[2] if cands else None
# linker: the HPO term for a phrase the LLM proposed. Dictionary first (the rule reader's labels and synonyms); else the
# nearest HPO labels/synonyms by SapBERT embedding; between the two thresholds the LLM picks one of the top candidates
# or none (its reply is constrained to those ids, so it cannot invent one)
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Extractor, tokens
from .llm import _json, chat_many, require_local
from .model import ROOT, Ontology

SAPBERT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
# words a phrase may carry beyond the matched label and still count as fully matched ("intermittent squint")
MODIFIERS = set("mild moderate severe marked slight significant progressive recurrent intermittent chronic acute "
                "bilateral unilateral left right generalized generalised diffuse focal transient persistent episodic "
                "frequent occasional early late congenital of the a an and in with".split())


class SapBert:
    # [CLS] vectors of a SapBERT model, L2-normalised; runs on CPU unless told otherwise
    def __init__(self, model: str = SAPBERT, device: str = "cpu", batch: int = 128):
        self.model_name, self.device, self.batch = model, device, batch
        self._tok = self._model = None

    @property
    def name(self) -> str:
        return re.sub(r"[^A-Za-z0-9.]+", "-", self.model_name.split("/")[-1]).lower()

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        from transformers import AutoModel, AutoTokenizer
        if self._model is None:
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch):
                enc = self._tok(texts[i:i + self.batch], padding=True, truncation=True, max_length=32, return_tensors="pt")
                cls = self._model(**enc.to(self.device)).last_hidden_state[:, 0]
                out.append(torch.nn.functional.normalize(cls, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 768), np.float32)


def term_strings(ont: Ontology) -> list[tuple[str, str]]:
    # (term id, label or synonym) for every current phenotype term; rows of one term are adjacent
    out = []
    for tid in sorted(ont.descendants(ROOT) - {ROOT}):
        t = ont.terms.get(tid)
        if t is None or t.obsolete:
            continue
        seen = set()
        for s in [t.name, *t.synonyms, *(x for x, _ in t.extra_synonyms)]:
            if s.lower() not in seen:
                seen.add(s.lower())
                out.append((tid, s))
    return out


class HpoIndex:
    # vectors of all phenotype labels and synonyms, cached as data/kb/.cache/<embedder>-<hpo version>.npz
    def __init__(self, ont: Ontology, embedder, cache_dir: str | Path | None = None):
        self.embedder = embedder
        slug = re.sub(r"[^A-Za-z0-9.]+", "-", f"{embedder.name}-{ont.version}")
        self.path = Path(cache_dir) / f"{slug}.npz" if cache_dir else None
        rows = term_strings(ont)
        ids, strings = [r[0] for r in rows], [r[1] for r in rows]
        if self.path and self.path.exists():
            z = np.load(self.path, allow_pickle=False)
            if list(z["ids"]) == ids and list(z["strings"]) == strings:
                self.vecs = z["vecs"].astype(np.float32)
            else:
                self.vecs = None
        else:
            self.vecs = None
        if self.vecs is None:
            self.vecs = embedder.encode(strings)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(self.path, ids=np.array(ids), strings=np.array(strings), vecs=self.vecs.astype(np.float16))
        self.ids, self.strings = ids, strings
        self.starts = np.array([i for i in range(len(ids)) if i == 0 or ids[i] != ids[i - 1]])
        self.term_ids = [ids[i] for i in self.starts]
        self.label_row = {t: int(i) for t, i in zip(self.term_ids, self.starts)}    # a term's first row is its label

    def similarity(self, texts: list[str], hpo_ids: list[str]) -> list[float | None]:
        # cosine between each text and its term's label (None for a term not in the index)
        if not texts:
            return []
        q = self.embedder.encode(texts)
        return [float(q[i] @ self.vecs[self.label_row[h]]) if h in self.label_row else None for i, h in enumerate(hpo_ids)]

    def search(self, q: np.ndarray, k: int = 10, mask_rows: list | None = None) -> list[list[tuple[str, float, str]]]:
        # per query: the k best terms as (id, cosine of its best string, that string); mask_rows[i] lists string rows
        # query i may not use (calibration leaves a synonym out)
        out = []
        for i0 in range(0, len(q), 256):
            sc = q[i0:i0 + 256] @ self.vecs.T
            if mask_rows:
                for j, rows in enumerate(mask_rows[i0:i0 + 256]):
                    sc[j, rows] = -np.inf
            best = np.maximum.reduceat(sc, self.starts, axis=1)
            for j in range(len(sc)):
                top = np.argsort(-best[j])[:k]
                res = []
                for t in top:
                    a = self.starts[t]
                    b = self.starts[t + 1] if t + 1 < len(self.starts) else len(self.ids)
                    r = a + int(np.argmax(sc[j, a:b]))
                    res.append((self.term_ids[t], float(best[j, t]), self.strings[r]))
                out.append(res)
        return out


def calibrate(index: HpoIndex, n: int = 2000, k: int = 10, target: float = 0.9, seed: int = 576) -> dict:
    # leave one synonym out: query with a term's synonym, with that string (and same-word variants of it) removed,
    # and see whether the term comes back. hi: the lowest top-1 score whose accepted links are >= target correct;
    # lo: the lowest score band (0.02 wide) above which every band still has the true term in the top k at least
    # half the time. Synonyms resemble the LLM's standard-term phrases, so this needs no PMC case
    norm = [tuple(tokens(s)) for s in index.strings]
    groups = {}
    for i, t in enumerate(index.ids):
        groups.setdefault(t, []).append(i)
    pool = [i for i, t in enumerate(index.ids) if len({norm[j] for j in groups[t]}) > 1]
    rows = random.Random(seed).sample(pool, min(n, len(pool)))
    masks = [[j for j in groups[index.ids[i]] if norm[j] == norm[i]] for i in rows]
    res = index.search(index.vecs[rows], k, masks)
    recs = [{"score": r[0][1], "margin": r[0][1] - r[1][1] if len(r) > 1 else 1.0, "correct": r[0][0] == index.ids[i],
             "in_top_k": index.ids[i] in [x[0] for x in r]} for i, r in zip(rows, res)]
    edges = np.round(np.arange(0.0, 1.0001, 0.02), 2)
    bands = []
    for a in edges[:-1]:                    # the first band also holds scores below 0, the last a score of 1
        b = [x for x in recs if (a == 0 or x["score"] >= a) and (x["score"] < a + 0.02 or a == edges[-2])]
        bands.append({"from": float(a), "n": len(b), "correct": sum(x["correct"] for x in b),
                      "in_top_k": sum(x["in_top_k"] for x in b)})
    # accept: top-1 score >= hi and ahead of the second term by >= margin; the pair accepting the most queries
    # with >= target correct (a margin lets a lower score through when no rival term is close)
    hi, margin, best = 1.0, 0.0, (0, 0.0)
    for t in np.round(np.arange(0.80, 0.9901, 0.01), 2):
        for m in np.round(np.arange(0.0, 0.1001, 0.01), 2):
            acc = [x for x in recs if x["score"] >= t and x["margin"] >= m]
            prec = sum(x["correct"] for x in acc) / len(acc) if acc else 0.0
            if acc and prec >= target and (len(acc), prec) > best:
                hi, margin, best = float(t), float(m), (len(acc), prec)
    lo = hi
    for band in reversed([b for b in bands if b["from"] < hi]):
        if band["n"] and band["in_top_k"] / band["n"] < 0.5:
            break
        lo = band["from"]
    acc = [x for x in recs if x["score"] >= hi and x["margin"] >= margin]
    return {"n": len(recs), "k": k, "target": target, "hi": hi, "margin": margin, "lo": lo,
            "accepted_share": len(acc) / len(recs), "accepted_correct": sum(x["correct"] for x in acc) / max(len(acc), 1),
            "top1_correct": sum(x["correct"] for x in recs) / len(recs),
            "in_top_k": sum(x["in_top_k"] for x in recs) / len(recs), "bands": bands}


@dataclass(frozen=True)
class Link:
    hpo_id: str | None
    method: str          # dict:<label kind> | embedding | llm_choice | unlinked | borderline (no LLM to choose)
    grounding: str       # strong | supported | weak ("" when unlinked)
    matched: str = ""    # the label/synonym it matched
    score: float | None = None
    candidates: tuple = ()


CHOOSE_PROMPT = """A clinical case report describes a finding as:
  phrase: "{phrase}"
  words in the report: "{quote}"

Which Human Phenotype Ontology term below names this finding? Answer "none" if none of them fits.
{options}"""
# v2 (frozen configuration): the whole sentence, and broader terms (direct parents of the candidates) for a finding
# no candidate names exactly; still only listed ids or "none"
CHOOSE_PROMPT_V2 = """A clinical case report describes a finding as:
  phrase: "{phrase}"
  words in the report: "{quote}"
  sentence: "{sentence}"

Which Human Phenotype Ontology term below names this finding? Prefer a candidate that names it exactly. If none
does, choose a broader term that is still true of it. If no term is true of it, or the sentence is not about a
finding of the patient, answer "none". Do not choose a term that adds details the sentence does not state.

Candidates:
{options}

Broader terms:
{broader}"""
CHOOSER_VERSIONS = (1, 2)


class Linker:
    def __init__(self, ont: Ontology, ext: Extractor, index: HpoIndex | None = None, llm=None,
                 hi: float = 1.0, lo: float = 1.0, k: int = 10, margin: float = 0.02, workers: int = 8,
                 chooser_version: int = 1):
        self.ont, self.ext, self.index, self.llm = ont, ext, index, llm
        self.hi, self.lo, self.k, self.margin, self.workers = hi, lo, k, margin, workers
        if chooser_version not in CHOOSER_VERSIONS:
            raise ValueError(f"chooser version must be one of {CHOOSER_VERSIONS}")
        self.chooser_version = chooser_version

    def broader(self, ids: list[str]) -> list[str]:
        # direct parents of the candidates that are phenotype terms and not candidates themselves, in candidate order
        out = []
        for t in ids:
            for p in self.ont.terms[t].parents if t in self.ont.terms else ():
                if p not in ids and p not in out and self.ont.is_phenotype(p):
                    out.append(p)
        return out

    def dictionary(self, phrase: str) -> list[Link]:
        # the rule reader's labels and synonyms, only if they account for every word of the phrase except modifiers
        ms = self.ext.mentions(phrase)
        covered = [False] * len(phrase)
        for m in ms:
            covered[m.start:m.end] = [True] * (m.end - m.start)
        rest = "".join(ch if not c else " " for ch, c in zip(phrase, covered))
        if not ms or any(w not in MODIFIERS for w in tokens(rest)):
            return []
        out = {}
        for m in ms:
            out.setdefault(m.hpo_id, Link(m.hpo_id, f"dict:{m.link[0]}", m.grounding, m.link[1] or m.quote))
        return list(out.values())

    def link_many(self, items: list[tuple]) -> list[list[Link]]:
        # items: (phrase, quote) or (phrase, quote, sentence); one list of links per item (several when the phrase
        # names several terms); chooser v2 shows the sentence
        out = [self.dictionary(it[0]) for it in items]
        todo = [i for i, links in enumerate(out) if not links]
        if not todo:
            return out
        if self.index is None:
            for i in todo:
                out[i] = [Link(None, "unlinked", "")]
            return out
        found = self.index.search(self.index.embedder.encode([items[i][0] for i in todo]), self.k)
        ask = []
        for i, cands in zip(todo, found):
            (t1, s1, m1), s2 = cands[0], cands[1][1] if len(cands) > 1 else -1.0
            cand = tuple((t, round(s, 4)) for t, s, _ in cands)
            if s1 >= self.hi and s1 - s2 >= self.margin:
                out[i] = [Link(t1, "embedding", "supported", m1, round(s1, 4), cand)]
            elif s1 >= self.lo:
                out[i] = [Link(None, "borderline", "", m1, round(s1, 4), cand)]
                ask.append(i)
            else:
                out[i] = [Link(None, "unlinked", "", m1, round(s1, 4), cand)]
        if ask and self.llm is not None:
            require_local(self.llm)
            calls = []
            offered = {}
            for i in ask:
                ids = [t for t, _ in out[i][0].candidates]
                up = self.broader(ids) if self.chooser_version == 2 else []
                offered[i] = ids + up
                opts = "\n".join(f"{t}: {self.ont.name(t)}" for t in ids)
                schema = {"type": "object", "additionalProperties": False, "required": ["choice"],
                          "properties": {"choice": {"type": "string", "enum": offered[i] + ["none"]}}}
                if self.chooser_version == 1:
                    user = CHOOSE_PROMPT.format(phrase=items[i][0], quote=items[i][1][:300], options=opts)
                else:
                    sent = items[i][2] if len(items[i]) > 2 and items[i][2] else items[i][1]
                    user = CHOOSE_PROMPT_V2.format(phrase=items[i][0], quote=items[i][1][:300], sentence=sent[:600],
                                                   options=opts, broader="\n".join(f"{t}: {self.ont.name(t)}" for t in up)
                                                   or "(none)")
                calls.append({"user": user, "schema": schema})
            for i, reply in zip(ask, chat_many(self.llm, calls, self.workers)):
                obj = _json(reply, "{") if reply else None
                choice = obj.get("choice") if isinstance(obj, dict) else None
                old = out[i][0]
                if choice in offered[i]:         # a broader term has no score of its own
                    out[i] = [Link(choice, "llm_choice" if choice in dict(old.candidates) else "llm_choice:broader",
                                   "weak", self.ont.name(choice), dict(old.candidates).get(choice), old.candidates)]
                else:
                    out[i] = [Link(None, "unlinked" if choice == "none" else "borderline", "", old.matched, old.score,
                                   old.candidates)]
        return out


# cross-check: the rule reader on the LLM's quote alone
FILLER = set("a an the of in on at to by for from with without and or but as is was were are be been being has have had "
             "he she his her him their they it its this that these those patient child boy girl also both which who "
             "there showed revealed noted found presented observed seen no not none never negative normal absent "
             "absence denies denied".split())


@dataclass(frozen=True)
class Check:
    verdict: str                 # agree | no_rule_term | disagree | explained_by_other_terms
    rule_terms: tuple = ()       # terms the rules find in the quote
    similarity: float | None = None
    low_similarity: bool = False


def crosscheck(ont: Ontology, ext: Extractor, hpo_id: str, quote: str, similarity: float | None = None,
               lo: float | None = None) -> Check:
    # agree: the rules find this term (or a parent/child) in the quote. no_rule_term: they find nothing (a paraphrase;
    # kept). explained_by_other_terms: they find only unrelated terms and those account for every content word of
    # the quote, so nothing is left to support this term (dropped). disagree: unrelated terms, but words remain
    # (kept as weak). low_similarity (SapBERT quote vs the term's label below lo) is only a flag: paraphrases score
    # low by nature ("testes were not palpable" vs Cryptorchidism)
    ms = ext.mentions(quote)
    terms = tuple(dict.fromkeys(m.hpo_id for m in ms))
    low = similarity is not None and lo is not None and similarity < lo
    if any(t == hpo_id or t in ont.ancestors(hpo_id) or hpo_id in ont.ancestors(t) for t in terms):
        return Check("agree", terms, similarity, low)
    if not ms:
        return Check("no_rule_term", terms, similarity, low)
    covered = [False] * len(quote)
    for m in ms:
        covered[m.start:m.end] = [True] * (m.end - m.start)
    rest = [w for w in tokens("".join(ch if not c else " " for ch, c in zip(quote, covered)))
            if w not in FILLER and w not in MODIFIERS and not w.isdigit()]
    return Check("disagree" if rest else "explained_by_other_terms", terms, similarity, low)


def calibrated(index: HpoIndex, rebuild: bool = False) -> dict:
    # thresholds saved next to the index (<index>.calibration.json); computed once
    p = index.path.with_suffix(".calibration.json") if index.path else None
    if p and p.exists() and not rebuild:
        return json.loads(p.read_text())
    cal = calibrate(index)
    if p:
        p.write_text(json.dumps(cal, indent=1))
    return cal


def build_linker(ont: Ontology, ext: Extractor, llm=None, cache_dir: str | Path | None = None, embedder=None,
                 workers: int = 8, chooser_version: int = 1) -> tuple[Linker, dict]:
    index = HpoIndex(ont, embedder or SapBert(), cache_dir)
    cal = calibrated(index)
    return Linker(ont, ext, index, llm, hi=cal["hi"], lo=cal["lo"], k=cal["k"], margin=cal["margin"], workers=workers,
                  chooser_version=chooser_version), cal

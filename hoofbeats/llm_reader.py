# llm_reader: the LLM text reader for intake: candidates (llm_extract) -> HPO links (linker) -> cross-check -> findings
from __future__ import annotations

from dataclasses import asdict

from .data import Extractor
from .grounding import RANK, GroundedFinding
from .linker import Linker, crosscheck
from .llm_extract import PROMPT_VERSION, extract_candidates
from .model import Ontology


class LLMReader:
    # call it like a reader: reader(text, source, field) -> findings with reader="llm". Every candidate, linked or not,
    # kept or dropped, goes to .log (with .case_id); extraction counts go to .reports. Results are remembered per text,
    # so several configurations of one run cost one set of requests
    def __init__(self, llm, linker: Linker, ont: Ontology, ext: Extractor, prompt_version: int = PROMPT_VERSION,
                 max_chars: int = 2500, workers: int = 8):
        self.llm, self.linker, self.ont, self.ext = llm, linker, ont, ext
        self.prompt_version, self.max_chars, self.workers = prompt_version, max_chars, workers
        self.case_id, self.log, self.reports, self._memo = "", [], [], {}

    def __call__(self, text: str | None, source: str, field: str) -> list[GroundedFinding]:
        if not text or not text.strip():
            return []
        key = (text, source, field)
        if key not in self._memo:
            self._memo[key] = self._read(text, source, field)
        return list(self._memo[key])

    def _read(self, text: str, source: str, field: str) -> list[GroundedFinding]:
        cands, rep = extract_candidates(self.llm, text, field, self.max_chars, self.workers, self.prompt_version)
        self.reports.append({"case": self.case_id, "field": field, **{k: v for k, v in asdict(rep).items()
                                                                       if k != "not_found_examples"},
                             "not_found_examples": rep.not_found_examples})
        links = self.linker.link_many([(c.phrase, c.quote) for c in cands])
        pairs = [(c, link) for c, ls in zip(cands, links) for link in ls]
        linked = [(c, self.ont.resolve(link.hpo_id)) for c, link in pairs if link.hpo_id]
        sims = (self.linker.index.similarity([c.quote for c, h in linked if h], [h for c, h in linked if h])
                if self.linker.index else [])
        sim = iter(sims)
        out, seen = [], set()
        for c, link in pairs:
            rec = {"case": self.case_id, "field": field, "phrase": c.phrase, "status": c.status.value, "quote": c.quote,
                   "start": c.start, "end": c.end, "link_method": link.method, "link_score": link.score,
                   "hpo_id": link.hpo_id or "", "label": self.ont.name(link.hpo_id) if link.hpo_id else "",
                   "candidates": "; ".join(f"{t} {self.ont.name(t)} {s}" for t, s in link.candidates[:5])}
            h = self.ont.resolve(link.hpo_id) if link.hpo_id else None
            if h is None:
                self.log.append({**rec, "verdict": "unlinked" if not link.hpo_id else "not_a_phenotype", "kept": False})
                continue
            s = next(sim, None) if self.linker.index else None
            chk = crosscheck(self.ont, self.ext, h, c.quote, s, self.linker.lo if self.linker.index else None)
            grounding = link.grounding
            if chk.verdict == "agree" and RANK[grounding] < RANK["supported"]:
                grounding = "supported"
            if chk.verdict == "disagree" or chk.low_similarity:
                grounding = "weak"
            kept = chk.verdict != "explained_by_other_terms"
            self.log.append({**rec, "hpo_id": h, "label": self.ont.name(h), "verdict": chk.verdict,
                             "rule_terms": "; ".join(f"{t} {self.ont.name(t)}" for t in chk.rule_terms),
                             "similarity": None if s is None else round(s, 4), "low_similarity": chk.low_similarity,
                             "grounding": grounding if kept else "", "kept": kept})
            if not kept or (h, c.status, c.start, c.end) in seen:
                continue
            seen.add((h, c.status, c.start, c.end))
            out.append(GroundedFinding(h, c.status, source, c.quote, quote=c.quote, field=field, start=c.start, end=c.end,
                                       link=(link.method, link.matched), grounding=grounding, reader="llm"))
        return out

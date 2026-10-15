from __future__ import annotations

import base64
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .data import Extractor
from .model import Case, Finding, LRRanker, Ranking, Status, explain

DEFAULT_MODEL = "google/medgemma-1.5-4b-it"
SYSTEM = "You are a careful clinical geneticist. Answer only with the JSON requested."
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class PolicyError(RuntimeError):
    pass


class MedGemma:
    # endpoint=None loads the model in-process; otherwise a local OpenAI-compatible server

    def __init__(self, model: str = DEFAULT_MODEL, endpoint: str | None = None, max_new_tokens: int = 2048):
        self.model, self.endpoint, self.max_new_tokens = model, endpoint, max_new_tokens
        self._pipe = None

    @property
    def is_local(self) -> bool:
        return self.endpoint is None or urlparse(self.endpoint).hostname in LOCAL_HOSTS

    def _load(self):
        import torch
        from transformers import pipeline
        # the 4B models are image-text models; the 27B text model is a plain causal LM
        task = "text-generation" if "text" in self.model else "image-text-to-text"
        self._pipe = pipeline(task, model=self.model, torch_dtype=torch.bfloat16, device_map="auto")

    @property
    def sees_images(self) -> bool:
        return "text" not in self.model      # medgemma-27b-text-it is text-only

    def chat(self, user: str, system: str = SYSTEM, images: tuple[str, ...] = ()) -> str:
        # images: local file paths (image models only)
        if images and not self.sees_images:
            raise ValueError(f"{self.model} is text-only; use google/medgemma-1.5-4b-it for images")
        if self.endpoint:
            content = user
            if images:   # OpenAI-compatible servers (vLLM) take images as base64 data URLs
                content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                            base64.b64encode(Path(p).read_bytes()).decode()}} for p in images]
                content.append({"type": "text", "text": user})
            body = {"model": self.model, "temperature": 0, "max_tokens": self.max_new_tokens,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
            req = urllib.request.Request(self.endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"]
        if self._pipe is None:
            self._load()
        if self._pipe.task == "image-text-to-text":   # message format from the MedGemma model cards
            from PIL import Image
            parts = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in images]
            msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": parts + [{"type": "text", "text": user}]}]
            out = self._pipe(text=msgs, max_new_tokens=self.max_new_tokens, do_sample=False)
        else:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            out = self._pipe(msgs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"]


def check_policy(llm, cases: list[Case], r: LRRanker | None = None) -> None:
    restricted = any(c.meta.get("policy") == "local_only" for c in cases) or (r is not None and r.policy != "open")
    if not llm.is_local and restricted:
        raise PolicyError("credentialed MIMIC data may only go to an in-process model or a localhost server")


def _json(text: str, opener: str):
    # first JSON array or object in a reply; repairs fences, comments, trailing commas, cut-off endings
    closer = "]" if opener == "[" else "}"
    i = text.find(opener)
    if i < 0:
        return None
    body = re.sub(r"```[a-zA-Z]*", "", text[i:])
    body = re.sub(r"(?m)(?<=[,\[{\s])//[^\n]*$", "", body)
    body = re.sub(r",\s*([}\]])", r"\1", body)
    j = body.rfind(closer)
    for cand in ([body[:j + 1]] if j > 0 else []) + _repairs(body):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def _repairs(body: str, tries: int = 40):
    # cut back to a complete value and close any open brackets
    out, cuts = [], ([len(body)] + [m.end() for m in re.finditer(r'[}\]"]', body)][::-1])[:tries]
    for cut in cuts:
        piece, stack, in_str, esc = body[:cut].rstrip().rstrip(","), [], False, False
        for ch in piece:
            if in_str:
                esc, in_str = (not esc and ch == "\\"), in_str and not (ch == '"' and not esc)
                continue
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]" and stack:
                stack.pop()
        piece = piece + '"' if in_str else re.sub(r",\s*$", "", re.sub(r':\s*$', ': null', piece))
        out.append(piece + "".join(reversed(stack)))
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


# 1. extraction
EXTRACT_PROMPT = """Read the clinical note and list every sign, symptom, physical finding, or abnormal
test result mentioned about the patient or their relatives.

For each one output an object:
  "finding": short standard medical term (e.g. "ectopia lentis", "hyperammonemia")
  "status": "present" (patient has it), "absent" (explicitly not present), "family" (a relative has it),
            or "uncertain" (suspected, possible, to rule out)
  "quote": the exact words from the note that support it, copied verbatim

Return only a JSON array of these objects.

Note:
\"\"\"
{note}
\"\"\""""


@dataclass
class ExtractionReport:
    proposed: int = 0
    quote_not_in_note: int = 0
    unmapped: list[str] = field(default_factory=list)


def extract_findings(llm, ext: Extractor, case: Case, note: str) -> tuple[list[Finding], ExtractionReport]:
    check_policy(llm, [case])
    items = _json(llm.chat(EXTRACT_PROMPT.format(note=note)), "[") or []
    rep, out, note_n = ExtractionReport(), {}, _norm(note)
    statuses = {s.value: s for s in Status}
    for it in items:
        if not isinstance(it, dict):
            continue
        rep.proposed += 1
        phrase, quote = str(it.get("finding", "")), str(it.get("quote", ""))
        status = statuses.get(str(it.get("status", "")).lower())
        if not quote or _norm(quote) not in note_n:
            rep.quote_not_in_note += 1
            continue
        hpos = {f.hpo_id for f in ext.findings(phrase)} or {f.hpo_id for f in ext.findings(quote)}
        if not hpos or status is None:
            rep.unmapped.append(phrase)
            continue
        for h in hpos:
            if h not in out:
                out[h] = Finding(h, status, "note:medgemma", quote[:300])
    return list(out.values()), rep


# 2. shortlist re-ranking
RERANK_PROMPT = """A phenotype likelihood model produced this shortlist of diseases (rare and common) for a patient.
Each line of evidence has an id. Reorder the shortlist from most to least likely using ONLY this
evidence and standard medical knowledge about how specific each finding is. Do not add diseases.

{evidence}

Return only JSON, no markdown. Use the disease ids exactly as written above (e.g. "{example}"):
{{"ranking": ["<disease id>", ...],
  "explanations": {{"<disease id>": "1-3 sentences; end every sentence with the evidence ids it uses, e.g. [C2][C5]"}}}}"""

_STOP = set("the and with that this have has for are was were from into than been also which patient "
            "cases case disease diseases shows show consistent supports support suggests".split())


def _content(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3 and w not in _STOP}


@dataclass
class RerankResult:
    ranking: list[str]
    explanations: dict[str, list[str]]     # kept (verified) sentences per disease
    dropped: dict[str, list[str]]          # rejected sentences per disease, with reason
    faithfulness: float                    # kept sentences / all sentences
    raw: str = ""
    parsed: bool = True                    # False when the reply had no usable ranking or explanations


def _to_id(x, shortlist: list[str], names: dict[str, str]) -> str | None:
    # accept "OMIM:618890", "618890", "OMIM 618890" or the disease name
    x = str(x).strip()
    if x in shortlist:
        return x
    num = re.fullmatch(r"(?:[A-Za-z]+[\s:_-]*)?(\d+)", x)
    if num:
        hit = [d for d in shortlist if d.split(":")[-1] == num.group(1)]
        if hit:
            return hit[0]
    low = x.lower()
    hit = [d for d in shortlist if names.get(d, "").lower() == low] or \
          [d for d in shortlist if len(low) >= 6 and (low in names.get(d, "").lower() or names.get(d, "").lower() in low)]
    return hit[0] if hit else None


def rerank(llm, r: LRRanker, case: Case, ranking: Ranking, exclude=frozenset(), k: int = 10) -> RerankResult:
    check_policy(llm, [case], r)   # prompts include common-disease frequencies learned from MIMIC
    claims, lines, owner = {}, [], {}
    for d, p, cls in explain(r, case, ranking, k, exclude):
        common = f" | common: {r.db.prior[d]:.1%} of hospital admissions" if d in r.db.prior else ""
        lines.append(f"\n{d} | {r.db.names[d]} | model posterior {p:.1%}{common}")
        for cl in cls:
            cid = f"C{len(claims) + 1}"
            claims[cid], owner[cid] = cl.text, d
            lines.append(f"  [{cid}] {cl.text}")
    shortlist = [d for d, _, _ in ranking.top(k)]
    raw = llm.chat(RERANK_PROMPT.format(evidence="\n".join(lines), example=shortlist[0] if shortlist else "OMIM:100100"))
    obj = _json(raw, "{")
    obj = obj if isinstance(obj, dict) else {}
    names = r.db.names
    ranked = obj.get("ranking") or []
    ranked = [x.get("id") or x.get("disease") if isinstance(x, dict) else x for x in ranked] if isinstance(ranked, list) else []
    order = [d for d in dict.fromkeys(_to_id(x, shortlist, names) for x in ranked) if d]
    used_order = bool(order)
    order += [d for d in shortlist if d not in order]          # nothing added, nothing lost
    expl = obj.get("explanations") or {}
    if isinstance(expl, list):                                 # [{"id": ..., "text": ...}, ...]
        expl = {x.get("id") or x.get("disease"): x.get("text") or x.get("explanation") for x in expl if isinstance(x, dict)}
    kept, dropped, n_all = {}, {}, 0
    for key, text in (expl.items() if isinstance(expl, dict) else []):
        d = _to_id(key, shortlist, names)
        if isinstance(text, list):
            text = " ".join(map(str, text))
        if not d or not isinstance(text, str):
            continue
        for sent in filter(None, (s.strip() for s in re.split(r"(?<=[.!?\]])\s+(?=[A-Z])", text))):
            n_all += 1
            cited = re.findall(r"\[(C\d+)\]", sent)
            body = _content(re.sub(r"\[C\d+\]", "", sent))
            support = _content(" ".join(claims.get(c, "") + " " + r.db.names[d] for c in cited))
            if not cited:
                why = "no citation"
            elif any(c not in claims or owner[c] != d for c in cited):
                why = "cites evidence that does not exist or belongs to another disease"
            elif body and len(body & support) / len(body) < 0.5:
                why = "cited evidence does not support the sentence"
            else:
                kept.setdefault(d, []).append(sent)
                continue
            dropped.setdefault(d, []).append(f"{sent}  <- {why}")
    n_kept = sum(map(len, kept.values()))
    return RerankResult(order, kept, dropped, n_kept / n_all if n_all else float("nan"), raw,
                        parsed=used_order or n_all > 0)


# 3. image reading (MedPix)
IMAGE_SYSTEM = "You are a careful radiologist. Answer only with the JSON requested."
IMAGE_PROMPT = """This is a {modality} image{plane} of the {location}{patient}.
Describe what you see. Do not state a final diagnosis.

Return only JSON:
{{"report": "2-4 sentences describing the abnormal imaging findings in plain radiology language",
  "findings": [{{"finding": "short standard term, e.g. pleural effusion, hydrocephalus, splenomegaly",
                 "status": "present" or "absent"}}]}}"""


def image_prompt(img: dict) -> str:
    patient = f" of a {img['age']}-year-old {img['sex']}" if img.get("age") and img.get("sex") else ""
    plane = f" ({img['plane']} plane)" if img.get("plane") else ""
    return IMAGE_PROMPT.format(modality=img.get("modality") or "medical", plane=plane,
                               location=(img.get("location") or "body").lower(), patient=patient)


def read_images(llm, ext: Extractor, case: Case, max_images: int = 4, cache: dict | None = None):
    # read each image, map report and findings to HPO; cache maps image id to raw reply
    check_policy(llm, [case])
    cache = {} if cache is None else cache
    statuses = {s.value: s for s in Status}
    best, records = {}, []
    for img in case.meta.get("images", [])[:max_images]:
        if img["id"] not in cache:
            cache[img["id"]] = llm.chat(image_prompt(img), IMAGE_SYSTEM, images=(img["path"],))
        obj = _json(cache[img["id"]], "{") or {}
        report = str(obj.get("report", ""))
        found = [f for f in ext.findings(report, f"image:{img['id']}") if f.status in (Status.PRESENT, Status.ABSENT)]
        for it in obj.get("findings") or []:
            if isinstance(it, dict) and str(it.get("status", "")).lower() in statuses:
                st = statuses[str(it["status"]).lower()]
                if st in (Status.PRESENT, Status.ABSENT):
                    found += [Finding(f.hpo_id, st, f"image:{img['id']}", str(it.get("finding", ""))[:200])
                              for f in ext.findings(str(it.get("finding", "")))]
        for f in found:
            if f.hpo_id not in best or (f.status == Status.PRESENT and best[f.hpo_id].status != Status.PRESENT):
                best[f.hpo_id] = f
        records.append({"image": img["id"], "report": report, "hpo": sorted({f.hpo_id for f in found}),
                        "parsed": bool(obj)})
    return list(best.values()), records
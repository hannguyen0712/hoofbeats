from __future__ import annotations

import base64
import hashlib
import json
import re
import threading
import time
import urllib.error
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


class QuotaExhausted(Exception):
    # not a RuntimeError on purpose: benchmarks stop instead of skipping to the next case
    pass


def quota_info(body: bytes) -> dict:
    # read Google-style 429 details: which quota, its limit, and the suggested wait
    try:
        obj = json.loads(body)
    except (ValueError, TypeError):
        return {"message": body[:300].decode(errors="replace") if isinstance(body, bytes) else ""}
    err = (obj[0] if isinstance(obj, list) and obj else obj).get("error", {}) if obj else {}
    out = {"message": str(err.get("message", ""))[:500], "quotas": [], "retry": None}
    for d in err.get("details", []) or []:
        for v in d.get("violations", []) or []:
            out["quotas"].append((str(v.get("quotaId", "")), str(v.get("quotaValue", ""))))
        if d.get("retryDelay"):
            m = re.match(r"([\d.]+)s", str(d["retryDelay"]))
            out["retry"] = float(m.group(1)) if m else None
    return out


class MedGemma:
    # any chat model: endpoint=None loads it in-process; otherwise any OpenAI-compatible server (vLLM, Ollama, Gemini)

    def __init__(self, model: str = DEFAULT_MODEL, endpoint: str | None = None, max_new_tokens: int = 2048,
                 api_key: str | None = None, vision: bool | None = None, extra: dict | None = None,
                 rpm: float | None = None, cache_dir: str | Path | None = None, dry_run: bool = False,
                 offline: bool = False):
        self.model, self.endpoint, self.max_new_tokens = model, endpoint, max_new_tokens
        self.api_key, self.vision, self.extra = api_key, vision, extra or {}
        self.rpm, self._last = rpm, 0.0                 # rpm: max requests per minute (free tiers)
        self.dry_run = dry_run                          # count requests that would be sent, send none
        self.offline = offline                          # answer from the cache only; a miss is counted and gives ""
        self.n_calls, self.seconds, self.errors = 0, 0.0, 0
        self.requests, self.cache_hits, self.retries, self.planned, self.missed = 0, 0, 0, 0, 0
        self._pipe, self._cache, self._cache_file = None, {}, None
        self._lock = threading.Lock()                   # chat() may run in several threads (chat_many)
        if cache_dir:                                   # replies keyed by a hash of everything sent
            self._cache_file = Path(cache_dir) / f"{self.slug}.jsonl"
            if self._cache_file.exists():
                for line in self._cache_file.read_text().splitlines():
                    try:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec["reply"]
                    except (ValueError, KeyError):
                        continue

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
        if self.vision is not None:
            return self.vision
        return "text" not in self.model      # medgemma-27b-text-it is text-only

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9.]+", "-", self.model.split("/")[-1]).strip("-").lower()

    def _key(self, user, system, images, schema=None) -> str:
        imgs = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in images]
        parts = [self.model, self.max_new_tokens, self.extra, system, user, imgs] + ([schema] if schema else [])
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def chat(self, user: str, system: str = SYSTEM, images: tuple[str, ...] = (), schema: dict | None = None) -> str:
        # images: local file paths (image models only); schema: a JSON schema the server must follow (guided decoding;
        # an in-process model ignores it); identical requests are answered from the cache
        if images and not self.sees_images:
            raise ValueError(f"{self.model} is text-only; pick a vision model for images")
        key = self._key(user, system, images, schema)
        with self._lock:
            if key in self._cache:
                self.cache_hits += 1
                return self._cache[key]
            if self.dry_run or self.offline:
                self.planned += self.dry_run
                self.missed += self.offline
                return ""
        t0 = time.time()
        try:
            reply = (self._endpoint_chat(user, system, images, **({"schema": schema} if schema else {})) if self.endpoint
                     else self._local_chat(user, system, images))
        finally:
            with self._lock:
                self.n_calls, self.seconds = self.n_calls + 1, self.seconds + time.time() - t0
        with self._lock:
            self._cache[key] = reply
            if self._cache_file:
                self._cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self._cache_file, "a") as fh:
                    fh.write(json.dumps({"key": key, "reply": reply}) + "\n")
        return reply

    def usage(self) -> dict:
        return {"requests": self.requests, "retries": self.retries, "cache_hits": self.cache_hits,
                "would_send": self.planned, "offline_misses": self.missed,
                "sec_per_request": round(self.seconds / max(self.n_calls, 1), 2)}

    def _endpoint_chat(self, user, system, images, schema=None) -> str:
        content = user
        if images:   # OpenAI-compatible servers take images as base64 data URLs
            content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                        base64.b64encode(Path(p).read_bytes()).decode()}} for p in images]
            content.append({"type": "text", "text": user})
        body = {"model": self.model, "temperature": 0, "max_tokens": self.max_new_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}], **self.extra}
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": True}}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(6):   # free tiers rate-limit: back off on 429/5xx
            if self.rpm:            # pace requests so the per-minute limit is never hit
                gap = 60.0 / self.rpm - (time.time() - self._last)
                if gap > 0:
                    time.sleep(gap)
            self._last = time.time()
            with self._lock:
                self.requests += 1
                self.retries += attempt > 0
            req = urllib.request.Request(self.endpoint, json.dumps(body).encode(), headers)
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    return json.loads(resp.read())["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as e:
                raw = e.read()
                info = quota_info(raw) if e.code == 429 else {}
                daily = [(q, v) for q, v in info.get("quotas", []) if "perday" in q.lower()]
                if daily:           # waiting minutes will not help: stop the run
                    self.errors += 1
                    lim = f" (limit {daily[0][1]} requests per day)" if daily[0][1] else ""
                    raise QuotaExhausted(f"{self.model}: daily free-tier quota used up{lim}. Try again after the "
                                         f"daily reset, use another model, or check https://ai.dev/rate-limit") from None
                if e.code not in (429, 500, 502, 503, 504) or attempt == 5:
                    self.errors += 1
                    msg = info.get("message") or raw[:300].decode(errors="replace")
                    if e.code == 429:
                        raise QuotaExhausted(f"{self.model}: still rate-limited after 6 tries: {msg}") from None
                    raise RuntimeError(f"{self.model}: HTTP {e.code} {msg}") from None
                wait = min(90, info.get("retry") or 5 * 2 ** attempt)
                print(f"\n[llm] HTTP {e.code} ({'rate limit' if e.code == 429 else 'server busy'}), "
                      f"retrying in {wait}s (attempt {attempt + 1}/6)", flush=True)
                time.sleep(wait)

    def _local_chat(self, user, system, images) -> str:
        self.requests += 1
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


def require_local(llm) -> None:
    # finding extraction and HPO linking send patient text: in-process or localhost servers only, whatever the dataset
    if not llm.is_local:
        raise PolicyError(f"{getattr(llm, 'endpoint', '')}: extraction and linking only use an in-process model or a "
                          "localhost server")


def chat_many(llm, calls: list[dict], workers: int = 8) -> list[str]:
    # calls: chat() keyword arguments; replies in the same order. Several requests in flight let a vLLM server
    # batch them; a client without threads (a test fake) is fine too
    if workers <= 1 or len(calls) <= 1:
        return [llm.chat(**c) for c in calls]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(lambda c: llm.chat(**c), calls))


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
  "explanations": {{"<disease id>": "1-3 sentences; end every sentence with the evidence ids it uses, e.g. [C2][L1]"}}}}"""

_STOP = set("the and with that this have has for are was were from into than been also which patient "
            "cases case disease diseases shows show consistent supports support suggests "
            # reasoning words that make no factual claim of their own (claims state LRs as numbers)
            "likelihood ratio ratios very high highly strong strongly extremely features feature finding findings "
            "matches matching match key hallmark classic typical presenting presentation exhibits evidence".split())


_CITE_GROUP = re.compile(r"\[[^\]]*\b[CL]\d+\b[^\]]*\]")


def _cites(sent: str) -> list[str]:
    # evidence ids in [C2][L1] or [C2, C5] or [C2; L1] form
    return [x for grp in _CITE_GROUP.findall(sent) for x in re.findall(r"\b([CL]\d+)\b", grp)]


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


def rerank(llm, r: LRRanker, case: Case, ranking: Ranking, exclude=frozenset(), k: int = 10,
           literature: dict | None = None) -> RerankResult:
    # literature: disease id -> verified PubMed claims (pubmed.literature_claims), cited as [L#]
    check_policy(llm, [case], r)   # prompts include common-disease frequencies learned from MIMIC
    claims, lines, owner, n_lit = {}, [], {}, 0
    for d, p, cls in explain(r, case, ranking, k, exclude):
        common = f" | common: {r.db.prior[d]:.1%} of hospital admissions" if d in r.db.prior else ""
        lines.append(f"\n{d} | {r.db.names[d]} | model posterior {p:.1%}{common}")
        for cl in cls:
            cid = f"C{sum(c.startswith('C') for c in claims) + 1}"
            claims[cid], owner[cid] = cl.text, d
            lines.append(f"  [{cid}] {cl.text}")
        for lc in (literature or {}).get(d, []):
            n_lit += 1
            claims[f"L{n_lit}"], owner[f"L{n_lit}"] = lc.text, d
            lines.append(f"  [L{n_lit}] {lc.text}")
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
            cited = _cites(sent)
            body = _content(_CITE_GROUP.sub("", sent))
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


# 2b. model-only baseline: diagnose from the findings with no ranker and no evidence
DIRECT_PROMPT = """A patient has these clinical findings (HPO terms).

Present: {present}
Explicitly absent: {absent}

List the {k} most likely diagnoses, most likely first. Prefer specific named diseases (with the OMIM
number if you know it) over broad categories.

Return only JSON, no markdown:
{{"diagnoses": [{{"name": "<disease name>", "omim": "<OMIM:number or empty>"}}, ...]}}"""


def direct_diagnose(llm, r: LRRanker, case: Case, matcher, k: int = 10) -> tuple[list[set[str]], bool, str]:
    # each answer slot -> knowledge-base ids, by OMIM number, else by name (a name like "Noonan syndrome"
    # matches all its numbered subtypes); unmatched answers keep their slot as an empty set
    check_policy(llm, [case], r)
    names = lambda st: ", ".join(sorted({r.ont.name(t) for t in case.terms(st) if r.ont.resolve(t)})) or "none"
    raw = llm.chat(DIRECT_PROMPT.format(present=names(Status.PRESENT), absent=names(Status.ABSENT), k=k))
    obj = _json(raw, "{")
    items = obj.get("diagnoses") if isinstance(obj, dict) else None
    out = []
    for it in (items if isinstance(items, list) else [])[:k]:
        name, num = (it.get("name", ""), str(it.get("omim", ""))) if isinstance(it, dict) else (str(it), "")
        m = re.search(r"(\d+)", num)
        ids = {f"OMIM:{m.group(1)}"} if m and f"OMIM:{m.group(1)}" in r.db.names else set()
        out.append(ids | set(matcher.match(str(name))))
    return out, isinstance(items, list), raw


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


MULTI_PROMPT = """These are {n} images from one patient{patient}:
{listing}
Describe what you see across them. Do not state a final diagnosis.

Return only JSON:
{{"report": "2-4 sentences describing the abnormal imaging findings in plain radiology language",
  "findings": [{{"finding": "short standard term, e.g. pleural effusion, hydrocephalus, splenomegaly",
                 "status": "present" or "absent"}}]}}"""


def multi_prompt(imgs: list[dict]) -> str:
    first = imgs[0]
    patient = f", a {first['age']}-year-old {first['sex']}" if first.get("age") and first.get("sex") else ""
    listing = "\n".join(f"{i}. {img.get('modality') or 'medical'} image" + (f" ({img['plane']} plane)" if img.get("plane") else "")
                        + f" of the {(img.get('location') or 'body').lower()}" for i, img in enumerate(imgs, 1))
    return MULTI_PROMPT.format(n=len(imgs), patient=patient, listing=listing)


def read_images(llm, ext: Extractor, case: Case, max_images: int = 4, cache: dict | None = None, per_call: int = 1):
    # read the case's images, per_call images per request; map report and findings to HPO
    # cache maps an image id (or ids joined by "+") to the raw reply
    check_policy(llm, [case])
    cache = {} if cache is None else cache
    statuses = {s.value: s for s in Status}
    best, records = {}, []
    imgs = case.meta.get("images", [])[:max_images]
    groups = [imgs[i:i + max(per_call, 1)] for i in range(0, len(imgs), max(per_call, 1))]
    for grp in groups:
        gid = "+".join(img["id"] for img in grp)
        reply = cache.get(gid)
        if reply is None:
            prompt = image_prompt(grp[0]) if len(grp) == 1 else multi_prompt(grp)
            reply = llm.chat(prompt, IMAGE_SYSTEM, images=tuple(img["path"] for img in grp))
            if not getattr(llm, "dry_run", False):      # a dry run must not leave empty replies in the cache
                cache[gid] = reply
        img = {"id": gid}
        obj = _json(reply, "{") or {}
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
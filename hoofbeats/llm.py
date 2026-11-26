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


def image_type(path) -> str:
    # media type from the file's first bytes, so JPEG/WebP inputs are not sent labelled as PNG
    head = Path(path).read_bytes()[:12]
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[:4] == b"GIF8":
        return "image/gif"
    return "image/png"


# --json-mode asks the server for JSON only (response_format json_object). Ollama and OpenAI support it. Gemini's
# OpenAI-compatible endpoint documents only schema-based JSON, so it is not sent there; Gemini answers in JSON from
# the prompt alone. Add a host here if another server rejects it.
NO_JSON_MODE = ("generativelanguage.googleapis.com",)


def with_json_mode(json_mode: bool, endpoint, extra, model: str):
    # extra request fields, plus response_format when --json-mode is on and the server supports it
    if not json_mode:
        return extra
    if urlparse(endpoint or "").hostname in NO_JSON_MODE:
        print(f"[llm] {model}: --json-mode not sent (this server does not document it); the prompt asks for JSON")
        return extra
    return {**(extra or {}), "response_format": {"type": "json_object"}}


class MedGemma:
    # any chat model: endpoint=None loads it in-process; otherwise any OpenAI-compatible server (vLLM, Ollama, Gemini)

    def __init__(self, model: str = DEFAULT_MODEL, endpoint: str | None = None, max_new_tokens: int = 2048,
                 api_key: str | None = None, vision: bool | None = None, extra: dict | None = None,
                 rpm: float | None = None, cache_dir: str | Path | None = None, dry_run: bool = False,
                 timeout: float = 600, offline: bool = False):
        self.model, self.endpoint, self.max_new_tokens = model, endpoint, max_new_tokens
        self.timeout = timeout                          # seconds to wait for one reply (CPU models can need more)
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
            content = [{"type": "image_url", "image_url": {"url": f"data:{image_type(p)};base64," +
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
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return reply_text(json.loads(resp.read()))
            except TimeoutError:        # a failed request (benchmarks skip the case), not a crash
                self.errors += 1
                raise RuntimeError(f"{self.model}: no reply within {self.timeout:.0f} s. A slow CPU or a reply stuck "
                                   f"repeating itself: raise --llm-timeout or lower --llm-max-tokens") from None
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
            except ConnectionError as e:          # the server dropped the connection mid-reply (e.g. it crashed)
                self.errors += 1
                raise RuntimeError(f"{self.model}: the connection to {self.endpoint} was lost ({e}). If the server "
                                   "crashed (e.g. out of memory), restart it and run again") from None
            except urllib.error.URLError as e:     # not reachable: no server, or no reply while connecting
                self.errors += 1
                why = "timed out" if isinstance(e.reason, TimeoutError) else str(e.reason)
                raise RuntimeError(f"{self.model}: cannot reach {self.endpoint} ({why}). Is the server (e.g. Ollama) "
                                   "running?") from None

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


def reply_text(resp: dict) -> str:
    # the answer text of an OpenAI-style reply. Thinking models served by Ollama/vLLM put their thinking in a
    # separate field ("reasoning" / "reasoning_content") and can leave "content" empty when they stop while still
    # thinking; then the thinking is the only text there is, and the JSON parser looks for an answer in it.
    # <think>...</think> blocks inside content are dropped.
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    text = re.sub(r"(?s)<think>.*?</think>", "", msg.get("content") or "").strip()
    return text or (msg.get("reasoning") or msg.get("reasoning_content") or "")


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


# 4. image context (Stage A) -> Stage 1
# Stage A: a vision model writes a detailed image context for each image (findings, not the disease behind them).
# The text goes into the case's image_context field, where the Stage 1 readers (rules, numbers, LLM reader) turn it
# into HPO findings like any note; those findings are grounded, fused and scored by the ranker with everything else.
# The images stay on the case as records (modality, plane, location). No model ranks diagnoses from the images.
# The prompt was written after small models were seen to (a) confuse left and right, (b) call an enlarged ventricle a
# mass, (c) deny "hydrocephalus" while describing enlarged ventricles because they were told not to diagnose.
# Stage A prompts. Radiology (most images, and all of MedPix) keeps the radiologist prompt it was evaluated with;
# every other kind of image gets a specialist of its own (KIND_ROLES) and its own parts and rules (IMAGE_KINDS).
RADIOLOGY_SYSTEM = """You are an experienced, board-certified radiologist. You write the findings section of a report for a \
clinician who cannot see the images. Software then turns each of your findings into a standard phenotype term (Human \
Phenotype Ontology), so exact wording matters as much as accurate looking.

How to look:
- Examine the whole image before writing: every structure in the field of view, both sides, and the edges of the image.
- Compare left with right, and judge each size against what is normal for the patient's age and sex.
- Tell normal anatomy, normal variants and artefacts (motion, positioning, windowing, beam hardening) apart from disease.
- Use the clinical question only to decide where to look most carefully. Report what the image shows, even if it does \
not fit the question, and never report something because the question expects it.
- If the image is limited (cropped, low resolution, a single slice, an unknown sequence), say so and lower your \
confidence. Never describe what lies outside the image.

How to write:
- Describe what you see, not the disease behind it: never name a disease, syndrome or diagnosis.
- Use standard radiological and medical terms (for example ventriculomegaly, hydronephrosis, pleural effusion, ectopia \
lentis, hepatomegaly, cerebral calcification), one finding per item, each with its location, side and size.
- Give sizes in millimetres or centimetres only when the image lets you estimate them; never invent a measurement.
- State absent findings plainly as "No ..." (for example "No hydrocephalus", "No pleural effusion").
- Put your certainty in the confidence field (definite, probable or possible), not in words such as "may represent".
- If nothing is abnormal, say so: an empty findings list with pertinent negatives is a valid answer.

Answer only with the JSON requested: no text before or after it."""
RADIOLOGY_PROMPT = """{intro}{patient}{question}

Write the findings section of a radiology report for {it}. Go through {it} systematically:
- technique: modality, view or plane, contrast phase, image quality and limitations
- anatomy: the body region and every structure visible
- each structure: size, shape, left-right symmetry, density / signal / echogenicity{checklist}
- findings: every abnormality, with location, side, approximate size, margins and effect on nearby structures,
  and how sure you are (definite, probable or possible)
- pertinent negatives: important structures that look normal, or signs that are absent
- summary: 2-4 sentences on the key abnormal features

Rules:
- Use the patient's left and right. On axial and coronal images the patient's right is on the left of the image.
- Before calling something a mass or lesion, check whether it is a normal structure that is enlarged or displaced
  (a ventricle, vessel, cyst, bowel loop, bladder or fluid collection). Fluid is dark on CT and bright on T2 MRI.
- Name imaging findings and patterns when you see them (for example pleural effusion, pneumothorax,
  hydronephrosis, ventricular enlargement, mass effect), but not the disease or cause behind them.
- Report only what you can see. Your pertinent negatives must not contradict your findings.
- Reply with the JSON only: no plan, checklist or notes before or after it.

Return only JSON:
{{"technique": "...", "anatomy": "...",
  "findings": [{{"finding": "...", "location": "...", "details": "...", "confidence": "definite|probable|possible"}}],
  "pertinent_negatives": ["..."],
  "summary": "..."}}"""

KIND_SYSTEM = """You are {role}. You write the findings section of a report for a clinician who cannot see the image. \
Software then turns each of your findings into a standard phenotype term (Human Phenotype Ontology), so exact wording \
matters as much as accurate looking.

How to look:
{first}- Examine the whole image before writing: everything in the field of view or on the tracing, both sides, and the edges.
- Compare left with right where there are two sides, and judge every size, rate or interval against what is normal \
for the patient's age and sex.
- Tell normal anatomy, normal variants and artefacts (motion, positioning, lighting, windowing, lead or electrode \
problems, staining or cutting artefacts) apart from disease.
- Use the clinical question only to decide where to look most carefully. Report what the image shows, even if it does \
not fit the question, and never report something because the question expects it.
- If the image is limited (cropped, low resolution, a single slice or lead, unknown settings), say so and lower your \
confidence. Never describe what lies outside the image.

How to write:
- Describe what you see, not the disease behind it: never name a disease, syndrome or diagnosis.
- Use the standard medical terms of that field, one finding per item, each with its location (side, structure, lead \
or body part) and its size or value: for example ventriculomegaly, hydronephrosis, pleural effusion, prolonged QT \
interval, left axis deviation, ST segment elevation, hypertelorism, cafe-au-lait spots, arachnodactyly, retinal \
hemorrhage, ectopia lentis, cerebral calcification.
- Give sizes, rates and intervals only when the image lets you measure or estimate them (from a scale, the ECG grid or \
known anatomy); never invent a number.
- State absent findings plainly as "No ..." (for example "No hydrocephalus", "No ST segment elevation").
- Put your certainty in the confidence field (definite, probable or possible), not in words such as "may represent".
- If nothing is abnormal, say so: an empty findings list with pertinent negatives is a valid answer.

Answer only with the JSON requested: no text before or after it."""
KIND_PROMPT = """{intro}{patient}{question}

Write the findings section of a report for {it}. Go through {it} systematically:
- technique: what kind of image it is, the view, plane, leads or stain, settings (contrast phase, paper speed and
  gain, magnification), image quality and limitations
- anatomy: what the image covers (body region and structures, or the leads and length of a tracing)
- each part: {parts}{checklist}
- findings: every abnormality, with location and side, size or value, appearance, and how sure you are (definite,
  probable or possible)
- pertinent negatives: important structures or features that look normal, or signs that are absent
- summary: 2-4 sentences on the key abnormal features

Rules:{rules}
- Name findings and patterns when you see them, but not the disease or cause behind them.
- Report only what you can see. Your pertinent negatives must not contradict your findings.
- Reply with the JSON only: no plan, checklist or notes before or after it.

Return only JSON:
{{"technique": "...", "anatomy": "...",
  "findings": [{{"finding": "...", "location": "...", "details": "...", "confidence": "definite|probable|possible"}}],
  "pertinent_negatives": ["..."],
  "summary": "..."}}"""

KIND_ROLES = {
    "ecg": "an experienced, board-certified cardiologist who reads ECGs and rhythm strips",
    "eeg": "an experienced, board-certified clinical neurophysiologist who reads EEGs",
    "eye": "an experienced, board-certified ophthalmologist who reads fundus photographs, slit-lamp images and OCT",
    "microscopy": "an experienced, board-certified pathologist who reads histology, cytology and blood smears",
    "endoscopy": "an experienced, board-certified endoscopist",
    "photo": ("an experienced, board-certified clinical geneticist and dermatologist who describes clinical "
              "photographs of the face, skin, hands, feet and body"),
    "general": ("an experienced, board-certified physician who reads every kind of medical image: radiology, ECG "
                "and EEG tracings, clinical photographs, eye images, microscopy and endoscopy"),
}
CONTEXT_SYSTEM = RADIOLOGY_SYSTEM      # the default (radiology) system prompt


def context_system(kind: str) -> str:
    if kind == "radiology":
        return RADIOLOGY_SYSTEM
    first = ("- First decide what kind of image this is, and read it the way a specialist in that kind of image "
             "would.\n") if kind == "general" else ""
    return KIND_SYSTEM.replace("{role}", KIND_ROLES.get(kind, KIND_ROLES["general"])).replace("{first}", first)


# Images whose labels name no kind (image_kind -> "general"): the vision model is asked what the image is first
CLASSIFY_PROMPT = """What kind of medical image is this? Answer with one word from this list:
radiology (X-ray, CT, MRI, ultrasound, echocardiography, nuclear medicine), ecg (ECG or rhythm strip), eeg,
eye (fundus photograph, slit lamp, OCT), microscopy (histology, cytology, blood smear), endoscopy,
photo (a photograph of a person's face, skin, hands, feet or body), other.
Reply with the one word only."""


def classify_image(llm, grp: list[dict]) -> str:
    raw = llm.chat(CLASSIFY_PROMPT, "", images=tuple(g["path"] for g in grp))
    words = re.findall(r"[a-z]+", str(raw or "").lower())
    kinds = [k for k, *_ in IMAGE_KINDS]
    return next((w for w in words if w in kinds), "general")


# Kinds of image (matched on the image's modality and location; first match wins). Each kind has what to describe
# for every part of the image and its own reading rules; radiology also gets a checklist by body region (CHECKLISTS).
IMAGE_KINDS = (
    ("ecg", r"\b(ecg|ekg|electrocardio\w*|rhythm strip|holter)\b",
     "rate, rhythm, P waves, PR interval, QRS duration and axis, ST segments, T waves, QT and QTc interval, voltage",
     ("Read the calibration first (normally 25 mm/s and 10 mm/mV): one small square is 40 ms and 0.1 mV. Measure "
      "the rate and every interval from the grid and give them in ms or beats per minute.",
      "Go through the leads in groups (I, aVL, V5-V6; II, III, aVF; V1-V4; aVR) and name the leads of every "
      "change.")),
    ("eeg", r"\b(eeg|electroencephalo\w*)\b",
     "background rhythm and frequency, symmetry, reactivity, sleep features, focal or generalized slowing, "
     "epileptiform discharges (spikes, sharp waves, spike-wave), their location and frequency",
     ("Read the montage, time scale and sensitivity first, and name the electrodes or regions of every change.",)),
    ("eye", r"\b(fundus|fundoscop\w*|retina\w*|slit[- ]?lamp|optical coherence|oct|ophthalm\w*|eye photo\w*)\b",
     "optic disc (size, cup-to-disc ratio, colour, margins), vessels, macula and fovea, retina in each quadrant; on "
     "slit lamp: lids, conjunctiva, cornea, iris, pupil, lens position and clarity",
     ("Say which eye it is if the image shows it.",)),
    ("microscopy", r"\b(histo\w*|patholog\w*|biops\w*|microscop\w*|smear|cytolog\w*|h&e|stain\w*|marrow)\b",
     "stain and magnification, tissue or cell type, architecture, cell size and shape, nuclei, inclusions or "
     "deposits, inflammation, abnormal cells (for a gross specimen: size, colour, surface and cut surface, lesions)",
     ("Describe cells and tissue as a pathologist would; give sizes only against a scale bar or known cell size.",)),
    ("endoscopy", r"\b(endo|colono|gastro|broncho|laryngo|cysto|arthro|naso|esophagogastroduodeno)scop\w*|capsule",
     "the segment shown, mucosa (colour, pattern, vessels), lumen, folds, lesions (size, shape, surface), bleeding",
     ("Name the segment if the image shows it.",)),
    ("photo", r"\b(photo\w*|clinical image|clinical picture|dermatolog\w*|skin|facial|face|dysmorph\w*|"
              r"whole body|hand photo|foot photo|nail\w*|hair)\b",
     "for the face: head shape, forehead, eyes and their spacing, nose, philtrum, lips, mouth, chin, ears; for skin: "
     "each lesion's type, colour, size, number, shape, border and distribution; for limbs: proportions, fingers and "
     "toes, joints, nails",
     ("Use the standard terms of dysmorphology (Elements of Morphology) and dermatology (macule, papule, plaque, "
      "vesicle; cafe-au-lait spot, hypopigmentation).",
      "Describe what the photograph shows, not the person's identity, and only the features you can see.")),
    ("radiology", r"\b(ct|mri?|mr[a-z]?|x-?ray|xr|radiograph\w*|ultraso\w*|us|sonogra\w*|echo\w*|doppler|mammo\w*|"
                  r"nuclear|pet|spect|angiogra\w*|fluoro\w*|ugi|upper gi|barium|contrast study|dexa|t1|t2|flair|dwi|"
                  r"scan)\b",
     "size, shape, left-right symmetry, density / signal / echogenicity",
     ("Use the patient's left and right. On axial and coronal images the patient's right is on the left of the image.",
      "Before calling something a mass or lesion, check whether it is a normal structure that is enlarged or "
      "displaced (a ventricle, vessel, cyst, bowel loop, bladder or fluid collection). Fluid is dark on CT and "
      "bright on T2 MRI.")),
)
GENERAL_PARTS = "size, shape, colour or density, symmetry, and anything that differs from normal"


def image_kind(imgs: list[dict]) -> str:
    # "ecg", "eeg", "eye", "microscopy", "endoscopy", "photo", "radiology" or "general" (nothing matched). The modality
    # decides first (a CT of the orbit is radiology, whatever its location says); the location only if it does not
    for text in (" ".join(str(img.get("modality") or "") for img in imgs),
                 " ".join(f"{img.get('modality') or ''} {img.get('location') or ''}" for img in imgs)):
        kind = next((k for k, pat, *_ in IMAGE_KINDS if re.search(pat, text.lower())), None)
        if kind:
            return kind
    return "general"


def details_of(case: Case) -> dict:
    # patient details: meta["details"] plus age and sex from the image records (MedPix keeps them there)
    d = dict(case.meta.get("details") or {})
    for img in case.meta.get("images", []):
        for key in ("age", "sex"):
            if img.get(key) and not d.get(key):
                d[key] = str(img[key]).strip()
    return {k: v for k, v in d.items() if v not in (None, "", [])}


def details_text(d: dict) -> str:
    # {"age": 73, "sex": "male", "setting": "ED"} -> "73-year-old male. Setting: ED."
    d = dict(d)
    age, sex = d.pop("age", None), str(d.pop("sex", "") or "").lower()
    head = f"{age}-year-old {sex}." if age and sex else f"Age {age}." if age else f"Sex: {sex}." if sex else ""
    return " ".join([head] + [f"{str(k).replace('_', ' ').capitalize()}: {v}." for k, v in d.items()]).strip()


def _img_label(img: dict) -> str:
    plane = f"{img['plane'].lower()} plane" if img.get("plane") else ""
    return ", ".join(x for x in (img.get("modality") or "image", plane, (img.get("location") or "").lower()) if x)


# structures a small model should not skip, by body region (matched on the image's location and modality)
CHECKLISTS = (
    (("brain", "neuro", "head ct", "skull", "cranial"), "lateral ventricles (each side), third and fourth ventricles, "
     "midline, sulci and cisterns, grey-white matter, extra-axial spaces, posterior fossa"),
    (("orbit", "eye"), "globes, lenses, optic nerves, extraocular muscles, orbital fat, bony walls"),
    (("head and neck", "neck", "sinus", "pharyn", "larynx"), "airway, pharynx and larynx, salivary glands, thyroid, "
     "lymph nodes by level, vessels, bones"),
    (("chest", "lung", "pulmonary", "thorax", "cardio", "heart"), "lungs (each side), airways, pleura, heart size, "
     "mediastinum and hila, diaphragm, bones and soft tissues"),
    (("breast", "mammo"), "skin, parenchyma, masses, calcifications, nipple, axilla"),
    (("abdomen", "pelvi", "gastro", "liver", "kidney", "renal", "genitourinary", "bladder", "bowel"),
     "liver, gallbladder and bile ducts, pancreas, spleen, kidneys and ureters, bladder, bowel, free fluid or air, "
     "vessels, lymph nodes, bones"),
    (("spine", "spinal", "vertebra"), "alignment, vertebral bodies, discs, spinal canal and cord, neural foramina, "
     "paraspinal soft tissues"),
    (("musculoskeletal", "bone", "joint", "knee", "hip", "shoulder", "hand", "foot", "extremit"),
     "bones and cortex, joints and alignment, soft tissues, any fracture, lesion or periosteal reaction"),
)


def _checklist(imgs: list[dict]) -> str:
    # body-region checklists are for radiology; the other kinds list their parts in IMAGE_KINDS
    if image_kind(imgs) not in ("radiology", "general"):
        return ""
    where = " ".join(f"{img.get('location') or ''} {img.get('modality') or ''}" for img in imgs).lower()
    found = [items for keys, items in CHECKLISTS if any(k in where for k in keys)]
    return "".join(f"\n- check in particular: {items}" for items in dict.fromkeys(found))


def context_prompt(imgs: list[dict], details: dict, clinical: str = "", kind: str | None = None) -> str:
    # the patient details and the clinical question (the history) are given, as a specialist would get them.
    # kind: from image_kind (or classify_image); radiology gets the radiologist prompt, the others KIND_PROMPT
    if len(imgs) == 1:
        intro, it = f"This is one medical image: {_img_label(imgs[0])}.", "this image"
    else:
        listing = "\n".join(f"{i}. {_img_label(img)}" for i, img in enumerate(imgs, 1))
        intro, it = f"These are {len(imgs)} images from one patient:\n{listing}", "these images"
    pt = details_text(details)
    question = " ".join(clinical.split())[:1500]
    kind = kind or image_kind(imgs)
    patient, question = f"\nPatient: {pt}" if pt else "", f"\nClinical question: {question}" if question else ""
    if kind == "radiology":
        return RADIOLOGY_PROMPT.format(intro=intro, patient=patient, it=it, question=question,
                                       checklist=_checklist(imgs))
    spec = next((k for k in IMAGE_KINDS if k[0] == kind), None)
    parts, rules = (spec[2], spec[3]) if spec else (GENERAL_PARTS, ())
    return KIND_PROMPT.format(intro=intro, patient=patient, it=it, question=question, parts=parts,
                              checklist=_checklist(imgs) if kind == "general" else "",
                              rules="".join(f"\n- {r}" for r in rules))


def render_context(obj, raw: str, label: str, lean: bool = False) -> str:
    # the context JSON as readable text; the raw reply if it was not JSON. lean (--context-read findings): only the
    # findings list (finding, location, confidence) and the pertinent negatives. The details, summary, technique and
    # anatomy are left out: there the model hedges ("may indicate edema"), names interpretations of a finding it already
    # listed ("suggesting hydrocephalus" for enlarged ventricles) and repeats the history, and Stage 1 would read each
    # of those as one more finding.
    if not isinstance(obj, dict):
        return f"{label}: {raw.strip()[:3000]}" if raw.strip() else ""
    lines = [f"{label}:"]
    for key in () if lean else ("technique", "anatomy"):
        if str(obj.get(key) or "").strip():
            lines.append(f"{key.capitalize()}: {str(obj[key]).strip()}")
    found = obj.get("findings") or []
    if isinstance(found, list) and found:
        lines.append("Findings:")
        for f in found:
            if isinstance(f, dict) and str(f.get("finding") or "").strip():
                loc, det = str(f.get("location") or "").strip(), str(f.get("details") or "").strip()
                conf = str(f.get("confidence") or "").strip().lower()
                lines.append(f"- {str(f['finding']).strip()}" + (f" ({loc})" if loc else "")
                             + (f" [{conf}]" if conf else "") + (f": {det}" if det and not lean else ""))
            elif isinstance(f, str) and f.strip():
                lines.append(f"- {f.strip()}")
    neg = obj.get("pertinent_negatives") or []
    neg = [neg] if isinstance(neg, str) else neg if isinstance(neg, list) else []
    if any(str(x).strip() for x in neg):
        lines.append("Pertinent negatives: " + "; ".join(str(x).strip() for x in neg if str(x).strip()))
    if not lean and str(obj.get("summary") or "").strip():
        lines.append(f"Summary: {str(obj['summary']).strip()}")
    return "\n".join(lines) if len(lines) > 1 else (f"{label}: {raw.strip()[:3000]}" if raw.strip() else "")


@dataclass
class ImageContext:
    images: list[str]          # image ids described together
    label: str                 # "Image 1 (CT - noncontrast, axial plane, head)"
    text: str                  # what Stage 1 reads (the checked description when the check is on)
    parsed: bool
    model: str = ""
    cut_off: bool = False      # the reply stopped before the JSON was finished
    unchecked: str = ""        # the description before the check (same as text when the check is off)
    checked: int = 0           # findings the model was asked to confirm
    kept: int = 0              # findings it confirmed (answered yes)
    definite: str = ""         # the description with only the findings the model called definite
    kind: str = ""             # the kind of image (image_kind, or classify_image when the labels name none)


# --context-confidence: keep only findings the describing model rated at least this sure (its "confidence" field;
# a finding without one counts as "possible"). The summary is dropped when a finding is, since it restates them.
CONFIDENCE = {"possible": 1, "probable": 2, "definite": 3}


def by_confidence(obj: dict, level: str) -> dict:
    if level not in CONFIDENCE or not isinstance(obj, dict):
        return obj
    found = _findings(obj)
    keep = [f for f in found if CONFIDENCE.get(str(f.get("confidence", "")).strip().lower(), 1) >= CONFIDENCE[level]]
    out = {k: v for k, v in obj.items() if k != "findings"}
    out["findings"] = keep
    if len(keep) < len(found):
        out.pop("summary", None)
    return out


# The check (--context-verify): a second request with the same image(s) asks, for each finding of the description,
# whether it is really visible. Only findings answered "yes" are kept. Small vision models describe many things that
# are not there (precision 6% against the radiologist on MedPix with Qwen2.5-VL-7B); a yes/no question about one
# named finding is easier than writing a report, so the check should remove more wrong findings than right ones.
VERIFY_SYSTEM = """You are an experienced, board-certified radiologist reviewing a draft report written by a less \
experienced reader. The draft was written quickly and often lists things that are not in the image: a normal structure \
taken for disease, an artefact taken for a lesion, or a finding that was expected rather than seen. Each finding you \
confirm goes to software that uses it as evidence for a diagnosis, so a wrong "yes" does more harm than a wrong "no".

How to check:
- Treat every finding as unproven until you have found it yourself in the image.
- Judge each finding on its own. Do not confirm one because another was confirmed, or because together they \
suggest a disease.
- Look at the stated location and side. A real finding in the wrong place, or on the wrong side, is a "no".
- Check whether the image could show what is described at all: the right body region, modality, plane and sequence. \
If the structure is outside the image, or the image cannot show that kind of finding, answer "unsure".
- Rule out the usual mimics: normal anatomy and normal variants, normal age-related change, partial volume, motion, \
positioning, windowing, beam hardening and other artefacts, contrast or calcification in normal places.
- Check the details as well as the name: if the size, number, density, signal or shape stated clearly does not \
match the image, answer "no".

How to answer:
- "yes" only when you can point to the finding in the image, at the stated location, and it is abnormal.
- "no" when it is not there, when the image shows something else (a normal structure, a variant or an artefact), or \
when it is in a different place.
- "unsure" when the image does not let you tell: too small, too low in resolution, cut off, or the wrong \
plane or sequence for this finding.
- You cannot add findings, correct them or explain your answers here: only answer for the findings listed.

Answer only with the JSON requested: no text before or after it."""
VERIFY_PROMPT = """{intro}

A draft report lists the findings below. Check each one against the image, one at a time, as you were told:
look for it at the stated location and side, compare it with the opposite side and with what is normal for this
region, rule out normal anatomy, variants and artefacts, and check that the details match what you see.

Findings to check:
{listing}

For each number answer:
- "yes": you can clearly see this abnormality in the image, at that location, and the details fit
- "no": it is not there, it is a normal structure, variant or artefact, or it is somewhere else
- "unsure": the image does not let you tell (outside the field of view, too small, wrong plane or sequence)

Return only JSON, with one answer for every number and nothing else:
{{"answers": {{{example}}}}}"""


def verify_prompt(imgs: list[dict], found: list[dict]) -> str:
    intro = (f"This is one medical image: {_img_label(imgs[0])}." if len(imgs) == 1 else
             f"These are {len(imgs)} images from one patient: " + "; ".join(_img_label(g) for g in imgs) + ".")
    listing = "\n".join(f"{i}. {f['finding']}" + (f" - location: {f['location']}" if f.get("location") else "")
                        + (f" - details: {f['details']}" if f.get("details") else "") for i, f in enumerate(found, 1))
    example = ", ".join(f'"{i}": "yes|no|unsure"' for i in range(1, len(found) + 1))
    return VERIFY_PROMPT.format(intro=intro, listing=listing, example=example)


def _findings(obj) -> list[dict]:
    # the description's findings as dicts with a "finding" text (plain strings become {"finding": text})
    out = []
    for f in (obj.get("findings") or []) if isinstance(obj, dict) else []:
        if isinstance(f, dict) and str(f.get("finding") or "").strip():
            out.append({k: str(v).strip() for k, v in f.items() if v is not None})
        elif isinstance(f, str) and f.strip():
            out.append({"finding": f.strip()})
    return out


def verify_findings(llm, grp: list[dict], obj: dict) -> tuple[dict, int, int]:
    # (the description with only the confirmed findings, findings asked, findings kept). The summary is dropped when
    # a finding was dropped, since it restates them; technique, anatomy and pertinent negatives are kept as written.
    found = _findings(obj)
    if not found:
        return obj, 0, 0
    raw = llm.chat(verify_prompt(grp, found), VERIFY_SYSTEM, images=tuple(g["path"] for g in grp))
    ans = reply_json(raw, ("answers",))
    ans = ans.get("answers") if isinstance(ans, dict) else None
    ans = ans if isinstance(ans, dict) else {}
    keep = [f for i, f in enumerate(found, 1) if str(ans.get(str(i), "")).strip().lower().startswith("yes")]
    out = {k: v for k, v in obj.items() if k != "findings"}
    out["findings"] = keep
    if len(keep) < len(found):
        out.pop("summary", None)
    return out, len(found), len(keep)


def image_context(llm, case: Case, max_images: int = 4, per_call: int = 1,
                  verify: bool | None = None) -> list[ImageContext]:
    # Stage A: per_call images per request (1 = one context per image, best for small models). verify: check each
    # finding with a second request (None = the model's own setting, llm.verify, set by --context-verify)
    check_policy(llm, [case])
    verify = getattr(llm, "verify", False) if verify is None else verify
    level = getattr(llm, "min_confidence", "all")       # --context-confidence
    lean = getattr(llm, "read", "all") == "findings"    # --context-read
    imgs = case.meta.get("images", [])[:max_images]
    if imgs and not llm.sees_images:
        raise ValueError(f"{llm.model} cannot read images; the image-context model must be a vision model")
    out, step, details = [], max(per_call, 1), details_of(case)
    clinical = str(case.meta.get("history") or "")
    for i in range(0, len(imgs), step):
        grp = imgs[i:i + step]
        kind = image_kind(grp)
        if kind == "general":         # the labels name no kind: the vision model says what the image is
            kind = classify_image(llm, grp)
        raw = llm.chat(context_prompt(grp, details, clinical, kind), context_system(kind),
                       images=tuple(g["path"] for g in grp))
        obj = reply_json(raw, ("findings", "summary", "technique"))
        label = (f"Image {i + 1} ({_img_label(grp[0])})" if len(grp) == 1 else
                 f"Images {i + 1}-{i + len(grp)} (" + "; ".join(_img_label(g) for g in grp) + ")")
        limit = getattr(llm, "max_new_tokens", None)
        stopped = cut_off(raw, limit)
        # free text is passed on only if it is a finished, short answer, not a reply that ran out of tokens
        usable = isinstance(obj, dict) or (raw.strip() and not stopped and len(raw) < 4000)
        text = render_context(obj, raw, label, lean=lean) if usable else ""
        definite = render_context(by_confidence(obj, "definite"), "", label, lean=lean) if isinstance(obj, dict) else ""
        if level in CONFIDENCE and isinstance(obj, dict):
            obj = by_confidence(obj, level)
            text = render_context(obj, "", label, lean=lean)        # no raw-text fallback: it would bring the dropped ones back
        unchecked, asked, kept = text, 0, 0
        if verify and isinstance(obj, dict):
            checked, asked, kept = verify_findings(llm, grp, obj)
            if asked:      # no raw-text fallback: it would bring back the findings the check removed
                text = render_context(checked, "", label, lean=lean)
        out.append(ImageContext([g.get("id", g["path"]) for g in grp], label, text, isinstance(obj, dict),
                                getattr(llm, "model", ""), stopped, unchecked, asked, kept, definite, kind))
    return out


def reply_json(raw: str, keys: tuple[str, ...]):
    # the JSON object holding one of `keys`. Models that think aloud (MedGemma 1.5) can write a draft JSON in their
    # plan and the final one after it, so a span from the first "{" to the last "}" is not valid JSON (or is the
    # draft). Try each object start from the end: the first that holds a key is the final answer, or the part of it
    # a cut-off reply managed to write.
    for m in reversed(list(re.finditer(r'\{\s*"', raw))):
        obj = _json(raw[m.start():], "{")
        if isinstance(obj, dict) and any(k in obj for k in keys):
            return obj
    return _json(raw, "{")


def cut_off(raw: str, max_tokens: int | None = None) -> bool:
    # the reply stopped before its JSON was finished: it ends inside the JSON, or (given the token limit) it is as
    # long as the limit allows and holds no JSON at all, usually a model stuck repeating itself while thinking
    tail = raw.rstrip().rstrip("`").rstrip()
    if bool(tail) and "{" in raw and not tail.endswith(("}", "]")):
        return True
    return bool(max_tokens) and len(raw) >= 3 * max_tokens and _json(raw, "{") is None
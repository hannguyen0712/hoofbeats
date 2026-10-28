# llm_extract: findings proposed by a language model, each tied to the exact words (and offsets) it came from;
# HPO linking is separate (linker.py)
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .data import sentence_spans
from .llm import _json, chat_many, require_local
from .model import Status

PROMPT_VERSION = 1
SYSTEM = "You extract clinical findings from case reports. Answer only with the JSON requested."
PROMPT = """Read this text from a clinical case report and list the clinical findings it states about the patient
being described, or about the patient's relatives.

A finding is a sign, symptom, physical or imaging finding, abnormal test result, or a developmental or behavioural
feature. Do not list the diagnosis itself, genes, variants, treatments, or laboratory methods.

For each finding give:
- "phrase": a short standard medical term for it (e.g. "cryptorchidism" for "testes were not palpable in the scrotum")
- "status": "present" (the patient has it), "absent" (the text says the patient does not have it, or a normal
  result rules it out), "family" (a relative has it, not the patient), or "uncertain" (suspected or possible)
- "quote": the shortest exact words from the text that state it, copied character for character

Normal results: list one only when the text reports a normal examination or test that clearly rules out a specific
abnormality, e.g. "blood pressure was normal" -> "hypertension", absent; "cognitive development is normal" ->
"developmental delay", absent.

Skip findings about other patients, about the disease in general, or from the literature.

Text:
\"\"\"
{text}
\"\"\""""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["findings"],
          "properties": {"findings": {"type": "array", "items": {
              "type": "object", "additionalProperties": False, "required": ["phrase", "status", "quote"],
              "properties": {"phrase": {"type": "string"}, "status": {"type": "string", "enum": [s.value for s in Status]},
                             "quote": {"type": "string"}}}}}}


@dataclass(frozen=True)
class Candidate:
    # one proposed finding; text[start:end] == quote in the original field
    phrase: str
    status: Status
    quote: str
    start: int
    end: int
    field: str
    chunk: int
    match: str          # how the quote was found: exact | normalised


@dataclass
class ExtractReport:
    chunks: int = 0
    unreadable: int = 0            # replies that were not the requested JSON
    proposed: int = 0
    bad_item: int = 0              # missing phrase or quote, or a status outside the four
    quote_not_found: int = 0
    duplicates: int = 0
    kept: int = 0
    not_found_examples: list = field(default_factory=list)


def chunks(text: str, max_chars: int = 2500) -> list[tuple[int, int]]:
    # paragraphs (blank-line separated), long ones cut at sentence ends, then joined in order up to max_chars;
    # a single sentence longer than max_chars stays whole
    pieces = []
    for m in re.finditer(r"\S(?:.*?\S)?(?=\s*\n\s*\n|\s*$)", text, re.S):
        a, b = m.start(), m.end()
        if b - a <= max_chars:
            pieces.append((a, b))
            continue
        cur = None
        for s, e in sentence_spans(text, a, b):
            if cur and e - cur[0] > max_chars:
                pieces.append(cur)
                cur = None
            cur = (cur[0], e) if cur else (s, e)
        if cur:
            pieces.append(cur)
    out = []
    for a, b in pieces:
        if out and b - out[-1][0] <= max_chars:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


_SAME = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-"}


def _normalised(text: str, a: int, b: int) -> tuple[str, list[int]]:
    # lower case, one space for any run of whitespace, curly quotes and dashes made plain; idx maps back to text
    out, idx, space = [], [], False
    for i in range(a, b):
        ch = text[i]
        if ch.isspace():
            if not space and out:
                out.append(" ")
                idx.append(i)
            space = True
            continue
        space = False
        out.append(_SAME.get(ch, ch).lower())
        idx.append(i)
    return "".join(out), idx


def locate(text: str, a: int, b: int, quote: str) -> tuple[int, int, str] | None:
    # where the model's quote is in text[a:b]: verbatim first, then ignoring case, spacing, quote and dash styles;
    # a trailing full stop or comma the model added is allowed
    for q in dict.fromkeys((quote.strip(), quote.strip().rstrip(".,;:").strip())):
        if not q:
            continue
        i = text.find(q, a, b)
        if i >= 0:
            return i, i + len(q), "exact"
        norm, idx = _normalised(text, a, b)
        nq, _ = _normalised(q, 0, len(q))
        j = norm.find(nq.strip())
        if nq.strip() and j >= 0:
            return idx[j], idx[j + len(nq.strip()) - 1] + 1, "normalised"
    return None


def parse_reply(reply: str) -> list | None:
    obj = _json(reply, "{")
    if isinstance(obj, dict) and isinstance(obj.get("findings"), list):
        return obj["findings"]
    arr = _json(reply, "[")
    return arr if isinstance(arr, list) else None


def extract_candidates(llm, text: str, field: str = "note_text", max_chars: int = 2500,
                       workers: int = 8) -> tuple[list[Candidate], ExtractReport]:
    require_local(llm)
    rep, out, seen = ExtractReport(), [], set()
    if not text or not text.strip():
        return out, rep
    spans = chunks(text, max_chars)
    rep.chunks = len(spans)
    replies = chat_many(llm, [{"user": PROMPT.format(text=text[a:b]), "system": SYSTEM, "schema": SCHEMA}
                              for a, b in spans], workers)
    statuses = {s.value: s for s in Status}
    for k, ((a, b), reply) in enumerate(zip(spans, replies)):
        items = parse_reply(reply) if reply else None
        if items is None:
            rep.unreadable += bool(reply)
            continue
        for it in items:
            rep.proposed += 1
            phrase = str(it.get("phrase", "")).strip() if isinstance(it, dict) else ""
            quote = str(it.get("quote", "")) if isinstance(it, dict) else ""
            status = statuses.get(str(it.get("status", "")).strip().lower()) if isinstance(it, dict) else None
            if not phrase or not quote.strip() or status is None:
                rep.bad_item += 1
                continue
            hit = locate(text, a, b, quote)
            if hit is None:
                rep.quote_not_found += 1
                if len(rep.not_found_examples) < 20:
                    rep.not_found_examples.append({"phrase": phrase, "quote": quote[:200]})
                continue
            s, e, how = hit
            key = (phrase.lower(), status, s, e)
            if key in seen:
                rep.duplicates += 1
                continue
            seen.add(key)
            out.append(Candidate(phrase, status, text[s:e], s, e, field, k, how))
    rep.kept = len(out)
    return out, rep

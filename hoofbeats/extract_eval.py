# extract-eval: compare what each reader extracts from a case file with gold findings (HPO id, status, quote)
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .data import Extractor
from .intake import load_case
from .model import Ontology

TEXT_FIELDS = {"note": "note_text", "image_context": "image_context"}   # readers whose spans point into a text
TEXT_READERS = ("note", "image_context", "image_findings")


def reader_of(source: str) -> str:
    if source == "input:hpo":
        return "hpo_terms"
    if source.startswith("image:"):
        return "image_findings"
    if source.startswith("lab:"):
        return "labs"
    if source.startswith("vitals:"):
        return "vitals"
    return source


def predictions(case, raw: dict) -> dict[str, list[dict]]:
    # reader -> [{hpo_id, status, evidence, span}] from provenance (every contribution, before fusion drops any)
    out = defaultdict(list)
    for h, entries in case.meta["provenance"].items():
        if not h.startswith("HP:"):
            continue
        for e in entries:
            r = reader_of(e["source"])
            span = None
            if e.get("start") is not None and e.get("field") == TEXT_FIELDS.get(r):
                span = (e["start"], e["end"])
            elif e.get("start") is None and r in TEXT_FIELDS and e.get("evidence"):
                i = (raw.get(TEXT_FIELDS[r]) or "").find(e["evidence"])
                span = (i, i + len(e["evidence"])) if i >= 0 else None
            method = (e.get("readers") or [None])[0] or ("rules" if r in TEXT_FIELDS else r)
            out[r].append({"hpo_id": h, "status": e["status"], "evidence": e.get("evidence", ""), "span": span, "method": method})
    return out


def locate_quote(text: str, g: dict) -> int | None:
    # start of the gold quote in text: its own offsets if given and valid, else the first whole-word match
    # ("CP" is not found inside "PCP")
    q = g.get("quote")
    if not q:
        return None
    if isinstance(g.get("start"), int) and text[g["start"]:g["start"] + len(q)] == q:
        return g["start"]
    pat = (r"(?<![A-Za-z0-9])" if q[0].isalnum() else "") + re.escape(q) + (r"(?![A-Za-z0-9])" if q[-1].isalnum() else "")
    m = re.search(pat, text)
    return m.start() if m else None


def _related(ont: Ontology, a: str, b: str) -> bool:
    return a != b and (a in ont.ancestors(b) or b in ont.ancestors(a))


def score_reader(ont: Ontology, gold: list[dict], preds: list[dict], text: str | None) -> dict:
    # exact = same id (or an alternative) and status; near = ancestor/descendant with the same status
    res = {"gold": len(gold), "exact": 0, "near": 0, "missed": 0, "pred": 0, "pred_exact": 0, "pred_near": 0, "extra": 0,
           "span_checked": 0, "span_overlap": 0, "span_chars": [], "items": [], "extras": [],
           "by_method": defaultdict(lambda: {"exact": 0, "near": 0, "pred": 0, "extra": 0})}
    uniq, methods = {}, defaultdict(set)
    for p in preds:
        uniq.setdefault((p["hpo_id"], p["status"]), p)
        methods[(p["hpo_id"], p["status"])].add(p.get("method") or "?")
    preds = list(uniq.values())
    used = defaultdict(set)
    for g in gold:
        ids = [g["hpo_id"], *g.get("alternatives", [])]
        same = [p for p in preds if p["status"] == g["status"]]
        hit = [p for p in same if p["hpo_id"] in ids]
        near = [] if hit else [p for p in same if any(_related(ont, p["hpo_id"], i) for i in ids)]
        kind = "exact" if hit else "near" if near else "missed"
        res[kind] += 1
        for p in hit + near:
            used[(p["hpo_id"], p["status"])].add(kind)
        for m in {m for p in (hit or near) for m in methods[(p["hpo_id"], p["status"])]}:   # which methods found it
            res["by_method"][m][kind] += 1
        overlap = None
        q = g.get("quote")
        if text is not None and q and (hit or near):
            qs = locate_quote(text, g)
            spans = [p["span"] for p in hit + near if p["span"]]
            overlap = qs is not None and any(s < qs + len(q) and qs < e for s, e in spans)
            res["span_checked"] += 1
            res["span_overlap"] += overlap
        res["items"].append({**g, "result": kind, "got": [p["hpo_id"] for p in hit + near], "span_overlap": overlap})
    for p in preds:
        k = used.get((p["hpo_id"], p["status"]), set())
        for m in methods[(p["hpo_id"], p["status"])]:
            res["by_method"][m]["pred"] += 1
            res["by_method"][m]["extra"] += not k
        res["pred"] += 1
        res["pred_exact"] += "exact" in k
        res["pred_near"] += "near" in k and "exact" not in k
        if not k:
            res["extra"] += 1
            res["extras"].append(p)
        if p["span"]:
            res["span_chars"].append(p["span"][1] - p["span"][0])
    return res


def prf(tp_pred: int, n_pred: int, tp_gold: int, n_gold: int) -> tuple[float, float, float]:
    p = tp_pred / n_pred if n_pred else 0.0
    r = tp_gold / n_gold if n_gold else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def summarize(rows: list[dict]) -> dict:
    s = {k: sum(r[k] for r in rows) for k in ("gold", "exact", "near", "missed", "pred", "pred_exact", "pred_near", "extra",
                                              "span_checked", "span_overlap")}
    chars = [c for r in rows for c in r["span_chars"]]
    s["strict"] = prf(s["pred_exact"], s["pred"], s["exact"], s["gold"])
    s["lenient"] = prf(s["pred_exact"] + s["pred_near"], s["pred"], s["exact"] + s["near"], s["gold"])
    s["span_mean_chars"] = sum(chars) / len(chars) if chars else None
    return s


def evaluate(cases_dir, gold_path, ont: Ontology, ext: Extractor | None = None,
             disease_names: dict[str, str] | None = None) -> tuple[list[dict], list[str]]:
    # one row per (case file, reader); gold readers with no predictions and predicting readers with no gold both count
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    ext = ext or Extractor(ont)
    rows, problems = [], []
    for fn, by_reader in gold.items():
        if fn.startswith("_"):
            continue
        path = Path(cases_dir) / fn
        if not path.exists():
            problems.append(f"{fn}: case file not found in {cases_dir}")
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        case = load_case(path, ont, ext, disease_names=disease_names)
        preds = predictions(case, raw)
        for r in sorted(set(by_reader) | set(preds)):
            text = raw.get(TEXT_FIELDS[r]) if r in TEXT_FIELDS else None
            for g in by_reader.get(r, []):
                if text is not None and g.get("quote") and locate_quote(text, g) is None:
                    problems.append(f"{fn}/{r}: gold quote not in text: {g['quote']!r}")
            rows.append({"case": fn, "reader": r, **score_reader(ont, by_reader.get(r, []), preds.get(r, []), text)})
    return rows, problems


def method_breakdown(rows: list[dict]) -> list[tuple[str, str, dict]]:
    # per reader and method (rules, numbers, labs, vitals, input): gold found and extras, summed over cases;
    # a gold finding found by two methods counts for both
    acc = defaultdict(lambda: {"exact": 0, "near": 0, "pred": 0, "extra": 0})
    for r in rows:
        for m, v in r["by_method"].items():
            for k in v:
                acc[(r["reader"], m)][k] += v[k]
    return [(rd, m, v) for (rd, m), v in sorted(acc.items())]


def _pct(x) -> str:
    return f"{100 * x:.0f}%"


def report(rows: list[dict], ont: Ontology, details: bool = False) -> str:
    out = ["| case | reader | gold | exact | parent/child | missed | pred | extra | span overlap | mean span chars |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sp = f"{r['span_overlap']}/{r['span_checked']}" if r["span_checked"] else "–"
        mc = f"{sum(r['span_chars']) / len(r['span_chars']):.0f}" if r["span_chars"] else "–"
        out.append(f"| {r['case']} | {r['reader']} | {r['gold']} | {r['exact']} | {r['near']} | {r['missed']} | "
                   f"{r['pred']} | {r['extra']} | {sp} | {mc} |")
    out += ["", "| scope | gold | pred | strict P | strict R | strict F1 | lenient P | lenient R | lenient F1 | extras | span overlap |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, sel in (("text readers", [r for r in rows if r["reader"] in TEXT_READERS]), ("all readers", rows)):
        s = summarize(sel)
        sp = f"{s['span_overlap']}/{s['span_checked']}" if s["span_checked"] else "–"
        out.append(f"| {name} | {s['gold']} | {s['pred']} | " + " | ".join(_pct(x) for x in s["strict"] + s["lenient"])
                   + f" | {s['extra']} | {sp} |")
    out += ["", "| reader | method | gold found exact | gold found parent/child | pred | extra |", "|---|---|---|---|---|---|"]
    for reader, method, m in method_breakdown(rows):
        out.append(f"| {reader} | {method} | {m['exact']} | {m['near']} | {m['pred']} | {m['extra']} |")
    if details:
        for r in rows:
            out.append(f"\n{r['case']} / {r['reader']}")
            for it in r["items"]:
                got = f" (got {', '.join(ont.name(h) for h in it['got'])})" if it["result"] == "near" else ""
                span = "" if it["span_overlap"] is None else f" span {'ok' if it['span_overlap'] else 'MISS'}"
                out.append(f"  {it['result']:6s} {it['status']:9s} {it['label']}{got}{span} <- {it.get('quote')!r}")
            for p in r["extras"]:
                out.append(f"  EXTRA  {p['status']:9s} {ont.name(p['hpo_id'])} <- {p['evidence'][:90]!r}")
    return "\n".join(out)

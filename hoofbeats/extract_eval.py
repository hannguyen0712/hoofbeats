# extract-eval: compare what each reader extracts from a case file with gold findings (HPO id, status, quote)
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from .data import Extractor, sentence_spans, tokens
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
            method = "+".join(sorted(e.get("readers") or [])) or ("rules" if r in TEXT_FIELDS else r)   # llm+rules: both
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
        found_by = sorted({m for p in (hit or near) for m in methods[(p["hpo_id"], p["status"])]})
        for m in found_by:
            res["by_method"][m][kind] += 1
        overlap = None
        q = g.get("quote")
        if text is not None and q and (hit or near):
            qs = locate_quote(text, g)
            spans = [p["span"] for p in hit + near if p["span"]]
            overlap = qs is not None and any(s < qs + len(q) and qs < e for s, e in spans)
            res["span_checked"] += 1
            res["span_overlap"] += overlap
        res["items"].append({**g, "result": kind, "got": [p["hpo_id"] for p in hit + near], "found_by": found_by,
                             "span_overlap": overlap})
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
             disease_names: dict[str, str] | None = None, **load_opts) -> tuple[list[dict], list[str]]:
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
        case = load_case(path, ont, ext, disease_names=disease_names, **load_opts)
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


# PMC test set: the note reader on each manifest case's located text, against its phenopacket features
def _norm(s: str) -> str:
    # lower case, word characters only, trailing plural "s" dropped (as the extractor's tokens)
    return " " + " ".join(tokens(s)) + " "


def vocab_visible(ont: Ontology, hpo_id: str, norm_text: str) -> bool:
    # the term's label or one of its synonyms (EXACT, RELATED, layperson) appears as whole words, plurals allowed
    t = ont.terms.get(hpo_id)
    if t is None:
        return False
    names = (_norm(n) for n in [t.name, *t.synonyms, *(x for x, _ in t.extra_synonyms)])
    return any(n.strip() and n in norm_text for n in names)


def evaluate_pmc(manifest: dict, ont: Ontology, ext: Extractor | None = None, disease_names: dict[str, str] | None = None,
                 split: str = "dev", **load_opts) -> list[dict]:
    from .intake import parse_case
    from .pmc import load_case_text
    ext = ext or Extractor(ont)
    rows = []
    for c in manifest["cases"]:
        if c["split"] != split:
            continue
        text = load_case_text(c)
        case = parse_case({"case_id": c["case_id"], "note_text": text}, ont, ext, disease_names=disease_names, **load_opts)
        norm = _norm(text)
        gold = [{"hpo_id": g["hpo_id"], "label": g["label"], "status": g["status"],
                 "vocab_visible": vocab_visible(ont, g["hpo_id"], norm)} for g in c["gold"]]
        preds = predictions(case, {"note_text": text}).get("note", [])
        res = score_reader(ont, gold, preds, None)
        sents = sentence_spans(text, 0, len(text))
        for p in res["extras"]:     # quote and whole sentence, for hand-checking
            if p["span"]:
                a, b = p["span"]
                ss = next(((x, y) for x, y in sents if x <= a < y), (a, b))
                p["quote"], p["sentence"] = text[a:b], text[ss[0]:max(ss[1], b)].strip()
        methods = {}
        for p in preds:
            methods.setdefault((p["hpo_id"], p["status"]), set()).add(p["method"])
        res["preds"] = [{"hpo_id": h, "status": st, "method": "+".join(sorted(ms))} for (h, st), ms in methods.items()]
        rows.append({"case": c["case_id"], "reader": "note", "paper_kind": c["paper_kind"], **res})
    return rows


def recall(items: list[dict]) -> dict:
    n = len(items)
    ex = sum(it["result"] == "exact" for it in items)
    near = sum(it["result"] == "near" for it in items)
    return {"n": n, "strict": ex / n if n else None, "lenient": (ex + near) / n if n else None}


def pmc_summary(rows: list[dict]) -> dict:
    s = summarize(rows)
    items = [it for r in rows for it in r["items"]]
    out = {"cases": len(rows), "gold": s["gold"], "pred": s["pred"], "extra": s["extra"],
           "precision": {"strict": s["strict"][0], "lenient": s["lenient"][0]},
           "recall_all_gold": recall(items), "recall_vocab_visible": recall([it for it in items if it["vocab_visible"]]),
           "vocab_visible_share": sum(it["vocab_visible"] for it in items) / len(items) if items else None,
           "by_status": {st: recall([it for it in items if it["status"] == st]) for st in ("present", "absent")},
           "by_paper_kind": {k: recall([it for r in rows if r["paper_kind"] == k for it in r["items"]])
                             for k in ("single", "multi")},
           "by_method": {m: v for _, m, v in method_breakdown(rows)}}
    return out


def pmc_report(summ: dict) -> str:
    f = lambda x: "–" if x is None else f"{100 * x:.1f}%"
    r, v = summ["recall_all_gold"], summ["recall_vocab_visible"]
    out = [f"{summ['cases']} cases, {summ['gold']} gold findings, {summ['pred']} predictions, {summ['extra']} extras", "",
           "| measure | n | strict | lenient |", "|---|---|---|---|",
           f"| recall, all gold | {r['n']} | {f(r['strict'])} | {f(r['lenient'])} |",
           f"| recall, vocabulary-visible gold | {v['n']} | {f(v['strict'])} | {f(v['lenient'])} |",
           f"| precision (extras unadjudicated) | {summ['pred']} | {f(summ['precision']['strict'])} | {f(summ['precision']['lenient'])} |"]
    for k, x in [*summ["by_status"].items(), *summ["by_paper_kind"].items()]:
        out.append(f"| recall, all gold, {k} | {x['n']} | {f(x['strict'])} | {f(x['lenient'])} |")
    out += ["", f"vocabulary-visible share of gold: {f(summ['vocab_visible_share'])}", "",
            "| method | gold found exact | gold found parent/child | pred | extra |", "|---|---|---|---|---|"]
    for m, x in summ["by_method"].items():
        out.append(f"| {m} | {x['exact']} | {x['near']} | {x['pred']} | {x['extra']} |")
    return "\n".join(out)

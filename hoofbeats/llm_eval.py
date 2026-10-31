# llm-eval: rules vs LLM vs rules+LLM (both status policies) on PMC dev, the six example cases and the task-C
# regression set, from one set of LLM requests per prompt version
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

from .extract_eval import evaluate, evaluate_pmc, pmc_summary, predictions, recall, summarize, TEXT_READERS
from .intake import load_case, parse_case
from .pmc import load_case_text

EXAMPLES = (("examples", "dev"), ("examples/heldout", "held_out"))


def configs(reader, policies=("llm", "rules")) -> dict[str, dict]:
    # parse_case options per configuration; "rules" is the reader as before (rules + numbers); rules+llm once per
    # status policy (whose status wins when the two readers disagree on the same words)
    out = {"rules": {}}
    if reader is not None:
        out["llm"] = {"llm_reader": reader, "rules_text": False}
        out.update({f"rules+llm/status={p}": {"llm_reader": reader, "status_policy": p} for p in policies})
    return out


def _read(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def hand_checked(rows: list[dict], task_b: list[dict]) -> dict:
    # recall over task-B terms expressed in the text (dev only): all, label (the term's label or a synonym is in the
    # text) and paraphrase (it is not)
    res = {(r["case"], it["hpo_id"], it["status"]): it for r in rows for it in r["items"]}
    yes = [res[k] for k in ((b["case_id"], b["hpo_id"], b["gold_status"]) for b in task_b if b["expressed"] == "yes")
           if k in res]
    return {"all": recall(yes), "label": recall([it for it in yes if it["vocab_visible"]]),
            "paraphrase": recall([it for it in yes if not it["vocab_visible"]])}


def absent_stats(rows: list[dict]) -> dict:
    # absent findings on their own: how many, how many match an absent gold term, and absent-gold recall
    preds = [p for r in rows for p in r["preds"] if p["status"] == "absent"]
    extra = [p for r in rows for p in r["extras"] if p["status"] == "absent"]
    by = Counter(p["method"] for p in preds)
    items = [it for r in rows for it in r["items"] if it["status"] == "absent"]
    return {"pred": len(preds), "extra": len(extra), "matched_share": 1 - len(extra) / len(preds) if preds else None,
            "by_method": dict(by), "gold_recall": recall(items)}


def regression(manifest: dict, rows: list[dict], ont, ext, names, opts: dict) -> list[dict]:
    # each task-C error row: is the wrong (term, status) gone from the sentence, and is the expected one there?
    # expected matches exactly or as a parent/child; "none" rows only need the wrong one gone
    cases = {c["case_id"]: c for c in manifest["cases"]}
    rel = lambda a, b: a == b or a in ont.ancestors(b) or b in ont.ancestors(a)
    out = []
    for r in rows:
        text = load_case_text(cases[r["case_id"]])
        case = parse_case({"case_id": r["case_id"], "note_text": text}, ont, ext, disease_names=names, **opts)
        a, b = int(r["sentence_start"]), int(r["sentence_end"])
        here = [p for p in predictions(case, {"note_text": text}).get("note", []) if p["span"] and p["span"][0] < b
                and a < p["span"][1]]
        wrong = any(p["hpo_id"] == r["predicted_hpo"] and p["status"] == r["predicted_status"] for p in here)
        want = r["expected_status"] == "none" or any(rel(p["hpo_id"], r["expected_hpo"]) and p["status"] == r["expected_status"]
                                                     for p in here)
        out.append({"row_id": r["row_id"], "error": r["error"], "was": f"{r['predicted_label']} {r['predicted_status']}",
                    "expected": f"{ont.name(r['expected_hpo']) if r['expected_hpo'] else 'no finding'} "
                                f"{r['expected_status'] if r['expected_status'] != 'none' else ''}".strip(),
                    "outcome": "fixed" if not wrong and want else "partly" if not wrong else "not fixed",
                    "in_sentence": "; ".join(sorted({f"{ont.name(p['hpo_id'])} {p['status']} ({p['method']})" for p in here}))})
    return out


def example_ranks(ranker, ont, ext, names, opts: dict, without_llm_absent: bool = False) -> dict:
    # truth rank of each example case (OMIM, --common none setup); optionally without the absent findings that only
    # the LLM gave
    out = {}
    for folder, _ in EXAMPLES:
        for fn in json.loads(Path(folder, "intended_findings.json").read_text()):
            if fn.startswith("_"):
                continue
            c = load_case(Path(folder) / fn, ont, ext, disease_names=names, **opts)
            if without_llm_absent:
                prov = c.meta["provenance"]
                keep = lambda f: not (f.status.value == "absent" and all(
                    e.get("readers") == ["llm"] for e in prov.get(f.hpo_id, []) if e["status"] == "absent"))
                c = type(c)(c.case_id, [f for f in c.findings if keep(f)], c.truth, c.genes, meta=c.meta)
            out[fn] = ranker.score(c, frozenset()).rank_of(c.truth)
    return out


def evaluate_config(name, opts, manifest, ont, ext, names, task_b, reg_rows, ranker) -> dict:
    rows = evaluate_pmc(manifest, ont, ext, names, "dev", **opts)
    summ = pmc_summary(rows)
    summ["hand_checked"] = hand_checked(rows, task_b)
    summ["absent"] = absent_stats(rows)
    ex = {}
    for folder, split in EXAMPLES:
        er, _ = evaluate(folder, Path(folder) / "intended_findings.json", ont, ext, names, **opts)
        t = summarize([r for r in er if r["reader"] in TEXT_READERS])
        a = summarize(er)
        ex[split] = {"text": {"strict": t["strict"], "lenient": t["lenient"], "extra": t["extra"]},
                     "all": {"strict": a["strict"], "lenient": a["lenient"], "extra": a["extra"]}}
    summ["examples"] = ex
    summ["ranks"] = example_ranks(ranker, ont, ext, names, opts)
    if opts.get("llm_reader") is not None:
        summ["ranks_without_llm_absent"] = example_ranks(ranker, ont, ext, names, opts, without_llm_absent=True)
    summ["regression"] = regression(manifest, reg_rows, ont, ext, names, opts)
    summ["regression_outcomes"] = dict(Counter(r["outcome"] for r in summ["regression"]))
    return {"config": name, "rows": rows, "summary": summ}


def evaluate_split(name, opts, manifest, ont, ext, names, task_b, split: str) -> dict:
    # PMC metrics of one configuration on one split (the frozen test run): hand-checked recall on that split's task-B
    # terms (label vs paraphrase), all-gold and vocabulary-visible recall, precision with extras not judged, by status,
    # paper kind and method, and absent findings on their own
    rows = evaluate_pmc(manifest, ont, ext, names, split, **opts)
    summ = pmc_summary(rows)
    summ["hand_checked"] = hand_checked(rows, [b for b in task_b if b["split"] == split])
    summ["absent"] = absent_stats(rows)
    summ["split"] = split
    return {"config": name, "rows": rows, "summary": summ}


def estimated_judged_precision(summ: dict, items: list[dict], rules_share: float, llm_share: float | None) -> dict:
    # precision if extras were as often correct as on dev: rule-found extras at rules_share (task C), LLM-only extras
    # at llm_share (task C for LLM extras). An estimate for test, which has no judged extras
    extras = [r for r in items if r["kind"] == "extra"]
    n_llm = sum(r["method"] == "llm" for r in extras)
    add = rules_share * (len(extras) - n_llm) + (llm_share or 0.0) * n_llm
    return {k: (summ["precision"][k] * summ["pred"] + add) / summ["pred"] if summ["pred"] else None
            for k in ("strict", "lenient")}


GAP_ROWS = (("hand-checked recall", lambda s: s["hand_checked"]["all"]),
            ("… label", lambda s: s["hand_checked"]["label"]),
            ("… paraphrase", lambda s: s["hand_checked"]["paraphrase"]),
            ("all-gold recall", lambda s: s["recall_all_gold"]),
            ("vocabulary-visible recall", lambda s: s["recall_vocab_visible"]),
            ("present-gold recall", lambda s: s["by_status"]["present"]),
            ("absent-gold recall", lambda s: s["by_status"]["absent"]),
            ("single-patient recall", lambda s: s["by_paper_kind"]["single"]),
            ("multi-patient recall", lambda s: s["by_paper_kind"]["multi"]),
            ("precision, extras not judged", lambda s: {"n": s["pred"], **s["precision"]}),
            ("precision, estimated with dev judging", lambda s: {"n": s["pred"], **s["precision_estimated"]}))


def gap_table(results: dict) -> str:
    # results[(config, split)] = summary; one block per configuration: dev, test and test - dev (points)
    f = lambda x: "–" if x is None else f"{100 * x:.1f}"
    d = lambda a, b: "–" if a is None or b is None else f"{100 * (b - a):+.1f}"
    out = []
    for name in dict.fromkeys(c for c, _ in results):
        dev, test = results.get((name, "dev")), results.get((name, "test"))
        out += [f"### {name}", "", "| measure | dev n | dev strict / lenient | test n | test strict / lenient | "
                "gap strict / lenient |", "|---|---|---|---|---|---|"]
        for label, get in GAP_ROWS:
            a, b = get(dev), get(test)
            out.append(f"| {label} | {a.get('n', '')} | {f(a['strict'])} / {f(a['lenient'])} | {b.get('n', '')} | "
                       f"{f(b['strict'])} / {f(b['lenient'])} | {d(a['strict'], b['strict'])} / "
                       f"{d(a['lenient'], b['lenient'])} |")
        for split, s in (("dev", dev), ("test", test)):
            ab = s["absent"]
            out.append(f"\n{split}: {s['cases']} cases, {s['gold']} gold, {s['pred']} predictions, {s['extra']} extras; "
                       f"absent predictions {ab['pred']} ({ab['extra']} not in gold); by method "
                       + ", ".join(f"{m} {v['exact']}+{v['near']} found / {v['pred']} pred" for m, v in s["by_method"].items()))
        out.append("")
    return "\n".join(out)


def absent_sample(rows: list[dict], n: int = 25, seed: int = 576) -> list[dict]:
    # LLM absent findings not in the gold, with quote and sentence, for hand-judging (dev only)
    ex = [(r["case"], p) for r in rows for p in r["extras"] if p["status"] == "absent" and "llm" in p["method"]]
    pick = random.Random(seed).sample(ex, min(n, len(ex)))
    return [{"row_id": f"N{i:03d}", "case_id": c, "hpo_id": p["hpo_id"], "method": p["method"], "quote": p.get("quote", ""),
             "sentence": p.get("sentence", ""), "verdict": "", "notes": ""} for i, (c, p) in enumerate(pick, 1)]

# pmc_checks: hand-check sheets for the PMC test set and the scores computed from the filled-in answers
#   A  is the located text about this patient?           -> locator accuracy
#   B  is each gold term expressed in the located text?  -> hand-checked recall, inter-annotator agreement
#   C  is each extra (prediction not in gold) right?     -> precision with adjudicated extras
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

from .pmc import bioc_document, load_case_text

SEED = 576
ANSWERS = {"is_this_patient": {"yes", "no", "partly"}, "expressed": {"yes", "no", "unsure"},
           "status_matches": {"yes", "no", ""},
           "verdict": {"correct_not_in_gold", "wrong_term", "wrong_status", "not_this_patient", "unsure"}}


def _write(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _read(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]


def _marked_text(c: dict) -> str:
    # located paragraphs, each preceded by its rule and headings
    doc = bioc_document(c["pmcid"])
    ps = doc.get("passages", [])
    return "\n\n".join(f"[{i} | {p['rule']} | {' > '.join(p['headings']) or p['section_type']}]\n{ps[p['index']]['text'].strip()}"
                       for i, p in enumerate(c["passages"], 1))


def make_task_a(manifest: dict, out: Path, n_single: int = 20, n_multi: int = 20) -> Path:
    rng = random.Random(SEED)
    cases = []
    for kind, n in (("single", n_single), ("multi", n_multi)):
        pool = [c for c in manifest["cases"] if c["paper_kind"] == kind]
        cases += rng.sample(pool, min(n, len(pool)))
    rows = [{"row_id": f"A{i:03d}", "case_id": c["case_id"], "split": c["split"], "pmid": c["pmid"], "pmcid": c["pmcid"],
             "article_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{c['pmcid']}/", "paper_kind": c["paper_kind"],
             "subject_id": c["subject_id"], "disease_label": c["disease_label"], "rules": ";".join(c["rules"]),
             "located_text": _marked_text(c), "is_this_patient": "", "notes": ""} for i, c in enumerate(cases, 1)]
    _write(out / "task_a_locator.csv", rows)
    return out / "task_a_locator.csv"


def make_task_b(manifest: dict, out: Path, n: int = 30, n_double: int = 10) -> list[Path]:
    rng = random.Random(SEED + 1)
    cases = rng.sample(manifest["cases"], min(n, len(manifest["cases"])))
    rows, k = [], 0
    for c in cases:
        text = load_case_text(c)
        gold = sorted(c["gold"], key=lambda g: (g["status"] != "present", g["label"].lower()))
        for j, g in enumerate(gold):
            k += 1
            rows.append({"row_id": f"B{k:04d}", "case_id": c["case_id"], "split": c["split"], "hpo_id": g["hpo_id"],
                         "label": g["label"], "gold_status": g["status"], "case_text": text if j == 0 else "",
                         "expressed": "", "quote": "", "status_matches": "", "notes": ""})
    _write(out / "task_b_gold_in_text.csv", rows)
    double = {c["case_id"] for c in cases[:n_double]}
    _write(out / "task_b_second_annotator.csv", [r for r in rows if r["case_id"] in double])
    return [out / "task_b_gold_in_text.csv", out / "task_b_second_annotator.csv"]


def make_task_c(items_csv, out: Path, n: int = 100) -> Path:
    extras = [r for r in _read(items_csv) if r["kind"] == "extra"]
    rows = [{"row_id": f"C{i:03d}", "case_id": r["case"], "hpo_id": r["hpo_id"], "label": r["label"],
             "predicted_status": r["status"], "method": r["method"], "quote": r["quote"], "sentence": r["evidence"],
             "verdict": "", "notes": ""} for i, r in enumerate(random.Random(SEED + 2).sample(extras, min(n, len(extras))), 1)]
    _write(out / "task_c_extras.csv", rows)
    return out / "task_c_extras.csv"


def kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else None


def score(checks: Path, items_csv=None, run_summary=None) -> dict:
    # answers -> locator accuracy, hand-checked recall, agreement, precision with adjudicated extras;
    # problems lists unanswered rows and values outside the allowed answers
    out, problems = {}, []

    def answered(rows, col):
        good = []
        for r in rows:
            v = r[col].lower()
            if not v:
                continue
            if v not in ANSWERS[col]:
                problems.append(f"{r['row_id']}: {col}={r[col]!r} not one of {sorted(ANSWERS[col] - {''})}")
                continue
            good.append({**r, col: v})
        if len(good) < len(rows):
            problems.append(f"{len(rows) - len(good)} of {len(rows)} rows have no valid {col}")
        return good

    pa = checks / "task_a_locator.csv"
    if pa.exists():
        a = answered(_read(pa), "is_this_patient")
        out["task_a"] = {"answered": len(a), "answers": dict(Counter(r["is_this_patient"] for r in a)),
                         "by_kind": {k: dict(Counter(r["is_this_patient"] for r in a if r["paper_kind"] == k)) for k in ("single", "multi")},
                         "by_rule": {k: dict(Counter(r["is_this_patient"] for r in a if r["rules"] == k)) for k in {r["rules"] for r in a}}}
    pb = checks / "task_b_gold_in_text.csv"
    if pb.exists():
        b = answered(_read(pb), "expressed")
        res = {}
        if items_csv:
            res = {(r["case"], r["hpo_id"], r["status"]): r["result"] for r in _read(items_csv) if r["kind"] == "gold"}
        yes = [r for r in b if r["expressed"] == "yes"]
        run = [r for r in yes if (r["case_id"], r["hpo_id"], r["gold_status"]) in res]
        ex = sum(res[(r["case_id"], r["hpo_id"], r["gold_status"])] == "exact" for r in run)
        near = sum(res[(r["case_id"], r["hpo_id"], r["gold_status"])] == "near" for r in run)
        sm = [r["status_matches"].lower() for r in yes if r["status_matches"]]
        out["task_b"] = {"answered": len(b), "expressed": dict(Counter(r["expressed"] for r in b)),
                         "expressed_share": len(yes) / len(b) if b else None,
                         "status_matches_yes": sm.count("yes") / len(sm) if sm else None,
                         "hand_checked_recall": {"n_in_run": len(run), "n_not_in_run": len(yes) - len(run),
                                                 "strict": ex / len(run) if run else None,
                                                 "lenient": (ex + near) / len(run) if run else None}}
        p2 = checks / "task_b_second_annotator.csv"
        if p2.exists():
            second = {r["row_id"]: r for r in answered(_read(p2), "expressed")}
            first = {r["row_id"]: r for r in b}
            both = [k for k in second if k in first]
            x, y = [first[k]["expressed"] for k in both], [second[k]["expressed"] for k in both]
            out["task_b"]["agreement"] = {"rows": len(both), "raw": sum(i == j for i, j in zip(x, y)) / len(both) if both else None,
                                          "cohen_kappa": kappa(x, y)}
    pc = checks / "task_c_extras.csv"
    if pc.exists():
        c = answered(_read(pc), "verdict")
        share = sum(r["verdict"] == "correct_not_in_gold" for r in c) / len(c) if c else None
        out["task_c"] = {"answered": len(c), "verdicts": dict(Counter(r["verdict"] for r in c)), "correct_share": share}
        if run_summary and share is not None:
            s = json.loads(Path(run_summary).read_text())
            pred, extra = s["pred"], s["extra"]
            tp_strict, tp_len = s["precision"]["strict"] * pred, s["precision"]["lenient"] * pred
            out["task_c"]["precision_adjusted"] = {"strict": (tp_strict + share * extra) / pred,
                                                   "lenient": (tp_len + share * extra) / pred}
    out["problems"] = problems
    return out

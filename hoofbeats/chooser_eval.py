# chooser_eval: the linker's LLM-choice step, old vs new, on PMC dev. Each LLM candidate is matched across the two runs
# by (case, field, phrase, status, start, end). Old chooser extras judged in task C keep their verdict when their
# term is unchanged; a changed term, and a sample of extras the new chooser links that the old one did not, go on
# one small blind sheet (task_c_chooser.csv). Precision is estimated per stratum and weighted by population counts
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

SEED = 579
CORRECT = "correct_not_in_gold"


def _read(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _cand(r) -> tuple:
    return (r["case"], r["field"], r["phrase"], r["status"], str(r["start"]), str(r["end"]))


def _linked(r) -> str | None:
    return r["hpo_id"] if str(r.get("kept")) == "True" and r.get("hpo_id") else None


def _chooser(r) -> bool:
    return str(r.get("link_method", "")).startswith("llm_choice")


class Runs:
    # one run's LLM log and rules+llm items, as lookups
    def __init__(self, folder: Path):
        self.log = _read(folder / "llm_log.csv")
        items = _read(folder / "rules+llm_status-llm" / "pmc_dev_items.csv")
        self.extras = {(r["case"], r["hpo_id"], r["status"]): r for r in items if r["kind"] == "extra"}
        self.llm_only = {k for k, r in self.extras.items() if r["method"] == "llm"}
        self.by_cand = {_cand(r): r for r in self.log}
        # LLM-only extras that came (at least once) through the chooser, and those that never did
        via = {}
        for r in self.log:
            h = _linked(r)
            if h:
                via.setdefault((r["case"], h, r["status"]), set()).add(_chooser(r))
        self.chooser_extras = {k for k in self.llm_only if True in via.get(k, set())}
        self.other_llm_extras = self.llm_only - self.chooser_extras


def compare(old_dir: Path, new_dir: Path, checks: Path, version: str = "v1", n_new: int = 20) -> dict:
    old, new = Runs(old_dir), Runs(new_dir)
    key = json.loads((checks / "task_c_llm_key.json").read_text())
    sheet = {r["row_id"]: r for r in _read(checks / "task_c_llm.csv")}
    sampled = {(sheet[rid]["case_id"], sheet[rid]["hpo_id"], sheet[rid]["predicted_status"]): sheet[rid]["verdict"]
               for rid, ss in key["rows"].items() if {"version": version, "stratum": "random"} in ss}
    rows, judge = [], []
    for k, verdict in sampled.items():
        if k not in old.chooser_extras:
            continue
        if k in new.llm_only:
            rows.append({"extra": k, "outcome": "unchanged", "verdict": verdict})
            continue
        # where did this extra's chooser candidates go?
        targets = {_linked(new.by_cand[_cand(r)]) for r in old.log
                   if _linked(r) == k[1] and (r["case"], r["status"]) == (k[0], k[2]) and _chooser(r)
                   and _cand(r) in new.by_cand}
        targets.discard(None)
        if not targets:
            rows.append({"extra": k, "outcome": "dropped", "verdict": verdict})
            continue
        for t in sorted(targets):
            nk = (k[0], t, k[2])
            outcome = ("changed_to_judge" if nk in new.llm_only else "changed_also_rules" if nk in new.extras
                       else "changed_to_gold")
            rows.append({"extra": k, "outcome": outcome, "verdict": verdict, "new": nk})
            if outcome == "changed_to_judge":
                judge.append(("changed", nk))
    # newly linked: LLM-only chooser extras of the new run that come only from candidates the old run left unlinked
    # (a changed term is not newly linked: it is judged above through the old sample)
    from_unlinked, from_linked = set(), set()
    for r in new.log:
        h = _linked(r)
        if h and _chooser(r):
            o = old.by_cand.get(_cand(r))
            (from_linked if o is not None and _linked(o) else from_unlinked).add((r["case"], h, r["status"]))
    fresh = sorted((new.chooser_extras & from_unlinked) - from_linked - old.llm_only)
    pick = random.Random(SEED).sample(fresh, min(n_new, len(fresh)))
    judge += [("new", k) for k in pick]
    # population counts of chooser candidates whose outcome changed
    moved = Counter()
    for c, r in old.by_cand.items():
        n = new.by_cand.get(c)
        if n is None or not (_chooser(r) or _chooser(n)):
            continue
        a, b = _linked(r), _linked(n)
        moved["unchanged" if a == b else "dropped" if b is None else "newly_linked" if a is None else "changed"] += 1
    return {"old": old, "new": new, "rows": rows, "judge": judge, "fresh": len(fresh), "moved": dict(moved)}


def make_sheet(old_dir, new_dir, checks: Path, version: str = "v1", n_new: int = 20) -> dict:
    cmp = compare(Path(old_dir), Path(new_dir), Path(checks), version, n_new)
    new = cmp["new"]
    seen, out = set(), []
    for why, k in cmp["judge"]:
        if k in seen:
            continue
        seen.add(k)
        r = new.extras[k]
        out.append({"key": k, "why": why, "row": {"case_id": k[0], "hpo_id": k[1], "label": r["label"],
                                                   "predicted_status": k[2], "quote": r["quote"],
                                                   "sentence": r["evidence"], "verdict": "", "notes": ""}})
    random.Random(SEED + 1).shuffle(out)
    for i, o in enumerate(out, 1):
        o["row"] = {"row_id": f"E{i:03d}", **o["row"]}
    _write(Path(checks) / "task_c_chooser.csv", [o["row"] for o in out],
           ["row_id", "case_id", "hpo_id", "label", "predicted_status", "quote", "sentence", "verdict", "notes"])
    (Path(checks) / "task_c_chooser_key.json").write_text(json.dumps(
        {"old": str(old_dir), "new": str(new_dir), "version": version, "n_new": n_new,
         "rows": {o["row"]["row_id"]: o["why"] for o in out}}, indent=1))
    return {"sheet_rows": len(out), "outcomes": dict(Counter(r["outcome"] for r in cmp["rows"])),
            "fresh_chooser_extras": cmp["fresh"], "candidates": cmp["moved"]}


def _share(verdicts: list[str]) -> float | None:
    v = [x for x in verdicts if x]
    return v.count(CORRECT) / len(v) if v else None


def score(checks: Path, rules_share: float) -> dict:
    # chooser precision (old vs new) and precision with every extra judged, both from the same strata formula:
    # LLM-only extras = chooser stratum + other stratum, each at its judged share; other extras at rules_share
    checks = Path(checks)
    k = json.loads((checks / "task_c_chooser_key.json").read_text())
    cmp = compare(Path(k["old"]), Path(k["new"]), checks, k["version"], k["n_new"])
    ans = {(r["case_id"], r["hpo_id"], r["predicted_status"]): r["verdict"].strip().lower()
           for r in _read(checks / "task_c_chooser.csv")}
    old, new = cmp["old"], cmp["new"]
    key = json.loads((checks / "task_c_llm_key.json").read_text())
    sheet = {r["row_id"]: r for r in _read(checks / "task_c_llm.csv")}
    sampled = {(sheet[rid]["case_id"], sheet[rid]["hpo_id"], sheet[rid]["predicted_status"]): sheet[rid]["verdict"]
               for rid, ss in key["rows"].items() if {"version": k["version"], "stratum": "random"} in ss}
    other = _share([v for x, v in sampled.items() if x in old.other_llm_extras])   # non-chooser LLM extras: unchanged
    ch_old = _share([v for x, v in sampled.items() if x in old.chooser_extras])
    # new chooser stratum: the old sample after the change (dropped rows leave; a term now in the gold is correct;
    # one now also found by the rules leaves this stratum) plus the newly linked sample
    after = []
    for r in cmp["rows"]:
        if r["outcome"] == "unchanged":
            after.append(r["verdict"])
        elif r["outcome"] == "changed_to_judge":
            after.append(ans.get(r["new"], ""))
        elif r["outcome"] == "changed_to_gold":
            after.append(CORRECT)
    fresh = [ans.get(x, "") for why, x in cmp["judge"] if why == "new"]
    n_sample = sum(1 for x in sampled if x in old.chooser_extras)
    kept_frac = len(after) / n_sample if n_sample else 0.0
    pop_old_kept, pop_fresh = len(old.chooser_extras) * kept_frac, cmp["fresh"]
    p_after, p_fresh = _share(after), _share(fresh)
    ch_new = None
    if p_after is not None and (p_fresh is not None or not pop_fresh):
        ch_new = (pop_old_kept * p_after + pop_fresh * (p_fresh or 0)) / (pop_old_kept + pop_fresh)

    def judged(run: Runs, ch: float | None, folder: str) -> dict | None:
        if ch is None or other is None:
            return None
        s = json.loads((Path(folder) / "rules+llm_status-llm" / "summary.json").read_text())
        n_ch, n_ot = len(run.chooser_extras), len(run.other_llm_extras)
        add = ch * n_ch + other * n_ot + rules_share * (len(run.extras) - n_ch - n_ot)
        return {m: (s["precision"][m] * s["pred"] + add) / s["pred"] for m in ("strict", "lenient")}
    return {"sample_outcomes": dict(Counter(r["outcome"] for r in cmp["rows"])), "candidates": cmp["moved"],
            "chooser_extras": {"old": len(old.chooser_extras), "new": len(new.chooser_extras), "fresh": pop_fresh},
            "chooser_precision": {"old": ch_old, "new": ch_new, "old_sample_after_change": p_after,
                                  "newly_linked_sample": p_fresh, "judged": len([v for v in fresh + after if v])},
            "other_llm_extras_precision": other,
            "precision_judged": {"old": judged(old, ch_old, k["old"]), "new": judged(new, ch_new, k["new"])},
            "unanswered": sum(1 for x in ans.values() if not x)}

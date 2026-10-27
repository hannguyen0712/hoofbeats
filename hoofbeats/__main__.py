from __future__ import annotations

import argparse
import math
import random
import shlex
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import json
import re

from .data import (DiseaseMatcher, Extractor, download, learn_common, load_medpix, load_mimic,
                   load_phenopackets, orphadata_icd10)
from .model import (AnnotationDB, Case, Finding, LRRanker, Ontology, Status, explain,
                    most_frequent_test, suggest_tests, verify_claim)


# run record: every command logs its settings, metrics and console output under <out>/runs/
RUN = {"metrics": {}, "files": [], "settings": {}}


def note(section: str, **metrics):
    # store numbers for the run record (read later by `report`)
    clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in metrics.items()}
    RUN["metrics"].setdefault(section, {}).update(clean)


def wrote(path):
    RUN["files"].append(str(path))
    return path


class _Tee:
    # copy console output to the run log
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for st in self.streams:
            st.write(text)

    def flush(self):
        for st in self.streams:
            st.flush()


def _out_dir(a) -> Path:
    out = Path(getattr(a, "out", None) or "results")
    return out.parent if out.suffix else out


def _run_dir(a) -> Path:
    return _out_dir(a) / "runs"


def usage_line(llm) -> str:
    u = llm.usage()
    note("requests", **u)
    if llm.dry_run:
        return f"[llm] dry run: would send {u['would_send']} requests; {u['cache_hits']} already answered from the cache"
    return (f"[llm] requests sent {u['requests']} (retries {u['retries']}), answered from cache {u['cache_hits']}, "
            f"{u['sec_per_request']} s per request")


# helpers
def get_llm(a):
    if not a.llm:
        return None
    import os
    from .llm import MedGemma
    key = os.environ.get(a.llm_key_env) if a.llm_key_env else None
    if a.llm_key_env and not key:
        raise SystemExit(f"environment variable {a.llm_key_env} is not set")
    vision = {"auto": None, "yes": True, "no": False}[a.llm_vision]
    extra = json.loads(a.llm_extra) if a.llm_extra else None
    cache = None if a.llm_cache == "none" else Path(a.llm_cache if a.llm_cache != "auto" else Path(_out_dir(a)) / "llm_cache")
    llm = MedGemma(a.llm_model, a.llm_endpoint, a.llm_max_tokens, key, vision, extra, a.llm_rpm, cache, a.dry_run)
    RUN["settings"].update({"model": llm.model, "endpoint": llm.endpoint or "in-process",
                            "max_tokens": a.llm_max_tokens, "extra": extra or {}})
    print(f"[llm] {llm.model} via {llm.endpoint or 'transformers (in-process)'}"
          + (f"; {len(llm._cache)} cached replies in {cache}" if cache else "")
          + ("; DRY RUN, nothing is sent" if a.dry_run else ""))
    if not llm.is_local:
        print("[llm] remote service: only public datasets will be sent; credentialed data is refused")
    return llm


def log_compare(a, row: dict):
    # one summary row per (model, task) run, appended to <out>/llm_compare.csv
    path = Path(a.out) / "llm_compare.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "run": RUN.get("id", ""), "note": RUN.get("note", ""), **row}
    old = pd.read_csv(path) if path.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    print(f"[compare] summary appended to {path}")


def get_pubmed(a):
    if not a.pubmed:
        return None
    from .pubmed import PubMed
    pm = PubMed(a.pubmed_cache)
    print(f"[pubmed] NCBI E-utilities, cache {a.pubmed_cache}"
          + ("" if pm.api_key else " (set NCBI_API_KEY for faster requests)"))
    return pm


def gather_literature(a, pm, r: LRRanker, ext, c: Case, ranking, ex, k: int = 10) -> dict:
    # verified PubMed sentences for the top-k diseases; case paper and (optionally) later papers excluded
    from .pubmed import case_year, literature_claims, verify_lit
    maxdate = None
    if a.pubmed_before_case:
        y = case_year(pm, c)
        maxdate = f"{y - 1}/12/31" if y else None
    out = {}
    for d, _, _ in ranking.top(k):
        name = r.db.names[d]
        out[d] = [lc for lc in literature_claims(pm, r.ont, ext, c, d, name, ex, a.pubmed_per_disease, maxdate=maxdate)
                  if verify_lit(pm, r.ont, ext, c, lc, name, ex)[0]]
    pm.save()
    return out


def load_kb(a, prefixes=("OMIM",)):
    kb = Path(a.kb)
    ont = Ontology(kb / "hp.obo")
    genes = kb / "genes_to_disease.txt"
    db = AnnotationDB(kb / "phenotype.hpoa", ont, prefixes, genes if genes.exists() else None)
    n_rare, rare_mass, a.profiles = len(db.names), 0.04, None
    if a.common and Path(a.common).exists():
        a.profiles = json.loads(Path(a.common).read_text())
        db.add_common(a.profiles)
        if a.profiles["n_rare_admissions"]:
            rare_mass = a.profiles["n_rare_admissions"] / a.profiles["n_admissions"]
    RUN["settings"].update({"hpo": ont.version, "annotations": db.version, "prefixes": list(prefixes),
                            "common": a.profiles["source"] if a.profiles else None, "prior_weight": a.prior_weight})
    print(f"[kb] HPO {ont.version}, annotations {db.version}: {n_rare} rare + {len(db.names) - n_rare} common diseases"
          + (f" (common from {a.profiles['source']}, rare prior mass {rare_mass:.1%})" if a.profiles else ""))
    return LRRanker(ont, db, prior_weight=a.prior_weight, rare_mass=rare_mass)


def label(r: LRRanker, d: str) -> str:
    tag = f" [common, {r.db.prior[d]:.1%} of admissions]" if d in r.db.prior else ""
    return f"{r.db.names[d]} ({d}){tag}"


def leak(c: Case) -> frozenset[str]:
    return frozenset(c.source_pmids)


def evaluable(r: LRRanker, c: Case) -> bool:
    # gold disease still has annotations after removing the case's own paper
    return any(r.db.annotations(t, leak(c)) for t in c.truth if t in r.db.names)


def pick_cases(r: LRRanker, root, n, per_disease=5, seed=13) -> list[Case]:
    cases = [c for c in load_phenopackets(root) if c.present and any(t in r.db.names for t in c.truth)]
    random.Random(seed).shuffle(cases)
    count, out = defaultdict(int), []
    for c in cases:                       # cap cases per disease so a few diseases can't dominate
        count[c.truth[0]] += 1
        if count[c.truth[0]] <= per_disease:
            out.append(c)
    return out[:n] if n else out


def topk(ranks) -> dict:
    n = len(ranks)
    m = {f"top{k}": sum(1 for x in ranks if x and x <= k) / n for k in (1, 3, 10)} if n else {}
    m["mrr"] = sum(1 / x for x in ranks if x) / n if n else 0
    return m


def fmt(m: dict) -> str:
    return " | ".join(f"{m[k]:.3f}" for k in ("top1", "top3", "top10", "mrr"))


# commands
def read_case(a, r: LRRanker) -> tuple[Case, frozenset[str]]:
    # a hoofbeats case file (intake format) or a phenopacket; case files have no PMIDs to exclude
    from .intake import IntakeError, intake_summary, is_intake, load_case
    path = Path(a.case)
    try:
        obj = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        obj = None
    if is_intake(obj):
        try:
            c = load_case(path, r.ont, Extractor(r.ont), disease_names=r.db.names)
        except IntakeError as e:
            raise SystemExit(str(e))
        print(intake_summary(c, r.ont))
        note("intake", case=c.case_id, findings=len(c.findings), warnings=len(c.meta["intake_warnings"]),
             subsumed=len(c.meta["provenance"]["subsumed_by"]))
        return c, frozenset()
    found = load_phenopackets(a.case) if path.exists() else []
    if not found:
        raise SystemExit(f"no phenopacket or hoofbeats case found at {a.case}")
    return found[0], leak(found[0])


def cmd_diagnose(a):
    r = load_kb(a)
    c, ex = read_case(a, r)
    ranking = r.score(c, ex)
    print(f"\n{c.case_id}: truth {c.truth}, rank {ranking.rank_of(c.truth)}\n")
    n_ok = n_all = 0
    for i, (d, p, claims) in enumerate(explain(r, c, ranking, a.k, ex), 1):
        print(f"{i}. {label(r, d)} posterior {p:.1%}")
        for cl in claims:
            n_all += 1
            n_ok += verify_claim(r, c, cl, ex)[0]
            print("   -", cl.text)
    note("diagnose", case=c.case_id, truth=c.truth[0] if c.truth else None, rank=ranking.rank_of(c.truth),
         claims=n_all, claims_verified=n_ok)
    print(f"\n{n_ok}/{n_all} claims verified against the record and knowledge base\n\nNext checks:")
    for t, eig, by in suggest_tests(r, ranking, c, ex):
        print(f"   - {r.ont.name(t)} ({t}), {eig:.2f} bits; " +
              ", ".join(f"{r.db.names[d][:30]} {f:.0%}" for d, f in by[:3]))
    pm, lit = get_pubmed(a), {}
    if pm:
        lit = gather_literature(a, pm, r, Extractor(r.ont), c, ranking, ex)
        print(f"\nPubMed evidence (verified sentences from abstracts, {pm.calls} new requests):")
        for i, (d, _, _) in enumerate(ranking.top(a.k), 1):
            print(f"{i}. {r.db.names[d]} ({d})")
            for lc in lit.get(d, []):
                print(f"   - {lc.text}")
            if not lit.get(d):
                print("   - none found")
    llm = get_llm(a)
    if llm:
        from .llm import rerank
        print(f"\n[llm] asking {llm.model} to re-rank the top 10 (can take a minute for thinking models)...", flush=True)
        res = rerank(llm, r, c, ranking, ex, k=10, literature=lit)
        if not res.parsed:
            print(f"\n[llm] could not read MedGemma's reply; it began:\n{res.raw[:1500]}\n")
        print(usage_line(llm))
        pos = next((i for i, d in enumerate(res.ranking, 1) if d in c.truth), None)
        note("diagnose", case=c.case_id, rank=ranking.rank_of(c.truth), rerank=pos,
             faithfulness=res.faithfulness, parsed=res.parsed)
        where = f"truth now at {pos or '>10'}" if c.truth else "no truth given"
        print(f"\n{llm.model} re-ranking ({where}), faithfulness {res.faithfulness:.0%}:")
        for i, d in enumerate(res.ranking[: a.k], 1):
            print(f"{i}. {label(r, d)}")
            for sent in res.explanations.get(d, []):
                print("   +", sent)
            for sent in res.dropped.get(d, []):
                print("   x", sent)


def cmd_benchmark(a):
    r = load_kb(a)
    cases = pick_cases(r, a.phenopackets, a.sample, seed=a.seed)
    ev = [evaluable(r, c) for c in cases]
    no_absent = lambda c: replace(c, findings=[f for f in c.findings if f.status != Status.ABSENT])
    runs = {"present only": (no_absent, True, False), "present + absent": (lambda c: c, True, False),
            "present + absent, NO leakage control": (lambda c: c, False, False),
            "present + absent + gene": (lambda c: c, True, True)}
    if r.db.prior:   # common diseases loaded: also show ranking without prevalence weighting
        runs["present + absent, evidence only (no prevalence)"] = (lambda c: c, True, False)
    rows, scores = [], {}
    print(f"[data] {len(cases)} cases, {len({c.truth[0] for c in cases})} diseases, {sum(ev)} evaluable\n")
    note("data", cases=len(cases), diseases=len({c.truth[0] for c in cases}), evaluable=int(sum(ev)))
    print("| condition | Top-1 | Top-3 | Top-10 | MRR | evaluable Top-1 |\n|---|---|---|---|---|---|")
    for name, (prep, control, genes) in runs.items():
        r.use_genes, ranks = genes, []
        r.prior_weight = 0.0 if "no prevalence" in name else a.prior_weight
        for c in cases:
            rk = r.score(prep(c), leak(c) if control else frozenset())
            ranks.append(rk.rank_of(c.truth))
            scores.setdefault(name, []).append(rk.log_scores)
            rows.append({"condition": name, "case": c.case_id, "truth": c.truth[0], "rank": ranks[-1], "evaluable": ev[len(ranks) - 1]})
        print(f"| {name} | {fmt(topk(ranks))} | {topk([x for x, e in zip(ranks, ev) if e])['top1']:.3f} |")
        note("conditions", **{name: {**topk(ranks), "evaluable": topk([x for x, e in zip(ranks, ev) if e])}})
    r.use_genes, r.prior_weight = False, a.prior_weight

    print("\nWeight on absent evidence (evaluable cases):\n| weight | Top-1 | Top-10 |\n|---|---|---|")
    gold = [np.array([d in c.truth for d in r.diseases]) for c in cases]
    for w in (0, 0.25, 0.5, 1):
        ranks = []
        for sp, sa, g, e in zip(scores["present only"], scores["present + absent"], gold, ev):
            if e:
                s = sp + w * (sa - sp)
                ranks.append(int((s[~g] >= s[g].max()).sum()) + 1)
        m = topk(ranks)
        print(f"| {w} | {m['top1']:.3f} | {m['top10']:.3f} |")
        note("absent_weight", **{str(w): m})

    # calibration: fit temperature on half the diseases, measure on the other half
    dis = sorted({c.truth[0] for c in cases})
    random.Random(a.seed).shuffle(dis)
    dev = set(dis[: len(dis) // 2])
    S, G = scores["present + absent"], [int(np.argmax(g)) for g in gold]
    T = r.calibrate([s for s, c in zip(S, cases) if c.truth[0] in dev], [g for g, c in zip(G, cases) if c.truth[0] in dev])
    conf, hit = [], []
    for s, g, c in zip(S, G, cases):
        if c.truth[0] not in dev:
            p = np.exp((s - s.max()) / T)
            conf.append(p.max() / p.sum())
            hit.append(float(s[g] >= s.max()))
    conf, hit = np.array(conf), np.array(hit)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(conf[bins == b].mean() - hit[bins == b].mean()) * (bins == b).mean() for b in set(bins))
    rel = [{"bin": int(b), "confidence": float(conf[bins == b].mean()), "accuracy": float(hit[bins == b].mean()),
            "n": int((bins == b).sum())} for b in sorted(set(bins))]
    note("calibration", temperature=T, ece=float(ece), reliability=rel,
         top1_confident_half=float(hit[np.argsort(-conf)[: len(conf) // 2]].mean()))
    print(f"\nCalibration on held-out diseases: temperature {T:.2f}, ECE {ece:.3f}, "
          f"Top-1 when answering the 50% most confident: {hit[np.argsort(-conf)[: len(conf) // 2]].mean():.3f}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(wrote(Path(a.out) / "benchmark.csv"), index=False)

    llm = get_llm(a)
    if llm:
        llm_benchmark(a, r, llm, [c for c, e in zip(cases, ev) if e][: a.llm_cases])


def cmd_llm_bench(a):
    # only the model comparison: same evaluable cases for every model, no ranker ablations
    a.llm = True
    r = load_kb(a)
    cases = [c for c in pick_cases(r, a.phenopackets, a.sample, seed=a.seed) if evaluable(r, c)]
    print(f"[data] {len(cases)} evaluable cases; sending the first {min(a.llm_cases, len(cases))} to the model")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    llm_benchmark(a, r, get_llm(a), cases[: a.llm_cases])


def llm_benchmark(a, r: LRRanker, llm, sub: list[Case]):
    # same cases for every model: ranker alone, ranker + LLM re-rank, LLM alone (no ranker, no evidence)
    from .llm import QuotaExhausted, direct_diagnose, rerank
    pm = get_pubmed(a)
    ext = Extractor(r.ont) if pm else None
    matcher = DiseaseMatcher(r.db.names)
    out, failed, streak, stopped = [], 0, 0, ""
    for i, c in enumerate(sub, 1):
        rk = r.score(c, leak(c))
        shortlist = [d for d, _, _ in rk.top(10)]
        pos = lambda order: next((j for j, d in enumerate(order, 1) if d in c.truth), None)
        row = {"case": c.case_id, "truth": c.truth[0], "ranker": pos(shortlist)}
        try:
            if row["ranker"] is None and not a.rerank_all:   # re-ranking cannot bring the truth into the top 10
                row.update({"rerank": None, "rerank_skipped": True})
            else:
                lit = gather_literature(a, pm, r, ext, c, rk, leak(c)) if pm else {}
                res = rerank(llm, r, c, rk, leak(c), k=10, literature=lit)
                row.update({"rerank": pos(res.ranking), "rerank_parsed": res.parsed, "faithfulness": res.faithfulness,
                            "pubmed_sentences": sum(map(len, lit.values())), "rerank_skipped": False})
            if not a.skip_direct:
                slots, ok, _ = direct_diagnose(llm, r, c, matcher, k=10)
                row.update({"direct": next((j for j, s in enumerate(slots, 1) if s & set(c.truth)), None),
                            "direct_parsed": ok, "direct_mapped": sum(bool(s) for s in slots) / max(len(slots), 1)})
            streak = 0
        except QuotaExhausted as e:    # quota gone: keep what is done, stop asking
            stopped = str(e)
            print(f"\n[llm] stopping early: {e}")
            break
        except RuntimeError as e:      # a failed call is counted, not fatal
            failed, streak = failed + 1, streak + 1
            print(f"\n[llm] {c.case_id}: {e}")
            if streak == 3:
                stopped = "3 failed cases in a row"
                print(f"\n[llm] stopping early: {stopped}")
                break
        out.append(row)
        print(f"\r[llm] {i}/{len(sub)}", end="", flush=True)
    if llm.dry_run:
        print("\n" + usage_line(llm))
        return
    out = [x for x in out if "rerank" in x]      # only cases the model actually answered
    if not out:
        raise SystemExit(f"no case was answered by {llm.model}" + (f": {stopped}" if stopped else ""))
    df = pd.DataFrame(out)
    path = Path(a.out) / f"llm_rerank_{llm.slug}.csv"
    df.to_csv(wrote(path), index=False)
    print(f"\n\n{llm.model} on {len(df)} evaluable cases (rows in {path}):")
    print("| ranking | Top-1 | Top-3 | Top-10 | MRR |\n|---|---|---|---|---|")
    summary = {"model": llm.model, "task": "phenopacket", "cases": len(df), "failed_calls": failed,
               "stopped_early": stopped, "sec_per_call": llm.usage()["sec_per_request"],
               "requests": llm.requests, "cache_hits": llm.cache_hits,
               "reranked": int((~df.rerank_skipped.fillna(False).astype(bool)).sum())}
    for col, label in (("ranker", "ranker alone"), ("rerank", "ranker + LLM re-rank"), ("direct", "LLM alone")):
        if col not in df:
            continue
        m = topk([None if pd.isna(x) else int(x) for x in df[col]])
        print(f"| {label} | {fmt(m)} |")
        summary.update({f"{col}_{k}": round(v, 3) for k, v in m.items()})
    if "rerank_parsed" in df:
        asked = df[~df.rerank_skipped.fillna(False).astype(bool)]
        summary["rerank_parse_rate"] = round(asked.rerank_parsed.fillna(False).astype(bool).mean(), 3) if len(asked) else None
        summary["faithfulness"] = round(df.faithfulness.mean(), 3)
        print(f"Re-ranked {summary['reranked']} cases (the others had the truth outside the top 10); replies parsed: "
              f"{summary['rerank_parse_rate'] or 0:.0%}; "
              f"explanation sentences kept after checking: {summary['faithfulness']:.0%}")
    if "direct_parsed" in df:
        summary["direct_parse_rate"] = round(df.direct_parsed.fillna(False).mean(), 3)
        summary["direct_mapped"] = round(df.direct_mapped.mean(), 3)
        print(f"LLM-alone answers matched to a knowledge-base disease: {summary['direct_mapped']:.0%}")
    print(usage_line(llm))
    note("llm", **summary)
    log_compare(a, summary)


def simulate(r: LRRanker, c: Case, policy: str, rounds: int, seed: int):
    # hide half the findings, ask one HPO term per round, answer from the record
    rng = random.Random(f"{seed}:{c.case_id}")
    present = list(c.present)
    rng.shuffle(present)
    k = max(1, math.ceil(len(present) / 2))
    yes = set().union(*(r.ont.ancestors(t) for t in present[k:] if r.ont.resolve(t)))
    no = set().union(*(r.ont.descendants(t) for t in c.absent if r.ont.resolve(t)))
    cur = Case(c.case_id, [Finding(t, Status.PRESENT) for t in present[:k]], c.truth)
    ex, asked, answered = leak(c), set(), 0
    rk = r.score(cur, ex)
    ranks = [rk.rank_of(c.truth)]
    for _ in range(rounds):
        if policy == "eig":
            s = suggest_tests(r, rk, cur, ex, asked, n=1)
            q = s[0][0] if s else None
        elif policy == "frequent":
            q = most_frequent_test(r, rk, cur, ex, asked)
        else:
            s = suggest_tests(r, rk, cur, ex, asked, n=10_000)
            q = rng.choice(s)[0] if s else None
        if q:
            asked.add(q)
            ans = Status.PRESENT if q in yes else Status.ABSENT if q in no else None
            if ans:
                answered += 1
                cur.findings.append(Finding(q, ans, "oracle"))
                rk = r.score(cur, ex)
        ranks.append(rk.rank_of(c.truth))
    return ranks, answered


def cmd_sequential(a):
    r = load_kb(a)
    cases = [c for c in pick_cases(r, a.phenopackets, None, seed=a.seed)
             if len(c.present) >= 2 and evaluable(r, c)][: a.sample]
    print(f"[data] {len(cases)} evaluable cases, {a.rounds} questions each\n")
    print("| policy | Top-1 start | Top-1 end | Top-10 start | Top-10 end | answered |\n|---|---|---|---|---|---|")
    rows = []
    for pol in ("eig", "frequent", "random"):
        res = [simulate(r, c, pol, a.rounds, a.seed) for c in cases]
        s, e = topk([x[0][0] for x in res]), topk([x[0][-1] for x in res])
        rate = sum(x[1] for x in res) / (len(res) * a.rounds)
        print(f"| {pol} | {s['top1']:.3f} | {e['top1']:.3f} | {s['top10']:.3f} | {e['top10']:.3f} | {rate:.0%} |")
        note("sequential", **{pol: {"start": s, "end": e, "answered": rate}})
        rows += [{"policy": pol, "case": c.case_id, "truth": c.truth[0], "round": i, "rank": rk, "answered": ans}
                 for c, (ranks, ans) in zip(cases, res) for i, rk in enumerate(ranks)]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(wrote(Path(a.out) / "sequential.csv"), index=False)


def cmd_learn_common(a):
    icd = orphadata_icd10(a.orphadata_xml)[0] if a.orphadata_xml else None
    prof = learn_common(a.mimic, icd, min_adm=a.min_adm, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(prof))
    note("common", diseases=len(prof["diseases"]), admissions=prof["n_admissions"],
         rare_admissions=prof["n_rare_admissions"], held_out_patients=len(prof["test_subjects"]), source=prof["source"])
    top = sorted(prof["diseases"].items(), key=lambda x: -x[1]["n_adm"])[:8]
    print(f"[common] {len(prof['diseases'])} diseases with >= {a.min_adm} training admissions "
          f"({prof['n_admissions']} admissions, {len(prof['test_subjects'])} patients held out) -> {a.out}")
    for cat, p in top:
        print(f"   {cat}  {p['name'][:60]}  ({p['n_adm']} admissions)")
    if prof["policy"] != "open":
        print("   credentialed-derived file: keep it private, never commit or upload it")


def cmd_mimic(a):
    r = load_kb(a, ("ORPHA",))
    icd, names = orphadata_icd10(a.orphadata_xml) if a.orphadata_xml else (None, None)
    ext = Extractor(r.ont) if a.notes else None
    llm, fn = get_llm(a), None
    if llm and ext:
        from .llm import extract_findings
        fn = lambda text, policy: extract_findings(llm, ext, Case("note", [], meta={"policy": policy}), text)[0]
    held_out = set(a.profiles["test_subjects"]) if a.profiles else None   # never test on training patients
    cases = load_mimic(a.mimic, icd, names, r.db.disease_genes, a.notes, ext, labeled_only=not a.all,
                       extract_fn=fn, common=a.profiles, subjects=held_out)
    rows = []
    for c in cases:
        rk = r.score(c)
        rows.append({"case": c.case_id, "truth": ";".join(c.truth), "rank": rk.rank_of(c.truth),
                     "rare_truth": any(t.startswith("ORPHA") for t in c.truth),
                     "top3": "; ".join(f"{d} {r.db.names[d][:40]}" for d, _, _ in rk.top(3)),
                     "n_present": len(c.present), "n_absent": len(c.absent), "policy": c.meta["policy"],
                     "delabel_residual": ";".join(c.meta.get("delabel", {}).get("residual", []))})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(wrote(Path(a.out) / "mimic.csv"), index=False)
    print(f"[mimic] {len(cases)} admissions" + (" from held-out patients" if held_out is not None else "")
          + f", policy {cases[0].meta['policy'] if cases else '-'}")
    for name, sel in (("rare diagnosis", lambda x: x["rare_truth"]),
                      ("common diagnosis", lambda x: x["truth"] and not x["rare_truth"])):
        ranks = [x["rank"] for x in rows if sel(x)]
        if ranks:
            note("mimic", **{name: {"n": len(ranks), **topk(ranks)}})
            print(f"   {name}: n={len(ranks)}, Top-1 {topk(ranks)['top1']:.3f}, Top-3 {topk(ranks)['top3']:.3f}, "
                  f"Top-10 {topk(ranks)['top10']:.3f}")


def cmd_extract_eval(a):
    # reader outputs vs gold findings, per case file and reader
    from .extract_eval import TEXT_READERS, evaluate, report, summarize
    ont = Ontology(Path(a.kb) / "hp.obo")
    names = AnnotationDB(Path(a.kb) / "phenotype.hpoa", ont).names     # same disease names as diagnose (OMIM)
    if a.pmc:
        return pmc_extract_eval(a, ont, names)
    RUN["settings"].update({"hpo": ont.version, "cases": a.cases, "gold": a.gold})
    rows, problems = evaluate(a.cases, a.gold, ont, Extractor(ont), names)
    print(f"[extract-eval] {len({r['case'] for r in rows})} cases from {a.cases}, gold {a.gold}\n")
    print(report(rows, ont, a.details))
    for p in problems:
        print("[gold]", p)
    for name, sel in (("text", [r for r in rows if r["reader"] in TEXT_READERS]), ("all", rows)):
        s = summarize(sel)
        note("extract_eval", **{name: {k: v for k, v in s.items()}})
    items = [{"case": r["case"], "reader": r["reader"], "kind": "gold", "hpo_id": it["hpo_id"], "label": it["label"],
              "status": it["status"], "result": it["result"], "got": ";".join(it["got"]), "quote": it.get("quote"),
              "span_overlap": it["span_overlap"]} for r in rows for it in r["items"]]
    items += [{"case": r["case"], "reader": r["reader"], "kind": "extra", "hpo_id": p["hpo_id"], "label": ont.name(p["hpo_id"]),
               "status": p["status"], "result": "extra", "quote": p["evidence"]} for r in rows for p in r["extras"]]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(items).to_csv(wrote(Path(a.out) / "extract_eval.csv"), index=False)


def pmc_extract_eval(a, ont, names):
    # the note reader on the frozen PMC manifest; the test split only with --allow-test, every run logged with its commit
    import subprocess
    from .extract_eval import evaluate_pmc, pmc_report, pmc_summary
    if a.split == "test" and not a.allow_test:
        raise SystemExit("the PMC test split is run only at agreed milestones; add --allow-test")
    manifest = json.loads(Path(a.pmc).read_text())
    RUN["settings"].update({"hpo": ont.version, "pmc_manifest": a.pmc, "locator_version": manifest["locator_version"],
                            "split": a.split})
    rows = evaluate_pmc(manifest, ont, Extractor(ont), names, a.split)
    summ = pmc_summary(rows)
    print(f"[extract-eval] PMC manifest v{manifest['locator_version']} ({a.pmc}), split {a.split}\n")
    print(pmc_report(summ))
    note("extract_eval_pmc", **summ)
    items = [{"case": r["case"], "kind": "gold", "hpo_id": it["hpo_id"], "label": it["label"], "status": it["status"],
              "result": it["result"], "got": ";".join(it["got"]), "vocab_visible": it["vocab_visible"], "method": "",
              "quote": "", "evidence": ""} for r in rows for it in r["items"]]
    items += [{"case": r["case"], "kind": "extra", "hpo_id": p["hpo_id"], "label": ont.name(p["hpo_id"]), "status": p["status"],
               "result": "extra", "got": "", "vocab_visible": "", "method": p["method"], "quote": p.get("quote", ""),
               "evidence": p.get("sentence") or p["evidence"]} for r in rows for p in r["extras"]]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(items).to_csv(wrote(out / f"pmc_{a.split}_items.csv"), index=False)
    (out / f"pmc_{a.split}_summary.json").write_text(json.dumps(summ, indent=1))
    wrote(out / f"pmc_{a.split}_summary.json")
    git = lambda *x: subprocess.run(["git", *x], capture_output=True, text=True).stdout.strip()
    with open(Path(a.pmc).parent / "runs.jsonl", "a") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": git("rev-parse", "HEAD"),
                             "dirty": bool(git("status", "--porcelain", "hoofbeats")), "manifest": a.pmc,
                             "locator_version": manifest["locator_version"], "split": a.split,
                             "recall_all_gold": summ["recall_all_gold"], "recall_vocab_visible": summ["recall_vocab_visible"],
                             "precision": summ["precision"], "pred": summ["pred"], "extra": summ["extra"]}) + "\n")


def cmd_pmc_build(a):
    # fetch (cached), locate and freeze the PMC manifest; refuses uncommitted locator code
    from .pmc import LOCATOR_VERSION, build_manifest
    ont = Ontology(Path(a.kb) / "hp.obo")
    manifest, excluded = build_manifest(a.phenopackets, ont)
    out = Path(a.dest)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"manifest_v{LOCATOR_VERSION}.json"
    if path.exists() and not a.force:
        raise SystemExit(f"{path} exists (frozen); bump LOCATOR_VERSION to rebuild, or --force before it is used")
    path.write_text(json.dumps(manifest, indent=1))
    pd.DataFrame(excluded).to_csv(wrote(out / f"excluded_v{LOCATOR_VERSION}.csv"), index=False)
    print(json.dumps(manifest["counts"], indent=1))
    note("pmc_build", **manifest["counts"])
    print(f"[pmc] manifest: {wrote(path)}")


def cmd_pmc_checks(a):
    # pre-filled hand-check sheets: A and B from the manifest; C from an extract-eval items file (dev run)
    from .pmc_checks import make_task_a, make_task_b, make_task_c
    manifest = json.loads(Path(a.manifest).read_text())
    out = Path(a.dest)
    made = [] if a.items else [make_task_a(manifest, out), *make_task_b(manifest, out)]
    if a.items:
        made.append(make_task_c(a.items, out))
    for f in made:
        print(f"[pmc] wrote {wrote(f)}")


def cmd_pmc_score(a):
    from .pmc_checks import score
    res = score(Path(a.checks), a.items, a.summary)
    print(json.dumps(res, indent=1))
    note("pmc_score", **res)


def image_agreement(r: LRRanker, predicted: set[str], reference: set[str]) -> tuple[float | None, float | None]:
    # precision/recall of image findings vs radiologist (ancestor matches count)
    agree = lambda a, b: a in r.ont.ancestors(b) or b in r.ont.ancestors(a)
    p = sum(any(agree(x, y) for y in reference) for x in predicted) / len(predicted) if predicted else None
    rc = sum(any(agree(y, x) for x in predicted) for y in reference) / len(reference) if reference else None
    return p, rc


def cmd_medpix(a):
    r = load_kb(a, ("OMIM", "ORPHA"))
    ext = Extractor(r.ont)
    syn = orphadata_icd10(a.orphadata_xml)[1] if a.orphadata_xml else {}
    cases = load_medpix(a.medpix, DiseaseMatcher(r.db.names, syn), ext, a.split, a.labels)
    if a.only_labeled:
        cases = [c for c in cases if c.truth]
    if a.limit:
        cases = cases[: a.limit]
    llm = get_llm(a)
    if llm and not llm.sees_images:
        raise SystemExit(f"{llm.model} cannot read images; pick a vision model or pass --llm-vision yes")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # image reads are cached per model so different models never share answers
    cache_path = Path(a.cache or out / f"medpix_image_reads_{llm.slug if llm else 'none'}.jsonl")
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line); cache[rec["image"]] = rec["raw"]
    n_cached, failed = len(cache), 0

    def save_cache():
        with open(cache_path, "w") as fh:
            for k, v in cache.items():
                fh.write(json.dumps({"image": k, "raw": v}) + "\n")

    rows = []
    for i, c in enumerate(cases, 1):
        text = [f for f in c.findings]
        ref = [f for f in ext.findings(c.meta["reference_text"], "medpix:radiologist") if f.status == Status.PRESENT]
        have = {f.hpo_id for f in text}
        oracle = Case(c.case_id, text + [f for f in ref if f.hpo_id not in have], c.truth)
        row = {"case": c.case_id, "title": c.meta["title"], "truth": ";".join(c.truth),
               "rank_text": r.score(c).rank_of(c.truth) if c.truth else None,
               "rank_radiologist": r.score(oracle).rank_of(c.truth) if c.truth else None,
               "n_text": len(text), "n_radiologist": len(ref), "n_images": len(c.meta["images"])}
        if llm:
            from .llm import read_images
            from .llm import QuotaExhausted
            try:
                img, recs = read_images(llm, ext, c, a.max_images, cache, a.images_per_call)
            except QuotaExhausted as e:   # quota gone: keep finished cases, stop
                print(f"\n[llm] stopping early: {e}")
                break
            except RuntimeError as e:     # a failed call is counted, not fatal
                failed += 1
                print(f"\n[llm] {c.case_id}: {e}")
                rows.append(row)
                continue
            both = Case(c.case_id, text + [f for f in img if f.hpo_id not in have], c.truth)
            p, rc = image_agreement(r, {f.hpo_id for f in img if f.status == Status.PRESENT}, {f.hpo_id for f in ref})
            row.update({"rank_image": r.score(Case(c.case_id, img, c.truth)).rank_of(c.truth) if c.truth else None,
                        "rank_multimodal": r.score(both).rank_of(c.truth) if c.truth else None,
                        "n_image": len(img), "image_precision": p, "image_recall": rc,
                        "unparsed_images": sum(not x["parsed"] for x in recs),
                        "report": " | ".join(x["report"] for x in recs)[:1000]})
            print(f"\r[medpix] {i}/{len(cases)} cases read", end="", flush=True)
            if i % 10 == 0:
                save_cache()
        rows.append(row)
    if llm and llm.dry_run:
        print("\n" + usage_line(llm))
        return
    if llm:
        print()
        save_cache()   # raw replies: reruns reuse them instead of calling the model
        print(f"[cache] {len(cache) - n_cached} new image reads, {len(cache)} total -> {cache_path}")
    df = pd.DataFrame(rows)
    df.to_csv(wrote(out / (f"medpix_{llm.slug}.csv" if llm else "medpix.csv")), index=False)
    df[["case", "title", "truth"]].to_csv(out / "medpix_labels.csv", index=False,
                                          header=["U_id", "title", "truth"])
    lab = df[df.truth != ""]
    print(f"[medpix] {len(df)} cases ({a.split}), {len(lab)} with a diagnosis the ranker knows; "
          f"median findings from history/exam {df.n_text.median():.0f}, from radiologist report {df.n_radiologist.median():.0f}")
    print("\n| input | Top-1 | Top-3 | Top-10 | MRR |\n|---|---|---|---|---|")
    conds = [("history + exam text", "rank_text"), ("text + radiologist image findings (upper bound)", "rank_radiologist")]
    if llm:
        conds += [("LLM image findings only", "rank_image"), ("text + LLM image findings", "rank_multimodal")]
    summary = {"model": llm.model if llm else "none", "task": "medpix", "cases": len(df), "labeled": len(lab),
               "failed_calls": failed}
    for name, col in conds:
        if col not in lab:
            continue
        m = topk([None if pd.isna(x) else int(x) for x in lab[col]])
        print(f"| {name} | {fmt(m)} |")
        summary.update({f"{col}_{k}": round(v, 3) for k, v in m.items()})
    note("medpix", **summary)
    if llm:
        summary.update({"image_precision": round(df.image_precision.mean(), 3), "image_recall": round(df.image_recall.mean(), 3),
                        "unparsed_images": int(df.unparsed_images.fillna(0).sum()),
                        "sec_per_call": llm.usage()["sec_per_request"], "requests": llm.requests,
                        "cache_hits": llm.cache_hits})
        print(f"\nLLM image findings vs radiologist: precision {summary['image_precision']:.1%}, "
              f"recall {summary['image_recall']:.1%}; unparsed replies {summary['unparsed_images']}; "
              f"{summary['sec_per_call']} s per request")
        print(usage_line(llm))
        note("medpix", **summary)
        log_compare(a, summary)
    print(f"\nReview and correct diagnosis labels in {out / 'medpix_labels.csv'} (pass back with --labels).")


def main(argv=None):
    p = argparse.ArgumentParser(prog="hoofbeats")
    p.add_argument("--kb", default="data/kb", help="folder with hp.obo, phenotype.hpoa, genes_to_disease.txt")
    p.add_argument("--note", default="", help="free-text note saved with this run, shown in reports")
    p.add_argument("--common", default="data/kb/common_profiles.json",
                   help="common-disease profiles from learn-common; used whenever the file exists")
    p.add_argument("--prior-weight", type=float, default=1.0, help="0 = ignore how common each disease is")
    p.add_argument("--pubmed", action="store_true", help="add verified PubMed abstract sentences as evidence")
    p.add_argument("--pubmed-cache", default="data/kb/pubmed_cache.json")
    p.add_argument("--pubmed-per-disease", type=int, default=3)
    p.add_argument("--pubmed-before-case", action="store_true",
                   help="only papers published before the case's own paper (stricter leakage control)")
    p.add_argument("--llm", action="store_true", help="use a language model (MedGemma by default; see README)")
    p.add_argument("--llm-model", default="google/medgemma-1.5-4b-it")
    p.add_argument("--llm-endpoint", help="OpenAI-compatible chat URL, e.g. http://localhost:11434/v1/chat/completions; "
                                          "omit to load the model in-process with transformers")
    p.add_argument("--llm-key-env", help="name of the environment variable holding an API key, e.g. GEMINI_API_KEY")
    p.add_argument("--llm-vision", default="auto", choices=["auto", "yes", "no"], help="whether the model reads images")
    p.add_argument("--llm-max-tokens", type=int, default=2048)
    p.add_argument("--llm-rpm", type=float, help="max model requests per minute (match your free-tier limit, e.g. 5)")
    p.add_argument("--llm-cache", default="auto", help="folder of saved model replies (default <out>/llm_cache; 'none' to disable)")
    p.add_argument("--dry-run", action="store_true", help="count the model requests a command would send, send none")
    p.add_argument("--llm-extra", help='extra JSON fields for the request body, e.g. \'{"reasoning_effort": "low"}\'')
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("download"); s.add_argument("--dest", default="data")
    s = sub.add_parser("diagnose"); s.add_argument("case", help="phenopacket JSON or hoofbeats case file (see examples/)")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--out", default="results", help="run log goes to <out>/runs/")
    for name in ("benchmark", "sequential", "llm-bench"):
        s = sub.add_parser(name)
        s.add_argument("--phenopackets", default="data/phenopacket-store")
        s.add_argument("--sample", type=int, default=500 if name == "benchmark" else 300)
        s.add_argument("--seed", type=int, default=13)
        s.add_argument("--rounds", type=int, default=5)
        s.add_argument("--out", default="results")
        if name in ("benchmark", "llm-bench"):
            s.add_argument("--llm-cases", type=int, default=50, help="evaluable cases sent to the model")
            s.add_argument("--skip-direct", action="store_true", help="skip the LLM-alone baseline")
            s.add_argument("--rerank-all", action="store_true",
                           help="also re-rank cases whose truth is outside the top 10 (cannot change accuracy)")
    s = sub.add_parser("learn-common")
    s.add_argument("--mimic", required=True, help="folder containing hosp/")
    s.add_argument("--orphadata-xml", help="so codes of rare diseases are not learned as common ones")
    s.add_argument("--min-adm", type=int, default=5, help="minimum training admissions per disease (use ~50 on full MIMIC)")
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--out", default="data/kb/common_profiles.json")
    s = sub.add_parser("mimic")
    s.add_argument("--mimic", required=True, help="folder containing hosp/")
    s.add_argument("--orphadata-xml", help="Orphadata en_product1.xml (ICD-10 -> ORPHA labels)")
    s.add_argument("--notes", help="folder with discharge.csv.gz (credentialed only)")
    s.add_argument("--all", action="store_true", help="include admissions without a rare-disease label")
    s.add_argument("--out", default="results")
    s = sub.add_parser("medpix")
    s.add_argument("--medpix", default="data/MedPix-2-0", help="MedPix-2-0 folder (Case_topic.json, Descriptions.json, images/)")
    s.add_argument("--split", default="all", choices=["all", "train", "train_1", "train_2", "dev", "test"])
    s.add_argument("--labels", help="CSV U_id,truth overriding automatic diagnosis matching")
    s.add_argument("--orphadata-xml", help="adds Orphanet synonyms for diagnosis matching")
    s.add_argument("--max-images", type=int, default=4, help="images per case sent to the model")
    s.add_argument("--images-per-call", type=int, default=1, help="images sent together in one request (fewer requests)")
    s.add_argument("--only-labeled", action="store_true", help="only cases whose diagnosis is in the knowledge base")
    s.add_argument("--limit", type=int, help="first N cases (quick runs)")
    s.add_argument("--cache", help="jsonl of raw image reads (default: <out>/medpix_image_reads_<model>.jsonl)")
    s.add_argument("--out", default="results")
    s = sub.add_parser("extract-eval", help="score reader outputs on case files against gold findings")
    s.add_argument("--cases", default="examples", help="folder with the case files named in the gold file")
    s.add_argument("--gold", default="examples/intended_findings.json")
    s.add_argument("--details", action="store_true", help="list every gold finding and extra")
    s.add_argument("--out", default="results/extract_eval")
    s.add_argument("--pmc", help="PMC manifest (pmc_eval/manifest_v1.json): score the note reader on its located text")
    s.add_argument("--split", default="dev", choices=["dev", "test"])
    s.add_argument("--allow-test", action="store_true", help="needed to run on the PMC test split")
    s = sub.add_parser("pmc-build", help="fetch, locate and freeze the PMC test-set manifest")
    s.add_argument("--phenopackets", default="data/phenopacket-store")
    s.add_argument("--dest", default="pmc_eval")
    s.add_argument("--force", action="store_true", help="overwrite an existing manifest of the same version")
    s = sub.add_parser("pmc-checks", help="write the pre-filled hand-check CSVs")
    s.add_argument("--manifest", default="pmc_eval/manifest_v1.json")
    s.add_argument("--items", help="extract-eval pmc_dev_items.csv: write task C (extras) instead of A and B")
    s.add_argument("--dest", default="data/pmc/checks")
    s = sub.add_parser("pmc-score", help="scores from the filled-in hand-check CSVs")
    s.add_argument("--checks", default="data/pmc/checks")
    s.add_argument("--items", help="extract-eval pmc_<split>_items.csv, for hand-checked recall")
    s.add_argument("--summary", help="extract-eval pmc_<split>_summary.json, for precision with adjudicated extras")
    s.add_argument("--out", default="results")
    s = sub.add_parser("report", help="tables, figures and notes from saved runs")
    s.add_argument("--results", default="results", help="folder searched (recursively) for runs and CSVs")
    s.add_argument("--out", default="reports/report", help="base name; each run writes <base>-1, <base>-2, ...")
    s.add_argument("--latex", action="store_true", help="also write LaTeX tables")
    a = p.parse_args(argv)
    if a.cmd == "download":
        download(a.dest)
    elif a.cmd == "report":
        from .report import build_report
        build_report(Path(a.results), Path(a.out), latex=a.latex)
    else:
        _logged(a)


def _clean_console(raw: str) -> str:
    # keep what the terminal finally showed: progress lines drawn with \r collapse to their last state
    lines = []
    for line in raw.split("\n"):
        parts = [x for x in line.split("\r") if x.strip()]
        lines.append(parts[-1] if parts else "")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def _write_txt(path: Path, body: str, trace: str):
    # readable log: header with settings, the console output, then status and metrics
    rule = "=" * 72
    st = RUN["settings"]
    head = [f"run       {RUN['id']}", f"started   {RUN['started']}", f"command   {RUN['command']}", f"note      {RUN['note'] or '-'}",
            f"status    {RUN['status']}", f"duration  {RUN['seconds'] / 60:.1f} min",
            f"argv      python -m hoofbeats {shlex.join(RUN['argv'])}", ""]
    head += [f"{k:<14}{v}" for k, v in st.items()]
    tail = ["", rule, "metrics (also in the .json file)", rule, json.dumps(RUN["metrics"], indent=1, default=str)]
    if RUN["files"]:
        tail += ["", "files written:"] + [f"  {f}" for f in RUN["files"]]
    if trace:
        tail += ["", rule, "error", rule, trace]
    path.write_text("\n".join(head + ["", rule, "console output", rule, "", body.rstrip()] + tail) + "\n")


def _logged(a):
    # run the command; save <out>/runs/<id>.txt (readable log) and <id>.json (settings + metrics)
    import traceback
    model = re.sub(r"[^A-Za-z0-9.]+", "-", a.llm_model.split("/")[-1]).lower() if a.llm or a.cmd == "llm-bench" else ""
    RUN.update({"id": time.strftime("%Y%m%d-%H%M%S") + f"_{a.cmd}" + (f"_{model}" if model else ""),
                "command": a.cmd, "argv": sys.argv[1:], "note": a.note, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    RUN["settings"].update({k: v for k, v in vars(a).items()
                            if k in ("sample", "seed", "llm_cases", "rounds", "split", "limit", "max_images", "only_labeled")})
    runs = _run_dir(a)
    runs.mkdir(parents=True, exist_ok=True)
    partial = runs / f"{RUN['id']}.partial.txt"     # raw output while running; replaced by the clean log at the end
    log = open(partial, "w")
    stdout, t0, status, trace = sys.stdout, time.time(), "ok", ""
    sys.stdout = _Tee(stdout, log)
    try:
        {"diagnose": cmd_diagnose, "benchmark": cmd_benchmark, "sequential": cmd_sequential,
         "learn-common": cmd_learn_common, "mimic": cmd_mimic, "medpix": cmd_medpix,
         "llm-bench": cmd_llm_bench, "extract-eval": cmd_extract_eval, "pmc-build": cmd_pmc_build,
         "pmc-checks": cmd_pmc_checks, "pmc-score": cmd_pmc_score}[a.cmd](a)
    except KeyboardInterrupt:
        status = "stopped (Ctrl-C)"
        raise
    except SystemExit as e:
        status = f"stopped: {e}"
        raise
    except Exception as e:
        if type(e).__name__ != "QuotaExhausted":
            status, trace = f"failed: {type(e).__name__}: {e}", traceback.format_exc()
            raise
        status = f"stopped: {e}"
        print(f"\n[llm] {e}")
        raise SystemExit(1) from None
    except BaseException as e:
        status, trace = f"failed: {type(e).__name__}: {e}", traceback.format_exc()
        raise
    finally:
        sys.stdout = stdout
        log.close()
        early = RUN["metrics"].get("llm", {}).get("stopped_early")
        if status == "ok" and early:
            status = f"partial (stopped early: {early})"
        RUN.update({"status": status, "seconds": round(time.time() - t0, 1)})
        (runs / f"{RUN['id']}.json").write_text(json.dumps(RUN, indent=1, default=str))
        txt = runs / f"{RUN['id']}.txt"
        _write_txt(txt, _clean_console(partial.read_text()), trace)
        partial.unlink()
        (runs / "latest.txt").write_text(txt.read_text())
        print(f"[run] log: {txt}  ({status})")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import json

from .data import (DiseaseMatcher, Extractor, download, learn_common, load_medpix, load_mimic,
                   load_phenopackets, orphadata_icd10)
from .model import (AnnotationDB, Case, Finding, LRRanker, Ontology, Status, explain,
                    most_frequent_test, suggest_tests, verify_claim)


# helpers
def get_llm(a):
    if not a.llm:
        return None
    from .llm import MedGemma
    llm = MedGemma(a.llm_model, a.llm_endpoint)
    print(f"[llm] {llm.model} via {llm.endpoint or 'transformers (in-process)'}")
    return llm


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
def cmd_diagnose(a):
    r = load_kb(a)
    c = load_phenopackets(a.case)[0]
    ex = leak(c)
    ranking = r.score(c, ex)
    print(f"\n{c.case_id}: truth {c.truth}, rank {ranking.rank_of(c.truth)}\n")
    n_ok = n_all = 0
    for i, (d, p, claims) in enumerate(explain(r, c, ranking, a.k, ex), 1):
        print(f"{i}. {label(r, d)} posterior {p:.1%}")
        for cl in claims:
            n_all += 1
            n_ok += verify_claim(r, c, cl, ex)[0]
            print("   -", cl.text)
    print(f"\n{n_ok}/{n_all} claims verified against the record and knowledge base\n\nNext checks:")
    for t, eig, by in suggest_tests(r, ranking, c, ex):
        print(f"   - {r.ont.name(t)} ({t}), {eig:.2f} bits; " +
              ", ".join(f"{r.db.names[d][:30]} {f:.0%}" for d, f in by[:3]))
    llm = get_llm(a)
    if llm:
        from .llm import rerank
        res = rerank(llm, r, c, ranking, ex, k=10)
        if not res.parsed:
            print(f"\n[llm] could not read MedGemma's reply; it began:\n{res.raw[:1500]}\n")
        print(f"\nMedGemma re-ranking (truth now at {res.ranking.index(c.truth[0]) + 1 if c.truth[0] in res.ranking else '>10'}), "
              f"faithfulness {res.faithfulness:.0%}:")
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
    print(f"\nCalibration on held-out diseases: temperature {T:.2f}, ECE {ece:.3f}, "
          f"Top-1 when answering the 50% most confident: {hit[np.argsort(-conf)[: len(conf) // 2]].mean():.3f}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(a.out) / "benchmark.csv", index=False)

    llm = get_llm(a)
    if llm:
        from .llm import rerank
        sub = [c for c, e in zip(cases, ev) if e][: a.llm_cases]
        out = []
        for i, c in enumerate(sub, 1):
            rk = r.score(c, leak(c))
            res = rerank(llm, r, c, rk, leak(c), k=10)
            shortlist = [d for d, _, _ in rk.top(10)]
            pos = lambda order: next((j for j, d in enumerate(order, 1) if d in c.truth), None)
            out.append({"case": c.case_id, "ranker": pos(shortlist), "medgemma": pos(res.ranking),
                        "faithfulness": res.faithfulness})
            print(f"\r[llm] {i}/{len(sub)}", end="", flush=True)
        df = pd.DataFrame(out)
        df.to_csv(Path(a.out) / "medgemma_rerank.csv", index=False)
        print(f"\n\nMedGemma re-ranking of the top-10 shortlist ({len(df)} evaluable cases):")
        print("| ranking | Top-1 | Top-3 | Top-10 |\n|---|---|---|---|")
        for col in ("ranker", "medgemma"):
            m = topk([None if pd.isna(x) else int(x) for x in df[col]])
            print(f"| {col} | {m['top1']:.3f} | {m['top3']:.3f} | {m['top10']:.3f} |")
        print(f"Claim-level faithfulness of MedGemma explanations: {df.faithfulness.mean():.1%}")


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
        rows += [{"policy": pol, "case": c.case_id, "truth": c.truth[0], "round": i, "rank": rk, "answered": ans}
                 for c, (ranks, ans) in zip(cases, res) for i, rk in enumerate(ranks)]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(a.out) / "sequential.csv", index=False)


def cmd_learn_common(a):
    icd = orphadata_icd10(a.orphadata_xml)[0] if a.orphadata_xml else None
    prof = learn_common(a.mimic, icd, min_adm=a.min_adm, seed=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(prof))
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
    pd.DataFrame(rows).to_csv(Path(a.out) / "mimic.csv", index=False)
    print(f"[mimic] {len(cases)} admissions" + (" from held-out patients" if held_out is not None else "")
          + f", policy {cases[0].meta['policy'] if cases else '-'}")
    for name, sel in (("rare diagnosis", lambda x: x["rare_truth"]),
                      ("common diagnosis", lambda x: x["truth"] and not x["rare_truth"])):
        ranks = [x["rank"] for x in rows if sel(x)]
        if ranks:
            print(f"   {name}: n={len(ranks)}, Top-1 {topk(ranks)['top1']:.3f}, Top-3 {topk(ranks)['top3']:.3f}, "
                  f"Top-10 {topk(ranks)['top10']:.3f}")


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
    llm = get_llm(a)
    if llm and not llm.sees_images:
        raise SystemExit(f"{llm.model} cannot read images; use --llm-model google/medgemma-1.5-4b-it")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache_path = Path(a.cache or out / "medpix_image_reads.jsonl")
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line); cache[rec["image"]] = rec["raw"]
    n_cached = len(cache)
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
            img, recs = read_images(llm, ext, c, a.max_images, cache)
            both = Case(c.case_id, text + [f for f in img if f.hpo_id not in have], c.truth)
            p, rc = image_agreement(r, {f.hpo_id for f in img if f.status == Status.PRESENT}, {f.hpo_id for f in ref})
            row.update({"rank_image": r.score(Case(c.case_id, img, c.truth)).rank_of(c.truth) if c.truth else None,
                        "rank_multimodal": r.score(both).rank_of(c.truth) if c.truth else None,
                        "n_image": len(img), "image_precision": p, "image_recall": rc,
                        "unparsed_images": sum(not x["parsed"] for x in recs),
                        "report": " | ".join(x["report"] for x in recs)[:1000]})
            print(f"\r[medpix] {i}/{len(cases)} cases read", end="", flush=True)
        rows.append(row)
    if llm:
        print()
        with open(cache_path, "w") as fh:   # raw MedGemma replies: reruns reuse them instead of calling the model
            for k, v in cache.items():
                fh.write(json.dumps({"image": k, "raw": v}) + "\n")
        print(f"[cache] {len(cache) - n_cached} new image reads, {len(cache)} total -> {cache_path}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "medpix.csv", index=False)
    df[["case", "title", "truth"]].to_csv(out / "medpix_labels.csv", index=False,
                                          header=["U_id", "title", "truth"])
    lab = df[df.truth != ""]
    print(f"[medpix] {len(df)} cases ({a.split}), {len(lab)} with a diagnosis the ranker knows; "
          f"median findings from history/exam {df.n_text.median():.0f}, from radiologist report {df.n_radiologist.median():.0f}")
    print("\n| input | Top-1 | Top-3 | Top-10 | MRR |\n|---|---|---|---|---|")
    conds = [("history + exam text", "rank_text"), ("text + radiologist image findings (upper bound)", "rank_radiologist")]
    if llm:
        conds += [("MedGemma image findings only", "rank_image"), ("text + MedGemma image findings", "rank_multimodal")]
    for name, col in conds:
        m = topk([None if pd.isna(x) else int(x) for x in lab[col]])
        print(f"| {name} | {fmt(m)} |")
    if llm:
        print(f"\nMedGemma image findings vs radiologist: precision {df.image_precision.mean():.1%}, "
              f"recall {df.image_recall.mean():.1%}; unparsed replies {int(df.unparsed_images.sum())}")
    print(f"\nReview and correct diagnosis labels in {out / 'medpix_labels.csv'} (pass back with --labels).")


def main(argv=None):
    p = argparse.ArgumentParser(prog="hoofbeats")
    p.add_argument("--kb", default="data/kb", help="folder with hp.obo, phenotype.hpoa, genes_to_disease.txt")
    p.add_argument("--common", default="data/kb/common_profiles.json",
                   help="common-disease profiles from learn-common; used whenever the file exists")
    p.add_argument("--prior-weight", type=float, default=1.0, help="0 = ignore how common each disease is")
    p.add_argument("--llm", action="store_true", help="use MedGemma (see README)")
    p.add_argument("--llm-model", default="google/medgemma-1.5-4b-it")
    p.add_argument("--llm-endpoint", help="local OpenAI-compatible server, e.g. http://localhost:8000/v1/chat/completions; "
                                          "omit to load the model in-process with transformers")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("download"); s.add_argument("--dest", default="data")
    s = sub.add_parser("diagnose"); s.add_argument("case"); s.add_argument("-k", type=int, default=5)
    for name in ("benchmark", "sequential"):
        s = sub.add_parser(name)
        s.add_argument("--phenopackets", default="data/phenopacket-store")
        s.add_argument("--sample", type=int, default=500 if name == "benchmark" else 300)
        s.add_argument("--seed", type=int, default=13)
        s.add_argument("--rounds", type=int, default=5)
        s.add_argument("--out", default="results")
        if name == "benchmark":
            s.add_argument("--llm-cases", type=int, default=50, help="cases to re-rank with MedGemma when --llm is set")
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
    s.add_argument("--max-images", type=int, default=4, help="images per case sent to MedGemma")
    s.add_argument("--cache", help="jsonl of raw MedGemma image reads (default: <out>/medpix_image_reads.jsonl)")
    s.add_argument("--out", default="results")
    a = p.parse_args(argv)
    if a.cmd == "download":
        download(a.dest)
    else:
        {"diagnose": cmd_diagnose, "benchmark": cmd_benchmark, "sequential": cmd_sequential,
         "learn-common": cmd_learn_common, "mimic": cmd_mimic, "medpix": cmd_medpix}[a.cmd](a)


if __name__ == "__main__":
    main()
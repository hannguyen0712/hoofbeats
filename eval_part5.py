"""Part 5: data and evaluation. Every number this prints comes from a saved file, so the report can cite it.

    python eval_part5.py data                      # data section: dataset sizes, knowledge base, leakage
    python eval_part5.py tables                    # main results: Top-1/3/5/10 and MRR with 95% intervals
    python eval_part5.py all                       # both

Inputs (defaults): results/main/benchmark.csv and results/full/benchmark.csv (`benchmark`), results/main/sequential.csv
(`sequential`). Outputs: results/part5/{data,main_results}.{md,json} and LaTeX tables next to them.

Intervals are 95% bootstrap intervals that resample whole diseases (cases of one disease are not independent), using
`boot_ci` from hoofbeats/report.py. Differences between conditions are paired per case.
"""
import argparse, json, statistics
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from hoofbeats.report import boot_ci

K = (1, 3, 5, 10)
MAIN = "present + absent"            # the ranker as shipped: present and absent findings, leakage control on
BASELINE_POLICY = "random"


# ---------------------------------------------------------------- metrics

def hits(ranks, k: int) -> np.ndarray:
    # 1 if the true disease is in the top k; a missing rank (disease not scored) is a miss
    return np.array([1.0 if r is not None and r <= k else 0.0 for r in ranks])


def rr(ranks) -> np.ndarray:
    return np.array([1.0 / r if r else 0.0 for r in ranks])


def _ranks(col) -> list:
    return [None if pd.isna(x) else int(x) for x in col]


def summarize(ranks, groups, seed: int = 0) -> dict:
    """Top-k and MRR with a grouped 95% bootstrap interval for each."""
    out = {"n": len(ranks)}
    for name, v in [*((f"top{k}", hits(ranks, k)) for k in K), ("mrr", rr(ranks))]:
        lo, hi = boot_ci(v, seed=seed, groups=groups)
        out[name], out[f"{name}_lo"], out[f"{name}_hi"] = float(v.mean()) if len(v) else float("nan"), lo, hi
    return out


def paired_delta(a_ranks, b_ranks, groups, k: int = 1, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (hit_b - hit_a) at k over the same cases, with a grouped 95% interval."""
    d = hits(b_ranks, k) - hits(a_ranks, k)
    lo, hi = boot_ci(d, seed=seed, groups=groups)
    return float(d.mean()), lo, hi


# ---------------------------------------------------------------- benchmark tables

def benchmark_table(csv: Path, evaluable_only: bool) -> pd.DataFrame:
    df = pd.read_csv(csv)
    if evaluable_only:
        df = df[df["evaluable"].astype(str) == "True"]
    main = df[df["condition"] == MAIN].set_index("case")
    rows = []
    for cond, g in df.groupby("condition", sort=False):
        s = summarize(_ranks(g["rank"]), g["truth"].tolist())
        s["condition"] = cond
        if cond != MAIN and len(main):
            g = g.set_index("case").loc[main.index]          # same cases, same order
            s["d_top1"], s["d_top1_lo"], s["d_top1_hi"] = paired_delta(
                _ranks(main["rank"]), _ranks(g["rank"]), main["truth"].tolist())
        rows.append(s)
    return pd.DataFrame(rows).set_index("condition")


def sequential_table(csv: Path) -> pd.DataFrame:
    """Accuracy before and after the next-test loop, per question policy, paired against random questions."""
    df = pd.read_csv(csv)
    last = df["round"].max()
    rows, end = [], {}
    for pol, g in df.groupby("policy", sort=False):
        s0 = g[g["round"] == 0].set_index("case")
        s1 = g[g["round"] == last].set_index("case").loc[s0.index]
        grp = s0["truth"].tolist()
        end[pol] = (s1, grp)
        # "answered" is the case's total number of answered questions, repeated on each of its rows
        row = {"policy": pol, "n": len(s0), "rounds": int(last),
               "answered": float(s0["answered"].mean() / last) if last else float("nan")}
        for k in (1, 3, 5, 10):
            row[f"top{k}_start"] = float(hits(_ranks(s0["rank"]), k).mean())
            row[f"top{k}_end"] = float(hits(_ranks(s1["rank"]), k).mean())
        row["d_top1"], row["d_top1_lo"], row["d_top1_hi"] = paired_delta(_ranks(s0["rank"]), _ranks(s1["rank"]), grp)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("policy")
    if BASELINE_POLICY in end:
        base, grp = end[BASELINE_POLICY]
        for pol, (s1, _) in end.items():
            if pol != BASELINE_POLICY:
                s1 = s1.loc[base.index]
                d = paired_delta(_ranks(base["rank"]), _ranks(s1["rank"]), grp)
                out.loc[pol, ["vs_random", "vs_random_lo", "vs_random_hi"]] = d
    return out


# ---------------------------------------------------------------- formatting

def pct(x) -> str:
    return "–" if x is None or pd.isna(x) else f"{100 * x:.1f}"


def ci(row, name) -> str:
    return f"{pct(row[name])} ({pct(row[name + '_lo'])}–{pct(row[name + '_hi'])})"


def delta(row, name="d_top1") -> str:
    if name not in row or pd.isna(row[name]):
        return "–"
    return f"{100 * row[name]:+.1f} ({100 * row[name + '_lo']:+.1f} to {100 * row[name + '_hi']:+.1f})"


def md_benchmark(t: pd.DataFrame) -> str:
    lines = ["| condition | n | Top-1 | Top-3 | Top-5 | Top-10 | MRR | Δ Top-1 vs main (pts) |",
             "|---|---|---|---|---|---|---|---|"]
    for cond, r in t.iterrows():
        lines.append(f"| {cond} | {int(r['n'])} | {ci(r, 'top1')} | {ci(r, 'top3')} | {ci(r, 'top5')} | "
                     f"{ci(r, 'top10')} | {ci(r, 'mrr')} | {delta(r)} |")
    return "\n".join(lines)


def md_sequential(t: pd.DataFrame) -> str:
    lines = ["| policy | n | answered | Top-1 start → end | Top-3 | Top-5 | Top-10 | Δ Top-1 (pts) | vs random (pts) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for pol, r in t.iterrows():
        cells = [f"{pct(r[f'top{k}_start'])} → {pct(r[f'top{k}_end'])}" for k in (1, 3, 5, 10)]
        lines.append(f"| {pol} | {int(r['n'])} | {pct(r['answered'])}% | " + " | ".join(cells)
                     + f" | {delta(r)} | {delta(r, 'vs_random')} |")
    return "\n".join(lines)


def tex_benchmark(t: pd.DataFrame, caption: str, label: str) -> str:
    esc = lambda s: s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
    body = "\n".join(f"{esc(c)} & {pct(r['top1'])} & {pct(r['top3'])} & {pct(r['top5'])} & {pct(r['top10'])} & "
                     f"{pct(r['mrr'])} \\\\" for c, r in t.iterrows())
    return ("\\begin{table}[t]\n\\centering\\small\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
            "Condition & Top-1 & Top-3 & Top-5 & Top-10 & MRR \\\\\n\\midrule\n" + body +
            f"\n\\bottomrule\n\\end{{tabular}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")


def cmd_tables(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    md, js = ["# Part 5: main results", "",
              "Percentages; 95% bootstrap intervals in brackets, resampling whole diseases. "
              f"Main condition: `{MAIN}` (leakage control on). Ties count against the true disease.", ""], {}
    for csv in map(Path, a.benchmark):
        if not csv.exists():
            print(f"[part5] skipped, not found: {csv}")
            continue
        for ev in (False, True):
            t = benchmark_table(csv, ev)
            key = f"{csv.parent.name}{' evaluable' if ev else ''}"
            what = ("evaluable cases (the true disease keeps annotations after its own paper is removed)"
                    if ev else "all cases")
            md += [f"## Ranking, `{csv}`, {what}", "", md_benchmark(t), ""]
            js[key] = t.reset_index().to_dict(orient="records")
            (out / f"main_results_{csv.parent.name}{'_evaluable' if ev else ''}.tex").write_text(tex_benchmark(
                t, f"Ranking on Phenopacket Store ({what}, n={int(t['n'].iloc[0])}). Percentages.",
                f"tab:rank-{csv.parent.name}{'-ev' if ev else ''}"))
            print(f"[part5] {key}: {MAIN} Top-1 {pct(t.loc[MAIN, 'top1'])}, Top-3 {pct(t.loc[MAIN, 'top3'])}, "
                  f"Top-5 {pct(t.loc[MAIN, 'top5'])}, Top-10 {pct(t.loc[MAIN, 'top10'])} (n={int(t.loc[MAIN, 'n'])})")
    for csv in map(Path, a.sequential):
        if not csv.exists():
            print(f"[part5] skipped, not found: {csv}")
            continue
        t = sequential_table(csv)
        md += [f"## Next-test loop, `{csv}` (evaluable cases, {int(t['rounds'].iloc[0])} questions each)", "",
               md_sequential(t), "",
               "`answered`: share of questions the simulator could answer from the record; "
               "unanswered questions change nothing.", ""]
        js[f"sequential {csv.parent.name}"] = t.reset_index().to_dict(orient="records")
    (out / "main_results.md").write_text("\n".join(md))
    (out / "main_results.json").write_text(json.dumps(js, indent=1))
    print(f"[part5] wrote {out / 'main_results.md'}")


# ---------------------------------------------------------------- data section

def cmd_data(a):
    from hoofbeats.__main__ import evaluable, load_kb, pick_cases
    from hoofbeats.data import load_phenopackets
    from hoofbeats.model import Status
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    r = load_kb(SimpleNamespace(kb=a.kb, common="none", prior_weight=1.0, absent_weight=0.5, absent_min_freq=0.8,
                                use_onset=False, use_sex=False))
    d = {"knowledge_base": {
        "hpo_version": r.ont.version, "hpo_terms": len(r.ont.terms), "annotations_version": r.db.version,
        "rare_diseases_omim": len(r.db.names),
        "disease_phenotype_annotations": sum(len(v) for v in r.db.annos.values()),
        "diseases_with_a_gene": sum(1 for x in r.db.names if r.db.disease_genes.get(x)),
        "papers_cited_by_annotations": len(r.db.ref_index)}}

    allc = load_phenopackets(a.phenopackets)
    usable = [c for c in allc if c.present and any(t in r.db.names for t in c.truth)]
    ev = [c for c in usable if evaluable(r, c)]
    pmids = {p for c in usable for p in c.source_pmids}
    per_case = lambda cs, f: statistics.median(f(c) for c in cs) if cs else 0
    pp = {"phenopackets_read": len(allc),
          "usable_cases": len(usable),               # has a present finding and its disease is in the knowledge base
          "diseases": len({c.truth[0] for c in usable}),
          "genes": len({g for c in usable for g in c.genes}),
          "papers": len(pmids),
          "cases_with_a_paper": sum(1 for c in usable if c.source_pmids),
          "median_present_findings": per_case(usable, lambda c: len(c.terms(Status.PRESENT))),
          "median_absent_findings": per_case(usable, lambda c: len(c.terms(Status.ABSENT))),
          "cases_with_a_gene": sum(1 for c in usable if c.genes),
          "evaluable_cases": len(ev),
          "known_only_from_own_paper": len(usable) - len(ev)}
    for n in (500, 0):
        s = pick_cases(r, a.phenopackets, n, seed=13)
        pp[f"benchmark_sample_{n or 'all'}"] = {"cases": len(s), "diseases": len({c.truth[0] for c in s}),
                                                "evaluable": sum(evaluable(r, c) for c in s)}
    d["phenopacket_store"] = pp

    hosp = Path(a.mimic) / "hosp"
    if hosp.exists():
        rd = lambda t: pd.read_csv(hosp / f"{t}.csv.gz", low_memory=False)
        dx, labs = rd("diagnoses_icd"), rd("labevents")
        d["mimic_iv_demo"] = {"patients": len(rd("patients")), "admissions": len(rd("admissions")),
                              "diagnosis_rows": len(dx), "distinct_icd_codes": int(dx["icd_code"].nunique()),
                              "lab_events": len(labs), "distinct_lab_items": int(labs["itemid"].nunique()),
                              "lab_events_flagged_abnormal": int((labs["flag"].astype(str) == "abnormal").sum())}
    else:
        d["mimic_iv_demo"] = f"not found at {hosp}"

    mp = Path(a.medpix_csv)
    if mp.exists():
        m = pd.read_csv(mp)
        d["medpix"] = {"source": str(mp), "cases": len(m), "images": int(m["n_images"].sum()),
                       "diagnosis_mapped_to_kb": int(m["truth"].notna().sum()),
                       "with_radiologist_findings": int((m["n_radiologist"] > 0).sum())}

    ex = sorted(Path("examples").rglob("*.json"))
    cases = [p for p in ex if "case_id" in json.loads(p.read_text())]
    d["example_case_files"] = {"files": [str(p) for p in cases],
                               "with_note": sum(1 for p in cases if json.loads(p.read_text()).get("note_text"))}

    (out / "data.json").write_text(json.dumps(d, indent=1))
    kb, ph = d["knowledge_base"], d["phenopacket_store"]
    md = ["# Part 5: data", "",
          f"Knowledge base: HPO {kb['hpo_version']} ({kb['hpo_terms']:,} terms), annotations {kb['annotations_version']}: "
          f"{kb['rare_diseases_omim']:,} OMIM diseases, {kb['disease_phenotype_annotations']:,} disease-finding "
          f"annotations from {kb['papers_cited_by_annotations']:,} references; {kb['diseases_with_a_gene']:,} "
          "diseases have a known gene.", "",
          "| Phenopacket Store | value |", "|---|---|",
          *(f"| {k.replace('_', ' ')} | {v} |" for k, v in ph.items() if not isinstance(v, dict)),
          *(f"| {k.replace('_', ' ')} | {v['cases']} cases, {v['diseases']} diseases, {v['evaluable']} evaluable |"
            for k, v in ph.items() if isinstance(v, dict)), ""]
    for name in ("mimic_iv_demo", "medpix"):
        if isinstance(d.get(name), dict):
            md += [f"| {name.replace('_', ' ')} | value |", "|---|---|",
                   *(f"| {k.replace('_', ' ')} | {v} |" for k, v in d[name].items()), ""]
    md += [f"Example case files: {len(cases)} ({d['example_case_files']['with_note']} with a note).", ""]
    (out / "data.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"[part5] wrote {out / 'data.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=["data", "tables", "all"])
    ap.add_argument("--out", default="results/part5")
    ap.add_argument("--benchmark", nargs="*", default=["results/main/benchmark.csv", "results/full/benchmark.csv"])
    ap.add_argument("--sequential", nargs="*", default=["results/main/sequential.csv"])
    ap.add_argument("--kb", default="data/kb")
    ap.add_argument("--phenopackets", default="data/phenopacket-store")
    ap.add_argument("--mimic", default="data/mimic-iv-demo/2.2")
    ap.add_argument("--medpix-csv", default="results/medpix/medpix.csv")
    a = ap.parse_args()
    if a.what in ("data", "all"):
        cmd_data(a)
    if a.what in ("tables", "all"):
        cmd_tables(a)


if __name__ == "__main__":
    main()

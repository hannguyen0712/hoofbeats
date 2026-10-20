from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

# colour-blind-safe palette (Okabe-Ito)
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
K = (1, 3, 10)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
    return plt


def _ranks(col) -> list:
    return [None if pd.isna(x) else int(x) for x in col]


def topk(ranks) -> dict:
    n = len(ranks)
    m = {f"top{k}": sum(1 for x in ranks if x and x <= k) / n for k in K} if n else {}
    m["mrr"] = sum(1 / x for x in ranks if x) / n if n else 0.0
    return m


def boot_ci(hits: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    # 95% bootstrap interval of a mean (hits are 0/1 or paired differences)
    if len(hits) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(hits, (n_boot, len(hits))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def hit_at(ranks, k: int) -> np.ndarray:
    return np.array([1.0 if x and x <= k else 0.0 for x in ranks])


def _bars(ax, *args, **kw):
    # bar chart with value labels, so zero-height bars are still visible
    bars = ax.bar(*args, **kw)
    ax.bar_label(bars, labels=["" if np.isnan(v) else f"{100 * v:.0f}" for v in bars.datavalues], fontsize=7, padding=1)
    return bars


def _latest(paths: list[Path]) -> dict[str, Path]:
    # newest file per file name
    out = {}
    for p in sorted(paths, key=lambda p: p.stat().st_mtime):
        out[p.name] = p
    return out


def _pct(x) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


def _tex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    esc = lambda v: str(v).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
    cols = "l" + "r" * (len(df.columns) - 1)
    body = " \\\\\n".join(" & ".join(esc(v) for v in row) for row in df.astype(str).values.tolist())
    return (f"\\begin{{table}}[t]\n\\centering\n\\small\n\\begin{{tabular}}{{{cols}}}\n\\hline\n"
            f"{' & '.join(esc(c) for c in df.columns)} \\\\\n\\hline\n{body} \\\\\n\\hline\n\\end{{tabular}}\n"
            f"\\caption{{{esc(caption)}}}\n\\label{{{label}}}\n\\end{{table}}\n")


class Report:
    def __init__(self, out: Path, latex: bool):
        self.out, self.latex = out, latex
        (out / "figures").mkdir(parents=True, exist_ok=True)
        (out / "tables").mkdir(parents=True, exist_ok=True)
        self.md: list[str] = []
        self.notes: list[str] = []

    def section(self, title: str):
        self.md.append(f"\n## {title}\n")

    def text(self, s: str):
        self.md.append(s + "\n")

    def table(self, df: pd.DataFrame, name: str, caption: str):
        df.to_csv(self.out / "tables" / f"{name}.csv", index=False)
        if self.latex:
            (self.out / "tables" / f"{name}.tex").write_text(_tex_table(df, caption, f"tab:{name}"))
        self.md.append(f"*{caption}*\n\n{_md_table(df)}\n")

    def figure(self, fig, name: str, caption: str):
        for ext in ("png", "pdf"):
            fig.savefig(self.out / "figures" / f"{name}.{ext}", bbox_inches="tight")
        _plt().close(fig)
        self.md.append(f"![{caption}](figures/{name}.png)\n\n*{caption}*\n")

    def note(self, s: str):
        self.notes.append(s)


# sections
def runs_section(rep: Report, runs: list[dict]):
    if not runs:
        return
    rep.section("Runs")
    rows = [{"started": r.get("started", ""), "command": r.get("command", ""),
             "model": r.get("settings", {}).get("model", "") or "–",
             "status": r.get("status", ""), "minutes": round(r.get("seconds", 0) / 60, 1), "note": r.get("note", ""),
             "id": r.get("id", "")} for r in sorted(runs, key=lambda r: r.get("started", ""))]
    rep.table(pd.DataFrame(rows), "runs", "All logged runs (settings, metrics and readable logs are in runs/<id>.json and .txt)")


def benchmark_section(rep: Report, csv: Path | None, runs: list[dict]):
    if not csv:
        return
    plt = _plt()
    df = pd.read_csv(csv)
    rep.section("Rare-disease ranking (Phenopacket Store)")
    rows = []
    for cond, g in df.groupby("condition", sort=False):
        ev = g[g.evaluable.astype(bool)]
        m, me = topk(_ranks(g["rank"])), topk(_ranks(ev["rank"]))
        lo, hi = boot_ci(hit_at(_ranks(ev["rank"]), 1))
        rows.append({"condition": cond, "cases": len(g), "Top-1": _pct(m["top1"]), "Top-3": _pct(m["top3"]),
                     "Top-10": _pct(m["top10"]), "MRR": f"{m['mrr']:.3f}", "evaluable": len(ev),
                     "evaluable Top-1 (95% CI)": f"{_pct(me['top1'])} ({_pct(lo)}–{_pct(hi)})", "_m": me})
    tab = pd.DataFrame(rows)
    rep.table(tab.drop(columns="_m"), "benchmark", f"Ranker accuracy by condition ({csv})")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    w = 0.8 / len(K)
    x = np.arange(len(tab))
    for i, k in enumerate(K):
        _bars(ax, x + (i - 1) * w, [r[f"top{k}"] for r in tab["_m"]], w, label=f"Top-{k}", color=COLORS[i])
    ax.set_xticks(x, [c.replace(", ", ",\n") for c in tab.condition], fontsize=8)
    ax.set_ylabel("accuracy, evaluable cases (labels in %)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=3)
    rep.figure(fig, "benchmark_conditions", "Ranker accuracy by condition (evaluable cases)")

    by = {r["condition"]: r["_m"] for r in rows}
    if "present + absent" in by and "present + absent, NO leakage control" in by:
        a, b = by["present + absent"]["top1"], by["present + absent, NO leakage control"]["top1"]
        rep.note(f"Leakage control: Top-1 is {_pct(a)} with the case's own paper removed vs {_pct(b)} without; "
                 f"results without leakage control overstate accuracy by {100 * (b - a):.1f} points.")
    if "present only" in by and "present + absent" in by:
        a, b = by["present only"]["top1"], by["present + absent"]["top1"]
        rep.note(f"Absent findings {'help' if b > a else 'hurt'}: Top-1 {_pct(a)} with present findings only vs {_pct(b)} with absent findings added.")
    if "present + absent + gene" in by:
        rep.note(f"Adding the causal gene raises Top-1 to {_pct(by['present + absent + gene']['top1'])} (ablation only).")

    run = next((r for r in sorted(runs, key=lambda r: r.get("started", ""), reverse=True)
                if r.get("command") == "benchmark" and "calibration" in r.get("metrics", {})), None)
    if not run:
        return
    met = run["metrics"]
    if "absent_weight" in met:
        aw = met["absent_weight"]
        ws = sorted(aw, key=float)
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for i, k in enumerate(("top1", "top10")):
            ax.plot([float(w) for w in ws], [aw[w][k] for w in ws], "o-", color=COLORS[i], label=k.replace("top", "Top-"))
        ax.set_xlabel("weight on absent findings")
        ax.set_ylabel("accuracy")
        ax.legend(frameon=False)
        rep.figure(fig, "absent_weight", "Effect of the weight on absent findings")
        best = max(ws, key=lambda w: aw[w]["top1"])
        rep.note(f"Best absent-finding weight on this sample: {best} (Top-1 {_pct(aw[best]['top1'])}).")
    cal = met["calibration"]
    rel = cal.get("reliability", [])
    if rel:
        fig, ax = plt.subplots(figsize=(3.8, 3.6))
        ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1)
        ax.plot([b["confidence"] for b in rel], [b["accuracy"] for b in rel], "o-", color=COLORS[0])
        for b in rel:
            ax.annotate(str(b["n"]), (b["confidence"], b["accuracy"]), textcoords="offset points", xytext=(4, -10), fontsize=7)
        ax.set_xlabel("predicted probability of top disease")
        ax.set_ylabel("observed Top-1 accuracy")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
        rep.figure(fig, "calibration", f"Calibration on held-out diseases (T = {cal['temperature']:.2f}, ECE = {cal['ece']:.3f}); numbers are cases per bin")
        rep.note(f"Calibration: temperature {cal['temperature']:.2f}, expected calibration error {cal['ece']:.3f}.")


def llm_section(rep: Report, compare: pd.DataFrame | None, per_case: dict[str, Path]):
    if compare is None or compare.empty:
        return
    plt = _plt()
    ph = compare[compare.task == "phenopacket"].copy()
    if ph.empty:
        return
    ph = ph.sort_values("time").groupby("model").tail(1)       # latest run per model
    rep.section("Language models on the shortlist (Phenopacket Store)")
    rows, diffs = [], {}
    for _, r in ph.iterrows():
        slug = r.model.split("/")[-1]
        f = next((p for n, p in per_case.items() if n == f"llm_rerank_{_slug(r.model)}.csv"), None)
        ci = ""
        if f is not None:
            d = pd.read_csv(f)
            if "rerank" in d:
                delta = hit_at(_ranks(d.rerank), 1) - hit_at(_ranks(d.ranker), 1)
                lo, hi = boot_ci(delta)
                diffs[slug] = (delta.mean(), lo, hi, len(d))
                ci = f"{100 * delta.mean():+.1f} ({100 * lo:+.1f} to {100 * hi:+.1f})"
        rows.append({"model": slug, "cases": int(r.cases), "ranker Top-1": _pct(r.get("ranker_top1")),
                     "+ re-rank Top-1": _pct(r.get("rerank_top1")), "Δ Top-1 pts (95% CI)": ci or "–",
                     "model alone Top-1": _pct(r.get("direct_top1")), "re-rank Top-10": _pct(r.get("rerank_top10")),
                     "parsed": _pct(r.get("rerank_parse_rate")), "faithfulness": _pct(r.get("faithfulness")),
                     "s/call": r.get("sec_per_call"),
                     "requests": "–" if pd.isna(r.get("requests", np.nan)) else int(r.get("requests")), "note": r.get("note", "") if isinstance(r.get("note", ""), str) else ""})
    rep.table(pd.DataFrame(rows), "llm_compare", "Ranker alone, ranker + model re-rank, and model alone on the same evaluable cases")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    series = [("ranker", "ranker alone"), ("rerank", "ranker + re-rank"), ("direct", "model alone")]
    x = np.arange(len(ph))
    w = 0.8 / len(series)
    for ax, k in zip(axes, ("top1", "top10")):
        for i, (col, lab) in enumerate(series):
            vals = [r.get(f"{col}_{k}", np.nan) for _, r in ph.iterrows()]
            _bars(ax, x + (i - 1) * w, vals, w, label=lab, color=COLORS[i])
        ax.set_xticks(x, [m.split("/")[-1] for m in ph.model], rotation=20, ha="right", fontsize=8)
        ax.set_title(k.replace("top", "Top-"))
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy (labels in %)")
    axes[0].legend(frameon=False, fontsize=8)
    rep.figure(fig, "llm_accuracy", "Accuracy by model: the ranker alone, with the model re-ranking its top 10, and the model alone")

    fig, ax = plt.subplots(figsize=(6, 3))
    labels = [m.split("/")[-1] for m in ph.model]
    up, same, down = [], [], []
    for m in ph.model:
        f = per_case.get(f"llm_rerank_{_slug(m)}.csv")
        if f is None:
            up.append(0), same.append(0), down.append(0)
            continue
        d = pd.read_csv(f).dropna(subset=["ranker"])
        ch = d.ranker - d.rerank.fillna(11)
        up.append((ch > 0).sum()), same.append((ch == 0).sum()), down.append((ch < 0).sum())
    if not sum(up) + sum(same) + sum(down):
        _plt().close(fig)
        rep.text("No case had the correct diagnosis in the ranker's top 10, so re-ranking could not move it.")
        return _llm_notes(rep, ph, diffs)
    ax.barh(labels, up, color=COLORS[2], label="truth moved up")
    ax.barh(labels, same, left=up, color="#BBBBBB", label="unchanged")
    ax.barh(labels, down, left=np.add(up, same), color=COLORS[5], label="truth moved down")
    ax.set_xlabel("cases with the truth in the ranker's top 10")
    ax.set_xlim(0, None)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    rep.figure(fig, "llm_rank_changes", "Where re-ranking moved the correct diagnosis")
    _llm_notes(rep, ph, diffs)


def _llm_notes(rep: Report, ph: pd.DataFrame, diffs: dict):
    for slug, (m, lo, hi, n) in diffs.items():
        sig = "a real gain" if lo > 0 else "a real loss" if hi < 0 else "not distinguishable from no change"
        rep.note(f"{slug} re-ranking changes Top-1 by {100 * m:+.1f} points on {n} cases (95% CI {100 * lo:+.1f} to {100 * hi:+.1f}): {sig}.")
    best = ph.sort_values("rerank_top1", ascending=False).iloc[0] if "rerank_top1" in ph else None
    if best is not None:
        rep.note(f"Highest Top-1 after re-ranking: {best.model} ({_pct(best.rerank_top1)}); "
                 f"its explanations kept {_pct(best.get('faithfulness'))} of sentences after checking.")
    if "direct_top1" in ph:
        for _, r in ph.iterrows():
            if not pd.isna(r.get("direct_top1")):
                rel = "beats" if r.ranker_top1 > r.direct_top1 else "ties with" if r.ranker_top1 == r.direct_top1 else "is below"
                rep.note(f"The ranker alone {rel} {r.model.split('/')[-1]} used on its own (Top-1 {_pct(r.ranker_top1)} vs {_pct(r.direct_top1)}).")
    if ph.cases.max() < 100:
        rep.note("Model comparisons use fewer than 100 cases; treat differences inside the confidence intervals as noise.")


def medpix_section(rep: Report, compare: pd.DataFrame | None, medpix: dict[str, Path]):
    plt = _plt()
    mp = compare[compare.task == "medpix"] if compare is not None and not compare.empty else pd.DataFrame()
    base = medpix.get("medpix.csv")
    if mp.empty and base is None:
        return
    rep.section("Multimodal (MedPix 2.0)")
    if base is not None:
        d = pd.read_csv(base)
        lab = d[d.truth.fillna("") != ""]
        rows = [{"input": name, "Top-1": _pct(topk(_ranks(lab[c]))["top1"]), "Top-10": _pct(topk(_ranks(lab[c]))["top10"])}
                for name, c in (("history + exam text", "rank_text"), ("text + radiologist findings (upper bound)", "rank_radiologist"))
                if c in lab]
        rep.table(pd.DataFrame(rows), "medpix_text", f"Ranker on {len(lab)} labelled MedPix cases without images")
    if not mp.empty:
        mp = mp.sort_values("time").groupby("model").tail(1)
        num = lambda v: "–" if pd.isna(v) else int(v)
        rows = [{"model": r.model.split("/")[-1], "cases": num(r.get("labeled")),
                 "text Top-10": _pct(r.get("rank_text_top10")), "images only Top-10": _pct(r.get("rank_image_top10")),
                 "text + images Top-10": _pct(r.get("rank_multimodal_top10")),
                 "radiologist upper bound Top-10": _pct(r.get("rank_radiologist_top10")),
                 "finding precision": _pct(r.get("image_precision")), "finding recall": _pct(r.get("image_recall")),
                 "unparsed": num(r.get("unparsed_images")), "s/image": r.get("sec_per_call")} for _, r in mp.iterrows()]
        rep.table(pd.DataFrame(rows), "medpix_models", "Image reading by model: ranking accuracy and agreement with radiologist findings")
        fig, ax = plt.subplots(figsize=(7, 3.2))
        cols = [("rank_text_top10", "text"), ("rank_multimodal_top10", "text + model images"),
                ("rank_radiologist_top10", "text + radiologist (upper bound)")]
        x, w = np.arange(len(mp)), 0.8 / len(cols)
        for i, (c, lab) in enumerate(cols):
            _bars(ax, x + (i - 1) * w, [r.get(c, np.nan) for _, r in mp.iterrows()], w, label=lab, color=COLORS[i])
        ax.set_xticks(x, [m.split("/")[-1] for m in mp.model], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Top-10 accuracy (labels in %)")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8)
        rep.figure(fig, "medpix_models", "Do image findings help? Top-10 accuracy on labelled MedPix cases")
        for _, r in mp.iterrows():
            gain = r.get("rank_multimodal_top10", np.nan) - r.get("rank_text_top10", np.nan)
            rep.note(f"{r.model.split('/')[-1]} images change MedPix Top-10 by {100 * gain:+.1f} points; its findings agree "
                     f"with the radiologist at {_pct(r.get('image_precision'))} precision and {_pct(r.get('image_recall'))} recall.")


def sequential_section(rep: Report, csv: Path | None):
    if not csv:
        return
    plt = _plt()
    d = pd.read_csv(csv)
    rep.section("Next-test suggestions (simulated)")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    rows = []
    for i, (pol, g) in enumerate(d.groupby("policy", sort=False)):
        curve = g.groupby("round")["rank"].apply(lambda s: topk(_ranks(s))["top1"])
        ax.plot(curve.index, curve.values, "o-", color=COLORS[i], label=pol)
        rows.append({"policy": pol, "Top-1 before": _pct(curve.iloc[0]), "Top-1 after": _pct(curve.iloc[-1]),
                     "questions answered": _pct(g[g["round"] == g["round"].max()].answered.mean() / max(g["round"].max(), 1))})
    ax.set_xlabel("questions asked")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, min(1, max(0.1, d.groupby(["policy", "round"])["rank"].apply(lambda s: topk(_ranks(s))["top1"]).max() * 1.3)))
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(frameon=False)
    rep.figure(fig, "sequential", "Top-1 accuracy as suggested checks are answered")
    rep.table(pd.DataFrame(rows), "sequential", "Question policies compared")
    by = {r["policy"]: r for r in rows}
    if "eig" in by and "random" in by:
        rep.note(f"Information-gain questions: Top-1 {by['eig']['Top-1 before']} → {by['eig']['Top-1 after']} "
                 f"(random questions: → {by['random']['Top-1 after']}).")


def mimic_section(rep: Report, csv: Path | None):
    if not csv:
        return
    d = pd.read_csv(csv)
    rep.section("Rare and common diseases (MIMIC-IV)")
    rows = []
    for name, sel in (("rare diagnosis", d.rare_truth.astype(bool)),
                      ("common diagnosis", (~d.rare_truth.astype(bool)) & d.truth.notna())):
        g = d[sel]
        if len(g):
            m = topk(_ranks(g["rank"]))
            rows.append({"admissions": name, "n": len(g), "Top-1": _pct(m["top1"]), "Top-3": _pct(m["top3"]), "Top-10": _pct(m["top10"])})
    if rows:
        rep.table(pd.DataFrame(rows), "mimic", "One ranked list for rare and common diseases")


def _slug(model: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9.]+", "-", model.split("/")[-1]).strip("-").lower()


def next_numbered(base: Path) -> tuple[Path, int]:
    # reports/report -> reports/report-1, then report-2, ... (one past the highest existing number)
    nums = [int(m.group(1)) for p in base.parent.glob(f"{base.name}-*")
            if (m := re.fullmatch(re.escape(base.name) + r"-(\d+)", p.name))]
    n = max(nums, default=0) + 1
    return base.parent / f"{base.name}-{n}", n


def build_report(results: Path, base: Path, latex: bool = False) -> Path:
    if not results.exists():
        raise SystemExit(f"{results} not found")
    out, n = next_numbered(base)
    rep = Report(out, latex)
    runs = []
    for p in results.rglob("runs/*.json"):
        try:
            runs.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    csvs = _latest(list(results.rglob("*.csv")))
    compare_files = [p for p in results.rglob("llm_compare.csv")]
    compare = pd.concat([pd.read_csv(p) for p in compare_files], ignore_index=True) if compare_files else None
    if compare is not None and "time" in compare:
        compare = compare.drop_duplicates()
    per_case = {n: p for n, p in csvs.items() if n.startswith("llm_rerank_")}
    medpix = {n: p for n, p in csvs.items() if n.startswith("medpix") and not n.endswith("_labels.csv")}

    benchmark_section(rep, csvs.get("benchmark.csv"), runs)
    llm_section(rep, compare, per_case)
    medpix_section(rep, compare, medpix)
    sequential_section(rep, csvs.get("sequential.csv"))
    mimic_section(rep, csvs.get("mimic.csv"))
    runs_section(rep, runs)

    head = [f"# hoofbeats results #{n}\n", f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{results}`.\n"]
    if rep.notes:
        head += ["\n## Key findings\n"] + [f"- {n}" for n in rep.notes] + [""]
    md = out / f"{base.name}-{n}.md"
    md.write_text("\n".join(head + rep.md))
    (out / f"notes-{n}.md").write_text("\n".join(f"- {x}" for x in rep.notes) + "\n")
    print(f"[report] {len(rep.notes)} findings, {len(list((out / 'figures').glob('*.png')))} figures, "
          f"{len(list((out / 'tables').glob('*.csv')))} tables -> {md}")
    return md
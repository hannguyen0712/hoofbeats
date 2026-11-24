# evaluation on a frozen dev/test split: the absent-finding rule, calibration, score-gap bins and learned weights are
# chosen on dev; every number is reported on test, with 95% intervals that resample diseases.
#   python -m hoofbeats --common none evaluate --learn            -> results/eval/EVAL.md, eval.json, ranks.csv
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .model import Case, Finding, LRRanker, Status
from .report import boot_ci

SALT = "hoofbeats-eval-v1"
SECTIONS = ("ranking", "uncertainty", "errors", "robustness", "sequential", "end2end", "grounding")
KS = (1, 3, 5, 10, 50, 100)
INHERITANCE = {"HP:0000006": "autosomal dominant", "HP:0000007": "autosomal recessive", "HP:0001417": "X-linked",
               "HP:0001419": "X-linked recessive", "HP:0001423": "X-linked dominant", "HP:0001427": "mitochondrial"}
ABSENT_GRID = [(0.0, 0.0)] + [(w, m) for w in (0.25, 0.5, 1.0) for m in (0.0, 0.5, 0.8, 0.9)]   # (weight, min freq)


# split: cases sharing a disease or a paper form one group; each group goes to dev or test by a hash of its name
def assign_split(cases: list[Case], dev_share: float) -> dict[str, str]:
    from .train import _groups
    groups, names = _groups(cases), defaultdict(set)
    for c, g in zip(cases, groups):
        names[g] |= {*c.truth, *c.source_pmids}
    side = {g: "dev" if int(hashlib.md5(f"{SALT}:{min(k)}".encode()).hexdigest(), 16) % 1000 < dev_share * 1000
            else "test" for g, k in names.items()}
    return {c.case_id: side[g] for c, g in zip(cases, groups)}


def frozen_split(path: Path, cases: list[Case], dev_share: float, refreeze: bool) -> dict[str, str]:
    # the first run records a fingerprint of the split; later runs stop if the data moved cases between dev and test
    split = assign_split(cases, dev_share)
    sha = hashlib.sha1(json.dumps(sorted(split.items())).encode()).hexdigest()
    if path.exists() and not refreeze and json.loads(path.read_text())["sha1"] != sha:
        raise SystemExit(f"the dev/test split differs from {path} (other data or settings); --refreeze makes a new one")
    if refreeze or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"sha1": sha, "dev_share": dev_share, "cases": len(split),
                                    "dev": sum(v == "dev" for v in split.values()), "frozen": time.strftime("%Y-%m-%d")}))
    return split


# metrics; a missing rank (truth not in the knowledge base) is a miss
def hit(ranks, k: int) -> np.ndarray:
    return np.array([1.0 if x and x <= k else 0.0 for x in ranks])


def rr(ranks) -> np.ndarray:
    return np.array([1.0 / x if x else 0.0 for x in ranks])


def ci(v, groups) -> tuple[float, float, float]:
    v = np.asarray(v, float)
    return (float(v.mean()), *boot_ci(v, groups=list(groups))) if len(v) else (math.nan,) * 3


def pct(m, signed=False) -> str:
    f = (lambda x: f"{100 * x:+.1f}") if signed else (lambda x: f"{100 * x:.1f}")
    return "–" if math.isnan(m[0]) else f"{f(m[0])} ({f(m[1])} to {f(m[2])})"


def ece(conf, hits, bins=10) -> float:
    conf, hits = np.asarray(conf), np.asarray(hits)
    b = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(conf[b == i].mean() - hits[b == i].mean()) * (b == i).mean() for i in set(b)))


def auroc(y, s) -> float:
    # chance that a right case scores above a wrong one (ties count half)
    from scipy.stats import rankdata
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n1, n0 = y.sum(), (~y).sum()
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else math.nan


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    from scipy.stats import beta
    return (0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
            1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)))


def table(header, rows) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
                     + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def compare(conds: dict[str, list], groups, base: str) -> str:
    # Top-1/Top-10/MRR per condition and the paired change in Top-1 against `base` on the same cases
    b = conds[base]
    return table(["condition", "Top-1", "Top-10", "MRR", f"Δ Top-1 vs {base}, pts"],
                 [[k, pct(ci(hit(v, 1), groups)), pct(ci(hit(v, 10), groups)), pct(ci(rr(v), groups)),
                   "–" if k == base else pct(ci(hit(v, 1) - hit(b, 1), groups), True)] for k, v in conds.items()])


@contextmanager
def using(r: LRRanker, **attrs):
    # ranker settings for a block, then back
    old = {k: getattr(r, k) for k in attrs}
    for k, v in attrs.items():
        setattr(r, k, v)
    try:
        yield r
    finally:
        for k, v in old.items():
            setattr(r, k, v)


# noisy patients (robustness)
def keep_present(c: Case, frac: float, rng: random.Random) -> Case:
    pres = [i for i, x in enumerate(c.findings) if x.status == Status.PRESENT]
    kept = set(rng.sample(pres, max(1, round(frac * len(pres))))) if pres else set()
    return replace(c, findings=[x for i, x in enumerate(c.findings) if x.status != Status.PRESENT or i in kept])


def add_unrelated(c: Case, n: int, r: LRRanker, pool: list[str], rng: random.Random) -> Case:
    # n random terms that are neither ancestors nor descendants of the case's findings
    near = set().union(*(r.ont.ancestors(t) | r.ont.descendants(t) for t in map(r.ont.resolve, c.present) if t))
    add = []
    while len(add) < n:
        t = rng.choice(pool)
        if t not in near and t not in add:
            add.append(t)
    return replace(c, findings=c.findings + [Finding(t, Status.PRESENT, "noise") for t in add])


def broaden(c: Case, frac: float, r: LRRanker, rng: random.Random) -> Case:
    # a share of the present findings replaced by a direct parent term
    def up(x):
        t = r.ont.resolve(x.hpo_id)
        par = [p for p in (r.ont.terms[t].parents if t else []) if r.ont.is_phenotype(p)]
        return replace(x, hpo_id=rng.choice(par)) if x.status == Status.PRESENT and par and rng.random() < frac else x
    return replace(c, findings=[up(x) for x in c.findings])


def resnik(r: LRRanker, c: Case, ex: frozenset, ic: np.ndarray) -> np.ndarray:
    # Phenomizer-style baseline: each patient term scores the information content of its most specific ancestor the
    # disease is annotated with; the disease's score is the mean. Same leakage control as the ranker
    terms = [t for t in dict.fromkeys(map(r.ont.resolve, c.present)) if t]
    s = np.zeros(len(r.diseases))
    for t in terms:
        anc = [r.t_index[u] for u in r.ont.ancestors(t) if u in r.t_index]
        if anc:
            s += (r.M[anc] > 0).multiply(ic[anc][:, None]).max(axis=0).toarray().ravel()
    for d in {d for ref in ex for d in r.db.ref_index.get(ref, ())}:
        keys = r.implied(d, ex)[0]
        s[r.d_index[d]] = sum(max([ic[r.t_index[u]] for u in r.ont.ancestors(t) if u in keys and u in r.t_index],
                                  default=0.0) for t in terms)
    return s / max(len(terms), 1)


class Evaluation:
    def __init__(self, a, hb):
        self.a, self.hb, self.r = a, hb, hb.load_kb(a)
        cases = hb.pick_cases(self.r, a.phenopackets, a.sample, seed=a.seed)
        self.split = frozen_split(Path(a.split_file), cases, a.dev_share, a.refreeze)
        self.ev = {c.case_id: hb.evaluable(self.r, c) for c in cases}
        side = lambda s, ev=False: [c for c in cases if self.split[c.case_id] == s and (self.ev[c.case_id] or not ev)]
        self.dev, self.dev_ev, self.test, self.test_ev = side("dev"), side("dev", True), side("test"), side("test", True)
        n = lambda cs: f"{len(cs)} cases / {len({c.truth[0] for c in cs})} diseases"
        self.data = (f"Dev {n(self.dev)} ({len(self.dev_ev)} evaluable); test {n(self.test)} ({len(self.test_ev)} "
                     "evaluable). No disease or paper is in both.")
        self.md, self.js, self.headline, self.ranks = [], {}, [], defaultdict(dict)

    @staticmethod
    def g(cs):
        return [c.truth[0] for c in cs]

    def rank(self, c: Case, cfg=None, control=True):
        with using(self.r, **(cfg or self.cfg)):
            return self.r.score(c, self.hb.leak(c) if control else frozenset())

    def ranks_of(self, cs, name=None, prep=lambda c: c, **kw) -> list:
        out = [self.rank(prep(c), **kw).rank_of(c.truth) for c in cs]
        if name:
            self.ranks[name].update({c.case_id: x for c, x in zip(cs, out)})
        return out

    def add(self, title: str, body: str, data: dict | None = None):
        self.md.append(f"## {title}\n\n{body.strip()}\n")
        self.js[title] = data or {}
        print(self.md[-1])

    def tune(self):
        a = self.a
        base = {"use_genes": False, "use_onset": False, "use_sex": False, "prior_weight": a.prior_weight}
        grid = {}
        for w, m in ABSENT_GRID:
            x = self.ranks_of(self.dev_ev, cfg={**base, "absent_weight": w, "absent_min_freq": m})
            grid[(w, m)] = (hit(x, 1).mean(), rr(x).mean())
        best = max(grid, key=grid.get)
        se = math.sqrt(grid[best][0] * (1 - grid[best][0]) / len(self.dev_ev))
        # one-standard-error rule: keep the shipped rule unless another setting beats it by more than one SE
        shipped = (a.absent_weight, a.absent_min_freq)
        w, m = shipped if grid.get(shipped, (0,))[0] >= grid[best][0] - se else best
        self.cfg = {**base, "absent_weight": w, "absent_min_freq": m}
        from .workup import calibrate, ranker_probs
        with using(self.r, **self.cfg):
            self.calib = calibrate(self.r, self.dev, self.hb.leak)
        self.probs = lambda rk: ranker_probs(rk, self.calib)
        self.gap_edges = list(np.quantile([self.conf_gap(c)[1] for c in self.dev], [0.25, 0.5, 0.75]))
        self.ew = self.learn() if a.learn else None
        rows = [[w_, m_ if w_ else "any", f"{100 * t:.1f}", f"{100 * q:.1f}"] for (w_, m_), (t, q) in grid.items()]
        self.add("Tuning on dev", f"{self.data}\n\nAbsent findings (dev, evaluable, %):\n\n"
                 + table(["weight", "min disease frequency", "Top-1", "MRR"], rows)
                 + f"\n\nBest: {best} (one SE = {100 * se:.1f} pts). Rule: keep the shipped {shipped} unless another "
                 f"setting beats it by more than one SE. **Used: weight {w}, minimum frequency {m}.** Calibration: "
                 f"temperature {self.calib[0]:.2f}, truth in the top 20 {100 * self.calib[1]:.1f}%. Score-gap bins: "
                 + ", ".join(f"{x:.2f}" for x in self.gap_edges) + "."
                 + (f" Learned weights (l2 {a.l2}): " + ", ".join(f"{k} {v:.2f}" for k, v in self.ew.w.items() if v)
                    if self.ew else ""),
                 {"grid": {str(k): v for k, v in grid.items()}, "used": self.cfg, "calibration": self.calib})

    def learn(self):
        from .train import FEATURES, EvidenceWeights, candidates, default_weights, fit_weights
        with using(self.r, **self.cfg):
            data = [candidates(self.r, c, self.hb.leak(c), self.a.candidates, add_truth=True) for c in self.dev_ev]
            gold = [next(j for j, d in enumerate(ds) if d in c.truth) for (ds, _, _), c in zip(data, self.dev_ev)]
            w = fit_weights([X for _, X, _ in data], [p for _, _, p in data], gold, self.a.l2, default_weights(self.r))
        return EvidenceWeights(dict(zip(FEATURES, map(float, w))))

    def conf_gap(self, c: Case) -> tuple[float, float, bool]:
        # calibrated probability of the top disease, score gap to the second, and whether the top one is right
        rk = self.rank(c)
        o = rk.order
        return float(self.probs(rk)[o[0]]), float(rk.log_scores[o[0]] - rk.log_scores[o[1]]), rk.rank_of(c.truth) == 1

    def ranking(self):
        T, G = self.test_ev, self.g(self.test_ev)
        no_absent = lambda c: replace(c, findings=[f for f in c.findings if f.status != Status.ABSENT])
        conds = {"main": self.ranks_of(T, "main"),
                 "present findings only": self.ranks_of(T, "present only", no_absent),
                 "every absent finding at weight 1 (old rule)": self.ranks_of(
                     T, "old absent rule", cfg={**self.cfg, "absent_weight": 1.0, "absent_min_freq": 0.0}),
                 "no leakage control": self.ranks_of(T, "no leakage control", control=False),
                 "+ causal gene": self.ranks_of(T, "gene", cfg={**self.cfg, "use_genes": True}),
                 "+ age of onset": self.ranks_of(T, "onset", cfg={**self.cfg, "use_onset": True}),
                 "+ sex": self.ranks_of(T, "sex", cfg={**self.cfg, "use_sex": True})}
        if self.ew:
            from .train import rerank_learned
            with using(self.r, **self.cfg):
                conds["learned weights (dev)"] = [rerank_learned(self.r, c, self.ew, self.hb.leak(c), self.a.candidates)
                                                  .rank_of(c.truth) for c in T]
        ic = -np.log(np.maximum(np.asarray((self.r.M > 0).sum(axis=1)).ravel(), 1) / len(self.r.diseases))
        gold = lambda c: np.array([d in c.truth for d in self.r.diseases])
        by_score = lambda s, c: int((s[~gold(c)] >= s[gold(c)].max()).sum()) + 1     # ties count against the truth
        conds["baseline: random order"] = [by_score(np.random.default_rng(i).random(len(self.r.diseases)), c)
                                           for i, c in enumerate(T)]
        conds["baseline: Resnik similarity (Phenomizer-style)"] = [by_score(resnik(self.r, c, self.hb.leak(c), ic), c)
                                                                    for c in T]
        rows = [[name, len(cs)] + [pct(ci(hit(x, k), self.g(cs))) for k in KS] + [pct(ci(rr(x), self.g(cs)))]
                for name, cs, x in (("test, all", self.test, self.ranks_of(self.test)), ("test, evaluable", T, conds["main"]),
                                    ("dev, evaluable (tuned here)", self.dev_ev, self.ranks_of(self.dev_ev)))]
        s = {k: ci(hit(v, 1), G) for k, v in conds.items()}
        self.add("Disease ranking", "Ties count against the true disease. Evaluable: the true disease keeps annotations "
                 "after leakage control.\n\n" + table(["cases", "n", *(f"Top-{k}" for k in KS), "MRR"], rows)
                 + "\n\nAblations, options and baselines (test, evaluable, paired on the same cases; baselines use "
                 "present findings only):\n\n" + compare(conds, G, "main"), {"top1": s})
        self.headline += [["Ranking (test, evaluable)", f"Top-1 {pct(s['main'])}, Top-10 {pct(ci(hit(conds['main'], 10), G))}"],
                          ["Without leakage control", f"Top-1 {pct(s['no leakage control'])}"],
                          ["Resnik baseline", f"Top-1 {pct(s['baseline: Resnik similarity (Phenomizer-style)'])}"]]
        if self.ew:
            d = ci(hit(conds["learned weights (dev)"], 1) - hit(conds["main"], 1), G)
            self.headline.append(["Learned weights (fitted on dev)", f"{pct(d, True)} pts Top-1"])

    def uncertainty(self):
        T, G = self.test, np.array(self.g(self.test))
        conf, gap, hits = map(np.array, zip(*(self.conf_gap(c) for c in T)))
        hits = hits.astype(float)
        e, b, gb = ece(conf, hits), np.minimum((conf * 10).astype(int), 9), np.digitize(gap, self.gap_edges)
        rel = [[f"{10 * i}–{10 * i + 10}%", int((b == i).sum()), f"{100 * conf[b == i].mean():.1f}",
                pct(ci(hits[b == i], G[b == i]))] for i in sorted(set(b))]
        gaps = [[f"quartile {i + 1}", int((gb == i).sum()), pct(ci(hits[gb == i], G[gb == i]))] for i in range(4)]
        top = lambda s, f: np.argsort(-s, kind="stable")[:round(f * len(T))]
        sel = [[f"{int(100 * f)}%"] + [pct(ci(hits[top(s, f)], G[top(s, f)])) for s in (conf, gap)] for f in (0.25, 0.5, 0.75, 1)]
        a_c, a_g = auroc(hits, conf), auroc(hits, gap)
        self.add("Uncertainty", f"All {len(T)} test cases. Top disease's calibrated probability: mean "
                 f"{100 * conf.mean():.1f}% vs Top-1 {100 * hits.mean():.1f}%, ECE {e:.3f}.\n\n"
                 + table(["predicted", "cases", "mean predicted", "observed Top-1"], rel)
                 + "\n\nTop-1 by the score gap between the top two diseases (bins: dev quartiles):\n\n"
                 + table(["gap", "cases", "Top-1"], gaps) + f"\n\nAUROC for a correct Top-1: probability {a_c:.3f}, "
                 f"score gap {a_g:.3f}. Top-1 when answering only the most confident cases:\n\n"
                 + table(["coverage", "by probability", "by score gap"], sel), {"ece": e, "auroc_prob": a_c, "auroc_gap": a_g})
        self.headline += [["Calibration (test)", f"ECE {e:.3f}"], ["AUROC, probability / score gap", f"{a_c:.3f} / {a_g:.3f}"]]

    def errors(self):
        r, rows = self.r, []
        for c in self.test_ev:
            ex, d, rk = self.hb.leak(c), c.truth[0], self.rank(c)
            with using(r, **self.cfg):
                pres = [k for k in r.contributions(c, d, ex) if k.status == Status.PRESENT]
            top = r.diseases[rk.order[0]]
            inh = {INHERITANCE.get(t) for t, refs in r.db.inherit.get(d, []) if not refs or not refs <= ex} - {None}
            rows.append({"rank": rk.rank_of(c.truth) or 10 ** 9, "truth": d, "n_present": len(c.present),
                         "inheritance": "/".join(sorted(inh)) or "not annotated",
                         "annotations": len(r.db.annotations(d, ex)),
                         "exact": sum(k.kb_term == k.hpo_id for k in pres) / max(len(pres), 1),
                         "same_gene": top != d and bool(r.db.disease_genes.get(top, set()) & r.db.disease_genes.get(d, set()))})
        df, parts = pd.DataFrame(rows), []
        for col, title, edges in (("n_present", "present findings", (3, 7, 15)),
                                  ("annotations", "true disease's annotations after leakage control", (5, 20, 50)),
                                  ("exact", "share of findings annotated to the true disease itself", (0, 0.25, 0.5))):
            q = np.digitize(df[col], edges, right=True)
            lab = [f"≤ {edges[0]}"] + [f"{lo}–{hi}" for lo, hi in zip(edges, edges[1:])] + [f"> {edges[-1]}"]
            parts.append(f"By {title}:\n\n" + table(["", "cases", "Top-1", "Top-10"], [
                [lab[i], int((q == i).sum()), pct(ci(hit(df["rank"][q == i], 1), df.truth[q == i])),
                 pct(ci(hit(df["rank"][q == i], 10), df.truth[q == i]))] for i in range(4) if (q == i).any()]))
        inh = [[k, len(v), pct(ci(hit(v["rank"], 1), v.truth)), pct(ci(hit(v["rank"], 10), v.truth))]
               for k, v in sorted(df.groupby("inheritance"), key=lambda kv: -len(kv[1])) if len(v) >= 20]
        parts.append("By inheritance of the true disease (groups of 20+ cases):\n\n" + table(["", "cases", "Top-1", "Top-10"], inh))
        dis = df.assign(first=df["rank"] == 1).groupby("truth").agg(cases=("rank", "size"), first=("first", "sum"),
                                                                    median=("rank", "median"))
        worst = dis[(dis.cases >= 3) & (dis["first"] == 0)].sort_values("median", ascending=False).head(10)
        parts.append("Diseases never ranked first (3+ test cases, worst median rank):\n\n" + table(
            ["disease", "cases", "median rank"], [[f"{r.db.names.get(d, d)} ({d})", int(x.cases), f"{x['median']:.0f}"]
                                                   for d, x in worst.iterrows()]))
        far, top10 = df[df["rank"] > 50], df[df["rank"] <= 10]
        self.add("Error analysis", "Test, evaluable cases.\n\n" + "\n\n".join(parts)
                 + f"\n\nTruth outside the top 50 in {len(far)} of {len(df)} cases ({100 * len(far) / len(df):.1f}%); there a "
                 f"median {100 * far.exact.median():.0f}% of findings are annotated to the true disease itself, vs "
                 f"{100 * top10.exact.median():.0f}% when it is in the top 10. When the top disease is wrong, it shares a "
                 f"gene with the truth in {100 * df[df['rank'] > 1].same_gene.mean():.1f}% of cases.")
        self.headline.append(["Truth outside the top 50", f"{100 * len(far) / len(df):.1f}% of evaluable test cases"])

    def robustness(self):
        r, T, G = self.r, self.test_ev, self.g(self.test_ev)
        pool = sorted(t for t in r.t_index if r.ont.is_phenotype(t))
        R = lambda c, tag: random.Random(f"{tag}:{c.case_id}")      # the same change to a case in every run
        keep = lambda c, f: keep_present(c, f, R(c, "k"))
        noise = lambda c, n: add_unrelated(c, n, r, pool, R(c, "n"))
        broad = lambda c, f: broaden(c, f, r, R(c, "b"))
        ch = {"clean": lambda c: c, "keep 50% of present findings": lambda c: keep(c, 0.5),
              "keep 25%": lambda c: keep(c, 0.25), "+1 unrelated finding": lambda c: noise(c, 1),
              "+3 unrelated findings": lambda c: noise(c, 3), "every finding one level broader": lambda c: broad(c, 1.0),
              "keep 50% + 3 unrelated + half broader": lambda c: broad(noise(keep(c, 0.5), 3), 0.5)}
        conds = {k: self.ranks_of(T, prep=f) for k, f in ch.items()}
        self.add("Robustness to noisy patient data", "Test, evaluable cases. Unrelated findings are random HPO terms that "
                 "are neither ancestors nor descendants of the case's findings; broader means a direct parent term.\n\n"
                 + compare(conds, G, "clean"))
        t1 = lambda k: pct(ci(hit(conds[k], 1), G))
        self.headline.append(["Robustness", f"Top-1 {t1('keep 50% of present findings')} with half the findings, "
                                            f"{t1('+3 unrelated findings')} with 3 unrelated added"])

    def sequential(self):
        T, n = self.test_ev[: self.a.sequential_cases], self.a.rounds
        with using(self.r, **self.cfg):
            runs = {p: [self.hb.simulate(self.r, c, p, n, self.a.seed) for c in T] for p in ("eig", "frequent", "random")}
        conds = {"start (half the findings)": [x[0][0] for x in runs["eig"]],
                 **{f"after {n} questions, {p}": [x[0][-1] for x in v] for p, v in runs.items()}}
        answered = ", ".join(f"{p} {100 * sum(x[1] for x in v) / (len(T) * n):.1f}%" for p, v in runs.items())
        self.add("Next-question loop", f"{len(T)} test, evaluable cases. Half of each case's present findings are hidden; "
                 f"each round a policy asks one HPO term, answered from the hidden record. Questions answered: {answered}."
                 "\n\n" + compare(conds, self.g(T), f"after {n} questions, random"))

    def end2end(self):
        a, r = self.a, self.r
        from .data import Extractor, load_phenopackets
        from .intake import parse_case
        from .pmc import load_case_text
        if a.pmc_split == "test" and not a.allow_test:
            raise SystemExit("the PMC test split is read once, with the frozen Stage 1 configuration; add --allow-test")
        reader, s1 = self.hb.stage1_reader(a, r.ont)
        ext, text_r, gold_r, groups, missing = Extractor(r.ont), [], [], [], 0
        for m in json.loads(Path(a.pmc).read_text())["cases"]:
            if m["split"] != a.pmc_split:
                continue
            try:
                text = load_case_text(m)
            except SystemExit:
                missing += 1
                continue
            gc = (load_phenopackets(Path(a.phenopackets) / m["phenopacket_file"]) or [None])[0]
            if gc and self.hb.evaluable(r, gc):
                c = parse_case({"case_id": m["case_id"], "note_text": text}, r.ont, ext, disease_names=r.db.names,
                               llm_reader=reader, status_policy=s1["status_policy"])
                text_r.append(self.rank(replace(c, truth=gc.truth, source_pmids=gc.source_pmids)).rank_of(gc.truth))
                gold_r.append(self.rank(gc).rank_of(gc.truth))
                groups.append(gc.truth[0])
        if not text_r:
            return self.add("End to end", f"Not run: none of the PMC {a.pmc_split} texts is cached in data/pmc/ "
                            "(`python -m hoofbeats pmc-build` fetches them).")
        conds = {"curated Phenopacket findings": gold_r, "Stage 1 from the paper's text": text_r}
        self.add("End to end", f"{len(text_r)} PMC {a.pmc_split} cases ({missing} without cached text), read by Stage 1 "
                 f"({'rules + numbers + LLM' if reader else 'rules + numbers'}), against the same cases' curated findings."
                 "\n\n" + compare(conds, groups, "curated Phenopacket findings"), {"cases": len(text_r)})
        self.headline.append([f"End to end (PMC {a.pmc_split})", f"Top-1 {pct(ci(hit(text_r, 1), groups))} from text"])

    def grounding(self):
        # hand labels of Part 4's held-out model sentences, before and after the checks (exact binomial intervals)
        lab, LABEL, parts = Path(self.a.labels_dir), "label (1 = supported, 0 = not)", []

        def read(p):
            with open(p, newline="", encoding="utf-8-sig") as f:
                rows = {int(x["id"]): x for x in csv.DictReader(f)}
            return rows, {i: int(x[LABEL]) for i, x in rows.items() if x.get(LABEL, "").strip() in ("0", "1")}
        for name, tag, sheet, second in (("held-out run", "test", "test_final", "test_annotatorB_done"),
                                         ("fresh held-out run", "fresh", "fresh_A", None)):
            kp, gp = lab / f"label_key_v3_{tag}.recheck.csv", lab / f"label_sheet_v3_{sheet}.csv"
            if not (kp.exists() and gp.exists()):
                continue
            key, gold = read(kp)[0], read(gp)[1]
            rows = [["no checks", len(gold), f"{100 * sum(1 - v for v in gold.values()) / len(gold):.1f}", "–"]]
            for col, label in (("ok_basic", "original checks (V1–V4)"), ("ok_strict", "current checks (V1–V6)")):
                kept = [i for i in gold if key[i].get(col) == "1"]
                bad = sum(1 - gold[i] for i in kept)
                lo, hi = clopper_pearson(bad, len(kept))
                rows.append([label, len(kept), f"{100 * bad / len(kept):.1f} ({100 * lo:.1f} to {100 * hi:.1f})",
                             sum(1 for i in gold if gold[i] and key[i].get(col) == "0")])
            txt = table(["", "sentences shown", "unsupported among shown, %", "supported removed"], rows)
            if second and (lab / f"label_sheet_v3_{second}.csv").exists():
                A, B = read(lab / f"label_sheet_v3_{tag}.recheck.csv")[1], read(lab / f"label_sheet_v3_{second}.csv")[1]
                ids = sorted(set(A) & set(B))
                po, pa, pb = (np.mean([f(i) for i in ids]) for f in (lambda i: A[i] == B[i], A.get, B.get))
                pe = pa * pb + (1 - pa) * (1 - pb)
                txt += f"\n\nTwo annotators, {len(ids)} sentences: agreement {100 * po:.1f}%, Cohen's kappa {(po - pe) / (1 - pe):.2f}."
            parts.append(f"**{name.capitalize()}** ({len(gold)} sentences)\n\n{txt}")
            self.headline.append([f"Grounding, {name}", f"unsupported shown {rows[0][2]}% → {rows[-1][2].split()[0]}%"])
        self.add("Evidence grounding", "\n\n".join(parts) or f"Not run: no label files in {lab}.")


def cmd_evaluate(a, hb):
    want = SECTIONS if a.sections == "all" else a.sections.split(",")
    if set(want) - set(SECTIONS):
        raise SystemExit(f"sections: {', '.join(SECTIONS)} or all")
    E = Evaluation(a, hb)
    E.tune()
    for s in SECTIONS:
        if s in want:
            getattr(E, s)()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "EVAL.md").write_text(f"# Evaluation\n\n{time.strftime('%Y-%m-%d %H:%M')}, HPO {E.r.ont.version}. {E.data}\n\n"
                                 + table(["result", "value (95% CI)"], E.headline) + "\n\n" + "\n".join(E.md))
    (out / "eval.json").write_text(json.dumps({"headline": E.headline, **E.js}, indent=1, default=str))
    ids = [c.case_id for c in E.dev + E.test]
    pd.DataFrame({"case": ids, "split": [E.split[i] for i in ids], "evaluable": [E.ev[i] for i in ids],
                  **{k: [v.get(i) for i in ids] for k, v in E.ranks.items()}}).to_csv(out / "ranks.csv", index=False)
    print(f"[eval] {hb.wrote(out / 'EVAL.md')}")


def add_arguments(p, sub):
    s = sub.add_parser("evaluate", help="full evaluation on a frozen dev/test split (tuning on dev, results on test)")
    s.add_argument("--phenopackets", default="data/phenopacket-store")
    s.add_argument("--sample", type=int, default=0, help="cases (0 = all, at most 5 per disease)")
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--dev-share", type=float, default=0.3, help="share of disease/paper groups used for tuning")
    s.add_argument("--split-file", default="configs/eval_split_v1.json", help="fingerprint of the split, checked each run")
    s.add_argument("--refreeze", action="store_true", help="accept a new split (only if the data changed on purpose)")
    s.add_argument("--sections", default="all", help=f"comma list of {', '.join(SECTIONS)}, or all")
    s.add_argument("--learn", action="store_true", help="also learn evidence weights on dev and test them")
    s.add_argument("--candidates", type=int, default=50, help="top-k diseases re-scored by learned weights")
    s.add_argument("--l2", type=float, default=0.01,
                   help="pull of learned weights toward the defaults (0.01: picked by grouped CV inside dev)")
    s.add_argument("--sequential-cases", type=int, default=300)
    s.add_argument("--rounds", type=int, default=5)
    s.add_argument("--pmc", default="pmc_eval/manifest_v1.json")
    s.add_argument("--pmc-split", default="dev", choices=["dev", "test"])
    s.add_argument("--allow-test", action="store_true", help="needed for --pmc-split test")
    s.add_argument("--labels-dir", default="results/labels")
    s.add_argument("--out", default="results/eval")


def commands(hb) -> dict:
    return {"evaluate": lambda a: cmd_evaluate(a, hb)}
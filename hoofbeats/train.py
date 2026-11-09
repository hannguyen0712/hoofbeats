# learned evidence global weights on top of the likelihood ratios
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import Case, LRRanker, Ranking, Status
from .report import boot_ci, hit_at

HALLMARK = 0.8
FEATURES = ["present_exact", "present_partial", "present_unexplained", "absent_hallmark", "absent_other",
            "absent_excluded", "log_annotations", "log_references", "gene"]
DEFAULT_W = {"present_exact": 1.0, "present_partial": 1.0, "present_unexplained": math.log(0.05),
             "absent_hallmark": 1.0, "absent_other": 1.0, "absent_excluded": 1.0,
             "log_annotations": 0.0, "log_references": 0.0, "gene": 0.0}
# signs fixed so a weight can scale evidence but never flip its meaning
BOUNDS = {"present_exact": (0, 5), "present_partial": (0, 5), "present_unexplained": (-10, 0),
          "absent_hallmark": (0, 5), "absent_other": (0, 5), "absent_excluded": (0, 5),
          "log_annotations": (0, 0), "log_references": (0, 0), "gene": (0, 0)}   # genes off: near-answer ablation
# annotation/reference counts are off by default: leakage control strips the true disease's own-paper annotations
# but only one reference, so "few annotations per reference" points at the answer (--kb-size turns them on)
KB_SIZE = {"log_annotations": (-2, 2), "log_references": (-2, 2)}


def default_weights(r: LRRanker) -> dict:
    # the weights that reproduce the ranker's current absent-finding rule
    w = dict(DEFAULT_W)
    w["absent_hallmark"] = r.absent_weight if HALLMARK >= r.absent_min_freq else 0.0
    w["absent_other"] = r.absent_weight if r.absent_min_freq <= 0 else 0.0
    w["absent_excluded"] = r.absent_weight
    return w


def evidence_features(r: LRRanker, case: Case, d: str, exclude=frozenset()) -> np.ndarray:
    # one row per (case, disease): the raw score split by evidence type (every absent finding, weight 1)
    x = dict.fromkeys(FEATURES, 0.0)
    keep = r.absent_weight, r.absent_min_freq
    r.absent_weight, r.absent_min_freq = 1.0, 0.0
    try:
        contribs = r.contributions(case, d, exclude)
    finally:
        r.absent_weight, r.absent_min_freq = keep
    for c in contribs:
        if c.status == Status.PRESENT:
            if c.kb_term is None:
                x["present_unexplained"] += c.log_lr < 0          # out-of-scope findings score 0 and are skipped
            elif c.kb_term == c.hpo_id:
                x["present_exact"] += c.log_lr
            else:
                x["present_partial"] += c.log_lr
        elif c.kb_term is None:
            x["absent_excluded"] += c.log_lr                        # disease is annotated NOT: small bonus
        elif (c.kb_freq or 0) >= HALLMARK:
            x["absent_hallmark"] += c.log_lr
        else:
            x["absent_other"] += c.log_lr
    annos = r.db.annotations(d, exclude)
    x["log_annotations"] = math.log1p(len(annos))
    x["log_references"] = math.log1p(len({ref for a in annos for ref in a.refs}))
    x["gene"] = float(bool(set(case.genes) & r.db.disease_genes.get(d, set())))
    return np.array([x[f] for f in FEATURES])


@dataclass
class EvidenceWeights:
    w: dict
    meta: dict = field(default_factory=dict)

    @property
    def vec(self) -> np.ndarray:
        return np.array([self.w[f] for f in FEATURES])

    def save(self, path):
        import json
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"weights": self.w, "meta": self.meta}, indent=1))

    @classmethod
    def load(cls, path):
        import json
        obj = json.loads(Path(path).read_text())
        return cls({**DEFAULT_W, **obj["weights"]}, obj.get("meta", {}))


def candidates(r: LRRanker, case: Case, exclude=frozenset(), k=50, add_truth=False) -> tuple[list[str], np.ndarray, np.ndarray]:
    # top-k diseases by the default score (+ the truth when training), their features and fixed prior offset
    ranking = r.score(case, exclude)
    ds = [d for d, _, _ in ranking.top(k)]
    if add_truth:
        ds += [t for t in case.truth if t in r.d_index and t not in ds]
    X = np.array([evidence_features(r, case, d, exclude) for d in ds])
    prior = np.array([r.prior_weight * r.log_prior[r.d_index[d]] for d in ds])
    return ds, X, prior


def rerank_learned(r: LRRanker, case: Case, ew: EvidenceWeights, exclude=frozenset(), k=50) -> Ranking:
    # re-score the default top-k with learned weights; diseases below k keep their order underneath
    base = r.score(case, exclude)
    ds, X, prior = candidates(r, case, exclude, k)
    new = X @ ew.vec + prior
    idx = np.array([r.d_index[d] for d in ds])
    top = np.zeros(len(base.log_scores), bool)
    top[idx] = True
    s = base.log_scores.copy()
    if (~top).any():
        s[~top] = s[~top] - s[~top].max() + new.min() - 1.0
    s[idx] = new
    p = np.exp(s - s.max())
    return Ranking(r.diseases, s, p / p.sum(), np.lexsort((np.arange(len(s)), -s)))


def fit_weights(Xs: list[np.ndarray], priors: list[np.ndarray], gold: list[int], l2: float = 1.0,
                init: dict | None = None, bounds: dict | None = None) -> np.ndarray:
    # softmax over each case's candidates; L2 pull toward the default weights; signs bounded
    # present_exact is held at 1 and a free temperature T sets the overall scale, so weights are relative to an
    # exact match (otherwise the fit shrinks every weight to flatten the softmax and clips the small ones to 0)
    # returns the effective weights w / T (what re-scoring uses)
    from scipy.optimize import minimize
    w0 = np.array([(init or DEFAULT_W)[f] for f in FEATURES])
    w0[FEATURES.index("present_exact")] = 1.0
    scale = np.std(np.vstack(Xs), axis=0) + 1e-6               # penalty in units of each feature's spread
    n = len(Xs)

    def loss(v):
        w, T = v[:-1], math.exp(v[-1])
        tot, gw, gu = 0.0, np.zeros_like(w), 0.0
        for X, pr, g in zip(Xs, priors, gold):
            xw = X @ w
            z = xw / T + pr
            z = z - z.max()
            p = np.exp(z) / np.exp(z).sum()
            tot -= z[g] - math.log(np.exp(z).sum())
            e = p.copy(); e[g] -= 1
            gw += X.T @ e / T
            gu -= xw @ e / T
        d = (w - w0) * scale
        return tot / n + l2 * (d @ d), np.append(gw / n + 2 * l2 * d * scale, gu / n)

    bnd = [(1.0, 1.0) if f == "present_exact" else (bounds or BOUNDS)[f] for f in FEATURES] + [(0.0, 5.0)]   # T in [1, 150]: never sharper than the defaults
    res = minimize(loss, np.append(w0, 0.0), jac=True, method="L-BFGS-B", bounds=bnd)
    return res.x[:-1] / math.exp(res.x[-1])


def load_weights(a, run: dict | None = None):
    # weights from --weights, or None
    if not getattr(a, "weights", None):
        return None
    ew = EvidenceWeights.load(a.weights)
    m = ew.meta
    print(f"[weights] {a.weights}: trained on {m.get('dataset')} ({m.get('cases')} cases, {m.get('trained')})")
    if run is not None:
        run["settings"]["weights"] = a.weights
    return ew


def _groups(cases: list[Case]) -> list[int]:
    # cases sharing a disease or a paper end up in the same group (union-find)
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, c in enumerate(cases):
        for key in [*c.truth, *c.source_pmids]:
            parent[find(("c", i))] = find(("k", key))
    roots = {}
    return [roots.setdefault(find(("c", i)), len(roots)) for i in range(len(cases))]


def _train_cases(a, hb):
    # labelled cases for the chosen dataset, with the matching knowledge base
    if a.dataset == "phenopacket":
        r = hb.load_kb(a)
        return r, [c for c in hb.pick_cases(r, a.phenopackets, a.sample, seed=a.seed) if hb.evaluable(r, c)]
    if a.dataset == "medpix":
        r = hb.load_kb(a, ("OMIM", "ORPHA"))
        cases = hb.load_medpix(a.medpix, hb.DiseaseMatcher(r.db.names), hb.Extractor(r.ont), "all", a.labels)
        return r, [c for c in cases if c.truth and c.present][: a.sample or None]
    r = hb.load_kb(a, ("ORPHA",))
    icd, names = hb.orphadata_icd10(a.orphadata_xml) if a.orphadata_xml else (None, None)
    cases = hb.load_mimic(a.mimic, icd, names, r.db.disease_genes, common=a.profiles)
    return r, [c for c in cases if c.truth and c.present and any(t in r.db.names for t in c.truth)][: a.sample or None]


def _cand_rank(scores: np.ndarray, g: int, k: int) -> int | None:
    # rank of the truth among the first k candidates; ties count against it
    if g >= k:
        return None
    return int((np.delete(scores[:k], g) >= scores[g]).sum()) + 1


def cmd_train(a, hb):
    r, cases = _train_cases(a, hb)
    if len(cases) < 20:
        raise SystemExit(f"only {len(cases)} usable cases; need at least 20 to train")
    groups = _groups(cases)
    print(f"[train] {a.dataset}: {len(cases)} cases, {len(set(groups))} disease/paper groups, "
          f"{a.candidates} candidates per case, {a.folds} folds")
    data = []
    for i, c in enumerate(cases, 1):
        ex = hb.leak(c)
        ds, X, pr = candidates(r, c, ex, a.candidates, add_truth=True)
        g = next(j for j, d in enumerate(ds) if d in c.truth)
        data.append((X, pr, g, r.score(c, ex).rank_of(c.truth)))
        if i % 250 == 0 or i == len(cases):
            print(f"[train] features {i}/{len(cases)}", flush=True)
    in_top = sum(g < a.candidates for _, _, g, _ in data)
    rng = random.Random(a.seed)
    order = sorted(set(groups)); rng.shuffle(order)
    fold_of = {gid: i % a.folds for i, gid in enumerate(order)}
    dw = default_weights(r)
    w0 = np.array([dw[f] for f in FEATURES])
    bounds = {**BOUNDS, **(KB_SIZE if a.kb_size else {})}

    def fit(idx, l2):
        return fit_weights([data[i][0] for i in idx], [data[i][1] for i in idx], [data[i][2] for i in idx], l2, dw,
                           bounds)

    def nll(w, idx):
        tot = 0.0
        for i in idx:
            z = data[i][0] @ w + data[i][1]
            tot += np.log(np.exp(z - z.max()).sum()) + z.max() - z[data[i][2]]
        return tot / max(len(idx), 1)

    def pick_l2(idx):
        # inner cross-validation, also grouped
        sub = [groups[i] for i in idx]
        inner = {gid: j % 3 for j, gid in enumerate(sorted(set(sub)))}
        best = None
        for l2 in a.l2:
            loss = 0.0
            for f in range(3):
                tr = [i for i in idx if inner[groups[i]] != f]
                va = [i for i in idx if inner[groups[i]] == f]
                if tr and va:
                    loss += nll(fit(tr, l2), va)
            best = min(best or (loss, l2), (loss, l2))
        return best[1]

    base, learned, fold_w, held = [], [], [], []
    for f in range(a.folds):
        tr = [i for i in range(len(cases)) if fold_of[groups[i]] != f]
        te = [i for i in range(len(cases)) if fold_of[groups[i]] == f]
        l2 = pick_l2(tr)
        w = fit(tr, l2)
        fold_w.append(w)
        for i in te:
            X, pr, g, b = data[i]
            base.append(b)
            held.append(groups[i])
            learned.append(_cand_rank(X @ w + pr, g, a.candidates) or b)
        print(f"[train] fold {f + 1}/{a.folds}: {len(tr)} train, {len(te)} test cases, l2 {l2}")
    mb, ml = hb.topk(base), hb.topk(learned)
    diff = hit_at(learned, 1) - hit_at(base, 1)
    lo, hi = boot_ci(diff, groups=held)        # resample disease/paper groups
    print(f"\nHeld-out diseases and papers ({a.folds}-fold), truth in the top {a.candidates} for "
          f"{in_top}/{len(cases)} cases (re-scoring cannot help the rest):")
    print("| scoring | Top-1 | Top-3 | Top-10 | MRR |\n|---|---|---|---|---|")
    print(f"| default weights | {hb.fmt(mb)} |\n| learned weights | {hb.fmt(ml)} |")
    print(f"Top-1 change {100 * diff.mean():+.1f} points (95% CI {100 * lo:+.1f} to {100 * hi:+.1f})")
    hb.note("train", dataset=a.dataset, cases=len(cases), candidates=a.candidates, candidate_recall=in_top / len(cases),
         default=mb, learned=ml, top1_change=float(diff.mean()), top1_change_ci=[lo, hi])

    l2 = pick_l2(list(range(len(cases))))
    w = fit(list(range(len(cases))), l2)
    W = np.array(fold_w)
    print(f"\nFinal weights (all cases, l2 {l2}); fold spread shows how stable each one is:")
    print("| evidence | default | learned | fold min | fold max |\n|---|---|---|---|---|")
    pe = FEATURES.index("present_exact")
    rel, Wr = w / w[pe], W / W[:, pe:pe + 1]
    print(f"(relative to an exact present match; fitted temperature {1 / w[pe]:.2f})")
    for j, name in enumerate(FEATURES):
        print(f"| {name} | {w0[j]:.2f} | {rel[j]:.2f} | {Wr[:, j].min():.2f} | {Wr[:, j].max():.2f} |")
    meta = {"absent_weight": r.absent_weight, "absent_min_freq": r.absent_min_freq, "dataset": a.dataset, "kb_size": a.kb_size, "cases": len(cases), "candidates": a.candidates, "l2": l2, "folds": a.folds,
            "seed": a.seed, "hpo": hb.RUN["settings"].get("hpo"), "annotations": hb.RUN["settings"].get("annotations"),
            "trained": time.strftime("%Y-%m-%d %H:%M"), "cv_default": mb, "cv_learned": ml}
    EvidenceWeights(dict(zip(FEATURES, map(float, w))), meta).save(hb.wrote(Path(a.weights_out)))
    hb.note("weights", **dict(zip(FEATURES, map(float, w))))
    print(f"[train] weights -> {a.weights_out}  (use with --weights {a.weights_out})")


def add_arguments(p, sub):
    p.add_argument("--weights", help="learned evidence weights from `train` (re-scores the top candidates)")
    s = sub.add_parser("train", help="learn evidence weights from labelled cases (grouped cross-validation)")
    s.add_argument("--dataset", default="phenopacket", choices=["phenopacket", "medpix", "mimic"])
    s.add_argument("--phenopackets", default="data/phenopacket-store")
    s.add_argument("--medpix", default="data/medpix-src/MedPix-2-0")
    s.add_argument("--labels", help="MedPix CSV U_id,truth")
    s.add_argument("--mimic", help="folder containing hosp/")
    s.add_argument("--orphadata-xml")
    s.add_argument("--sample", type=int, default=500)
    s.add_argument("--candidates", type=int, default=50, help="top-k diseases re-scored per case")
    s.add_argument("--folds", type=int, default=5)
    s.add_argument("--l2", type=float, nargs="+", default=[0.01, 0.1, 1.0], help="pull toward default weights (picked by inner CV)")
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--kb-size", action="store_true", help="also learn weights on annotation/reference counts (see KB_SIZE)")
    s.add_argument("--out", default="results/train")
    s.add_argument("--weights-out", default="results/train/weights.json")


def commands(hb) -> dict:
    return {"train": lambda a: cmd_train(a, hb)}
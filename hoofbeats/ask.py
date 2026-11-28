# ask: targeted yes/no questions to the vision model (medpix-context condition "asked", --ask-*).
# After a first ranking (text + image description), the imaging findings the top diseases have, and that the case does
# not state yet, are asked about one by one, each twice: "Is there X?" and "Is the image free of X?". The two answers'
# yes-probabilities are combined as belief = (p_present + 1 - p_free) / 2, so a model that says yes to everything ends
# at 0.5, not at "present" (opposed questions, as in RadRef). A finding whose belief reaches --ask-threshold on any
# image is added as present. Only findings of the imaged body region are asked, and tumour types are never asked
# (the describing prompt does not name diseases either). Different model for the questions: --ask-model.
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .model import LRRanker, Ranking, Status

NEOPLASM = "HP:0002664"
# body region (matched on the image's location and modality) -> HPO subtrees an image of it can show
REGIONS = (
    (("brain", "neuro", "head", "skull", "cranial", "posterior fossa", "sella", "pituitar"),
     ("HP:0012443", "HP:0000929", "HP:0002143", "HP:0002597", "HP:0010766")),
    (("orbit", "eye"), ("HP:0012372", "HP:0012443")),
    (("spine", "spinal", "vertebra"), ("HP:0002143", "HP:0003468")),
    (("chest", "lung", "pulmonary", "thorax", "cardio", "heart", "mediastin"),
     ("HP:0002088", "HP:0030680", "HP:0002597")),
    (("abdomen", "pelvi", "gastro", "liver", "hepat", "kidney", "renal", "genitourinary", "bladder", "bowel", "adrenal",
      "pancrea", "spleen", "gallbladder"),
     ("HP:0410042", "HP:0012210", "HP:0000077", "HP:0001743", "HP:0012090", "HP:0002242", "HP:0005264", "HP:0011732",
      "HP:0001541", "HP:0002597", "HP:0010766")),
    (("musculoskeletal", "bone", "joint", "knee", "hip", "shoulder", "hand", "foot", "extremit", "skelet"),
     ("HP:0011842",)),
)
ALL_ROOTS = tuple(dict.fromkeys(r for _, roots in REGIONS for r in roots))

ASK_SYSTEM = ("You are an experienced, board-certified radiologist. Look at the image and answer the question with "
              "one word: Yes or No.")
ASK_PRESENT = "{intro}\n\nIs there {term} in {it}? Answer Yes or No."
ASK_FREE = "{intro}\n\nIs {it} free of {term}? Answer Yes or No."


@dataclass
class Asked:
    hpo_id: str
    name: str
    belief: float                       # best over the images asked
    p_present: float | None = None
    p_free: float | None = None
    per_image: list = field(default_factory=list)


def region_roots(imgs: list[dict], ont) -> set[str]:
    # HPO subtrees for the images' body regions (all regions when the labels name none)
    where = " ".join(f"{i.get('location') or ''} {i.get('modality') or ''}" for i in imgs).lower()
    roots = {r for keys, rs in REGIONS if any(k in where for k in keys) for r in rs}
    return {r for r in (roots or set(ALL_ROOTS)) if r in ont.terms}


def candidates(r: LRRanker, ranking: Ranking, imgs: list[dict], known: set[str], k: int = 5, n: int = 12) -> list[str]:
    # the findings to ask about: annotated to one of the top k diseases, under the images' body region, not a tumour
    # type, and neither stated nor implied by what the case already says (known: its HPO ids, any status). Ordered by
    # p(disease) x log(frequency / background), so findings that would move the leading diseases most come first.
    roots, ont = region_roots(imgs, r.ont), r.ont
    near = set()
    for t in known:
        near |= ont.ancestors(t) | ont.descendants(t)
    score: dict[str, float] = {}
    for d, _, p in ranking.top(k):
        for a in r.db.annotations(d):
            t = ont.resolve(a.hpo_id) if hasattr(ont, "resolve") else a.hpo_id
            if a.negated or not t or t in near or t not in r.t_index:
                continue
            anc = ont.ancestors(t)
            if NEOPLASM in anc or not (anc & roots):
                continue
            bg = float(r.bg[r.t_index[t]])
            s = max(p, 1e-3) * math.log(max(a.freq, 0.05) / bg)
            if s > score.get(t, -1e9):
                score[t] = s
    return [t for t, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))[:n]]


def belief(p_present: float | None, p_free: float | None) -> float | None:
    # opposed questions: (p_present + 1 - p_free) / 2; with one answer missing, the other alone
    if p_present is None and p_free is None:
        return None
    if p_free is None:
        return p_present
    if p_present is None:
        return 1 - p_free
    return (p_present + 1 - p_free) / 2


def ask_case(llm, r: LRRanker, ranking: Ranking, imgs: list[dict], known: set[str], k: int = 5, n: int = 12,
             per_call: int = 1) -> list[Asked]:
    # every candidate finding asked of every image group; the best belief over the groups is kept
    from .llm import _img_label
    terms = candidates(r, ranking, imgs, known, k, n)
    out = []
    groups = [imgs[i:i + max(per_call, 1)] for i in range(0, len(imgs), max(per_call, 1))]
    for t in terms:
        name = r.ont.name(t).lower()
        best = Asked(t, r.ont.name(t), -1.0)
        for grp in groups:
            if len(grp) == 1:
                intro, it = f"This is one medical image: {_img_label(grp[0])}.", "this image"
            else:
                intro = f"These are {len(grp)} images from one patient: " + "; ".join(_img_label(g) for g in grp) + "."
                it = "these images"
            paths = tuple(g["path"] for g in grp)
            pp = llm.yes_prob(ASK_PRESENT.format(intro=intro, term=name, it=it), ASK_SYSTEM, paths)
            pf = llm.yes_prob(ASK_FREE.format(intro=intro, term=name, it=it), ASK_SYSTEM, paths)
            b = belief(pp, pf)
            best.per_image.append({"images": [g.get("id", g["path"]) for g in grp], "p_present": pp, "p_free": pf,
                                   "belief": b})
            if b is not None and b > best.belief:
                best.belief, best.p_present, best.p_free = b, pp, pf
        if best.belief >= 0:
            out.append(best)
    return out


def asked_text(asked: list[Asked], threshold: float, label: str = "Image questions") -> str:
    # the findings answered present, written as their HPO names so Stage 1 reads exactly those
    keep = [a for a in asked if a.belief >= threshold]
    if not keep:
        return ""
    return f"{label} (answered yes, belief >= {threshold:g}):\n" + "\n".join(f"- {a.name}" for a in keep)


# Body-part filter for image findings. A finding read from an image description must belong to the organ systems
# of the imaged body part: a brain MRI gives no ECG or spine terms. Most findings outside the region are links the
# reader forced for words HPO has no term for (e.g. "midline shift" read as "Extreme axis deviation"), and each one
# moves hundreds of unrelated diseases above the true one. Wider than REGIONS (which picks what to ask about):
# whole organ systems, plus tumours, oedema and vessels, which can be anywhere.
ANY_REGION = ("HP:0002664", "HP:0011032", "HP:0002597")   # neoplasm, fluid regulation (oedema), vasculature
BODY_PARTS = (
    (("brain", "neuro", "nerve", "head", "skull", "cranial", "fossa", "sella", "pituitar", "orbit", "eye", "face",
      "neck"), ("HP:0000707", "HP:0000152", "HP:0000478")),
    (("spine", "spinal", "vertebra"), ("HP:0000707", "HP:0033127")),
    (("chest", "lung", "pulmonary", "thorax", "thoracic", "cardio", "heart", "mediastin", "breast"),
     ("HP:0002086", "HP:0001626", "HP:0045027")),
    (("abdomen", "abdominal", "pelvi", "gastro", "liver", "hepat", "kidney", "renal", "genitourinary", "bladder",
      "bowel", "adrenal", "pancrea", "spleen", "gallbladder"),
     ("HP:0025031", "HP:0000119", "HP:0001438", "HP:0000818", "HP:0001871")),
    (("musculoskeletal", "bone", "joint", "knee", "hip", "shoulder", "hand", "foot", "extremit", "skelet", "limb"),
     ("HP:0033127", "HP:0040064")),
)


def body_part_roots(images) -> set[str] | None:
    # the organ-system roots of the images' locations; None when no location is known (then nothing is filtered)
    words = " ".join(f"{i.get('location') or ''} {i.get('modality') or ''}" for i in images or []
                     if isinstance(i, dict)).lower()
    roots = {r for keys, rs in BODY_PARTS if any(k in words for k in keys) for r in rs}
    return roots | set(ANY_REGION) if roots else None


def in_body_part(ont, hpo_id: str, roots: set[str] | None) -> bool:
    if roots is None:
        return True
    t = ont.resolve(hpo_id)
    return t is None or bool((ont.ancestors(t) | {t}) & roots)

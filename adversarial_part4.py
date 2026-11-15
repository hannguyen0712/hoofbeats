"""Adversarial test set for the Part 4 checks: correct sentences written in other words (should be kept) and
sentences with a specific kind of error (should be removed), built from real evidence packs.

    python adversarial_part4.py 120            # build part4_adversarial.jsonl from 120 cases and score both checkers
    python adversarial_part4.py --score        # score the saved set again (e.g. after changing the checks)
    python adversarial_part4.py --score --nli cross-encoder/nli-deberta-v3-base   # also an NLI model (needs transformers;
                                                                                    # run on Sol or a laptop, not here)
Every sentence is judged against its cited evidence only; "should_pass" is set by how the sentence was made.
"""
import json, random, re, sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from hoofbeats.__main__ import load_kb
from hoofbeats.data import load_phenopackets
from hoofbeats.model import Case, Finding, Status, explain
import grounding as g

OUT = Path("part4_adversarial.jsonl")
N = next((int(x) for x in sys.argv[1:] if x.isdigit()), 120)
r = load_kb(SimpleNamespace(kb="data/kb", common="none", prior_weight=1.0, absent_weight=0.5, absent_min_freq=0.8,
                            use_onset=False, use_sex=False))
rng = random.Random(7)
ALL_NAMES = sorted(set(r.db.names.values()))


def nm(item):
    return r.ont.name(r.ont.resolve(item.hpo_id)).lower()


def freq_of(item):
    m = re.search(r"~(\d+)%", item.text)
    return m.group(1) if m else None


def lr_of(item):
    m = re.search(r"\bLR ([\d.]+)", item.text)
    return m.group(1) if m else None


def setup(f):
    case = load_phenopackets(f)[0]
    if not case.findings:
        return None
    ex = frozenset(case.source_pmids)
    ranking = r.score(case, ex)
    (d, _, p), (runner, _, _) = ranking.top(2)
    claims = explain(r, case, ranking, 1, ex, max_claims=8)[0][2]
    return case, ex, d, runner, g.build_pack(r, case, d, p, ex, runner, claims)


def make(f, case, d, runner, pack):
    D, R = r.db.names[d], r.db.names[runner]
    pid = {i.hpo_id: i.id for i in pack.values() if i.kind == "patient"}
    out = []

    def add(cat, ok, section, text, **extra):
        out.append({"category": cat, "should_pass": ok, "section": section, "sentence": text, "case": f.name,
                    "disease": d, "runner_up": runner, **extra})

    for k in pack.values():
        if k.kind != "knowledge" or not k.hpo_id:
            continue
        P, F, fr, lr = pid.get(k.hpo_id), nm(k), freq_of(k), lr_of(k)
        syns = [s.lower() for s in (getattr(r.ont.terms[r.ont.resolve(k.hpo_id)], "synonyms", []) or [])]
        if k.role == "for" and P and fr and k.via:
            V = r.ont.name(k.via).lower()
            add("paraphrase: broader term kept", True, "supports",
                f"The patient has {F}; about {fr}% of people with {D} have {V}, the broader HPO term it falls under [{P}, {k.id}].")
            add("broader term dropped", False, "supports",
                f"The patient has {F}, which is seen in about {fr}% of people with {D} [{P}, {k.id}].")
        elif k.role == "for" and P and fr:
            add("paraphrase: supports", True, "supports",
                f"The patient has {F}, which is seen in about {fr}% of people with {D} [{P}, {k.id}].")
            if syns:
                add("paraphrase: synonym", True, "supports",
                    f"This patient shows {rng.choice(syns)}, a feature of {D} in ~{fr}% of cases [{P}, {k.id}].")
            add("overclaim: certainty", False, "supports", f"The patient has {F}, which " + rng.choice(
                ["proves", "confirms", "establishes", "is diagnostic of", "is pathognomonic for"]) + f" {D} [{P}, {k.id}].")
            add("overclaim: causal", False, "supports", rng.choice(
                [f"{D} causes the patient's {F} [{P}, {k.id}].", f"The patient's {F} is due to {D} [{P}, {k.id}].",
                 f"The patient has {F} because of {D} [{P}, {k.id}]."]))
            other = rng.choice([n for n in ALL_NAMES if n not in (D, R) and len(n) >= 8])
            add("wrong disease", False, "supports",
                f"The patient has {F}, which is seen in about {fr}% of people with {other} [{P}, {k.id}].")
            extra = [i for i in pack.values() if i.role == "missing"]
            if extra:
                add("extra unsupported clause", False, "supports",
                    f"The patient has {F}, seen in about {fr}% of people with {D}, and also has {nm(extra[0])} [{P}, {k.id}].")
            if lr:
                add("number: LR given as a frequency", False, "supports",
                    f"The patient has {F}, which {D} shows in ~{lr}% of cases [{P}, {k.id}].")
        if k.role == "against" and P and fr:
            add("paraphrase: against", True, "against",
                f"{F.capitalize()} is absent in this patient, although {D} usually has it (~{fr}%) [{P}, {k.id}].")
            add("negation not tied to the finding", False, "against",
                f"No other explanation was found, but the patient has {F}, which {D} shows in ~{fr}% of cases [{P}, {k.id}].")
            add("wrong section", False, "supports",
                f"{F.capitalize()} is absent in this patient, although {D} usually has it (~{fr}%) [{P}, {k.id}].")
        if k.role == "missing" and fr:
            add("paraphrase: not recorded", True, "expected_but_missing",
                f"{F.capitalize()} is common in {D} (~{fr}%), but it is not mentioned in the record [{k.id}].")
            add("not recorded stated as absent", False, "expected_but_missing",
                f"The patient does not have {F}, which {D} shows in ~{fr}% of cases [{k.id}].")
            add("not recorded stated as present", False, "expected_but_missing",
                f"The patient has {F}, which {D} shows in ~{fr}% of cases [{k.id}].")
            add("wrong section", False, "supports",
                f"{F.capitalize()} is common in {D} (~{fr}%), but it is not mentioned in the record [{k.id}].")
        if k.role == "compare" and P:
            a, b = re.findall(r"likelihood ratio ([\d.]+) against ([\d.]+)", k.text)[0]
            add("paraphrase: vs runner-up", True, "vs_runner_up",
                f"{F.capitalize()} favours {D} over {R} (likelihood ratio {a} vs {b}) [{P}, {k.id}].")
            add("number: swapped LRs", False, "vs_runner_up",
                f"{F.capitalize()} fits {D} better than {R}: likelihood ratio {b} against {a} [{P}, {k.id}]." if a != b else
                f"{F.capitalize()} fits {D} better than {R}: likelihood ratio {a} against 1.00 [{P}, {k.id}].")
    # family history and uncertain findings: the same finding, recorded with another status
    for k in pack.values():
        P = pid.get(k.hpo_id) if k.kind == "knowledge" and k.role == "for" and not k.via else None
        if P:
            F, fr = nm(k), freq_of(k)
            for status, good in (("family", f"The patient's mother has {F}, a feature of {D} (~{fr}%) [{P}, {k.id}]."),
                                 ("uncertain", f"Possible {F} was noted; {D} shows it in ~{fr}% of cases [{P}, {k.id}].")):
                add(f"paraphrase: {status} finding", True, "supports", good, status_override=[P, status])
                add(f"{status} finding stated as the patient's", False, "supports",
                    f"The patient has {F}, which {D} shows in ~{fr}% of cases [{P}, {k.id}].", status_override=[P, status])
            break
    return out


HANDWRITTEN = [   # correct paraphrases in other words, Marfan note (rules reader): the name-matching V3 rejects these
    ("supports", "The echo shows a widened aortic root, which about 77% of people with Marfan syndrome have [P1, K1]."),
    ("supports", "His feet are flat, as in about 51% of people with Marfan syndrome [P6, K6]."),
    ("supports", "His arms span more than his height, which is seen in ~63% of people with Marfan syndrome [P8, K8]."),
    ("supports", "The mitral valve leaks, as it does in ~57% of people with Marfan syndrome [P4, K4]."),
    ("supports", "His teeth are crowded together, which Marfan syndrome shows in ~16% of cases [P7, K7]."),
    ("expected_but_missing", "Widening of the sac around the spinal cord, seen in ~80% of Marfan syndrome, is not mentioned in the record [K9]."),
]


def setup_any(name):
    if name == "marfan_note.json":
        from hoofbeats.data import Extractor
        from hoofbeats.intake import load_case
        case = load_case("examples/marfan_note.json", r.ont, Extractor(r.ont), disease_names=r.db.names)
        ex = frozenset(case.source_pmids)
        ranking = r.score(case, ex)
        (d, _, p), (runner, _, _) = ranking.top(2)
        claims = explain(r, case, ranking, 1, ex, max_claims=8)[0][2]
        return case, ex, d, runner, g.build_pack(r, case, d, p, ex, runner, claims)
    return setup(next(Path("data/phenopacket-store/0.1.27").glob(f"*/{name}")))


def context(x, cache):
    if x["case"] not in cache:
        cache[x["case"]] = setup_any(x["case"])
    case, ex, d, runner, pack = cache[x["case"]]
    if x.get("status_override"):
        P, status = x["status_override"]
        item = pack[P]
        pack = dict(pack) | {P: replace(item, status=status)}
        case = Case(case.case_id, [Finding(f.hpo_id, Status(status), f.source, f.evidence)
                                   if r.ont.resolve(f.hpo_id) == item.hpo_id else f for f in case.findings],
                    case.truth, case.genes, case.source_pmids, case.meta)
    return case, d, runner, pack


def score(rows, nli=None):
    cache, res = {}, defaultdict(Counter)
    for x in rows:
        case, d, runner, pack = context(x, cache)
        names = [r.db.names[d], r.db.names[runner]]
        basic = g.check_sentence(x["sentence"], pack, r, case, names, section=x["section"], strict=False)["ok"]
        strict = g.check_sentence(x["sentence"], pack, r, case, names, section=x["section"])
        cat = res[(x["should_pass"], x["category"])]
        cat["n"] += 1
        cat["basic right"] += basic == x["should_pass"]
        cat["strict right"] += strict["ok"] == x["should_pass"]
        if nli:
            ev = " ".join(pack[i].text for i in g.cites(x["sentence"]) if i in pack)
            ok = nli(ev, g.strip_cites(x["sentence"]))
            cat["nli right"] += ok == x["should_pass"]
            cat["strict+nli right"] += (strict["ok"] and ok) == x["should_pass"]
        if strict["ok"] != x["should_pass"] and cat["shown"] < 2:
            cat["shown"] += 1
            print(f"  strict wrong [{x['category']}]: {x['sentence'][:150]} -> {strict.get('reason') or 'kept'}")
    cols = ["basic right", "strict right"] + (["nli right", "strict+nli right"] if nli else [])
    print("\n| should be | category | n | " + " | ".join(c.replace(" right", "") for c in cols) + " |")
    print("|---|---|---|" + "---|" * len(cols))
    for (ok, cat), c in sorted(res.items(), key=lambda kv: (not kv[0][0], kv[0][1])):
        print(f"| {'kept' if ok else 'removed'} | {cat} | {c['n']} | " +
              " | ".join(f"{c[k] / c['n']:.0%}" for k in cols) + " |")
    for ok in (True, False):
        tot = [c for (o, _), c in res.items() if o == ok]
        n = sum(c["n"] for c in tot)
        print(f"{'correct sentences kept' if ok else 'wrong sentences removed'}: " +
              ", ".join(f"{k.replace(' right', '')} {sum(c[k] for c in tot)}/{n} ({sum(c[k] for c in tot) / n:.1%})" for k in cols))


def nli_model(name):
    from transformers import pipeline
    clf = pipeline("text-classification", model=name, top_k=None)

    def entails(premise, hypothesis, threshold=0.5):
        scores = {d["label"].lower(): d["score"] for d in clf({"text": premise, "text_pair": hypothesis})}
        return scores.get("entailment", 0.0) >= threshold
    return entails


if "--score" not in sys.argv:
    files = random.Random(0).sample(sorted(Path("data/phenopacket-store/0.1.27").glob("*/*.json")), N)
    rows = []
    for f in files:
        s = setup(f)
        if s:
            case, ex, d, runner, pack = s
            rows += make(f, case, d, runner, pack)
    # at most 60 per category, so no single kind dominates
    by = defaultdict(list)
    for x in rows:
        by[x["category"]].append(x)
    rows = [x for cat in sorted(by) for x in random.Random(1).sample(by[cat], min(60, len(by[cat])))]
    case, ex, d, runner, pack = setup_any("marfan_note.json")
    rows += [{"category": "paraphrase: other words (hand-written)", "should_pass": True, "section": sec, "sentence": t,
              "case": "marfan_note.json", "disease": d, "runner_up": runner} for sec, t in HANDWRITTEN]
    OUT.write_text("".join(json.dumps(x) + "\n" for x in rows))
    print(f"{len(rows)} sentences in {len(by)} categories -> {OUT}")
rows = [json.loads(l) for l in OUT.read_text().splitlines()]
nli = nli_model(sys.argv[sys.argv.index("--nli") + 1]) if "--nli" in sys.argv else None
score(rows, nli)

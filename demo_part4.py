"""Part 4 pipeline for one case: Stage 1 -> ranking -> evidence pack -> explanation (model, else template)
-> checks -> suggested tests (Part 3). Saves results/part4_demo.json for review_app.py.

    python demo_part4.py examples/marfan_note.json [--drop HP:0002616 ...] [--results results/part4_results.json]
    --results: test results entered by the clinician, [{"test": ..., "result": "positive"/"negative", "details": ...}]
    model: LLM_ENDPOINT, LLM_MODEL, PROMPT_VERSION (default v3); LLM_OFFLINE=1 uses saved replies only
    images: VISION_MODEL (+ VISION_ENDPOINT) names a local vision model that describes them (Stage A); without one
    the images are kept with the case but not described
"""
import argparse, json, os, re
from pathlib import Path
from types import SimpleNamespace
from hoofbeats.__main__ import load_kb
from hoofbeats.data import Extractor, load_phenopackets
from hoofbeats.intake import load_case
from hoofbeats.llm import MedGemma
from hoofbeats.model import Case, explain
import grounding as g
from hoofbeats.workup import ranker_probs
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1500"))   # same in llm_part4.py and walkthrough_part4.py (cache key)

ap = argparse.ArgumentParser()
ap.add_argument("case", nargs="?", default="examples/marfan_note.json")
ap.add_argument("--drop", nargs="*", default=[], help="HPO ids the clinician removed")
ap.add_argument("--results", default=None, help="JSON file with test results entered by the clinician")
ap.add_argument("--k", type=int, default=3)
a = ap.parse_args()

r = load_kb(SimpleNamespace(kb="data/kb", common=os.environ.get("COMMON_PROFILES", "none"), prior_weight=1.0, absent_weight=0.5, absent_min_freq=0.8,
                            use_onset=False, use_sex=False))
ext = Extractor(r.ont)
try:
    from hoofbeats.workup import done_texts, load_workupper, ranker_probs, workup_json
    wk = load_workupper(r, "data/kb")             # Part 3; None if the test data is not downloaded
except Exception as e:
    print(f"[part4] test suggestions unavailable: {e}")
    wk = None

VERSION = os.environ.get("PROMPT_VERSION", "v3")
make_prompt = {"v1": g.make_prompt, "v2": g.make_prompt_v2, "v3": g.make_prompt_v3}[VERSION]
llm = MedGemma(os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
               os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"),
               max_new_tokens=MAX_TOKENS, timeout=120, cache_dir="results/llm_cache",
               extra={"response_format": {"type": "json_object"}}, offline=os.environ.get("LLM_OFFLINE") == "1")
from hoofbeats.context import DEFAULT_CONTEXT_MODEL, DEFAULT_CONTEXT_ENDPOINT, VISION_NAMES
main_model = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
if os.environ.get("VISION_MODEL"):
    vision_model = os.environ["VISION_MODEL"]
    vision_endpoint = os.environ.get("VISION_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
elif VISION_NAMES.search(main_model):
    vision_model = main_model
    vision_endpoint = os.environ.get("LLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
else:
    vision_model, vision_endpoint = DEFAULT_CONTEXT_MODEL, DEFAULT_CONTEXT_ENDPOINT
describer = MedGemma(vision_model, vision_endpoint, max_new_tokens=1500, timeout=300, vision=True,
                     cache_dir="results/llm_cache", offline=os.environ.get("LLM_OFFLINE") == "1")


def load(path):
    raw = json.loads(Path(path).read_text()) if Path(path).suffix == ".json" else {}
    if "case_id" not in raw:
        return load_phenopackets(path)[0]                               # phenopacket
    if raw.get("images") and describer is not None:                     # images described first, then Stage 1
        try:
            c = load_case(path, r.ont, ext, disease_names=r.db.names, describer=describer)
            text = (c.meta.get("described") or {}).get("text")
            IMAGES.update(described_by=describer.model if text else None,
                          note="" if text else "The image model gave no description for these images.")
            return c
        except Exception as e:
            IMAGES["note"] = f"The image model could not be used ({type(e).__name__}); the images were not described."
    if raw.get("images"):                                               # no image model here: keep the images
        IMAGES["note"] = IMAGES["note"] or "No image model is running here, so the images are saved but not described."
        tmp = Path("results/_case_without_images.json")
        tmp.write_text(json.dumps({k: v for k, v in raw.items() if k != "images"}))
        return load_case(str(tmp), r.ont, ext, disease_names=r.db.names)
    return load_case(path, r.ont, ext, disease_names=r.db.names)        # note / case file -> findings (Stage 1)


def analyse(path, drop):
    case = load(path)
    if drop:                                                            # findings the clinician removed
        case = Case(case.case_id, [f for f in case.findings if r.ont.resolve(f.hpo_id) not in drop],
                    case.truth, case.genes, case.source_pmids, case.meta)
    ex = frozenset(case.source_pmids)                                   # leakage control
    ranking = r.score(case, ex)
    calibrated = ranker_probs(ranking) if wk is not None else None
    rare, common = [], []
    for d, _, p in ranking.top(400):
        (common if d in r.db.prior else rare).append((d, p))
    lists = {"rare": rare[:a.k], "common": common[:a.k]}
    tests = {"diseases": {}, "separate": []}
    if wk is not None:
        shown = [d for lst in lists.values() for d, _ in lst]
        case.meta["done_texts"] = done_texts(path)
        w = workup_json(wk.build(case, ranking, ex, len(shown), 3, shown=shown))
        tests = {"diseases": {x["id"]: x for x in w["diseases"]}, "separate": w["separate"]}
    prob = lambda d, p: float(calibrated[r.d_index[d]]) if calibrated is not None else p
    return case, ex, ranking, lists, tests, prob


dropped = {r.ont.resolve(h) for h in a.drop}
case_path = a.case
followup = None
results = json.loads(Path(a.results).read_text()) if a.results and Path(a.results).exists() else []
if results and Path(a.case).suffix == ".json" and "case_id" in json.loads(Path(a.case).read_text()):
    # Part 3's follow-up: read the results, turn them into findings, and analyse the merged case again
    from hoofbeats.followup import interpret, merge
    c0, ex0, rk0, lists0, tests0, prob0 = analyse(a.case, dropped)
    before = {"top": [{"id": d} for lst in lists0.values() for d, _ in lst], "tests": tests0}
    terms, genes, rows = interpret(wk, before, c0, ex0, results)
    orig = json.loads(Path(a.case).read_text())
    merged = merge(orig, {"test_results": results}, Path(a.case).parent.resolve(), Path(a.case).parent.resolve())
    merged["hpo_terms"] = list(merged.get("hpo_terms") or []) + terms
    merged["genes"] = list(dict.fromkeys(list(merged.get("genes") or []) + genes))
    case_path = "results/part4_followup_case.json"
    Path(case_path).write_text(json.dumps(merged, indent=1))
    r.use_genes = bool(merged["genes"])
    followup = {"results": rows, "before": {d: prob0(d, p) for lst in lists0.values() for d, p in lst}}

case, ex, ranking, lists, tests, prob = analyse(case_path, dropped)
if followup:
    followup["after"] = {d: prob(d, p) for lst in lists.values() for d, p in lst}
    followup["before"] = {d: followup["before"].get(d) for d in set(followup["before"]) | set(followup["after"])}
    # a gene result classified pathogenic / likely pathogenic confirms the shown diseases of that gene
    causal = {gn for x in rows if x.get("kind") == "positive" and (x.get("classification") or "") in
              ("pathogenic", "likely pathogenic") for gn in x.get("genes", [])}
    hits = [d for d in followup["after"] if wk is not None and causal & set(wk.genes.get(d, ()))]
    followup["confirmed"], followup["same_gene"] = hits[:1], hits[1:]     # several diseases of one gene: the best-ranked


def explain_with_model(pack, d, p, runner):
    try:
        trace = g.parse_trace(llm.chat(make_prompt(r, d, p, pack, runner)))
    except Exception as e:                                              # server not running, timeout
        print(f"   model not reachable ({type(e).__name__}); using the template explanation")
        trace = {}
    if sum(len(v) for v in trace.values()):
        return trace, "model"
    return g.template_trace(pack), "template"


raw = json.loads(Path(case_path).read_text())
texts = {k: v for k, v in raw.items() if k == "note_text" and isinstance(v, str) and v}
if case.meta.get("image_context") or raw.get("image_context"):      # the imaging report (+ the images' description)
    texts["image_context"] = case.meta.get("image_context") or raw["image_context"]
prov = case.meta.get("provenance", {})
# structured inputs (labs, vitals, terms entered by the clinician) as one more document, one line per value,
# so that their evidence can be shown and highlighted like the note's
lines = list(dict.fromkeys(e["quote"] for h, es in prov.items() if h.startswith("HP:") for e in es
                           if e.get("start") is None and e.get("quote")))
if lines:
    texts["structured_data"] = "\n".join(lines)


def spans_for(hpo_id):
    out = []
    for e in prov.get(hpo_id, []):
        if e.get("field") in texts and e.get("start") is not None:
            out.append({"field": e["field"], "start": e["start"], "end": e["end"], "quote": e["quote"],
                        "cue": e.get("status_cue"), "grounding": e.get("grounding"), "source": e.get("source")})
        elif e.get("quote") and "structured_data" in texts:
            start = texts["structured_data"].find(e["quote"])
            if start >= 0:
                out.append({"field": "structured_data", "start": start, "end": start + len(e["quote"]),
                            "quote": e["quote"], "cue": None, "grounding": e.get("grounding"), "source": e.get("source")})
    return out


def item_json(i):
    d = vars(i) | {"refs": list(i.refs)}
    if i.kind == "patient":
        d["spans"] = spans_for(i.hpo_id)
    return d


def strength(lr):
    return "very strong" if lr >= 100 else "strong" if lr >= 10 else "moderate" if lr >= 3 else "weak"


def key_points(pack):
    # plain-language summary straight from the evidence pack (no model): the strongest findings that fit,
    # and the usual features that are not recorded
    p_for = {i.hpo_id: i.id for i in pack.values() if i.kind == "patient"}
    fits, missing, against = [], [], []
    for i in pack.values():
        if i.kind != "knowledge":
            continue
        freq = re.search(r"~(\d+)%", i.text)
        name = r.ont.name(r.ont.resolve(i.hpo_id)) if i.hpo_id else ""
        if i.role == "for":
            lr = re.search(r"LR ([\d.]+)", i.text)
            lr = float(lr.group(1)) if lr else 1.0
            fits.append({"finding": name, "freq": int(freq.group(1)) if freq else None, "strength": strength(lr),
                         "lr": lr, "ids": [x for x in (p_for.get(i.hpo_id), i.id) if x],
                         "via": r.ont.name(i.via) if getattr(i, "via", "") else ""})
        elif i.role == "against":                               # evidence that counts against the disease
            lr = re.search(r"LR ([\d.]+)", i.text)
            against.append({"finding": name, "freq": int(freq.group(1)) if freq else None,
                            "lr": float(lr.group(1)) if lr else None, "absent": i.text.startswith("Lacks"),
                            "ids": [x for x in (p_for.get(i.hpo_id), i.id) if x]})
        elif i.role == "missing":
            missing.append({"finding": name, "freq": int(freq.group(1)) if freq else None, "ids": [i.id]})
    fits.sort(key=lambda x: -x["lr"])
    return fits, missing, against


details = raw.get("details") or {}
out = {"case": case.case_id, "case_path": a.case, "truth": case.truth, "dropped": sorted(dropped),
       "patient": {"age": raw.get("age") or details.get("age"), "sex": raw.get("sex") or details.get("sex")},
       "source": {"kind": "clinical note" if "note_text" in texts else "structured record", "texts": texts},
       "findings": [{"hpo_id": r.ont.resolve(f.hpo_id), "name": r.ont.name(r.ont.resolve(f.hpo_id)),
                     "status": f.status.value, "words": f.evidence or ""} for f in case.findings
                    if r.ont.resolve(f.hpo_id)],
       "images": [{"path": x["path"], "modality": x.get("modality", ""), "name": Path(x["path"]).name}
                  for x in raw.get("images") or []],
       "images_described_by": IMAGES["described_by"], "images_note": IMAGES["note"],
       "separate": tests["separate"], "followup": followup, "can_enter_results": "case_id" in json.loads(Path(a.case).read_text()),
       "lists": {}}
order = [d for d, _, _ in ranking.top(400)]
for label, lst in lists.items():
    out["lists"][label] = []
    for n, (d, p) in enumerate(lst, 1):
        runner = order[order.index(d) + 1]
        claims = explain(r, case, ranking, order.index(d) + 1, ex, max_claims=8)[order.index(d)][2]
        pack = g.build_pack(r, case, d, p, ex, runner, claims)
        trace, source = explain_with_model(pack, d, float(ranker_probs(ranking)[r.d_index[d]]), runner)
        kept, removed = g.check_trace(trace, pack, r, case, [r.db.names[d], r.db.names[runner]])
        fits, missing, against = key_points(pack)
        print(f"{label} {n}. {r.db.names[d]} ({d}) {prob(d, p):.1%} | {source} | kept {sum(len(v) for v in kept.values())}, "
              f"removed {len(removed)}")
        out["lists"][label].append({"id": d, "name": r.db.names[d], "probability": prob(d, p), "ranker_score": p,
                                    "runner_up": runner, "runner_up_name": r.db.names[runner],
                                    "explanation_source": source, "fits": fits, "missing": missing, "against": against,
                                    "pack": [item_json(i) for i in pack.values()], "explanation": kept,
                                    "removed": removed, "tests": tests["diseases"].get(d)})

Path("results").mkdir(exist_ok=True)
Path("results/part4_demo.json").write_text(json.dumps(out, indent=1))
print("saved results/part4_demo.json")

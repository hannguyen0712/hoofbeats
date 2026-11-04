# Image-context stage (Stage A): its model options, the `multimodal` command and the `medpix-context` experiment.
# Stage A (llm.image_context) describes each image; the description goes into the case's image_context field, Stage 1
# reads it like any note, and the ranker scores the findings with everything else. No model ranks diagnoses itself.
# Kept out of __main__.py, which only registers these (add_arguments, commands). Functions here get `hb`, the
# hoofbeats command-line module, for its shared helpers (run log, load_kb, read_case, topk, ...): passing it avoids a
# circular import and a second copy of the run log when hoofbeats runs as `python -m hoofbeats`.
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pandas as pd

from .data import DiseaseMatcher, Extractor, load_medpix, orphadata_icd10
from .model import LRRanker, Status


# the Stage A model when neither --context-model nor a vision --llm-model is given: MedGemma on a local Ollama
DEFAULT_CONTEXT_MODEL = "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M"
DEFAULT_CONTEXT_ENDPOINT = "http://localhost:11434/v1/chat/completions"
# model names that read images (used when --llm-vision is auto); a name matching none is treated as text-only
VISION_NAMES = re.compile(r"medgemma(?!.*27b-text)|gemini|gemma-?3|gpt-4o|gpt-4\.1|gpt-5|claude|llava|vision|"
                          r"[-_.]vl\b|qwen\d?(\.\d)?-?vl|pixtral|llama-?3\.2-?\d+b-vision|minicpm-v|moondream", re.I)


def reads_images(a) -> bool:
    # does the main model (--llm-model) read images? --llm-vision yes/no decides; auto goes by the name
    if not a.llm_model:
        return False
    if getattr(a, "llm_vision", "auto") != "auto":
        return a.llm_vision == "yes"
    return bool(VISION_NAMES.search(a.llm_model))


def context_choice(a) -> tuple[str, str | None, str]:
    # (model, endpoint, why): --context-model; else the main model if it reads images; else the default
    if getattr(a, "context_model", None):
        return a.context_model, a.context_endpoint or a.llm_endpoint, "--context-model"
    if reads_images(a):
        return a.llm_model, a.llm_endpoint, "the main model (--llm-model), which reads images"
    return DEFAULT_CONTEXT_MODEL, DEFAULT_CONTEXT_ENDPOINT, "the default (no --context-model, and the main model is text-only)"


def context_llm(hb, a):
    # the Stage A vision model: --context-model, else the main model if it reads images, else MedGemma on a local
    # Ollama. Its key and extra fields default to the --llm ones when the endpoint is the same.
    import os
    from .llm import MedGemma, with_json_mode
    model, endpoint, why = context_choice(a)
    same = endpoint == a.llm_endpoint
    key_env = a.context_key_env or (a.llm_key_env if same else None)
    key = os.environ.get(key_env) if key_env else None
    if key_env and not key:
        raise SystemExit(f"environment variable {key_env} is not set")
    extra = json.loads(a.context_extra) if a.context_extra else (json.loads(a.llm_extra) if a.llm_extra and same else None)
    extra = with_json_mode(a.json_mode, endpoint, extra, model)
    cache = None if a.llm_cache == "none" else Path(a.llm_cache if a.llm_cache != "auto" else Path(hb._out_dir(a)) / "llm_cache")
    llm = MedGemma(model, endpoint, a.context_max_tokens or a.llm_max_tokens, key, True, extra, a.llm_rpm,
                   cache, a.dry_run, a.llm_timeout, offline=a.llm_offline)
    hb.RUN["settings"].update({"context_model": llm.model, "context_endpoint": llm.endpoint or "in-process"})
    print(f"[llm] image context (Stage A) by {llm.model} via {llm.endpoint or 'transformers (in-process)'}: {why}")
    if not llm.is_local:
        print("[llm] remote service: only public datasets will be sent; credentialed data is refused")
    return llm


def context_usage(llm) -> str:
    u = llm.usage()
    if llm.dry_run:
        return f"[llm] image context: would send {u['would_send']} requests to {llm.model} ({u['cache_hits']} already cached)"
    return f"[llm] image context: {u['requests']} requests sent to {llm.model}, {u['cache_hits']} answered from cache"


STUCK = ("MedGemma 1.5 often gets stuck repeating itself while it thinks. Use --json-mode (the server then returns "
         "JSON only), keep --llm-max-tokens (or --context-max-tokens) at 4096 or more, and set Ollama's context "
         "length to 16k.")


def _nth(values, i: int) -> str:
    # per-image option: the i-th value, or the only value for every image
    if not values:
        return ""
    return values[i] if i < len(values) else values[0] if len(values) == 1 else ""


def cmd_multimodal(a, hb):
    # one patient from options: writes a case file with its images and runs it through diagnose
    # (Stage A -> Stage 1 -> disease scoring -> evidence -> next checks)
    text = Path(a.text_file).read_text(encoding="utf-8") if a.text_file else (a.text or "")
    details = {}
    for kv in a.detail or []:
        if "=" not in kv:
            raise SystemExit(f"--detail takes key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        details[k.strip()] = v.strip()
    imgs = []
    for i, p in enumerate(a.image or []):
        if not Path(p).is_file():
            raise SystemExit(f"image not found: {p}")
        imgs.append({k: v for k, v in {"path": str(Path(p).resolve()), "modality": _nth(a.modality, i),
                                      "plane": _nth(a.plane, i), "location": _nth(a.location, i)}.items() if v})
    if not (text.strip() or imgs or details or a.age or a.sex):
        raise SystemExit("give clinical text (--text / --text-file), patient details (--age, --sex, --detail) "
                         "or images (--image)")
    obj = {"case_id": a.case_id, "note_text": text, "age": a.age, "sex": a.sex, "details": details,
           "genes": [g.strip() for g in (a.genes or "").split(",") if g.strip()], "images": imgs,
           "policy": "local_only" if a.local_only else "open"}
    obj = {k: v for k, v in obj.items() if v not in (None, "", [], {})}
    path = Path(a.out) / "multimodal" / f"{a.case_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"[multimodal] case file -> {hb.wrote(path)} (run it again with: diagnose {path})")
    a.case = [str(path)]
    hb.cmd_diagnose(a)


# what Stage 1 reads in each medpix-context condition; the ranker then scores the findings
MM_CONDITIONS = {"text": "history/exam text",
                 "context": "text + image context (Stage A)",
                 "reference": "text + radiologist findings (upper bound for a context)"}


def medpix_context_case(c, conds, ctx_llm, r: LRRanker, ext, max_images=2, per_call=1, agreement=None,
                        reader=None) -> list[dict]:
    # one MedPix case under each condition: the same Stage 1 (parse_case) and the same ranker, only the
    # image_context text differs. agreement(r, predicted, reference) -> (precision, recall) scores the image context
    # against the radiologist's findings.
    from .data import delabel
    from .intake import parse_case
    from .llm import details_of, image_context
    det = details_of(c)
    base = {"case_id": c.case_id, "note_text": c.meta.get("history") or "", "policy": c.meta.get("policy", "open"),
            **{k: det[k] for k in ("age", "sex") if det.get(k)}}
    rows, ctx = [], None
    for cond in conds:
        obj = dict(base)
        if cond == "context":
            parts = image_context(ctx_llm, c, max_images, per_call)
            text = "\n\n".join(x.text for x in parts if x.text)
            _, info = delabel(text, c.meta.get("names") or [c.meta.get("title", "")], [], ext.vocab)
            got = {f.hpo_id for f in ext.findings(text) if f.status == Status.PRESENT}
            ref = {f.hpo_id for f in ext.findings(c.meta["reference_text"]) if f.status == Status.PRESENT}
            prec, rec = agreement(r, got, ref) if agreement else (None, None)
            ctx = {"context_named_diagnosis": bool(info["masked"]),
                   "context_parsed": bool(parts) and all(x.parsed for x in parts),
                   "context_cut_off": any(x.cut_off for x in parts),
                   "context_precision": prec, "context_recall": rec, "context": text[:3000]}
            obj["image_context"] = text
        elif cond == "reference":
            obj["image_context"] = c.meta["reference_text"]
        case = parse_case(obj, r.ont, ext, disease_names=r.db.names, llm_reader=reader)
        rk = r.score(case)
        row = {"case": c.case_id, "title": c.meta["title"], "truth": ";".join(c.truth), "condition": cond,
               "rank": rk.rank_of(c.truth), "findings": len(case.findings),
               "top3": " | ".join(r.db.names[d] for d, *_ in rk.top(3))}
        if cond == "context":
            row.update(ctx)
        rows.append(row)
    return rows


def cmd_medpix_context(a, hb):
    # does the image context (Stage A) help? Same MedPix cases (diagnosis in the knowledge base), same Stage 1 and
    # ranker; Stage 1 reads the history alone, with the image context, or with the radiologist's findings
    from .llm import QuotaExhausted
    r = hb.load_kb(a, ("OMIM", "ORPHA"))
    ext = Extractor(r.ont)
    syn = orphadata_icd10(a.orphadata_xml)[1] if a.orphadata_xml else {}
    matcher = DiseaseMatcher(r.db.names, syn)
    want = [x.strip() for x in (a.cases or "").split(",") if x.strip()]
    listed = bool(a.labels_only and not want)
    if a.labels_only:
        if not a.labels:
            raise SystemExit("--labels-only needs --labels FILE.csv")
        import csv
        if not want:
            want = [row["U_id"].strip() for row in csv.DictReader(Path(a.labels).read_text(encoding="utf-8").splitlines())]
    cases = []
    for c in load_medpix(a.medpix, matcher, ext, "all" if want else a.split, a.labels):
        c.meta["images"] = [i for i in c.meta["images"] if Path(i["path"]).exists()]
        if c.meta["images"] and c.truth and (not want or c.case_id in want):
            cases.append(c)
    if want:
        missing = [x for x in want if x not in {c.case_id for c in cases}]
        if missing and listed:          # the labelled list: run what is there, say what is not
            print(f"[medpix-context] skipped (not found or no images on disk): {', '.join(missing)}")
        elif missing:
            raise SystemExit(f"not found, no images on disk, or diagnosis not in the knowledge base: {', '.join(missing)}")
        cases.sort(key=lambda c: want.index(c.case_id))
        if not cases:
            raise SystemExit("none of the listed cases has images on disk")
    else:
        if not cases:
            raise SystemExit(f"no MedPix cases with images and a knowledge-base diagnosis under {a.medpix} "
                             "(clone the full repository; see README)")
        random.Random(a.seed).shuffle(cases)
        cases = cases[: a.limit] if a.limit else cases
    conds = [x.strip() for x in a.conditions.split(",") if x.strip()]
    bad = [x for x in conds if x not in MM_CONDITIONS]
    if bad:
        raise SystemExit(f"unknown condition(s) {', '.join(bad)}; choose from {', '.join(MM_CONDITIONS)}")
    ctx_llm = context_llm(hb, a) if "context" in conds else None
    hb.RUN["settings"].update({"conditions": conds, "images_per_call": a.images_per_call, "max_images": a.max_images})
    print(f"[medpix-context] {len(cases)} cases with images and a knowledge-base diagnosis "
          f"({'chosen' if want else a.split + ' split'}); up to {a.max_images} images each; Stage 1 with rules + numbers")
    rows, failed, stopped = [], 0, ""
    for i, c in enumerate(cases, 1):
        try:
            rows += medpix_context_case(c, conds, ctx_llm, r, ext, a.max_images, a.images_per_call,
                                        agreement=hb.image_agreement)
        except QuotaExhausted as e:      # quota gone: keep finished cases, stop
            stopped = str(e)
            print(f"\n[llm] stopping early: {e}")
            break
        except RuntimeError as e:        # a failed call is counted, not fatal
            failed += 1
            print(f"\n[llm] {c.case_id}: {e}")
        print(f"\r[medpix-context] {i}/{len(cases)} cases", end="", flush=True)
    print()
    if ctx_llm is not None and ctx_llm.dry_run:
        print(context_usage(ctx_llm))
        return
    if not rows:
        raise SystemExit("no case was answered" + (f": {stopped}" if stopped else ""))
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("medpix_context" + (f"_ctx-{ctx_llm.slug}" if ctx_llm is not None else "") + ".csv")
    df.to_csv(hb.wrote(path), index=False)
    n_cases = df.case.nunique()
    summary = {"model": ctx_llm.model if ctx_llm is not None else "", "context_model": ctx_llm.model if ctx_llm else "",
               "task": "medpix-context", "cases": n_cases, "failed_calls": failed, "stopped_early": stopped}
    print(f"\nRank of the MedPix diagnosis by the ranker (Stage 1 findings -> disease scoring), {n_cases} cases"
          + (f"; image context by {ctx_llm.model}" if ctx_llm is not None else "") + f" (rows in {path})\n")
    print("| Stage 1 reads | Top-1 | Top-3 | Top-10 | MRR | findings per case |\n|---|---|---|---|---|---|")
    ranks = lambda col: [None if pd.isna(x) else int(x) for x in col]
    for cond in conds:
        d = df[df.condition == cond]
        m = hb.topk(ranks(d["rank"]))
        print(f"| {MM_CONDITIONS[cond]} | {m['top1']:.3f} | {m['top3']:.3f} | {m['top10']:.3f} | {m['mrr']:.3f} "
              f"| {d.findings.mean():.1f} |")
        summary.update({f"{cond}_{k}": round(v, 3) for k, v in m.items()})
    if n_cases <= 5:     # few cases: where the diagnosis landed in each condition
        num = lambda x: "-" if pd.isna(x) else str(int(x))
        print("\nPer case: rank of the MedPix diagnosis ('-' = not found)")
        for case, g in df.groupby("case", sort=False):
            print(f"  {case} ({g.title.iloc[0]}): " + ", ".join(f"{x.condition} {num(x.rank)}" for x in g.itertuples()))
    cx = df[df.condition == "context"]
    if len(cx):
        summary.update({k: round(cx[k].mean(), 3) for k in ("context_named_diagnosis", "context_parsed",
                                                            "context_precision", "context_recall")
                        if cx[k].notna().any()})
        pct = lambda k: "–" if k not in summary else f"{summary[k]:.1%}"
        print(f"\nImage context vs the radiologist's findings (HPO terms): precision {pct('context_precision')}, "
              f"recall {pct('context_recall')}; contexts that named the diagnosis {pct('context_named_diagnosis')} "
              f"(disease names are never turned into findings); replies parsed {pct('context_parsed')}")
        if cx.context_cut_off.any():
            print(f"\n[llm] {cx.context_cut_off.mean():.0%} of image contexts stopped before their JSON was finished. "
                  + STUCK)
    if ctx_llm is not None:
        summary.update({"sec_per_call": ctx_llm.usage()["sec_per_request"], "requests": ctx_llm.requests,
                        "cache_hits": ctx_llm.cache_hits})
        print(context_usage(ctx_llm))
    hb.note("medpix_context", **summary)
    if stopped:
        hb.note("llm", stopped_early=stopped)
    hb.log_compare(a, summary)


def add_arguments(p, sub):
    # options of the image-context stage (p: main parser, sub: its subcommands)
    # general model options, used by every model client
    p.add_argument("--json-mode", action="store_true",
                   help="ask the models for JSON only (Ollama, OpenAI): no thinking text first, faster, no loops "
                        "while thinking. Not sent to Gemini's endpoint, which answers in JSON from the prompt")
    p.add_argument("--llm-timeout", type=float, default=600,
                   help="seconds to wait for one reply (default 600; raise it for large models on a CPU, e.g. 1800)")
    # the Stage A model: case files with images (diagnose, multimodal) and medpix-context
    p.add_argument("--context-model", help="vision model that describes the images (Stage A). Default: the --llm-model "
                                           "if it reads images, else MedGemma on a local Ollama")
    p.add_argument("--context-endpoint", help="its chat URL (default: --llm-endpoint)")
    p.add_argument("--context-key-env", help="its API key variable (default: --llm-key-env when the endpoint is the same)")
    p.add_argument("--context-max-tokens", type=int, help="default: --llm-max-tokens")
    p.add_argument("--context-extra", help="extra JSON request fields for it (default: --llm-extra when the endpoint is the same)")
    p.add_argument("--context-max-images", type=int, default=4, help="images described per case (diagnose, multimodal)")
    p.add_argument("--context-images-per-call", type=int, default=1, help="images described together in one request")
    s = sub.add_parser("multimodal", help="one patient from options (text, details, images) -> case file -> diagnose")
    s.add_argument("--text", help="clinical text: history, examination, results")
    s.add_argument("--text-file", help="clinical text from a file")
    s.add_argument("--image", action="append", help="image file (PNG, JPEG, WebP); repeat for more images")
    s.add_argument("--modality", action="append", help="per image, in order, e.g. 'CT - noncontrast' (one value = all)")
    s.add_argument("--plane", action="append", help="per image, e.g. axial")
    s.add_argument("--location", action="append", help="per image, body region, e.g. abdomen")
    s.add_argument("--age")
    s.add_argument("--sex", choices=["male", "female", "other", "unknown"])
    s.add_argument("--detail", action="append", help="other patient detail as key=value, e.g. setting=emergency")
    s.add_argument("--genes", help="comma-separated gene findings")
    s.add_argument("--case-id", default="patient")
    s.add_argument("--local-only", action="store_true", help="credentialed data: refuse remote models")
    s.add_argument("-k", type=int, default=10, help="diseases shown")
    s.add_argument("--out", default="results", help="case file in <out>/multimodal/, result in <out>/diagnose/")
    s = sub.add_parser("medpix-context", help="MedPix: does the image context (Stage A) help the ranker?")
    s.add_argument("--medpix", default="data/MedPix-2-0", help="MedPix-2-0 folder (Case_topic.json, Descriptions.json, images/)")
    s.add_argument("--split", default="test", choices=["all", "train", "train_1", "train_2", "dev", "test"])
    s.add_argument("--conditions", default=",".join(MM_CONDITIONS), help=f"comma-separated, from {', '.join(MM_CONDITIONS)}")
    s.add_argument("--labels", help="CSV U_id,truth overriding automatic diagnosis matching")
    s.add_argument("--labels-only", action="store_true", help="run exactly the cases in --labels")
    s.add_argument("--orphadata-xml", help="adds Orphanet synonyms for diagnosis matching")
    s.add_argument("--limit", type=int, default=20, help="cases (0 = all)")
    s.add_argument("--cases", help="comma-separated MedPix ids, e.g. MPX1544 (instead of --split / --limit)")
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--max-images", type=int, default=2, help="images per case")
    s.add_argument("--images-per-call", type=int, default=1, help="images described together in one request")
    s.add_argument("--out", default="results")


def commands(hb) -> dict:
    # command name -> function, for the dispatch in __main__.py
    return {"multimodal": lambda a: cmd_multimodal(a, hb), "medpix-context": lambda a: cmd_medpix_context(a, hb)}

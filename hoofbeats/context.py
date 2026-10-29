# Image-context stage: the `multimodal` and `medpix-context` commands and their options.
# Kept out of __main__.py, which only registers them (add_arguments, commands). The prompts and model calls are in
# llm.py, section 4. Functions here get `hb`, the hoofbeats command-line module, for its shared helpers (run log,
# get_llm, load_kb, topk, ...): passing it avoids a circular import and a second copy of the run log when hoofbeats
# runs as `python -m hoofbeats`.
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pandas as pd

from .data import DiseaseMatcher, Extractor, load_medpix, orphadata_icd10
from .model import Case, LRRanker, Status


def _other_llm(hb, a, llm, role: str, what: str):
    # a second model (role "context" or "judge") from --<role>-model; endpoint, key and extra fields default to
    # the --llm ones when the endpoint is the same
    import os
    from .llm import MedGemma, with_json_mode
    model, endpoint = getattr(a, f"{role}_model"), getattr(a, f"{role}_endpoint", None) or a.llm_endpoint
    same = endpoint == a.llm_endpoint
    key_env = getattr(a, f"{role}_key_env", None) or (a.llm_key_env if same else None)
    key = os.environ.get(key_env) if key_env else None
    if key_env and not key:
        raise SystemExit(f"environment variable {key_env} is not set")
    extra = getattr(a, f"{role}_extra", None)
    extra = json.loads(extra) if extra else (json.loads(a.llm_extra) if a.llm_extra and same else None)
    extra = with_json_mode(a.json_mode, endpoint, extra, model)
    cache = llm._cache_file.parent if llm._cache_file else None
    other = MedGemma(model, endpoint, getattr(a, f"{role}_max_tokens", None) or a.llm_max_tokens, key, None, extra,
                     a.llm_rpm, cache, a.dry_run, a.llm_timeout)
    hb.RUN["settings"].update({f"{role}_model": other.model, f"{role}_endpoint": other.endpoint or "in-process"})
    print(f"[llm] {what} by {other.model} via {other.endpoint or 'transformers (in-process)'}")
    if not other.is_local:
        print("[llm] remote service: only public datasets will be sent; credentialed data is refused")
    return other


def get_context_llm(hb, a, llm):
    # the model that writes the image context: the ranking model itself unless --context-model names another
    endpoint = a.context_endpoint or a.llm_endpoint
    if not a.context_model or (a.context_model == llm.model and endpoint == llm.endpoint):
        return llm
    return _other_llm(hb, a, llm, "context", "image context")


STUCK = ("MedGemma 1.5 often gets stuck repeating itself while it thinks. Use --json-mode (the server then returns "
         "JSON only), keep --llm-max-tokens at 4096 or more, and set Ollama's context length to 16k.")


def _nth(values, i: int) -> str:
    # per-image option: the i-th value, or the only value for every image
    if not values:
        return ""
    return values[i] if i < len(values) else values[0] if len(values) == 1 else ""


def cmd_multimodal(a, hb):
    # one patient through the image-context pipeline: images -> image context -> ranking model
    from .data import load_input, make_input
    from .llm import diagnose_multimodal
    a.llm = True
    try:
        if a.input:
            c = load_input(a.input)
        else:
            text = Path(a.text_file).read_text(encoding="utf-8") if a.text_file else (a.text or "")
            details = {"age": a.age, "sex": a.sex}
            for kv in a.detail or []:
                if "=" not in kv:
                    raise SystemExit(f"--detail takes key=value, got {kv!r}")
                k, v = kv.split("=", 1)
                details[k.strip()] = v.strip()
            imgs = [{"path": p, "modality": _nth(a.modality, i), "plane": _nth(a.plane, i),
                     "location": _nth(a.location, i)} for i, p in enumerate(a.image or [])]
            c = make_input(text, imgs, details, [g.strip() for g in (a.genes or "").split(",")], a.case_id)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise SystemExit(f"could not read the input: {e}") from None
    if a.local_only:
        c.meta["policy"] = "local_only"
    if not (c.meta["history"] or c.meta["images"] or c.meta["details"]):
        raise SystemExit("give clinical text (--text / --text-file), patient details (--age, --sex, --detail) "
                         "or images (--image), or a JSON file with --input")
    llm = hb.get_llm(a)
    ctx_llm = get_context_llm(hb, a, llm)
    matcher = None
    if a.map_ids:
        r = hb.load_kb(a, ("OMIM", "ORPHA"))
        matcher = DiseaseMatcher(r.db.names)
    from .llm import details_of, details_text
    imgs = c.meta["images"][: a.max_images]
    print(f"\n{c.case_id}\n  patient details: {details_text(details_of(c)) or 'none'}\n"
          f"  clinical text: {len(c.meta['history'].split())} words\n  images: {len(imgs)}"
          + (f" (first {len(imgs)} of {len(c.meta['images'])}; see --max-images)" if len(c.meta["images"]) > len(imgs) else ""))
    if imgs and not llm.sees_images:
        print(f"[llm] {llm.model} is text-only: it gets the image context but not the images")
    from .llm import PolicyError
    try:
        ctxs, res = diagnose_multimodal(llm, c, ctx_llm, a.k, a.max_images, a.images_per_call, not a.no_images, matcher)
    except (PolicyError, ValueError) as e:     # credentialed input to a remote model, or a text-only context model
        raise SystemExit(str(e)) from None
    if llm.dry_run:
        print(hb.usage_line(llm))
        if ctx_llm is not llm:
            print(f"[llm] image context: would send {ctx_llm.planned} requests to {ctx_llm.model}")
        return
    if ctxs:
        print(f"\nStage A: image context by {ctx_llm.model}\n")
        for x in ctxs:
            if x.text:
                print(x.text)
                if not x.parsed:
                    print("  (reply was not JSON; passed on as written)")
            else:
                print(f"{x.label}: (could not be read" + ("; it stopped at the token limit" if x.cut_off else "")
                      + "; not given to the ranking model)")
            print()
    attached = len(imgs) if imgs and not a.no_images and llm.sees_images else 0
    print(f"Stage B: ranking by {llm.model} (details + clinical text" + (" + image context" if ctxs else "")
          + (f" + {attached} image(s)" if attached else "") + ")\n")
    if not res.parsed:
        print(f"[llm] could not read the reply ({len(res.raw)} characters); it began:\n{res.raw[:800]}\n...\n"
              f"and ended:\n{res.raw[-800:]}\n")
    if res.cut_off or any(x.cut_off for x in ctxs):
        print(f"[llm] a reply stopped before its JSON was finished ({len(res.names)} of {a.k} diagnoses read). "
              + STUCK + "\n")
    if res.key_findings:
        print("Key findings:\n" + "\n".join(f"  - {x}" for x in res.key_findings) + "\n")
    if res.image_check:
        print("Image check (image context vs images):\n" + "\n".join(f"  - {x}" for x in res.image_check) + "\n")
    marks = {True: "+", False: "x", None: "?"}
    against = res.against or [[] for _ in res.names]
    for i, (name, ev, ids, ag) in enumerate(zip(res.names, res.evidence, res.slots, against), 1):
        print(f"{i:>2}. {name}" + (f"  [{', '.join(sorted(ids))}]" if ids else ""))
        for e in ev:
            print(f"      {marks[e['supported']]} {e['source']}: {e['detail']}")
        for e in ag:
            print(f"      {marks[e['supported']]} AGAINST {e['source']}: {e['detail']}")
    g = res.grounding
    print("\n+ found in the source it cites   x not found there   ? from the images (cannot be checked)"
          "\nimage=context: labelled 'image' but it repeats the image context, so it is checked against the context")
    if not math.isnan(g):
        print(f"Evidence found in its cited source: {g:.0%}")
    print(hb.usage_line(llm))
    if ctx_llm is not llm:
        u = ctx_llm.usage()
        print(f"[llm] image context: {u['requests']} requests sent, {u['cache_hits']} answered from cache")
    hb.note("multimodal", case=c.case_id, images=len(imgs), contexts=len(ctxs), diagnoses=len(res.names),
         parsed=res.parsed, grounding=g, context_model=ctx_llm.model)
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps({
            "case_id": c.case_id, "details": details_of(c), "ranking_model": llm.model, "context_model": ctx_llm.model,
            "image_context": [{"images": x.images, "label": x.label, "text": x.text, "parsed": x.parsed} for x in ctxs],
            "key_findings": res.key_findings, "image_check": res.image_check,
            "diagnoses": [{"name": n, "ids": sorted(ids), "evidence": ev, "against": ag}
                          for n, ev, ids, ag in zip(res.names, res.evidence, res.slots, against)],
            "parsed": res.parsed, "grounding": None if math.isnan(g) else g}, indent=2), encoding="utf-8")
        print(f"[multimodal] saved -> {hb.wrote(Path(a.save))}")


# what the ranking model gets in each medpix-context condition (always: patient details + history/exam)
MM_CONDITIONS = {"text": "details + text", "images": "details + text + images",
                 "context": "details + text + image context",
                 "context+images": "details + text + image context + images (the pipeline)",
                 "reference": "details + text + radiologist findings (upper bound for a context)"}


def medpix_context_case(c: Case, conds, llm, ctx_llm, r: LRRanker, ext, matcher, k=10, max_images=2,
                        per_call=1, mask=False, judge=None, agreement=None) -> list[dict]:
    # one MedPix case under each condition; ranks by knowledge-base id (labelled cases) and by name (all cases).
    # agreement(r, predicted, reference) -> (precision, recall) scores the image context against the radiologist
    from .data import delabel, same_name
    from .llm import image_context, judge_ranks, rank_multimodal
    imgs = tuple(i["path"] for i in c.meta["images"][:max_images])
    names = c.meta.get("names") or [c.meta.get("title", "")]             # every name: for spotting a named diagnosis
    dx = [n for n in c.meta.get("dx_names") or names if n]                # title + diagnosis: for scoring
    rows, ctx = [], None
    for cond in conds:
        context, attach = None, ()
        if cond in ("context", "context+images"):
            if ctx is None:          # one image context per case, shared by both conditions
                parts = image_context(ctx_llm, c, max_images, per_call)
                text = "\n\n".join(x.text for x in parts if x.text)
                masked, info = delabel(text, names, [], ext.vocab)
                got = {f.hpo_id for f in ext.findings(text) if f.status == Status.PRESENT}
                ref = {f.hpo_id for f in ext.findings(c.meta["reference_text"]) if f.status == Status.PRESENT}
                prec, rec = agreement(r, got, ref) if agreement else (None, None)
                ctx = {"text": masked if mask else text, "context_named_diagnosis": bool(info["masked"]),
                       "context_parsed": bool(parts) and all(x.parsed for x in parts),
                       "context_cut_off": any(x.cut_off for x in parts),
                       "context_precision": prec, "context_recall": rec}
            context = ctx["text"] or "No image context available."
        if cond in ("images", "context+images"):
            attach = imgs
        if cond == "reference":
            context = c.meta["reference_text"] or "No imaging findings recorded."
        res = rank_multimodal(llm, c, context, attach, k, matcher)
        row = {"case": c.case_id, "title": c.meta["title"], "truth": ";".join(c.truth), "condition": cond,
               "rank": next((j for j, s in enumerate(res.slots, 1) if s & set(c.truth)), None) if c.truth else None,
               "rank_name": next((j for j, n in enumerate(res.names, 1) if any(same_name(n, t) for t in dx)), None),
               "parsed": res.parsed, "cut_off": res.cut_off, "grounding": res.grounding,
               "images_to_ranker": len(attach), "n_predictions": len(res.names),
               "predictions": " | ".join(res.names), "image_check": " | ".join(res.image_check)}
        if judge is not None:
            row["rank_judge"], row["rank_judge_broader"] = judge_ranks(judge, res.names, dx)
        if cond in ("context", "context+images"):
            row.update({k_: v for k_, v in ctx.items() if k_ != "text"}, context=ctx["text"][:3000])
        rows.append(row)
    return rows


def cmd_medpix_context(a, hb):
    # does the image-context stage help the ranking model? same MedPix cases, same prompt, five inputs
    from .llm import QuotaExhausted
    a.llm = True
    r = hb.load_kb(a, ("OMIM", "ORPHA"))
    ext = Extractor(r.ont)
    syn = orphadata_icd10(a.orphadata_xml)[1] if a.orphadata_xml else {}
    matcher = DiseaseMatcher(r.db.names, syn)
    want = [x.strip() for x in (a.cases or "").split(",") if x.strip()]
    cases = []
    for c in load_medpix(a.medpix, matcher, ext, "all" if want else a.split, a.labels):
        c.meta["images"] = [i for i in c.meta["images"] if Path(i["path"]).exists()]
        if c.meta["images"] and (c.truth or not a.only_labeled) and (not want or c.case_id in want):
            cases.append(c)
    if want:
        missing = [x for x in want if x not in {c.case_id for c in cases}]
        if missing:
            raise SystemExit(f"not found, or no images on disk: {', '.join(missing)}")
        cases.sort(key=lambda c: want.index(c.case_id))
    else:
        if not cases:
            raise SystemExit(f"no MedPix cases with images under {a.medpix} (clone the full repository; see README)")
        random.Random(a.seed).shuffle(cases)
        cases = cases[: a.limit] if a.limit else cases
    llm = hb.get_llm(a)
    ctx_llm = get_context_llm(hb, a, llm)
    judge = _other_llm(hb, a, llm, "judge", "answers judged") if a.judge_model else None
    conds = [x.strip() for x in a.conditions.split(",") if x.strip()]
    bad = [x for x in conds if x not in MM_CONDITIONS]
    if bad:
        raise SystemExit(f"unknown condition(s) {', '.join(bad)}; choose from {', '.join(MM_CONDITIONS)}")
    for x in ("images", "context+images"):
        if x in conds and not llm.sees_images:
            conds.remove(x)
            print(f"[medpix-context] {x} skipped: {llm.model} is text-only")
    for x in ("context", "context+images"):
        if x in conds and not ctx_llm.sees_images:
            conds.remove(x)
            print(f"[medpix-context] {x} skipped: {ctx_llm.model} cannot read images")
    hb.RUN["settings"].update({"conditions": conds, "k": a.k, "images_per_call": a.images_per_call,
                            "mask_context": a.mask_context, "context_model": ctx_llm.model})
    print(f"[medpix-context] {len(cases)} cases with images ({'chosen' if want else a.split + ' split'}, "
          f"{sum(bool(c.truth) for c in cases)} "
          f"matched to the knowledge base); up to {a.max_images} images each; image context by {ctx_llm.model}"
          + (" (the ranking model itself)" if ctx_llm is llm else ""))
    rows, failed, stopped = [], 0, ""
    for i, c in enumerate(cases, 1):
        try:
            rows += medpix_context_case(c, conds, llm, ctx_llm, r, ext, matcher, a.k, a.max_images,
                                        a.images_per_call, a.mask_context, judge, agreement=hb.image_agreement)
        except QuotaExhausted as e:      # quota gone: keep finished cases, stop
            stopped = str(e)
            print(f"\n[llm] stopping early: {e}")
            break
        except RuntimeError as e:        # a failed call is counted, not fatal
            failed += 1
            print(f"\n[llm] {c.case_id}: {e}")
        print(f"\r[medpix-context] {i}/{len(cases)} cases", end="", flush=True)
    print()
    if llm.dry_run:
        print(hb.usage_line(llm))
        if ctx_llm is not llm:
            print(f"[llm] image context: would send {ctx_llm.planned} requests to {ctx_llm.model} "
                  f"({ctx_llm.cache_hits} already cached)")
        if judge is not None:
            print(f"[llm] judge: one short request to {judge.model} per ranking, once the rankings exist")
        return
    if not rows:
        raise SystemExit("no case was answered" + (f": {stopped}" if stopped else ""))
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / (f"medpix_context_{llm.slug}" + (f"_ctx-{ctx_llm.slug}" if ctx_llm is not llm else "") + ".csv")
    df.to_csv(hb.wrote(path), index=False)
    n_cases, lab = df.case.nunique(), df[df.truth != ""]
    summary = {"model": llm.model, "context_model": ctx_llm.model, "task": "medpix-context", "cases": n_cases,
               "labeled": lab.case.nunique(), "failed_calls": failed, "stopped_early": stopped,
               "mask_context": a.mask_context}
    print(f"\n{llm.model} ranking, image context by {ctx_llm.model}: {n_cases} cases (rows in {path})")
    judged, kb = "rank_judge" in df, bool(len(lab))
    print("Name match: a predicted name matches the MedPix diagnosis (strict; all cases)."
          + (f" Judge: {judge.model} decides, accepting synonyms; 'or broader' also counts a broader category"
             " (e.g. 'hydrocephalus' for 'non-communicating hydrocephalus')." if judged else "")
          + (f" KB match: the prediction maps to the case's OMIM/ORPHA disease ({summary['labeled']} cases)." if kb else "")
          + "\n")
    head = ["ranking model gets", "name Top-1", "name Top-3", "name Top-10"]
    head += ["judge Top-1", "judge Top-3", "judge Top-10", "judge or broader Top-10"] if judged else []
    head += ["KB Top-1", "KB Top-10"] if kb else []
    print("| " + " | ".join(head + ["evidence found"]) + " |\n|" + "---|" * (len(head) + 1))
    ranks = lambda col: [None if pd.isna(x) else int(x) for x in col]
    for cond in conds:
        d = df[df.condition == cond]
        mn = hb.topk(ranks(d.rank_name))
        cells = [MM_CONDITIONS[cond]] + [f"{mn[x]:.3f}" for x in ("top1", "top3", "top10")]
        summary.update({f"{cond}_name_{k_}": round(v, 3) for k_, v in mn.items()})
        if judged:
            mj, mb = hb.topk(ranks(d.rank_judge)), hb.topk(ranks(d.rank_judge_broader))
            cells += [f"{mj[x]:.3f}" for x in ("top1", "top3", "top10")] + [f"{mb['top10']:.3f}"]
            summary.update({f"{cond}_judge_{k_}": round(v, 3) for k_, v in mj.items()})
            summary.update({f"{cond}_broader_{k_}": round(v, 3) for k_, v in mb.items()})
        if kb:
            dl = d[d.truth != ""]
            mk = hb.topk(ranks(dl["rank"])) if len(dl) else {}
            cells += [f"{mk['top1']:.3f}", f"{mk['top10']:.3f}"] if mk else ["–", "–"]
            summary.update({f"{cond}_kb_{k_}": round(v, 3) for k_, v in mk.items()})
        g = d.grounding.mean()
        print("| " + " | ".join(cells + ["–" if pd.isna(g) else f"{g:.0%}"]) + " |")
        summary[f"{cond}_parsed"] = round(d.parsed.mean(), 3)
        summary[f"{cond}_grounding"] = None if pd.isna(g) else round(g, 3)
    if n_cases <= 5:     # few cases: show where the diagnosis landed in each condition
        num = lambda x: "-" if pd.isna(x) else str(int(x))
        print("\nPer case: rank of the MedPix diagnosis by name match" + (" / judge / judge or broader" if judged else "")
              + " ('-' = not in the list)")
        for case, g in df.groupby("case", sort=False):
            print(f"  {case} ({g.title.iloc[0]}): " + ", ".join(
                f"{r.condition} {num(r.rank_name)}" + (f"/{num(r.rank_judge)}/{num(r.rank_judge_broader)}" if judged else "")
                for r in g.itertuples()))
            print("      " + "\n      ".join(f"{r.condition}: {r.predictions}" for r in g.itertuples()))
    cx = df[df.condition.isin(["context", "context+images"])].drop_duplicates("case")
    if len(cx):
        summary.update({"context_named_diagnosis": round(cx.context_named_diagnosis.mean(), 3),
                        "context_parsed": round(cx.context_parsed.mean(), 3),
                        "context_precision": round(cx.context_precision.mean(), 3),
                        "context_recall": round(cx.context_recall.mean(), 3)})
        pct = lambda x: "–" if x is None or pd.isna(x) else f"{x:.1%}"
        print(f"\nImage context vs the radiologist's findings (HPO terms): precision {pct(summary['context_precision'])}, "
              f"recall {pct(summary['context_recall'])}; contexts that named the diagnosis "
              f"{pct(summary['context_named_diagnosis'])}" + (" (masked before ranking)" if a.mask_context else "")
              + f"; replies parsed {pct(summary['context_parsed'])}")
    cut = df.cut_off.mean()
    summary["ranking_cut_off"] = round(cut, 3)
    if cut > 0 or (len(cx) and cx.context_cut_off.any()):
        print(f"\n[llm] {cut:.0%} of ranking replies" + (" and some image contexts" if len(cx) and cx.context_cut_off.any()
              else "") + " stopped before their JSON was finished, so diagnoses are missing. " + STUCK)
    if judged:
        summary["judge_model"] = judge.model
    summary.update({"sec_per_call": llm.usage()["sec_per_request"], "requests": llm.requests + (
        ctx_llm.requests if ctx_llm is not llm else 0), "cache_hits": llm.cache_hits})
    print(hb.usage_line(llm))
    if ctx_llm is not llm:
        u = ctx_llm.usage()
        print(f"[llm] image context: {u['requests']} requests sent, {u['cache_hits']} answered from cache")
    if judge is not None:
        u = judge.usage()
        print(f"[llm] judge: {u['requests']} requests sent, {u['cache_hits']} answered from cache")
    hb.note("medpix_context", **summary)
    if stopped:
        hb.note("llm", stopped_early=stopped)
    hb.log_compare(a, summary)


def add_arguments(p, sub):
    # options of the image-context stage (p: main parser, sub: its subcommands)
    # general model options, used by get_llm for every command
    p.add_argument("--json-mode", action="store_true",
                   help="ask the models for JSON only (Ollama, OpenAI): no thinking text first, faster, no loops "
                        "while thinking. Not sent to Gemini's endpoint, which answers in JSON from the prompt")
    p.add_argument("--llm-timeout", type=float, default=600,
                   help="seconds to wait for one reply (default 600; raise it for large models on a CPU, e.g. 1800)")
    # second and third models
    p.add_argument("--context-model", help="vision model that writes the image context (multimodal, medpix-context); "
                                           "default: the --llm-model itself")
    p.add_argument("--context-endpoint", help="its chat URL (default: --llm-endpoint)")
    p.add_argument("--context-key-env", help="its API key variable (default: --llm-key-env when the endpoint is the same)")
    p.add_argument("--context-max-tokens", type=int, help="default: --llm-max-tokens")
    p.add_argument("--context-extra", help="extra JSON request fields for it (default: --llm-extra when the endpoint is the same)")
    p.add_argument("--judge-model", help="medpix-context: model that decides whether an answer names the true diagnosis "
                                         "(accepts synonyms), e.g. gemma3:4b; default: name matching only")
    p.add_argument("--judge-endpoint", help="its chat URL (default: --llm-endpoint)")
    p.add_argument("--judge-key-env", help="its API key variable (default: --llm-key-env when the endpoint is the same)")
    s = sub.add_parser("multimodal", help="one patient: images -> image context -> ranking model")
    s.add_argument("--input", help="JSON file: text, details, genes, images (see README); or use the options below")
    s.add_argument("--text", help="clinical text: history, examination, results")
    s.add_argument("--text-file", help="clinical text from a file")
    s.add_argument("--image", action="append", help="image file (PNG, JPEG, WebP); repeat for more images")
    s.add_argument("--modality", action="append", help="per image, in order, e.g. 'CT - noncontrast' (one value = all)")
    s.add_argument("--plane", action="append", help="per image, e.g. axial")
    s.add_argument("--location", action="append", help="per image, body region, e.g. abdomen")
    s.add_argument("--age")
    s.add_argument("--sex")
    s.add_argument("--detail", action="append", help="other patient detail as key=value, e.g. setting=emergency")
    s.add_argument("--genes", help="comma-separated gene findings")
    s.add_argument("--case-id", default="patient")
    s.add_argument("--local-only", action="store_true", help="credentialed data: refuse remote models")
    s.add_argument("--no-images", action="store_true", help="give the ranking model the image context only, not the images")
    s.add_argument("--map-ids", action="store_true", help="map answers to OMIM/ORPHA ids (needs the knowledge base)")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--max-images", type=int, default=4)
    s.add_argument("--images-per-call", type=int, default=1, help="images described together in one request")
    s.add_argument("--save", help="write the image context and ranking to this JSON file")
    s.add_argument("--out", default="results", help="run log goes to <out>/runs/, model replies to <out>/llm_cache/")
    s = sub.add_parser("medpix-context", help="MedPix: does the image-context stage help the ranking model?")
    s.add_argument("--medpix", default="data/MedPix-2-0", help="MedPix-2-0 folder (Case_topic.json, Descriptions.json, images/)")
    s.add_argument("--split", default="test", choices=["all", "train", "train_1", "train_2", "dev", "test"])
    s.add_argument("--conditions", default=",".join(MM_CONDITIONS), help=f"comma-separated, from {', '.join(MM_CONDITIONS)}")
    s.add_argument("--labels", help="CSV U_id,truth overriding automatic diagnosis matching")
    s.add_argument("--orphadata-xml", help="adds Orphanet synonyms for diagnosis matching")
    s.add_argument("--only-labeled", action="store_true", help="only cases whose diagnosis is in the knowledge base")
    s.add_argument("--limit", type=int, default=20, help="cases (0 = all)")
    s.add_argument("--cases", help="comma-separated MedPix ids, e.g. MPX1544 (instead of --split / --limit)")
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--max-images", type=int, default=2, help="images per case")
    s.add_argument("--images-per-call", type=int, default=1, help="images described together in one request")
    s.add_argument("--mask-context", action="store_true",
                   help="remove the diagnosis name from generated image contexts before ranking (stricter ablation)")
    s.add_argument("--out", default="results")


def commands(hb) -> dict:
    # command name -> function, for the dispatch in __main__.py
    return {"multimodal": lambda a: cmd_multimodal(a, hb), "medpix-context": lambda a: cmd_medpix_context(a, hb)}

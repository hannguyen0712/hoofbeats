# Image-context stage (Stage A): its model options, the `multimodal` command and the `medpix-context` experiment.
# Stage A (llm.image_context) describes each image; the description goes into the case's image_context field, Stage 1
# reads it like any note, and the ranker scores the findings with everything else. No model ranks diagnoses itself.
# Kept out of __main__.py, which only registers these (add_arguments, commands). Functions here get `hb`, the
# hoofbeats command-line module, for its shared helpers (run log, load_kb, read_case, topk, ...): passing it avoids a
# circular import and a second copy of the run log when hoofbeats runs as `python -m hoofbeats`.
from __future__ import annotations

import json
from dataclasses import replace
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
    llm.verify = bool(getattr(a, "context_verify", False))    # read by llm.image_context
    llm.min_confidence = getattr(a, "context_confidence", "all") or "all"
    llm.read = getattr(a, "context_read", "all") or "all"       # read by llm.image_context
    llm.samples = max(int(getattr(a, "context_samples", 1) or 1), 1)
    llm.agree = max(int(getattr(a, "context_agree", 0) or 0), 0)
    hb.RUN["settings"].update({"context_model": llm.model, "context_endpoint": llm.endpoint or "in-process",
                               "context_verify": llm.verify, "context_read": llm.read, "context_samples": llm.samples, "context_agree": llm.agree})
    print(f"[llm] image context (Stage A) by {llm.model} via {llm.endpoint or 'transformers (in-process)'}: {why}"
          + ("; each finding checked with a second request (--context-verify)" if llm.verify else ""))
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
                 "verified": "text + image context, findings checked by the vision model",
                 "definite": "text + image context, only findings the vision model called definite",
                 "consensus": "text + image context, only findings at least --context-agree of --context-samples descriptions report",
                 "asked": "text + image context + findings the vision model answered yes to when asked (--ask-*)",
                 "reference": "text + radiologist findings (upper bound for a context)"}
CONTEXT_CONDS = ("context", "verified", "definite", "consensus", "asked")     # conditions that need the image context
ASK_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)     # the summary scores the asked findings at each, to pick --ask-threshold


def stage1_findings(ext, reader, text: str, field: str = "image_context") -> list:
    # what Stage 1 makes of one text: the rule reader, plus the LLM reader (and SapBERT linker) when one is given
    # (medpix-context with --llm-endpoint), as in diagnose. Present and absent findings only.
    keep = (Status.PRESENT, Status.ABSENT)
    out = [f for f in ext.findings(text) if f.status in keep]
    if reader is not None and text and text.strip():
        out += [f for f in reader(text, "image_context", field) if f.status in keep]
    return out


def medpix_context_case(c, conds, ctx_llm, r: LRRanker, ext, max_images=2, per_call=1, agreement=None,
                        reader=None, no_history=False, ask=None, lenient=None) -> list[dict]:
    # one MedPix case under each condition: the same Stage 1 (parse_case) and the same ranker, only the
    # image_context text differs. agreement(r, predicted, reference) -> (precision, recall) scores the image context
    # against the radiologist's findings. "context" and "verified" share one description: "verified" keeps only the
    # findings the vision model confirmed in a second request. reader: the Stage 1 LLM reader (None = rules + numbers);
    # it reads the image context, the radiologist's findings and the history, and the scores use it too. ask: dict
    # (llm, k, n, threshold) for the "asked" condition; lenient: a second agreement that also credits close terms.
    from .data import delabel
    from .intake import parse_case
    from .llm import details_of, image_context
    det = details_of(c)
    base = {"case_id": c.case_id, "note_text": c.meta.get("history") or "", "policy": c.meta.get("policy", "open"),
            **{k: det[k] for k in ("age", "sex") if det.get(k)}}
    texts, ctx = {}, {}
    if any(x in conds for x in CONTEXT_CONDS):
        # no_history: the vision model gets the images, age and sex but not the history (the "clinical question"),
        # to tell findings it sees from findings it expects from the history; Stage 1 still reads the history
        seen = replace(c, meta={**c.meta, "history": ""}) if no_history else c
        if "consensus" in conds:
            if getattr(ctx_llm, "samples", 1) < 2:
                raise SystemExit("the consensus condition needs --context-samples 3 or more")
            ctx_llm.extract = lambda t: stage1_findings(ext, reader, t)
            ctx_llm.term_name, ctx_llm.use_consensus = r.ont.name, False
        parts = image_context(ctx_llm, seen, max_images, per_call, verify="verified" in conds)
        texts = {"context": "\n\n".join(x.unchecked for x in parts if x.unchecked),
                 "verified": "\n\n".join(x.text for x in parts if x.text),
                 "definite": "\n\n".join(x.definite for x in parts if x.definite),
                 "consensus": "\n\n".join(x.consensus for x in parts if x.consensus)}
        if "consensus" not in conds:
            texts.pop("consensus")
        present = lambda t: {f.hpo_id for f in stage1_findings(ext, reader, t) if f.status == Status.PRESENT}
        ref = present(c.meta["reference_text"])
        if "asked" in conds:
            texts["asked"], ctx["asked"] = _asked(c, conds, texts, base, r, ext, reader, ask, max_images, per_call,
                                                 agreement, ref, present)
        for cond, text in texts.items():
            _, info = delabel(text, c.meta.get("names") or [c.meta.get("title", "")], [], ext.vocab)
            got = present(text)
            prec, rec = agreement(r, got, ref) if agreement else (None, None)
            lp, lr = lenient(r, got, ref) if lenient else (None, None)
            ctx[cond] = {"context_named_diagnosis": bool(info["masked"]),
                         "context_parsed": bool(parts) and all(x.parsed for x in parts),
                         "context_cut_off": any(x.cut_off for x in parts),
                         "context_precision": prec, "context_recall": rec, "context": text[:3000],
                         "context_pred": len(got), "context_agree": round((prec or 0) * len(got)),
                         "context_precision_lenient": lp, "context_recall_lenient": lr,
                         **ctx.get(cond, {})}
        if "verified" in conds:
            ctx["verified"].update({"findings_checked": sum(x.checked for x in parts),
                                    "findings_kept": sum(x.kept for x in parts)})
    rows = []
    n_diseases = len(r.diseases)
    for cond in conds:
        row = {"case": c.case_id, "title": c.meta["title"], "truth": ";".join(c.truth), "condition": cond}
        obj = dict(base)
        if cond in texts:
            obj["image_context"] = texts[cond]
        elif cond == "reference":
            obj["image_context"] = c.meta["reference_text"]
        case = parse_case(obj, r.ont, ext, disease_names=r.db.names, llm_reader=reader,
                          image_meta=c.meta.get("images", [])[:max_images],
                          region_filter=False if cond == "reference" else None)   # the radiologist is trusted
        rk = r.score(case)
        prov = (case.meta or {}).get("provenance", {})
        row.update({"rank": rk.rank_of(c.truth), "findings": len(case.findings), "n_diseases": n_diseases,
                    # the scored HPO terms, so a run can be analysed per finding later (which ones hit the truth)
                    "terms": json.dumps([[f.hpo_id, f.status.value, f.source] for f in case.findings]),
                    "outside_body_part": len(prov.get("outside_body_part") or []),
                    "top3": " | ".join(r.db.names[d] for d, *_ in rk.top(3))})
        row.update(ctx.get(cond, {}))
        rows.append(row)
    return rows


def _asked(c, conds, texts, base, r, ext, reader, ask, max_images, per_call, agreement, ref, present):
    # the "asked" condition: rank with the image description (the consensus one when that condition runs), ask about
    # the top diseases' imaging findings the case does not state, add the ones answered yes. Returns (text, columns);
    # the columns keep every belief, so the summary can score each threshold
    import json as _json
    from .ask import ask_case, asked_text
    from .intake import parse_case
    if not ask or ask.get("llm") is None:
        raise SystemExit("the asked condition needs a vision model (--context-model, or --ask-model)")
    first = texts.get("consensus") if "consensus" in conds and texts.get("consensus") else texts.get("context", "")
    obj = {**base, "image_context": first}
    case = parse_case(obj, r.ont, ext, disease_names=r.db.names, llm_reader=reader,
                      image_meta=c.meta.get("images", [])[:max_images])
    known = {f.hpo_id for f in case.findings}
    got = ask_case(ask["llm"], r, r.score(case), c.meta.get("images", [])[:max_images], known,
                   ask.get("k", 5), ask.get("n", 12), per_call)
    extra = asked_text(got, ask.get("threshold", 0.7))
    sweep = {}
    for th in ASK_THRESHOLDS:
        ids = {x.hpo_id for x in got if x.belief >= th}
        sweep[f"asked_p@{th:g}"], sweep[f"asked_r@{th:g}"] = agreement(r, ids, ref) if agreement else (None, None)
        sweep[f"asked_n@{th:g}"] = len(ids)
    cols = {"asked_questions": len(got), "asked_findings": _json.dumps(
        [{"hpo_id": x.hpo_id, "name": x.name, "belief": round(x.belief, 3)} for x in got]), **sweep}
    return ("\n\n".join(t for t in (first, extra) if t), cols)


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
        skip = hb.read_case_ids(a.exclude_cases)
        if skip:
            n = len(cases)
            cases = [c for c in cases if c.case_id not in skip]
            print(f"[medpix-context] {n - len(cases)} cases left out (--exclude-cases)")
        if not cases:
            raise SystemExit(f"no MedPix cases with images and a knowledge-base diagnosis under {a.medpix} "
                             "(clone the full repository; see README)")
        random.Random(a.seed).shuffle(cases)
        cases = cases[: a.limit] if a.limit else cases
    conds = [x.strip() for x in a.conditions.split(",") if x.strip()]
    bad = [x for x in conds if x not in MM_CONDITIONS]
    if bad:
        raise SystemExit(f"unknown condition(s) {', '.join(bad)}; choose from {', '.join(MM_CONDITIONS)}")
    ctx_llm = context_llm(hb, a) if any(x in conds for x in CONTEXT_CONDS) else None
    reader = None
    if getattr(a, "llm_endpoint", None) or getattr(a, "llm_offline", False):
        reader, _ = hb.stage1_reader(a, r.ont)     # full Stage 1 (rules + numbers + LLM reader), as in diagnose
    ask = None
    if "asked" in conds:
        ask = {"llm": ask_llm(hb, a, ctx_llm), "k": a.ask_diseases, "n": a.ask_findings, "threshold": a.ask_threshold}
    stage1_desc = "rules + numbers + LLM reader" if reader is not None else "rules + numbers"
    hb.RUN["settings"].update({"conditions": conds, "images_per_call": a.images_per_call, "max_images": a.max_images,
                               "context_no_history": a.context_no_history, "stage1": stage1_desc,
                               **({"ask_diseases": a.ask_diseases, "ask_findings": a.ask_findings,
                                   "ask_threshold": a.ask_threshold, "ask_model": ask["llm"].model} if ask else {})})
    print(f"[medpix-context] {len(cases)} cases with images and a knowledge-base diagnosis "
          f"({'chosen' if want else a.split + ' split'}); up to {a.max_images} images each; Stage 1 with {stage1_desc}")
    rows, failed, stopped = [], 0, ""
    for i, c in enumerate(cases, 1):
        try:
            if reader is not None:
                reader.case_id = c.case_id
            rows += medpix_context_case(c, conds, ctx_llm, r, ext, a.max_images, a.images_per_call,
                                        agreement=hb.image_agreement, reader=reader, no_history=a.context_no_history,
                                        ask=ask, lenient=hb.image_agreement_lenient)
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
               "task": "medpix-context" + ("-no-history" if a.context_no_history else ""), "cases": n_cases, "failed_calls": failed, "stopped_early": stopped}
    print(f"\nRank of the MedPix diagnosis by the ranker (Stage 1 findings -> disease scoring), {n_cases} cases"
          + (f"; image context by {ctx_llm.model}" if ctx_llm is not None else "") + f" (rows in {path})\n")
    print("| Stage 1 reads | Top-1 | Top-3 | Top-10 | MRR | median rank | findings per case |\n|---|---|---|---|---|---|---|")
    ranks = lambda col: [None if pd.isna(x) else int(x) for x in col]
    for cond in conds:
        d = df[df.condition == cond]
        m = hb.topk(ranks(d["rank"]))
        m["median_rank"] = median_rank(ranks(d["rank"]), len(r.diseases))
        print(f"| {MM_CONDITIONS[cond]} | {m['top1']:.3f} | {m['top3']:.3f} | {m['top10']:.3f} | {m['mrr']:.3f} "
              f"| {m['median_rank']:g} | {d.findings.mean():.1f} |")
        summary.update({f"{cond}_{k}": round(v, 3) for k, v in m.items()})
    if "outside_body_part" in df and df.outside_body_part.sum():
        print("\nImage findings outside the imaged body part, not scored (--no-image-region-filter keeps them): "
              + ", ".join(f"{c} {df[df.condition == c].outside_body_part.sum():g}" for c in conds
                          if df[df.condition == c].outside_body_part.sum()))
    if "text" in conds and len(conds) > 1:
        base = df[df.condition == "text"].set_index("case")
        others = [(MM_CONDITIONS[c], df[df.condition == c].set_index("case")["rank"].reindex(base.index))
                  for c in conds if c != "text"]
        hb.paired_table(MM_CONDITIONS["text"], base["rank"], others, base["truth"].astype(str), "medpix_context_paired")
    if n_cases <= 5:     # few cases: where the diagnosis landed in each condition
        num = lambda x: "-" if pd.isna(x) else str(int(x))
        print("\nPer case: rank of the MedPix diagnosis ('-' = not found)")
        for case, g in df.groupby("case", sort=False):
            print(f"  {case} ({g.title.iloc[0]}): " + ", ".join(f"{x.condition} {num(x.rank)}" for x in g.itertuples()))
    pct = lambda k: "–" if k not in summary else f"{summary[k]:.1%}"
    for cond in ("context", "verified", "definite", "consensus", "asked"):
        cx = df[df.condition == cond]
        if not len(cx):
            continue
        pre = cond + "_"
        summary.update({pre + k.removeprefix("context_"): round(cx[k].mean(), 3)
                        for k in ("context_named_diagnosis", "context_parsed", "context_precision", "context_recall",
                                  "context_precision_lenient", "context_recall_lenient")
                        if k in cx and cx[k].notna().any()})
        what = {"context": "Image context", "verified": "Checked image context",
                "definite": "Image context, definite findings only",
                "consensus": "Image context, findings the descriptions agree on",
                "asked": "Image context plus answered questions"}[cond]
        print(f"\n{what} vs the radiologist's findings (HPO terms): precision {pct(pre + 'precision')}, "
              f"recall {pct(pre + 'recall')}; contexts that named the diagnosis {pct(pre + 'named_diagnosis')} "
              f"(disease names are never turned into findings); replies parsed {pct(pre + 'parsed')}")
        if pre + "precision_lenient" in summary:
            print(f"Lenient (close terms count too, e.g. ventriculomegaly for hydrocephalus): precision "
                  f"{pct(pre + 'precision_lenient')}, recall {pct(pre + 'recall_lenient')}")
        if cond == "asked" and "asked_questions" in cx:
            print(f"Questions: {int(cx.asked_questions.sum())} findings asked over {len(cx)} cases. Asked findings vs the "
                  "radiologist at each threshold (pick --ask-threshold on one split, report another):")
            for th in ASK_THRESHOLDS:
                p_, r_, n_ = (cx.get(f"asked_{x}@{th:g}") for x in ("p", "r", "n"))
                if p_ is None:
                    continue
                summary.update({f"asked_p@{th:g}": round(p_.mean(), 3) if p_.notna().any() else None,
                                f"asked_r@{th:g}": round(r_.mean(), 3) if r_.notna().any() else None})
                print(f"  belief >= {th:g}: {int(n_.sum())} findings, precision "
                      + (f"{p_.mean():.1%}" if p_.notna().any() else "–") + ", recall "
                      + (f"{r_.mean():.1%}" if r_.notna().any() else "–"))
        if "context_pred" in cx:
            agree, pred = int(cx.context_agree.fillna(0).sum()), int(cx.context_pred.fillna(0).sum())
            summary.update({pre + "findings": pred, pre + "findings_agree": agree})
            print(f"Pooled over all findings: {agree} of {pred} match the radiologist"
                  + (f" ({agree / pred:.1%})" if pred else ""))
            if a.reliability_out:      # one file per condition: their findings differ
                out_p = Path(a.reliability_out)
                hb.write_reliability(out_p.with_name(f"{out_p.stem}_{cond}{out_p.suffix or '.json'}"),
                                     "image_context", agree, pred, n_cases)
        if cond == "verified" and "findings_checked" in cx:
            asked, kept = int(cx.findings_checked.sum()), int(cx.findings_kept.sum())
            summary.update({"findings_checked": asked, "findings_kept": kept})
            print(f"Check: {kept} of {asked} described findings confirmed ({kept / max(asked, 1):.0%}); the rest dropped")
    cx = df[df.condition.isin(["context", "verified", "definite"])]
    if len(cx):
        if cx.context_cut_off.fillna(False).astype(bool).any():
            print(f"\n[llm] {cx.context_cut_off.fillna(False).astype(bool).mean():.0%} of image contexts stopped before their JSON was finished. "
                  + STUCK)
    if ctx_llm is not None:
        summary.update({"sec_per_call": ctx_llm.usage()["sec_per_request"], "requests": ctx_llm.requests,
                        "cache_hits": ctx_llm.cache_hits})
        print(context_usage(ctx_llm))
    hb.note("medpix_context", **summary)
    if stopped:
        hb.note("llm", stopped_early=stopped)
    hb.log_compare(a, summary)


def median_rank(ranks, n_diseases: int) -> float:
    # the diagnosis not found at all counts as last
    xs = sorted(x if x else n_diseases for x in ranks)
    if not xs:
        return float("nan")
    m = len(xs) // 2
    return float(xs[m]) if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def ask_llm(hb, a, ctx_llm):
    # the model that answers the questions: --ask-model (e.g. a medical model, a different family from the describer,
    # which also checks it), else the describing model
    if not getattr(a, "ask_model", None):
        return ctx_llm
    from .llm import MedGemma
    cache = None if a.llm_cache == "none" else Path(a.llm_cache if a.llm_cache != "auto" else Path(hb._out_dir(a)) / "llm_cache")
    extra = json.loads(a.ask_extra) if getattr(a, "ask_extra", None) else None
    llm = MedGemma(a.ask_model, a.ask_endpoint or (ctx_llm.endpoint if ctx_llm else None), 1, None, True, extra,
                   a.llm_rpm, cache, a.dry_run, a.llm_timeout, offline=a.llm_offline)
    print(f"[ask] questions answered by {llm.model} via {llm.endpoint or 'transformers (in-process)'}")
    return llm


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
    p.add_argument("--context-confidence", choices=("all", "probable", "definite"), default="all",
                   help="keep only image findings the vision model rated at least this sure (default: all)")
    p.add_argument("--context-read", choices=("all", "findings"), default="all",
                   help="what Stage 1 reads from each image description: all of it, or only the findings list and "
                        "pertinent negatives (no details, summary, technique or anatomy)")
    p.add_argument("--context-samples", type=int, default=1,
                   help="describe each image this many times (the first at temperature 0, the others sampled) and keep "
                        "only the findings at least --context-agree of them report; 1 = off (medpix-context: condition "
                        "consensus); 5 is the tested setting")
    p.add_argument("--context-agree", type=int, default=0,
                   help="with --context-samples: how many descriptions must report a finding to keep it "
                        "(default 2)")
    p.add_argument("--context-verify", action="store_true",
                   help="check each described finding with a second request to the vision model and keep only the "
                        "confirmed ones (diagnose, multimodal; medpix-context compares both with --conditions verified)")
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
    s.add_argument("--conditions", default="text,context,reference",
                   help=f"comma-separated, from {', '.join(MM_CONDITIONS)}")
    s.add_argument("--labels", help="CSV U_id,truth overriding automatic diagnosis matching")
    s.add_argument("--labels-only", action="store_true", help="run exactly the cases in --labels")
    s.add_argument("--orphadata-xml", help="adds Orphanet synonyms for diagnosis matching")
    s.add_argument("--limit", type=int, default=20, help="cases (0 = all)")
    s.add_argument("--cases", help="comma-separated MedPix ids, e.g. MPX1544 (instead of --split / --limit)")
    s.add_argument("--seed", type=int, default=13)
    s.add_argument("--max-images", type=int, default=2, help="images per case")
    s.add_argument("--images-per-call", type=int, default=1, help="images described together in one request")
    s.add_argument("--exclude-cases", help="leave these cases out: a CSV with a case column (e.g. an earlier run) or ids")
    s.add_argument("--context-no-history", action="store_true",
                   help="the vision model sees images, age and sex only, not the history (Stage 1 still reads it)")
    s.add_argument("--ask-model", help="condition asked: vision model that answers the yes/no questions "
                                       "(default: the --context-model). Its server must return logprobs")
    s.add_argument("--ask-endpoint", help="its chat URL (default: the --context-endpoint)")
    s.add_argument("--ask-extra", help="extra JSON request fields for it")
    s.add_argument("--ask-diseases", type=int, default=5, help="asked: findings of this many top diseases (default 5)")
    s.add_argument("--ask-findings", type=int, default=12, help="asked: at most this many findings per case (default 12)")
    s.add_argument("--ask-threshold", type=float, default=0.7,
                   help="asked: add a finding when (P(yes, present) + 1 - P(yes, free of it)) / 2 reaches this "
                        "(default 0.7; the summary scores 0.5-0.9 so it can be picked on another split)")
    s.add_argument("--reliability-out", help="write each condition's measured image-context reliability, for "
                                             "--reliability (FILE.json -> FILE_context.json, FILE_verified.json, ...)")
    s.add_argument("--out", default="results")


def commands(hb) -> dict:
    # command name -> function, for the dispatch in __main__.py
    return {"multimodal": lambda a: cmd_multimodal(a, hb), "medpix-context": lambda a: cmd_medpix_context(a, hb)}
# Follow-up: the patient comes back after a suggested test (or with any new information) and the case is analysed
# again. The follow-up file has the same fields as a case file (note_text, images, labs, vitals, genes, hpo_terms,
# details, ...) plus an optional list of test results:
#   "test_results": [{"test": "Echocardiogram", "result": "negative"},
#                    {"test": "FBN1 gene testing", "result": "positive", "genes": ["FBN1"],
#                     "classification": "pathogenic", "details": "c.4082G>A, heterozygous"}]
# "test" is a test's name or id as the suggestions showed it; "result" is positive / negative (or abnormal / normal,
# detected / not detected) or a short free-text result; "details" is free text.
#
# How it runs (only existing code: diagnose_one, the ranker, the workup):
#  1. before: the original case through the whole pipeline (Stage A, Stage 1, ranking, evidence, tests)
#  2. merge: original + follow-up into one case. The original note is kept as it is (so Stage 1's reading of it is
#     the same, from the reply cache); the follow-up's note, image_context and test results go into followup_text,
#     which Stage 1 reads as a text of its own. Images, labs, genes and HPO terms are added. A negative result of a test
#     marks absent the findings the test was suggested for (or, if it was not suggested, the findings it shows in the
#     top 3 diseases); a positive result marks the finding present when there is exactly one. A gene result classified
#     pathogenic or likely pathogenic adds the gene to the case (the ranker then counts it); a negative gene test makes
#     the diseases of those genes 10x less likely (LRRanker.GENE_NEG_LR).
#  3. after: the merged case through the whole pipeline again, from Stage A (images already described come from the
#     reply cache, so only new images are sent to the vision model)
#  4. compare: rank and probability of each disease before and after, the new findings, and a status per disease:
#     confirmed           a gene result classified pathogenic / likely pathogenic in one of the disease's genes (if
#                         several diseases share the gene, the best-ranked one; the others: "same gene, fits less")
#     strongly supported  probability >= 90% after the follow-up and a test that was a "confirm" test for it came back
#                         positive
#     not yet confirmed   otherwise; its next tests are the after-run's suggestions
#  The ranker's probability alone never makes a disease "confirmed".
import json
import re
from pathlib import Path

POSITIVE = re.compile(r"^\s*(positive|abnormal|detected|present|yes|confirmed|(likely )?pathogenic)\b", re.I)
NEGATIVE = re.compile(r"^\s*(negative|normal|not detected|absent|no|unremarkable|within normal limits)\b", re.I)
CAUSAL = {"pathogenic", "likely pathogenic"}
STRONG = 0.90
MAX_IMPLIED = 5     # a result with no details marks at most this many findings; a vaguer one must say which
CLASS = re.compile(r"\b(likely pathogenic|pathogenic|uncertain significance|vus|likely benign|benign)\b", re.I)


def classification(t: dict) -> str:
    # the variant classification: the field, else the words in the result or details ("pathogenic variant ...").
    # "likely pathogenic" is tested before "pathogenic", so it is never read as the stronger class
    cls = str(t.get("classification") or "").strip().lower()
    if cls:
        return cls
    m = CLASS.search(f"{t.get('result') or ''} {t.get('details') or ''}")
    return {"vus": "uncertain significance"}.get(m.group(1).lower(), m.group(1).lower()) if m else ""


def vague(wk, tid: str | None, n_found: int) -> bool:
    # a general exam or specialist evaluation, or a test that looks for many findings here
    from .workup import is_generic
    test = wk.cat.tests.get(tid) if wk and tid else None
    return n_found > MAX_IMPLIED or bool(test and (is_generic(test.name) or test.kind == "specialist"))


def result_kind(result) -> str:
    text = str(result or "")
    return "negative" if NEGATIVE.search(text) else "positive" if POSITIVE.search(text) else "other"


def result_line(t: dict) -> str:
    # one test result as a sentence for the note
    genes = ", ".join(t.get("genes") or [])
    extra = "; ".join(str(x) for x in (t.get("classification") and f"{t['classification']} variant"
                                        + (f" in {genes}" if genes else ""), t.get("details")) if x)
    return f"{t.get('test') or t.get('id') or 'Test'}: {t.get('result', '')}" + (f" ({extra})" if extra else "") + "."


def _abs_images(obj: dict, folder: Path) -> list[dict]:
    # image paths are relative to their case file's folder; the merged file lives elsewhere, so make them absolute
    out = []
    for img in obj.get("images") or []:
        if isinstance(img, dict) and img.get("path"):
            p = Path(img["path"])
            out.append({**img, "path": str(p if p.is_absolute() else (folder / p).resolve())})
    return out


def merge(orig: dict, new: dict, orig_dir: Path, new_dir: Path) -> dict:
    # the original case with the follow-up added; nothing in the original is removed
    m = {k: v for k, v in orig.items() if k != "images"}
    head = "Follow-up" + (f" ({new['date']})" if new.get("date") else "") + ":"
    results = [result_line(t) for t in new.get("test_results") or [] if isinstance(t, dict)]
    added = "\n".join(x for x in (new.get("note_text") or "", new.get("image_context") or "",
                                  "Test results:\n" + "\n".join(results) if results else "") if x)
    if added:        # read by Stage 1 as its own text; the original note_text is left as it is
        m["followup_text"] = ((orig.get("followup_text") or "").rstrip() + f"\n\n{head}\n{added}").strip()
    imgs, seen = [], set()
    for img in _abs_images(orig, orig_dir) + _abs_images(new, new_dir):
        if img["path"] not in seen:
            seen.add(img["path"])
            imgs.append(img)
    if imgs:
        m["images"] = imgs
    for k in ("labs", "hpo_terms", "image_findings"):
        if new.get(k):
            m[k] = list(orig.get(k) or []) + list(new[k])
    m["genes"] = list(dict.fromkeys(list(orig.get("genes") or []) + list(new.get("genes") or [])))
    if new.get("vitals"):
        m["vitals"] = {**(orig.get("vitals") or {}), **new["vitals"]}
    if new.get("details"):
        m["details"] = {**(orig.get("details") or {}), **new["details"]}
    for k in ("age", "sex"):
        if new.get(k) is not None:
            m[k] = new[k]
    if "local_only" in (orig.get("policy", "local_only"), new.get("policy")):      # the stricter policy wins
        m["policy"] = "local_only"
    return m


def find_test(wk, before: dict, t: dict) -> str | None:
    # the test id a result is for: its id, its exact name, or the one test suggested before whose name contains it
    # (or is contained in it). None: the result is still read from the note, but marks no findings by itself.
    q = str(t.get("id") or t.get("test") or "").strip()
    norm = lambda s: " ".join(re.sub(r"[^a-z0-9]+", " ", re.sub(r"\(loinc test\)", "", s.lower())).split())
    if not q:
        return None
    if wk and q in wk.cat.tests:
        return q
    if q.startswith("GENE:"):
        return q
    names = {}
    if wk:
        for tid, test in wk.cat.tests.items():
            names.setdefault(norm(test.name), tid)
    if norm(q) in names:
        return names[norm(q)]
    shown = {}
    for d in ((before.get("tests") or {}).get("diseases") or {}).values():
        for x in (d.get("tests") or []) + (d.get("review") or []) + ([d["gene_test"]] if d.get("gene_test") else []):
            shown[norm(x["test"])] = x["id"]
    for x in (before.get("tests") or {}).get("separate") or []:
        shown[norm(x["test"])] = x["id"]
    hits = {tid for n, tid in shown.items() if len(norm(q)) >= 4 and (norm(q) in n or n in norm(q))}
    return hits.pop() if len(hits) == 1 else None


def suggested_for(before: dict, tid: str, diseases: list[str] | None = None) -> set[str]:
    # the findings this test was suggested for (its "shows"), over the given diseases (default: every one shown)
    out = set()
    for k, d in ((before.get("tests") or {}).get("diseases") or {}).items():
        if diseases is not None and k not in diseases:
            continue
        for x in (d.get("tests") or []) + (d.get("review") or []):
            if x.get("id") == tid:
                out |= {s["hpo_id"] for s in x.get("shows") or []}
    return out


def looks_for(wk, case, ex, tid: str, diseases: list[str]) -> set[str]:
    # the findings a (phenotype) test would show in these diseases
    from .workup import known_sets
    yes, no = known_sets(wk.r.ont, case)
    out = set()
    for d in diseases:
        for s in wk.features(d, ex, yes, no, case.meta.get("sex")):
            u = wk.r.ont.resolve(s.hpo_id)
            if u and tid in wk.cat.observing(u):
                out.add(u)
    return out


def interpret(wk, before: dict, c0, ex0, results: list[dict]) -> tuple[list[dict], list[str], list[str], list[dict]]:
    # (HPO terms to add, genes with a causal variant, genes with a negative test, one row per result saying what it did)
    shown = [x["id"] for x in before.get("top", [])]
    terms, genes, negative, rows = [], [], [], []
    for t in results:
        if not isinstance(t, dict):
            continue
        kind, tid = result_kind(t.get("result")), find_test(wk, before, t)
        row = {"test": t.get("test") or t.get("id"), "result": t.get("result"), "kind": kind, "matched": tid,
               "effect": [], "warning": ""}
        gs = [g.strip() for g in (t.get("genes") or ([] if not tid or not tid.startswith("GENE:")
                                                      else tid[5:].split(","))) if g.strip()]
        cls = classification(t)
        if gs or (tid or "").startswith("GENE:"):
            row["genes"], row["classification"] = gs, cls or None
            if kind == "positive" and cls in CAUSAL:
                genes += [g for g in gs if g not in genes]
                row["effect"].append(f"genes added to the case: {', '.join(gs)}")
            elif kind == "positive":
                row["warning"] = ("a positive gene result counts only with a classification of pathogenic or likely "
                                  "pathogenic" + (f" (given: {cls})" if cls else ""))
            elif kind == "negative" and gs:
                negative += [g for g in gs if g not in negative]
                row["effect"].append(f"diseases of {', '.join(gs)} made 10x less likely (a negative gene test lowers, "
                                     "but does not rule out, a disease)")
        elif tid and wk and kind in ("positive", "negative"):
            # the findings it was suggested for in the top 3 diseases; if it was not, what it shows in them. A vague
            # result (a general or specialist evaluation, or a test looking for many findings) marks none by itself:
            # the details or the note must name them, or one vague line would add or remove many findings at once
            found = sorted(suggested_for(before, tid, shown[:3]) or looks_for(wk, c0, ex0, tid, shown[:3]))
            is_vague = vague(wk, tid, len(found))
            if kind == "negative" and found and not is_vague:
                terms += [{"id": u, "status": "absent"} for u in found]
                row["effect"].append("absent: " + ", ".join(wk.r.ont.name(u) for u in found))
            elif kind == "negative" and found and t.get("details"):
                row["effect"].append(f"looks for {len(found)} findings here; which are absent is read from the details")
            elif kind == "negative" and found:
                row["warning"] = (f"negative, but this test looks for {len(found)} findings here: say which were "
                                  "looked for in details, note_text or hpo_terms so they can be scored")
            elif kind == "positive" and len(found) == 1 and not is_vague:
                terms.append({"id": found[0], "status": "present"})
                row["effect"].append("present: " + wk.r.ont.name(found[0]))
            elif kind == "positive" and t.get("details"):
                row["effect"].append(f"looks for {len(found)} findings here; which one is read from the details")
            elif kind == "positive":
                row["warning"] = (f"positive, but this test looks for {len(found)} different findings here: say which "
                                  "in details, note_text or hpo_terms so they can be scored")
        if not row["matched"]:
            row["warning"] = row["warning"] or ("not one of the suggested tests: the result is read from the note "
                                                "only (give the test's name or id as shown in the suggestions)")
        rows.append(row)
    return terms, genes, negative, rows


def statuses(wk, before: dict, after: dict, rows: list[dict], probs: dict) -> list[dict]:
    # the status of every disease shown after the follow-up (see the top of the file)
    causal = {g for x in rows for g in (x.get("genes") or []) if x["kind"] == "positive"
              and (x.get("classification") or "") in CAUSAL}
    # a positive gene result supports a disease only when classified pathogenic / likely pathogenic
    is_gene = lambda x: "genes" in x or str(x.get("matched") or "").startswith("GENE:")
    pos = {x["matched"] for x in rows if x["kind"] == "positive" and x["matched"]
           and (not is_gene(x) or (x.get("classification") or "") in CAUSAL)}
    bt = ((before.get("tests") or {}).get("diseases") or {})
    at = ((after.get("tests") or {}).get("diseases") or {})
    out, gene_done = [], set()
    for x in after.get("top", []):            # best first, so the best-fitting disease of a gene is met first
        d = x["id"]
        dgenes = set(wk.gene_list(d)[0]) if wk else set()
        confirm_tests = {t["id"] for t in (bt.get(d) or {}).get("tests") or [] if "confirm" in (t.get("role") or "")}
        p = probs.get(d, 0.0)
        if causal & dgenes and not causal & dgenes <= gene_done:
            level, why = "confirmed", (f"disease-causing variant in {', '.join(sorted(causal & dgenes))}; of the "
                                       "diseases of this gene, the one the findings fit best")
            gene_done |= causal & dgenes
        elif causal & dgenes:
            level, why = "same gene, fits less", (f"variant in {', '.join(sorted(causal & dgenes))}, which also causes "
                                                  "a disease ranked higher (allelic disease): the findings fit that one better")
        elif p >= STRONG and pos & confirm_tests:
            level, why = "strongly supported", (f"probability {p:.0%} and a confirming test came back positive: "
                                                + ", ".join(sorted(t["test"] for t in bt[d]["tests"] if t["id"] in pos)))
        else:
            level, why = "not yet confirmed", (f"probability {p:.0%}" + ("" if pos & confirm_tests else
                                               "; no confirming test has come back positive"))
        from .workup import is_generic
        nxt = [t["test"] for t in ((at.get(d) or {}).get("tests") or [])
               if not is_generic(t["test"], len(t.get("shows") or []))][:3] if level == "not yet confirmed" else []
        out.append({"id": d, "name": x["name"], "status": level, "why": why, "next_tests": nxt})
    return out


def cmd_followup(a, hb):
    from .intake import is_intake
    from .workup import ranker_probs
    orig_path, new_path = Path(a.case), Path(a.followup)
    try:
        orig = json.loads(orig_path.read_text(encoding="utf-8"))
        new = json.loads(new_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SystemExit(f"cannot read the case or the follow-up: {e}")
    if not is_intake(orig) or not isinstance(new, dict):
        raise SystemExit("followup needs a hoofbeats case file (with case_id) and a follow-up file (see followup.py)")
    case_id = orig["case_id"]
    out = Path(a.out) / "followup" / case_id
    out.mkdir(parents=True, exist_ok=True)
    merged_path = out / "merged_case.json"
    merged = merge(orig, new, orig_path.parent.resolve(), new_path.parent.resolve())
    merged_path.write_text(json.dumps(merged, indent=1), encoding="utf-8")

    r, reader, s1, describer, wk = hb.diagnose_setup(a, [str(orig_path), str(merged_path)])
    use_genes = r.use_genes
    try:
        print(f"\n===== before the follow-up: {orig_path} =====")
        r.use_genes = bool(orig.get("genes"))
        c0, ex0, rk0 = hb.diagnose_one(a, r, orig_path, reader, s1, out / "before", describer, wk)
        before = json.loads((out / "before" / f"{c0.case_id}.json").read_text(encoding="utf-8"))
        terms, genes, negative, rows = interpret(wk, before, c0, ex0, new.get("test_results") or [])
        merged["hpo_terms"] = list(merged.get("hpo_terms") or []) + terms
        merged["genes"] = list(dict.fromkeys(list(merged.get("genes") or []) + genes))
        merged["genes_negative"] = list(dict.fromkeys(list(merged.get("genes_negative") or []) + negative))
        merged_path.write_text(json.dumps(merged, indent=1), encoding="utf-8")
        print(f"\n===== after the follow-up: {new_path} (merged: {merged_path}) =====")
        r.use_genes = bool(merged["genes"])
        c1, ex1, rk1 = hb.diagnose_one(a, r, merged_path, reader, s1, out / "after", describer, wk)
        after = json.loads((out / "after" / f"{c1.case_id}.json").read_text(encoding="utf-8"))
    finally:
        r.use_genes = use_genes

    p0, p1 = ranker_probs(rk0), ranker_probs(rk1)
    truth = [d for d in (c1.truth or []) if d in r.d_index]
    ids = list(dict.fromkeys([x["id"] for x in after["top"]] + [x["id"] for x in before["top"][:5]] + truth))
    changes = [{"id": d, "name": r.db.names[d], "rank_before": rk0.rank_of([d]), "rank_after": rk1.rank_of([d]),
                "p_before": round(float(p0[r.d_index[d]]), 4), "p_after": round(float(p1[r.d_index[d]]), 4),
                "truth": d in truth} for d in ids]
    old = {(f["hpo_id"], f["status"]) for f in before["findings"]}
    new_findings = [f for f in after["findings"] if (f["hpo_id"], f["status"]) not in old]
    probs = {x["id"]: x["p_after"] for x in changes}
    status = statuses(wk, before, after, rows, probs)
    result = {"case_id": case_id, "followup": str(new_path), "before": str(out / "before" / f"{case_id}.json"),
              "after": str(out / "after" / f"{case_id}.json"), "merged_case": str(merged_path),
              "test_results": rows, "new_findings": new_findings, "changes": changes, "status": status,
              "note": "probabilities are the ranker's calibrated ones (the tests' pre_test); 'confirmed' needs a gene "
                      "result classified pathogenic or likely pathogenic"}
    (out / "followup.json").write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(f"\n===== follow-up summary: {case_id} =====")
    for x in rows:
        print(f"test {x['test']}: {x['result']} -> " + ("; ".join(x["effect"]) or "read from the note")
              + (f"  [!] {x['warning']}" if x["warning"] else ""))
    print(f"\nNew findings ({len(new_findings)}): " + ("; ".join(f"{f['status']} {f['term']}" for f in new_findings) or "none"))
    print("\n| disease (shown now, top 5 before, truth) | rank before -> after | probability before -> after | status |"
          "\n|---|---|---|---|")
    st = {x["id"]: x for x in status}
    for x in changes:
        s = st.get(x["id"])
        print(f"| {'[truth] ' if x['truth'] else ''}{x['name'][:45]} | {x['rank_before']} -> {x['rank_after']} | "
              f"{x['p_before']:.1%} -> {x['p_after']:.1%} "
              f"| {s['status'] if s else 'not shown now'} |")
    best = next((x for x in status if x["status"] == "confirmed"), None) or \
        next((x for x in status if x["status"] == "strongly supported"), None) or (status[0] if status else None)
    if best:
        print(f"\n{best['name']}: {best['status']} ({best['why']})"
              + (f"; next tests: {', '.join(best['next_tests'])}" if best["next_tests"] else ""))
    hb.wrote(out / "followup.json")


def add_arguments(p, sub):
    s = sub.add_parser("followup", help="re-analyse a case with new information (test results, notes, images, labs, "
                                        "genes): the whole pipeline before and after, and what changed")
    s.add_argument("case", help="the original hoofbeats case file")
    s.add_argument("followup", help="the follow-up file: case-file fields plus test_results (see followup.py)")
    s.add_argument("-k", type=int, default=10, help="diseases shown")
    s.add_argument("--out", default="results", help="results in <out>/followup/<case_id>/")
    s.add_argument("--sources", default="OMIM", help="disease sources ranked: OMIM (default) or OMIM,ORPHA")


def commands(hb) -> dict:
    return {"followup": lambda a: cmd_followup(a, hb)}

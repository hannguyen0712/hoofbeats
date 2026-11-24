"""Paper highlights for the review screen (Part 4).

For a knowledge item (an HPO annotation of a disease), find the sentences in its cited papers' PubMed abstracts that
mention the item's finding, so the screen can show and highlight where the paper reports it.

    papers, note = fetch_papers(["PMID:123", ...])          # PubMed abstracts, with a local cache
    hits = highlights(ont, ext, item, papers, memo)          # [{"pmid", "title", "year", "sentence", "marks"}]

Only paper IDs are sent to PubMed, never patient data. A sentence counts when the Stage 1 reader finds the item's
finding (or a narrower form of it, or the broader term its frequency belongs to) stated as present; "no X" and
hedged mentions do not count. A highlight shows that the paper reports the finding for this disease; it does not
prove the diagnosis.
"""
import os, re, urllib.request
from hoofbeats.model import Status
from hoofbeats.pubmed import Paper, PubMed, SKIP_TYPES

CACHE = os.environ.get("PUBMED_CACHE", "data/kb/pubmed_cache.json")
COUNTS = {Status.PRESENT, Status.FAMILY}        # in a paper, "affected relatives had X" also reports X
NEGATION = re.compile(r"\b(no|not|without|absent|absence|none|neither|nor|lack\w*|never|negative|denied|excluded)\b", re.I)


def fetch_papers(pmids, cache=CACHE, offline=None, opener=None):
    # {pmid: Paper} for the PubMed ids given (with or without the "PMID:" prefix), and a note if PubMed failed
    offline = os.environ.get("PUBMED_OFFLINE") == "1" if offline is None else offline
    ids = sorted({str(p).split(":")[-1] for p in pmids if str(p).split(":")[-1].isdigit()})
    quick = lambda url, timeout=30: urllib.request.urlopen(url, timeout=min(timeout, 8))   # a slow network must not
    pm = PubMed(cache, opener=opener or quick)                                            # hold up the screen
    papers, note = {}, ""
    try:
        if not offline:
            for k in range(0, len(ids), 100):
                papers.update(pm.fetch(ids[k:k + 100]))
    except Exception as e:                       # offline, timeout, NCBI error: use what the cache has
        note = f"PubMed could not be reached ({type(e).__name__}), so only saved abstracts are used."
    finally:
        if not offline:
            pm.save()
    for p in ids:
        if p not in papers and p in pm.cache["papers"]:
            papers[p] = Paper(**pm.cache["papers"][p])
    return papers, note


def highlights(ont, ext, item, papers, memo, per_item=3):
    # sentences of the item's cited abstracts that mention its finding: one sentence per paper, best paper first
    # the finding itself or a narrower form of it; the broader term (via) only by name, since it can be very general
    term, via = ont.resolve(item.get("hpo_id")), ont.resolve(item.get("via") or None)
    if not term and not via:
        return []
    fits = lambda m: (term is not None and term in ont.ancestors(m.hpo_id)) or (via is not None and m.hpo_id == via)
    out = []
    for ref in item.get("refs") or []:
        pmid = str(ref).split(":")[-1]
        p = papers.get(pmid)
        if not str(ref).startswith("PMID:") or p is None or set(p.types) & SKIP_TYPES:
            continue
        for s in [p.title] + list(p.sentences):
            if s not in memo:
                memo[s] = ext.mentions(s)
            marks = [(m.start, m.end) for m in memo[s] if m.status in COUNTS and fits(m)
                     and not NEGATION.search(s[:m.start])]      # stricter than the reader: any negation before it
            if marks:
                out.append({"pmid": f"PMID:{pmid}", "title": p.title, "year": p.year, "sentence": s, "marks": marks})
                break
        if len(out) >= per_item:
            break
    return out


def add_to_lists(out, ont, ext, cache=CACHE, offline=None, opener=None):
    # adds "lit" (highlights) and "lit_papers" (abstracts found) to every knowledge item shown on the screen
    items = [i for lst in out["lists"].values() for d in lst for i in d["pack"]
             if i.get("kind") == "knowledge" and i.get("hpo_id")]
    pmids = [x for i in items for x in i.get("refs") or [] if str(x).startswith("PMID:")]
    if not pmids:
        return
    papers, note = fetch_papers(pmids, cache, offline, opener)
    memo = {}
    for i in items:
        i["lit"] = highlights(ont, ext, i, papers, memo)
        i["lit_papers"] = sum(1 for x in i.get("refs") or [] if str(x).startswith("PMID:")
                              and str(x).split(":")[-1] in papers)
        i["lit_retracted"] = [x for x in i.get("refs") or [] if str(x).startswith("PMID:")
                              and set(getattr(papers.get(str(x).split(":")[-1]), "types", ())) & SKIP_TYPES]
    out["literature"] = {"papers": len(papers), "note": note}

"""Paper highlights for the review screen (Part 4).

For a knowledge item (an HPO annotation of a disease), find sentences in its cited papers that mention the item's
finding, so the screen can show and highlight where the paper reports it. Abstracts come from PubMed; for open-access
papers, the full text (results, case reports, tables, figure legends) comes from PubMed Central's BioC service.

    papers, note = fetch_papers(["PMID:123", ...])          # abstracts, with a local cache
    full, note = fetch_fulltext(["PMID:123", ...])          # open-access full text, with a local cache
    hits = highlights(ont, ext, item, papers, memo, full)    # [{"pmid", "title", "year", "sentence", "marks", "where"}]

Only paper IDs are sent to NCBI, never patient data. A sentence counts when the Stage 1 reader finds the item's
finding (or a narrower form of it, or the broader term its frequency belongs to) stated as present; "no X" and
hedged mentions do not count. A highlight shows that the paper reports the finding for this disease; it does not
prove the diagnosis.
"""
import json, os, re, time, urllib.error, urllib.request
from pathlib import Path
from hoofbeats.model import Status
from hoofbeats.pubmed import Paper, PubMed, SKIP_TYPES, split_sentences

CACHE = os.environ.get("PUBMED_CACHE", "data/kb/pubmed_cache.json")
FULLTEXT_CACHE = os.environ.get("PMC_CACHE", "data/kb/pmc_fulltext_cache.json")
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{}/unicode"
SECTIONS = {"TITLE": "title", "ABSTRACT": "abstract", "INTRO": "introduction", "METHODS": "methods",
            "RESULTS": "results", "CASE": "case report", "DISCUSS": "discussion", "CONCL": "conclusion",
            "FIG": "figure legend", "TABLE": "table"}                # references, funding etc. are left out
COUNTS = {Status.PRESENT, Status.FAMILY}        # in a paper, "affected relatives had X" also reports X
NEGATION = re.compile(r"\b(no|not|without|absent|absence|none|neither|nor|lack\w*|never|negative|denied|excluded)\b", re.I)


def _quick(url, timeout=30):
    # a slow network must not hold up the screen
    return urllib.request.urlopen(url, timeout=min(timeout, 8))


def _offline(offline):
    return os.environ.get("PUBMED_OFFLINE") == "1" if offline is None else offline


def _ids(pmids):
    return sorted({str(p).split(":")[-1] for p in pmids if str(p).split(":")[-1].isdigit()})


def fetch_papers(pmids, cache=CACHE, offline=None, opener=None):
    # {pmid: Paper} for the PubMed ids given (with or without the "PMID:" prefix), and a note if PubMed failed
    offline, ids = _offline(offline), _ids(pmids)
    pm = PubMed(cache, opener=opener or _quick)
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


def parse_bioc(raw):
    # BioC JSON from PubMed Central -> {"pmc": "PMC...", "sentences": [[section, sentence], ...]}, or None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None                              # "No result can be found": not in the open-access subset
    pmc, out = "", []
    for coll in data if isinstance(data, list) else [data]:
        for doc in (coll or {}).get("documents", []) if isinstance(coll, dict) else []:
            for ps in doc.get("passages", []):
                info = ps.get("infons") or {}
                pmc = pmc or info.get("article-id_pmc", "") or doc.get("id", "")
                sec = SECTIONS.get(info.get("section_type", ""))
                if sec and sec not in ("title", "abstract"):
                    out += [[sec, s] for s in split_sentences(ps.get("text") or "")]
    if not out:
        return None
    pmc = str(pmc)
    return {"pmc": pmc if pmc.startswith("PMC") or not pmc else f"PMC{pmc}", "sentences": out}


def fetch_fulltext(pmids, cache=FULLTEXT_CACHE, offline=None, opener=None, limit=200):
    # {pmid: {"pmc", "sentences"}} for the open-access papers among the ids given; non-open-access papers are
    # remembered as None so they are not asked for again
    offline, ids, opener = _offline(offline), _ids(pmids), opener or _quick
    path = Path(cache)
    store = json.loads(path.read_text()) if path.exists() else {}
    note, asked, last = "", 0, 0.0
    if not offline:
        for p in ids:
            if p in store or asked >= limit:
                continue
            time.sleep(max(0.0, 0.34 - (time.time() - last)))
            try:
                with opener(BIOC.format(p), timeout=30) as resp:
                    store[p] = parse_bioc(resp.read().decode("utf-8", "replace"))
            except urllib.error.HTTPError:
                store[p] = None                  # not available as open access
            except Exception as e:               # offline or timeout: stop asking, use what is saved
                note = f"PubMed Central could not be reached ({type(e).__name__}), so only saved full texts are used."
                break
            last, asked = time.time(), asked + 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store))
    return {p: store[p] for p in ids if store.get(p)}, note


def _keys(ont, term, via):
    # cheap pre-filter: word stems of the finding's names (and its narrower forms) and of the broader term's names
    tids = ([*sorted(ont.descendants(term))[:300]] if term else []) + ([via] if via else [])
    words = set()
    for t in tids:
        info = ont.terms.get(t)
        if info:
            for label in [info.name, *info.synonyms, *(s for s, _ in info.extra_synonyms)]:
                words |= {w[:5] for w in re.findall(r"[a-z]{4,}", label.lower())}
    return words


def highlights(ont, ext, item, papers, memo, full=None, exclude=(), per_item=3):
    # sentences of the item's cited papers that mention its finding: one sentence per paper, abstract first
    term, via = ont.resolve(item.get("hpo_id")), ont.resolve(item.get("via") or None)
    if not term and not via:
        return []
    fits = lambda m: (term is not None and term in ont.ancestors(m.hpo_id)) or (via is not None and m.hpo_id == via)
    keys, skip, out = _keys(ont, term, via), {str(x).split(":")[-1] for x in exclude}, []
    for ref in item.get("refs") or []:
        pmid = str(ref).split(":")[-1]
        p = papers.get(pmid)
        if not str(ref).startswith("PMID:") or pmid in skip or (p is not None and set(p.types) & SKIP_TYPES):
            continue
        ft = (full or {}).get(pmid) or {}
        cands = ([("title", p.title)] + [("abstract", s) for s in p.sentences] if p else []) + \
            [(sec, s) for sec, s in ft.get("sentences", [])]
        for where, s in cands:
            low = s.lower()
            if not any(k in low for k in keys):
                continue
            if s not in memo:
                memo[s] = ext.mentions(s)
            marks = [(m.start, m.end) for m in memo[s] if m.status in COUNTS and fits(m)
                     and not NEGATION.search(s[:m.start])]      # stricter than the reader: any negation before it
            if marks:
                out.append({"pmid": f"PMID:{pmid}", "title": p.title if p else "", "year": p.year if p else None,
                            "sentence": s, "marks": marks, "where": where, "pmc": ft.get("pmc", "")})
                break
        if len(out) >= per_item:
            break
    return out


def add_to_lists(out, ont, ext, exclude=(), offline=None):
    # adds "lit" (highlights) and "lit_papers" (papers found) to every knowledge item shown on the screen
    items = [i for lst in out["lists"].values() for d in lst for i in d["pack"]
             if i.get("kind") == "knowledge" and i.get("hpo_id")]
    pmids = [x for i in items for x in i.get("refs") or [] if str(x).startswith("PMID:")]
    if not pmids:
        return
    papers, note = fetch_papers(pmids, offline=offline)
    full, note2 = fetch_fulltext(pmids, offline=offline)
    memo = {}
    for i in items:
        i["lit"] = highlights(ont, ext, i, papers, memo, full, exclude)
        refs = [str(x).split(":")[-1] for x in i.get("refs") or [] if str(x).startswith("PMID:")]
        i["lit_papers"] = sum(1 for x in refs if x in papers or x in full)
        i["lit_retracted"] = [f"PMID:{x}" for x in refs if set(getattr(papers.get(x), "types", ())) & SKIP_TYPES]
    out["literature"] = {"papers": len(papers), "full_text": len(full), "note": " ".join(x for x in (note, note2) if x)}

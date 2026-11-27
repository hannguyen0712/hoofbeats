from dataclasses import dataclass
from hoofbeats.model import verify_claim
from hoofbeats.common_hpo import is_estimated, is_symptom_profile
import math
import dataclasses

@dataclass
class Item:
    id: str
    kind: str
    text: str
    hpo_id:str = None
    status: str = None
    words: str = ""
    source: str = ""
    refs: tuple = ()
    role: str = ""
    via: str = ""        # knowledge items: the broader HPO term the frequency belongs to, if not the finding itself

def patient_items(r, case, claims):
    """Make one P item for each patient finding that claim uses."""
    items = {}
    by_term = {r.ont.resolve(f.hpo_id): f for f in case.findings}
    for cl in claims:
        f = by_term.get(cl.patient_term)
        if f is None:
            continue
        if any(i.hpo_id == cl.patient_term for i in items.values()):
            continue
        pid = f"P{len(items)+1}"
        name = r.ont.name(cl.patient_term)
        status = f.status.value
        text = f"Patient: '{name}' is {status} (source: {f.source})"
        words = f.evidence if f.evidence and f.evidence.lower() != name.lower() else ""
        items[pid] = Item(
            pid,
            "patient",
            text,
            cl.patient_term,
            status,
            words,
            f.source)
    return items

def lab_profile(r, d, refs=()):
    """Name of the lab profile a common disease's frequencies come from (MIMIC-IV admissions), or "" for a rare
    disease, whose frequencies come from HPO annotations."""
    if d not in r.db.scope:          # rare diseases and symptom profiles come from HPO annotations
        return ""
    return "MIMIC-IV demo" if any("demo" in x.lower() for x in refs) or not refs else "MIMIC-IV"


def common_source(r, d):
    return "HPO common-disease annotations" if is_symptom_profile(r.db, d) else "HPO Annotations"


def knowledge_items(r, case, d, claims, ex, start=1):
    """Make one K item for each knowledge claim."""
    items = {}
    lr_of = {c.hpo_id: c.log_lr for c in r.contributions(case, d, ex)}   # signed: > 0 supports d, < 0 counts against
    n = start
    for cl in claims:
        ok, reason = verify_claim(r, case, cl, ex)
        if not ok:
            print(" dropped", cl.text, "->", reason)
            continue
        # the sign of the finding's contribution decides the role: every claim has a patient term, so the term alone
        # cannot (a lacking hallmark with LR 0.45 is evidence against the disease)
        role = "for" if lr_of.get(cl.patient_term, 0.0) > 0 else "against"
        kid = f"K{n}"
        text = cl.text.split(" [")[0]
        # matched through a broader annotation: the frequency is the broader finding's, so the text names that
        # finding as the thing the disease shows, not the patient's narrower one
        via = r.ont.resolve(cl.kb_term) if cl.kb_term else None
        via = via if via and via != cl.patient_term else ""
        if via:
            text = re.sub(r"shows this in (~\d+% of cases) \(via '([^']+)'\)",
                          r"shows '\2', the broader HPO term it falls under, in \1", text)
        lab = lab_profile(r, d, cl.refs)
        if lab:     # a common disease: the frequency is the share of its admissions with this lab result
            text = text.replace(" of cases", f" of {lab} admissions with this diagnosis")
        if is_estimated(r.db, d):    # text-mined annotation without a frequency: state no number for it
            text = re.sub(r",? in ~\d+% of cases", " (frequency not recorded)", text)
        items[kid] = Item(
            kid,
            "knowledge",
            text,
            cl.patient_term,
            cl.status.value,
            source = f"{lab} lab profile" if lab else common_source(r, d),
            refs = tuple(sorted(cl.refs)),
            role = role,
            via = via)
        n += 1
    return items

def known_terms(r, case):
    """Return every HPO term the record already mentions in some form: each recorded finding (any status), its
    broader terms and its narrower terms. A missing item says "the patient record does not mention this finding",
    which is wrong when the record has a broader or narrower version of it (e.g. 'Inability to walk' present and
    'Inability to walk by childhood/adolescence' expected; 'Joint contracture of the hand' absent and 'Joint
    contracture' expected)."""
    known = set()
    for f in case.findings:
        t = r.ont.resolve(f.hpo_id)
        if t is None:
            continue
        known |= set(r.ont.ancestors(t)) | set(r.ont.descendants(t))
    return known

def missing_items(r, case, d, ex, start, max_missing = 3):
    """Make one M item for each missing finding in the case."""
    items = {}
    name = r.db.names[d]
    known = known_terms(r, case)

    candidates = []
    if is_estimated(r.db, d):       # no recorded frequencies, so nothing can be called "usually present"
        return items
    for a in r.db.annotations(d, ex):
        t = r.ont.resolve(a.hpo_id)
        if a.negated or a.freq < 0.5 or t is None or t in known:
            continue
        candidates.append((a.freq, t, a.refs))

    candidates.sort(key=lambda x: (-x[0], r.ont.name(x[1])))

    seen = set()
    n = start
    for freq, t, refs in candidates:
        if t in seen:
            continue
        seen.add(t)
        kid = f"K{n}"
        lab = lab_profile(r, d, refs)
        cases = f"{lab} admissions with this diagnosis" if lab else "cases"
        text = f"{name} shows '{r.ont.name(t)}' in ~{freq:.0%} of {cases}; the patient record does not mention this finding."
        items[kid] = Item(
            kid,
            "knowledge",
            text,
            t,
            source=f"{lab} lab profile" if lab else common_source(r, d),
            refs=tuple(sorted(refs)),
            role = "missing")
        n += 1
        if len(seen) >= max_missing:
            break
    return items

def runner_up_items(r, case, d, runner, ex, start, max_items = 2):
    """Make one R item for each runner-up disease."""
    items = {}
    mine = {c.hpo_id: c.log_lr for c in r.contributions(case, d, ex)}
    status = {c.hpo_id: c.status.value for c in r.contributions(case, d, ex)}
    theirs = {c.hpo_id: c.log_lr for c in r.contributions(case, runner, ex)}

    diffs = []
    for t in mine:
        if t in theirs:
            diffs.append((mine[t] - theirs[t], t))
    diffs.sort(reverse=True)

    n = start
    for gap, t in diffs[:max_items]:
        if gap <0.5:
            break
        kid = f"K{n}"
        if status[t] == "absent":
            text = (f"The absence of '{r.ont.name(t)}' counts less against {r.db.names[d]} than against "
                    f"{r.db.names[runner]}: likelihood ratio {math.exp(mine[t]):.2f} against {math.exp(theirs[t]):.2f}")
        else:
            text = (f"'{r.ont.name(t)}' fits {r.db.names[d]} better than {r.db.names[runner]}: "
                    f"likelihood ratio {math.exp(mine[t]):.2f} against {math.exp(theirs[t]):.2f}")
        items[kid] = Item(
            kid,
            "knowledge",
            text,
            t,
            source="ranker contributions",
            role = "compare")
        n += 1
    return items

def clean_claim(cl, ex):
    refs = tuple(x for x in cl.refs if x not in ex)
    return dataclasses.replace(cl, refs=refs) if refs else cl

def build_pack(r, case, d, p, ex, runner, claims):
    claims = [clean_claim(cl, ex) for cl in claims]
    pack = patient_items(r, case, claims)

    k = knowledge_items(r, case, d, claims, ex, start=1)
    pack.update(k)

    m = missing_items(r, case, d, ex, start=len(k)+1)
    pack.update(m)

    c = runner_up_items(r, case, d, runner, ex, start=len(k)+len(m)+1)
    pack.update(c)
    return pack
import re
from hoofbeats.llm import reply_json

SECTIONS = ("supports", "against", "expected_but_missing", "vs_runner_up")

PROMPT = """You explain one candidate diagnosis to a clinician, using ONLY the numbered evidence below.

Rules:
- Every sentence must end with the IDs it uses in square brackets, e.g. [P2, K5].
- Use P items for what the patient has, K items for what the knowledge base says.
- Do not add facts, numbers, findings or diagnoses that are not in the evidence.
- Copy numbers exactly as they appear in the evidence.
- Write 1 to 3 short sentences per field; leave a field as [] if no evidence fits it.

Disease: {name} ({did}), model probability {prob}
Runner-up: {runner}

Evidence:
{evidence}

Return only JSON, no markdown:
{{"supports": ["..."], "against": ["..."], "expected_but_missing": ["..."], "vs_runner_up": ["..."]}}"""


def make_prompt(r, d, p, pack, runner):
    lines = [f"[{i.id}] {i.text}" for i in pack.values()]
    runner_text = f"{r.db.names[runner]} ({runner})" if runner else "none"
    return PROMPT.format(name=r.db.names[d], did=d, prob=f"{p:.1%}", runner=runner_text, evidence="\n".join(lines))


def parse_trace(raw):
    obj = reply_json(raw or "", SECTIONS)
    if not isinstance(obj, dict):
        obj = {}
    out = {}
    for s in SECTIONS:
        v = obj.get(s) or []
        if isinstance(v, str):
            v = re.split(r"(?<=[.!?\]])\s+(?=[A-Z])", v)
        if not isinstance(v, list):
            v = []
        out[s] = [str(x).strip() for x in v if str(x).strip()]
    return out


def template_trace(pack):
    p_for = {i.hpo_id: i.id for i in pack.values() if i.kind == "patient"}
    where = {"for": "supports", "against": "against", "missing": "expected_but_missing", "compare": "vs_runner_up"}
    out = {s: [] for s in SECTIONS}
    for i in pack.values():
        if i.kind != "knowledge":
            continue
        pid = p_for.get(i.hpo_id)
        cite = f"[{pid}, {i.id}]" if pid and i.role != "missing" else f"[{i.id}]"
        out[where[i.role]].append(f"{i.text.rstrip('.')}. {cite}")
    return out


# ---------------- Session 4: checks
NEGATION = re.compile(r"\b(no|not|absent|absence|lacks?|lacking|without|negative|denies|does not|did not|ruled out)\b", re.I)
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def cites(sentence):
    groups = re.findall(r"\[([^\]]*)\]", sentence)
    return [x for grp in groups for x in re.findall(r"\b([PK]\d+)\b", grp)]


def strip_cites(sentence):
    return re.sub(r"\[[^\]]*\b[PK]\d+\b[^\]]*\]", "", sentence)

def negated(body, name):
    # a negation word up to 3 words before the finding's name: "does not have aortic root aneurysm"
    return re.search(NEGATION.pattern + r"(?:\W+\w+){0,3}?\W+" + re.escape(name), body, re.I) is not None

def names_of(r, item):
    if not item.hpo_id:
        return []
    t = r.ont.resolve(item.hpo_id)
    out = [r.ont.name(t)] + list(getattr(r.ont.terms[t], "synonyms", []) or [])
    if item.words:
        out.append(item.words)
    return [n.lower() for n in out if n]


def check_sentence_basic(s, pack, r, case, disease_names=()):
    # the original V1-V4 (Oct-Nov 10), kept to compare with the stricter checker
    ids = cites(s)
    body = strip_cites(s).lower()
    # V1: cites something, and every cited ID is in this disease's pack
    if not ids:
        return {"ok": False, "reason": "V1: no citation"}
    missing = [i for i in ids if i not in pack]
    if missing:
        return {"ok": False, "reason": f"V1: {', '.join(missing)} not in the evidence"}
    items = [pack[i] for i in ids]
    # V2: cited patient findings are still in the record with that status; absent ones are stated as absent
    status_now = {r.ont.resolve(f.hpo_id): f.status.value for f in case.findings}
    for it in items:
        if it.kind != "patient":
            continue
        if status_now.get(it.hpo_id) != it.status:
            return {"ok": False, "reason": f"V2: {it.id} is no longer in the record as {it.status}"}
        if it.status == "absent" and not NEGATION.search(body):
            return {"ok": False, "reason": f"V2: {it.id} is absent but the sentence does not say so"}
        if it.status == "present" and any(negated(body, n) for n in names_of(r, it)):
            return {"ok": False, "reason": f"V2: {it.id} is present but the sentence says it is not"}
    # V3: the sentence names the finding of each cited item that has one
    for it in items:
        names = names_of(r, it)
        if names and not any(n in body for n in names):
            return {"ok": False, "reason": f"V3: does not name the finding of {it.id}"}
    # V4: every number in the sentence appears in the cited evidence (or a disease name)
    allowed = set(NUMBER.findall(" ".join(it.text for it in items) + " " + " ".join(disease_names)))
    for num in NUMBER.findall(body):
        if num not in allowed:
            return {"ok": False, "reason": f"V4: {num} is not in the cited evidence"}
    kinds = {it.kind for it in items}
    grade = "Verified" if kinds == {"patient", "knowledge"} else "KB only" if kinds == {"knowledge"} else "Patient only"
    return {"ok": True, "grade": grade, "reason": ""}


def check_trace(trace, pack, r, case, disease_names=(), strict=True):
    kept, removed = {s: [] for s in trace}, []
    for sec, sents in trace.items():
        for s in sents:
            res = check_sentence(s, pack, r, case, disease_names, section=sec, strict=strict)
            if res["ok"]:
                kept[sec].append({"text": s, "grade": res["grade"], "sources": res.get("sources", [])})
            else:
                removed.append({"section": sec, "text": s, "reason": res["reason"]})
    return kept, removed



PROMPT_V2 = """You explain one candidate diagnosis to a clinician, using ONLY the numbered evidence below.

How to write each item:
- Each item in a list is ONE complete English sentence that names the finding in words.
- The sentence ends with the IDs of the evidence it uses, in square brackets, e.g. [P2, K2].
- Never write an item that is only IDs. Never put the IDs at the start.
- Copy IDs exactly as listed below: patient items start with P, knowledge items start with K.
- One finding per sentence. Copy numbers exactly as they appear in the evidence.
- Do not add facts, numbers, findings, genes, tests or diagnoses that are not in the evidence.
- When an item says the disease shows a broader HPO term that the patient's finding falls under, name that broader
  term in the sentence: the percentage is for the broader term, not for the patient's exact finding.

Which evidence goes where:
- "supports": knowledge items marked (supports), each with its patient item.
- "against": knowledge items marked (against), each with its patient item.
- "expected_but_missing": knowledge items marked (expected but not recorded).
- "vs_runner_up": knowledge items marked (vs runner-up).
Write one sentence per knowledge item, at most 8 per list; leave a list empty [] if nothing is marked for it.

Example of the format (made-up evidence, not about this patient):
{{"supports": ["The patient has short stature, which Example syndrome shows in ~60% of cases [P1, K1]."],
  "against": [],
  "expected_but_missing": ["Example syndrome shows hearing loss in ~70% of cases, but the record does not mention it [K5]."],
  "vs_runner_up": ["Short stature fits Example syndrome better than Other syndrome, likelihood ratio 12.50 against 2.10 [P1, K6]."]}}

Disease: {name} ({did}), model probability {prob}
Runner-up: {runner}

Evidence:
{evidence}

Return only the JSON object."""

ROLE_LABEL = {"for": "supports", "against": "against", "missing": "expected but not recorded", "compare": "vs runner-up"}


def make_prompt_v2(r, d, p, pack, runner):
    lines = []
    for i in pack.values():
        tag = f" ({ROLE_LABEL[i.role]})" if i.kind == "knowledge" else ""
        lines.append(f"[{i.id}]{tag} {i.text}")
    runner_text = f"{r.db.names[runner]} ({runner})" if runner else "none"
    return PROMPT_V2.format(name=r.db.names[d], did=d, prob=f"{p:.1%}", runner=runner_text, evidence="\n".join(lines))

def make_prompt_v3(r, d, p, pack, runner):
    p_for = {i.hpo_id: i.id for i in pack.values() if i.kind == "patient"}
    lines = []
    for i in pack.values():
        if i.kind == "knowledge":
            pid = p_for.get(i.hpo_id) if i.role != "missing" else None
            tag = f" ({ROLE_LABEL[i.role]}; cite as [{pid}, {i.id}])" if pid else f" ({ROLE_LABEL[i.role]}; cite as [{i.id}])"
        else:
            tag = ""
        lines.append(f"[{i.id}]{tag} {i.text}")
    runner_text = f"{r.db.names[runner]} ({runner})" if runner else "none"
    return PROMPT_V2.format(name=r.db.names[d], did=d, prob=f"{p:.1%}", runner=runner_text, evidence="\n".join(lines))


# ---------------- checks, version 2 (Nov 11)
# V1 citations, V5 section, V2 record status (negation scoped to the finding; not recorded vs absent; family and
# uncertain), V3 names the finding, V4 typed numbers (a frequency must be a frequency, an LR an LR), V6 nothing beyond
# the evidence (certainty or causal wording, another disease, a finding no cited item covers).
SECTION_ROLES = {"supports": {"for"}, "against": {"against"}, "expected_but_missing": {"missing"},
                 "vs_runner_up": {"compare"}}
SOURCE_TYPE = {"HPO Annotations": "HPO annotation", "ranker contributions": "ranker calculation",
               "PubMed": "PubMed passage"}
NOT_RECORDED = re.compile(r"\b(?:not|never)\b(?:\W+\w+){0,3}?\W+(?:mention|record|report|document|note|describ|list)\w*"
                          r"|\bno (?:mention|record|report)\b|\bunrecorded\b|\bundocumented\b", re.I)
AFTER_ABSENT = r"(?:\W+\w+){0,3}?\W+(?:absent|not present|negative|not seen|not found|not observed|ruled out)\b"
FAMILY_WORDS = re.compile(r"\b(family|familial|mother|father|parents?|siblings?|brother|sister|aunt|uncle|cousin|"
                          r"grand\w*|relatives?|maternal|paternal)\b", re.I)
HEDGE_WORDS = re.compile(r"\b(possible|possibly|suspected|uncertain|unclear|may|might|probable|probably|questionable|"
                         r"query)\b|\?", re.I)
OVERCLAIM = re.compile(r"\b(proves?|proven|proof|confirm\w*|diagnostic|diagnoses|diagnosed|definitive\w*|definite\w*|"
                       r"certain|certainly|conclusive\w*|establish\w*|pathognomonic|rules? out|excludes|causes?|caused|"
                       r"causing|due to|because|results? (?:from|in)|leads? to|led to|always|guarantee\w*|must|"
                       r"clearly|undoubtedly)\b", re.I)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.I)
LR_BEFORE = re.compile(r"(?:\blr\b|likelihood ratio|\bratio\b|\bagainst\b|\bvs\b|\bversus\b|compared with)\W{0,3}"
                       r"(?:\w+\W+){0,3}$", re.I)
_EXT, _NAMES = {}, {}


def says_absent(body, name):
    n = re.escape(name)
    return bool(re.search(NEGATION.pattern + r"(?:\W+\w+){0,4}?\W+" + n, body, re.I)
                or re.search(n + AFTER_ABSENT, body, re.I))


def source_type(item):
    return "patient record" if item.kind == "patient" else SOURCE_TYPE.get(item.source, item.source or "knowledge base")


def quantities(items):
    text = " ".join(i.text for i in items)
    pct = set(re.findall(r"(\d+(?:\.\d+)?)%", text))
    lrs = set(re.findall(r"\bLR (\d+(?:\.\d+)?)", text))
    lrs |= {x for pair in re.findall(r"likelihood ratio (\d+(?:\.\d+)?) against (\d+(?:\.\d+)?)", text) for x in pair}
    return pct, lrs


def _extractor(r):
    if id(r) not in _EXT:
        from hoofbeats.data import Extractor
        _EXT[id(r)] = Extractor(r.ont)
    return _EXT[id(r)]


def _disease_index(r):
    # knowledge-base disease names (8+ letters), indexed by first word, to spot a disease the evidence doesn't name
    if id(r) not in _NAMES:
        idx = {}
        for name in set(r.db.names.values()):
            low = name.lower()
            if len(low) >= 8:
                idx.setdefault(re.split(r"\W+", low)[0], []).append(low)
        _NAMES[id(r)] = idx
    return _NAMES[id(r)]


def other_diseases(body, r, allowed):
    idx, found = _disease_index(r), set()
    for m in re.finditer(r"\w+", body):
        for name in idx.get(m.group(0), ()):
            if body.startswith(name, m.start()) and name not in allowed:
                end = m.start() + len(name)
                if end == len(body) or not body[end].isalnum():
                    found.add(name)
    return {x for x in found if not any(x != y and x in y for y in found | allowed)}     # longest names only


def _strip_names(text, names):
    for n in sorted(names, key=len, reverse=True):
        text = re.sub(re.escape(n), " ", text, flags=re.I)
    return text


def _residual(body, items, r, disease_names):
    # the sentence without the disease names and without the cited findings' own names and quoted phrases, so that
    # words inside a name ("... due to ...", a disease called "Hypertelorism") are not read as claims
    phrases = set(disease_names) | {q for i in items for q in re.findall(r"'([^']+)'", i.text)}
    phrases |= {n for i in items for n in names_of(r, i)}
    return _strip_names(body, phrases)


def _terms(r, text):
    return {r.ont.resolve(f.hpo_id) for f in _extractor(r).findings(text) if r.ont.resolve(f.hpo_id)}


# "falls under" is how a broader-term item (and the model's rewording of it) links a finding to its broader term;
# "falls" there is not the finding 'Falls'
BROADER_PHRASE = re.compile(r"\bfall(?:s|ing)? under\b", re.I)


def unsupported_findings(body, items, r, disease_names):
    # findings the sentence names that no cited item covers (the cited term, a term in the cited text, or a
    # broader / narrower term of one of those)
    said = _terms(r, _strip_names(BROADER_PHRASE.sub(" ", body), disease_names))
    if not said:
        return []
    allowed = {r.ont.resolve(i.hpo_id) for i in items if i.hpo_id} - {None}
    allowed |= _terms(r, _strip_names(BROADER_PHRASE.sub(" ", " ".join(i.text for i in items)), disease_names))
    out = []
    for t in said:
        if t in allowed or any(t in r.ont.ancestors(a) or a in r.ont.ancestors(t) for a in allowed):
            continue
        out.append(r.ont.name(t))
    return out


def check_sentence(s, pack, r, case, disease_names=(), section=None, strict=True):
    if not strict:
        return check_sentence_basic(s, pack, r, case, disease_names)
    ids = cites(s)
    body = strip_cites(s).lower()
    # V1: cites something, and every cited ID is in this disease's pack
    if not ids:
        return {"ok": False, "reason": "V1: no citation"}
    missing = [i for i in ids if i not in pack]
    if missing:
        return {"ok": False, "reason": f"V1: {', '.join(missing)} not in the evidence"}
    items = [pack[i] for i in ids]
    # V5: each cited knowledge item belongs to the section the sentence is in
    if section in SECTION_ROLES:
        wrong = [i for i in items if i.kind == "knowledge" and i.role not in SECTION_ROLES[section]]
        if wrong:
            return {"ok": False, "reason": f"V5: {wrong[0].id} is '{wrong[0].role}' evidence, not '{section}'"}
    # V2: patient findings as recorded; negation tied to the finding; not recorded is not absent
    status_now = {r.ont.resolve(f.hpo_id): f.status.value for f in case.findings}
    for it in items:
        names = names_of(r, it)
        if it.kind == "patient":
            if status_now.get(it.hpo_id) != it.status:
                return {"ok": False, "reason": f"V2: {it.id} is no longer in the record as {it.status}"}
            if it.status == "absent" and not any(says_absent(body, n) for n in names):
                return {"ok": False, "reason": f"V2: {it.id} is absent but the sentence does not say that finding is absent"}
            if it.status == "present" and any(negated(body, n) for n in names):
                return {"ok": False, "reason": f"V2: {it.id} is present but the sentence says it is not"}
            if it.status == "family" and not FAMILY_WORDS.search(body):
                return {"ok": False, "reason": f"V2: {it.id} is in the family history but the sentence does not say so"}
            if it.status == "uncertain" and not HEDGE_WORDS.search(body):
                return {"ok": False, "reason": f"V2: {it.id} is uncertain but the sentence states it as fact"}
        elif it.role == "missing" and not NOT_RECORDED.search(body):
            if any(says_absent(body, n) for n in names):
                return {"ok": False, "reason": f"V2: {it.id} is not recorded, but the sentence says the patient lacks it"}
            return {"ok": False, "reason": f"V2: {it.id} is not recorded, but the sentence does not say so"}
    # V3: the sentence names the finding of each cited item that has one (whole words)
    for it in items:
        names = names_of(r, it)
        if names and not any(re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", body) for n in names):
            return {"ok": False, "reason": f"V3: does not name the finding of {it.id}"}
    # V3 (broader term): a frequency matched through a broader term belongs to that term; a sentence that drops it
    # gives the patient's exact finding a frequency the evidence does not have
    for it in items:
        if it.kind == "knowledge" and it.via:
            vn = names_of(r, Item("", "", "", it.via))
            # look outside the patient's own finding name: 'cleft palate' inside 'submucous cleft palate' doesn't count
            rest = body
            for n in sorted(names_of(r, it), key=len, reverse=True):          # whole words only
                rest = re.sub(r"(?<!\w)" + re.escape(n) + r"(?!\w)", " ", rest)
            if vn and not any(re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", rest) for n in vn):
                return {"ok": False, "reason": f"V3: the frequency in {it.id} is for the broader term "
                                               f"'{r.ont.name(it.via)}', which the sentence does not name"}
    # V4: numbers, by kind: a percentage must be a frequency in the cited items, a number after LR / likelihood
    # ratio / against must be one of their LRs, any other number must appear in them
    pct, lrs = quantities(items)
    ev = " ".join(it.text for it in items)
    ev = re.sub(r"\d+(?:\.\d+)?%|\bLR \d+(?:\.\d+)?|likelihood ratio \d+(?:\.\d+)? against \d+(?:\.\d+)?", " ", ev)
    allowed = set(NUMBER.findall(_strip_names(ev, disease_names)))     # numbers that are neither a frequency nor an LR
    text = _strip_names(body, disease_names)
    pairs = {pair for it in items for pair in re.findall(r"likelihood ratio (\d+(?:\.\d+)?) against (\d+(?:\.\d+)?)", it.text)}
    for a, b in re.findall(r"(\d+(?:\.\d+)?)\s*(?:against|vs\.?|versus|compared with)\s*(\d+(?:\.\d+)?)", text):
        if (a, b) not in pairs:          # this disease's LR first, the runner-up's second
            return {"ok": False, "reason": f"V4: '{a} against {b}' is not a likelihood-ratio pair in the cited evidence"}
    for m in NUMBER.finditer(text):
        num, after, before = m.group(0), text[m.end():m.end() + 9], text[max(0, m.start() - 30):m.start()]
        if re.match(r"\s*(?:%|percent\b)", after):
            if num not in pct:
                return {"ok": False, "reason": f"V4: {num}% is not a frequency in the cited evidence"}
        elif LR_BEFORE.search(before):
            if num not in lrs:
                return {"ok": False, "reason": f"V4: {num} is not a likelihood ratio in the cited evidence"}
        elif num not in allowed:
            return {"ok": False, "reason": f"V4: {num} is not in the cited evidence"}
    # V6: nothing beyond the evidence
    rest = _residual(body, items, r, disease_names)
    w = OVERCLAIM.search(rest)
    if w:
        return {"ok": False, "reason": f"V6: claims more than the evidence shows ('{w.group(0)}')"}
    others = other_diseases(rest, r, {n.lower() for n in disease_names})
    if others:
        return {"ok": False, "reason": f"V6: names a disease that is not in the evidence ({sorted(others)[0]})"}
    extra = unsupported_findings(body, items, r, disease_names)
    if extra:
        return {"ok": False, "reason": f"V6: mentions '{extra[0]}', which no cited item covers"}
    return {"ok": True, "grade": "Evidence-linked", "sources": sorted({source_type(i) for i in items}), "reason": ""}
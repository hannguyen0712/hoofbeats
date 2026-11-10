from dataclasses import dataclass
from hoofbeats.model import verify_claim
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

def knowledge_items(r, case, d, claims, ex, start=1):
    """Make one K item for each knowledge claim."""
    items = {}
    lr_of = {c.hpo_id: c.log_lr for c in r.contributions(case, d, ex)}
    n = start
    for cl in claims:
        ok, reason = verify_claim(r, case, cl, ex)
        if not ok:
            print(" dropped", cl.text, "->", reason)
            continue
        role = "for" if cl.patient_term else "against"
        kid = f"K{n}"
        text = cl.text.split(" [")[0]
        items[kid] = Item(
            kid,
            "knowledge",
            text,
            cl.patient_term,
            cl.status.value,
            source = "HPO Annotations",
            refs = tuple(sorted(cl.refs)),
            role = role)
        n += 1
    return items

def known_terms(r, case):
    """Return every HPO term the record already covers: present findings and their ancestors, absent findings and their descendants."""
    known = set()
    for f in case.findings:
        t = r.ont.resolve(f.hpo_id)
        if t is None:
            continue
        if f.status.value == "present":
            known |= r.ont.ancestors(t)
        elif f.status.value == "absent":
            known |= r.ont.descendants(t)
    return known

def missing_items(r, case, d, ex, start, max_missing = 3):
    """Make one M item for each missing finding in the case."""
    items = {}
    name = r.db.names[d]
    known = known_terms(r, case)

    candidates = []
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
        text = f"{name} shows '{r.ont.name(t)}' in ~{freq:.0%} of cases; the patient record does not mention this finding."
        items[kid] = Item(
            kid,
            "knowledge",
            text,
            t,
            source="HPO Annotations",
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


def check_sentence(s, pack, r, case, disease_names=()):
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


def check_trace(trace, pack, r, case, disease_names=()):
    kept, removed = {s: [] for s in trace}, []
    for sec, sents in trace.items():
        for s in sents:
            res = check_sentence(s, pack, r, case, disease_names)
            if res["ok"]:
                kept[sec].append({"text": s, "grade": res["grade"]})
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

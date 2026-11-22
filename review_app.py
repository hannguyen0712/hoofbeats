"""Diagnosis support screen (Part 4).   streamlit run review_app.py

Page 1: patient details -> page 2: the clinician enters the findings -> page 3: the diagnosis lists, with the
evidence behind each suggestion -> page 4: the source document with the evidence highlighted.
"""
import csv, html, io, json, os, re, subprocess, sys, time
from pathlib import Path
from urllib.parse import quote as urlquote
import streamlit as st
import streamlit.components.v1 as components

RESULT = Path("results/part4_demo.json")
ACTIONS = Path("results/part4_actions.json")        # pick / rule out, per case
RESULTS = Path("results/part4_results.json")        # test results entered, per case
CASE_DIR = Path("results/cases")                     # case files written from the forms
EXAMPLES = {"": "(start empty)",
            "examples/marfan_note.json": "Tall teenager with lens problems (genetics clinic)",
            "examples/tsc_imaging.json": "Child with seizures and brain imaging (neurology)",
            "examples/wilson_labs.json": "Young adult with liver and movement problems (hepatology)"}
# where a piece of evidence comes from, shown on every checked sentence and in the evidence popovers
SOURCE_STYLE = {"patient record": ("Patient record", "#B45309", "#FEF3C7"),
                "HPO annotation": ("HPO annotation", "#1D4ED8", "#DBEAFE"),
                "ranker calculation": ("Ranker calculation", "#6D28D9", "#EDE9FE"),
                "PubMed passage": ("PubMed passage", "#047857", "#D1FAE5")}
OLD_GRADES = {"Verified": ["patient record", "HPO annotation"], "KB only": ["HPO annotation"],
              "Patient only": ["patient record"]}          # results saved before Nov 11


def source_chips(sources):
    return "".join(f"<span class='pill' style='color:{fg};background:{bg}'>{lab}</span>"
                   for lab, fg, bg in (SOURCE_STYLE.get(x, (x, "#334155", "#E2E8F0")) for x in sources))
SECTIONS = {"supports": "What fits", "against": "What does not fit",
            "expected_but_missing": "Usually present, not recorded", "vs_runner_up": "Why it ranks above the next disease"}
UPLOADS = Path("results/uploads")                    # images uploaded with a case
MODALITIES = ["X-ray", "CT", "MRI", "Ultrasound", "Photo", "Other"]
FORM = {"case_id": "", "age": None, "sex": "unknown", "setting": "", "example": "", "note": "", "present": [],
        "absent": [], "other_terms": [], "labs": [], "vitals": {}, "genes": "", "image_report": "", "images": [],
        "image_findings": []}
DOC_NAMES = {"note_text": "Clinical note", "image_context": "Imaging report", "structured_data": "Lab results and other entered values"}

st.set_page_config(page_title="Diagnosis support", layout="wide")
st.markdown("""<style>
.block-container{padding-top:3.6rem;max-width:1320px}
.stepper{display:flex;gap:10px;margin:2px 0 18px 0}
.step{flex:1;border:1px solid #E2E8F0;border-radius:10px;padding:8px 12px;color:#64748B;font-size:0.86rem;background:#F8FAFC}
.step b{color:#0F172A}
.step.on{border-color:#2563EB;background:#EFF6FF;color:#1E3A8A}
.step.done{border-color:#A7F3D0;background:#ECFDF5;color:#065F46}
.patient{border:1px solid #E2E8F0;border-radius:12px;padding:12px 16px;background:#F8FAFC;margin-bottom:10px}
.chip{display:inline-block;border-radius:999px;padding:2px 10px;margin:3px 4px 0 0;font-size:0.8rem;border:1px solid #CBD5E1;background:#fff}
.chip.no{border-color:#FECACA;background:#FEF2F2;color:#991B1B}
.rank{display:inline-block;width:26px;height:26px;border-radius:50%;background:#0F172A;color:#fff;text-align:center;
      line-height:26px;font-weight:700;margin-right:8px;font-size:0.85rem}
.dname{font-size:1.08rem;font-weight:700;color:#0F172A}
.pct{float:right;font-weight:800;font-size:1.05rem;color:#1E3A8A}
.bar{height:10px;background:#E2E8F0;border-radius:6px;overflow:hidden;margin:8px 0 2px 0}
.bar > div{height:100%;background:#2563EB}
.muted{color:#64748B;font-size:0.84rem}
.pill{display:inline-block;border-radius:999px;padding:1px 9px;font-size:0.76rem;font-weight:700;margin-right:6px}
.rejected{opacity:0.45}
.doc{font-family:ui-monospace,Menlo,monospace;font-size:0.84rem;line-height:1.6;white-space:pre-wrap;
     background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;color:#1E293B}
.doc mark{background:#FDE68A;color:#1E293B;padding:0 2px;border-radius:3px}
.doc mark.cue{background:#FECACA}
.doc .dim{color:#94A3B8}
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("page", "patient")
ss.setdefault("form", dict(FORM))
for _k, _v in FORM.items():
    ss.form.setdefault(_k, _v)
for _k in ("rep_v", "img_v", "lab_v"):    # bumped after an upload is read, to clear that upload box
    ss.setdefault(_k, 0)


def go(page):
    ss.page = page
    st.rerun()


def load_json(p, default):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


@st.cache_data
def hpo_terms():
    # HPO term names, for the symptom pickers
    names, cur = {}, {}
    for line in Path("data/kb/hp.obo").read_text().splitlines():
        if line == "[Term]":
            if cur.get("id") and cur.get("name") and not cur.get("obsolete"):
                names[cur["name"]] = cur["id"]
            cur = {}
        elif line.startswith("id: HP:"):
            cur["id"] = line[4:]
        elif line.startswith("name: "):
            cur["name"] = line[6:]
        elif line.startswith("is_obsolete: true"):
            cur["obsolete"] = True
    if cur.get("id") and cur.get("name") and not cur.get("obsolete"):
        names[cur["name"]] = cur["id"]
    names.pop("All", None)
    return names


@st.cache_data
def lab_names():
    rows = [l.split("\t") for l in Path("hoofbeats/resources/lab_reference.tsv").read_text().splitlines()
            if l and not l.startswith("#")]
    return sorted(r[0] for r in rows[1:])


def save_upload(up, case_id):
    folder = UPLOADS / (case_id or "unsaved")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / re.sub(r"[^A-Za-z0-9._-]+", "_", up.name)
    path.write_bytes(up.getvalue())
    return str(path.resolve())


def guess_modality(name):
    n = name.lower()
    for key, m in (("xr", "X-ray"), ("x-ray", "X-ray"), ("xray", "X-ray"), ("ct", "CT"), ("mr", "MRI"), ("mri", "MRI"),
                   ("us", "Ultrasound"), ("echo", "Ultrasound"), ("photo", "Photo")):
        if re.search(rf"(^|[^a-z]){re.escape(key)}([^a-z]|$)", n):
            return m
    return "Other"


def labs_from_csv(data):
    # a lab results file -> rows for the lab table. Columns (any case): test/name, value/result, unit,
    # ref_low + ref_high or "normal from"/"normal to", or one "reference range" column such as 20-60
    def num(x):
        try:
            return float(str(x).strip())
        except (TypeError, ValueError):
            return None
    norm = lambda x: re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()
    known = {}
    for n in lab_names():                       # "Alanine Aminotransferase (ALT)" also matches "ALT"
        for alias in (n, re.sub(r"\s*\([^)]*\)", "", n), *re.findall(r"\(([^)]+)\)", n)):
            known.setdefault(norm(alias), n)
    rows, unknown = [], []
    for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig", errors="replace"))):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        get = lambda *ks: next((row[k] for k in ks if row.get(k)), "")
        name, value = get("test", "name", "analyte", "lab"), num(get("value", "result"))
        if not name or value is None:
            continue
        lo, hi = num(get("ref_low", "normal from", "low")), num(get("ref_high", "normal to", "high"))
        m = re.match(r"\s*([\d.]+)\s*[-–]\s*([\d.]+)", get("reference range", "range", "ref", "normal range"))
        if m and (lo is None or hi is None):
            lo, hi = float(m[1]), float(m[2])
        if norm(name) not in known:
            unknown.append(name)
            continue
        rows.append({"name": known[norm(name)], "value": value, "unit": get("unit", "units") or None,
                     "ref_low": lo, "ref_high": hi})
    return rows, unknown


def stepper(active):
    order = ["patient", "clinical", "results"]
    labels = [("1. Patient", "Who is the patient"), ("2. Findings", "Notes, symptoms, images, labs"),
              ("3. Diagnosis", "Likely diseases, evidence, next test")]
    idx = order.index(active) if active in order else 2
    st.markdown("<div class='stepper'>" + "".join(
        f"<div class='step {'on' if i == idx else 'done' if i < idx else ''}'><b>{a}</b><br>{b}</div>"
        for i, (a, b) in enumerate(labels)) + "</div>", unsafe_allow_html=True)


def run(case, drop=(), with_results=True):
    # run the pipeline (saved model replies only; a new question falls back to the template explanation)
    cmd = [sys.executable, "demo_part4.py", str(case)] + (["--drop", *drop] if drop else [])
    if with_results and load_json(RESULTS, {}).get(str(case)):
        Path("results/part4_results_now.json").write_text(json.dumps(load_json(RESULTS, {})[str(case)]))
        cmd += ["--results", "results/part4_results_now.json"]
    with st.spinner("Reading the findings and ranking the diseases..."):
        res = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "LLM_OFFLINE": "1", "VISION_OFFLINE": "0"})
    if res.returncode:
        st.error(res.stderr[-2000:])
        return False
    return True


# ====================================================================== page 1: patient
def page_patient():
    stepper("patient")
    st.title("New patient")
    f = ss.form
    with st.container(border=True):
        st.markdown("**Start from an example** (optional)")
        ex = st.selectbox("Example patient", list(EXAMPLES), format_func=lambda k: EXAMPLES[k],
                          index=list(EXAMPLES).index(f["example"]) if f["example"] in EXAMPLES else 0,
                          label_visibility="collapsed")
        if ex != f["example"]:
            f["example"] = ex
            if ex:
                c = load_json(ex, {})
                by_id = {v: k for k, v in hpo_terms().items()}
                terms = c.get("hpo_terms", [])
                pick = lambda st_: [by_id[t["id"]] for t in terms if t.get("status", "present") == st_ and t["id"] in by_id]
                f.update(case_id=c.get("case_id", ""), age=c.get("age"), sex=c.get("sex") or "unknown",
                         note=c.get("note_text", ""), labs=c.get("labs", []), vitals=c.get("vitals") or {},
                         genes=", ".join(c.get("genes", [])), present=pick("present"), absent=pick("absent"),
                         other_terms=[t for t in terms if t.get("status", "present") not in ("present", "absent")
                                      or t["id"] not in by_id],
                         image_report=c.get("image_context", ""), image_findings=c.get("image_findings", []),
                         images=[{"path": str((Path(ex).parent / i["path"]).resolve()), "modality": i.get("modality") or "Other"}
                                 for i in c.get("images", [])])
            st.rerun()
    with st.form("patient"):
        c1, c2, c3 = st.columns(3)
        case_id = c1.text_input("Patient ID (no names)", value=f["case_id"], placeholder="e.g. clinic-0042")
        age = c2.number_input("Age (years)", min_value=0, max_value=120, value=int(f["age"]) if f["age"] else None,
                              placeholder="e.g. 17")
        sexes = ["unknown", "female", "male", "other"]
        sex = c3.selectbox("Sex", sexes, index=sexes.index(f["sex"]) if f["sex"] in sexes else 0)
        setting = st.text_input("Clinic or setting (optional)", value=f["setting"], placeholder="e.g. genetics clinic")
        if st.form_submit_button("Next: enter the findings →", type="primary"):
            if not case_id.strip():
                st.error("Please enter a patient ID.")
            else:
                f.update(case_id=re.sub(r"[^A-Za-z0-9_-]+", "-", case_id.strip()), age=age, sex=sex, setting=setting)
                go("clinical")
    st.caption("Decision support only: the system helps you review a case; it does not replace clinical judgement "
               "and gives no treatment advice. Data entered here stays on this computer.")


# ====================================================================== page 2: findings
def page_clinical():
    stepper("clinical")
    f = ss.form
    who = " ".join(x for x in (f"{f['age']}-year-old" if f["age"] else "", f["sex"] if f["sex"] != "unknown" else "") if x)
    st.title("Findings")
    st.caption(f"Patient {f['case_id']}" + (f" · {who}" if who else "") + (f" · {f['setting']}" if f["setting"] else ""))
    names = hpo_terms()

    with st.container(border=True):
        st.markdown("**Clinical note**")
        note = st.text_area("Clinical note", value=f["note"], height=200, label_visibility="collapsed",
                            placeholder="History, examination and anything else, in your own words. "
                                        "Negations such as 'no chest pain' are understood.")
        c1, c2 = st.columns(2)
        present = c1.multiselect("Symptoms or signs present (type to search)", list(names), default=f["present"])
        absent = c2.multiselect("Checked and absent", list(names), default=f["absent"])
        if f["other_terms"]:
            st.caption(f"Also kept from the example: {len(f['other_terms'])} finding(s) marked as family history or uncertain.")

    with st.container(border=True):
        st.markdown("**Imaging** (optional)")
        i1, i2 = st.columns([3, 2])
        with i1:
            report = st.text_area("Imaging or radiology report", value=f["image_report"], height=150,
                                  key=f"report_{ss.rep_v}",
                                  placeholder="Paste the report, e.g. 'MRI brain: multiple cortical tubers ...'")
            txt = st.file_uploader("...or upload the report as a text file", type=["txt", "md"], key=f"rep_up_{ss.rep_v}")
            if txt is not None:
                f["image_report"] = txt.getvalue().decode("utf-8-sig", errors="replace")
                ss.rep_v += 1
                st.rerun()
        with i2:
            ups = st.file_uploader("Images (X-ray, CT, MRI, ultrasound, photos)", type=["png", "jpg", "jpeg", "webp"],
                                   accept_multiple_files=True, key=f"img_up_{ss.img_v}")
            if ups:
                have = {x["path"] for x in f["images"]}
                for up in ups:
                    path = save_upload(up, f["case_id"])
                    if path not in have:
                        f["images"].append({"path": path, "modality": guess_modality(up.name)})
                ss.img_v += 1
                st.rerun()
        if f["images"]:
            cols = st.columns(4)
            for k, im in enumerate(list(f["images"])):
                with cols[k % 4]:
                    if Path(im["path"]).exists():
                        st.image(im["path"], use_container_width=True)
                    st.caption(Path(im["path"]).name)
                    m = st.selectbox("Type", MODALITIES, key=f"mod-{im['path']}",
                                     index=MODALITIES.index(im["modality"]) if im["modality"] in MODALITIES else 5)
                    im["modality"] = m
                    if st.button("Remove", key=f"rm-{im['path']}", use_container_width=True):
                        f["images"].remove(im)
                        st.rerun()
            st.caption("Images are described by a local image model when one is running (on Sol); the description is "
                       "then read like the report above. On this computer they are kept with the case but not described.")

    with st.container(border=True):
        st.markdown("**Lab results** (optional)")
        labs = st.data_editor(f["labs"] or [{"name": None, "value": None, "unit": None, "ref_low": None, "ref_high": None}],
                              num_rows="dynamic", use_container_width=True, key=f"labs_editor_{ss.lab_v}",
                              column_config={"name": st.column_config.SelectboxColumn("Test", options=lab_names()),
                                             "value": st.column_config.NumberColumn("Value"),
                                             "unit": st.column_config.TextColumn("Unit"),
                                             "ref_low": st.column_config.NumberColumn("Normal from"),
                                             "ref_high": st.column_config.NumberColumn("Normal to")})
        lab_csv = st.file_uploader("...or upload a lab results file (.csv with columns test, value, unit, reference range)",
                                   type=["csv"], key=f"lab_up_{ss.lab_v}")
        if lab_csv is not None:
            rows, unknown = labs_from_csv(lab_csv.getvalue())
            f["labs"] = [r for r in labs if r.get("name")] + rows
            ss.lab_msg = (f"Added {len(rows)} lab result(s) from {lab_csv.name}."
                          + (f" Not recognised, so not added: {', '.join(unknown)}." if unknown else ""))
            ss.lab_v += 1
            st.rerun()
        if ss.get("lab_msg"):
            st.caption(ss.lab_msg)

    with st.container(border=True):
        st.markdown("**Vital signs and genes** (optional)")
        v = f["vitals"] or {}
        v1, v2, v3, v4, v5 = st.columns(5)
        bp = v1.text_input("Blood pressure", value=v.get("bp") or "", placeholder="120/80")
        temp = v2.number_input("Temperature °C", value=v.get("temp_c"), placeholder="37.0")
        hr = v3.number_input("Heart rate", value=v.get("hr"), placeholder="72")
        rr = v4.number_input("Breathing rate", value=v.get("rr"), placeholder="16")
        spo2 = v5.number_input("SpO2 %", value=v.get("spo2"), placeholder="98")
        genes = st.text_input("Known gene results", value=f["genes"], placeholder="e.g. FBN1")

    b1, b2 = st.columns([1, 4])
    back = b1.button("← Back")
    submit = b2.button("Evaluate →", type="primary")
    if back:
        f.update(note=note, present=present, absent=absent, image_report=report, genes=genes)
        go("patient")
    if submit:
        if not (note.strip() or report.strip() or present or absent or f["images"] or any(r.get("name") for r in labs)):
            st.error("Please enter a note, an imaging report, some findings, an image or a lab result.")
            return
        f.update(note=note, present=present, absent=absent, genes=genes, image_report=report,
                 labs=[r for r in labs if r.get("name") and r.get("value") is not None],
                 vitals={k: x for k, x in {"bp": bp.strip() or None, "temp_c": temp, "hr": hr, "rr": rr,
                                           "spo2": spo2}.items() if x not in (None, "")})
        case = {"case_id": f["case_id"], "note_text": note, "image_context": report.strip(), "age": f["age"],
                "sex": f["sex"],
                "hpo_terms": [{"id": names[n], "status": "present"} for n in present] +
                             [{"id": names[n], "status": "absent"} for n in absent] + f["other_terms"],
                "image_findings": f["image_findings"],
                "images": [{"path": x["path"], "modality": x["modality"]} for x in f["images"]],
                "labs": [{k: x for k, x in r.items() if x not in (None, "")} for r in f["labs"]],
                "genes": [g.strip().upper() for g in genes.split(",") if g.strip()],
                "details": {"setting": f["setting"]} if f["setting"] else {}, "policy": "local_only"}
        if f["vitals"]:
            case["vitals"] = f["vitals"]
        if f["example"]:
            case["truth"] = load_json(f["example"], {}).get("truth", [])
        CASE_DIR.mkdir(parents=True, exist_ok=True)
        path = CASE_DIR / f"{f['case_id']}.json"
        path.write_text(json.dumps({k: x for k, x in case.items() if x not in (None, "", [], {})}, indent=1))
        if run(path):
            go("results")


# ====================================================================== page 3: diagnosis
def plain(sentence):
    s = re.sub(r"\s*\[[^\]]*\b[PK]\d+\b[^\]]*\]?\s*$", "", sentence).strip()
    s = re.sub(r"\.?\s*LR [\d.]+\.?$", ".", s)
    s = re.sub(r"^Has '([^']+)';", r"The patient has \1;", s)
    return s.rstrip(".") + "."


def highlight(text, spans, start=0, end=None, target=None):
    end = len(text) if end is None else end
    marks = []
    for s in spans:
        marks.append((s["start"], s["end"], "hit" if target is not None and s["start"] == target else ""))
        if s.get("cue"):
            marks.append((s["cue"]["start"], s["cue"]["end"], "cue"))
    marks = sorted(m for m in marks if m[0] >= start and m[1] <= end)
    out, pos = [], start
    for a, b, cls in marks:
        if a < pos:
            continue
        out.append(html.escape(text[pos:a]))
        anchor = " id='hit'" if cls == "hit" else ""
        out.append(f"<mark class='{cls}'{anchor}>{html.escape(text[a:b])}</mark>")
        pos = b
    out.append(html.escape(text[pos:end]))
    return "".join(out)


def literature_link(ref, finding):
    # PubMed page; the #:~:text= part asks the browser to highlight the finding's name if the abstract mentions it
    pmid = ref.split(":")[-1]
    frag = f"#:~:text={urlquote(finding)}" if finding else ""
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/{frag}"


def evidence_links(ids, pack, key):
    # one line per piece of evidence: patient evidence opens the source document, literature opens PubMed
    for cid in ids:
        item = pack.get(cid, {})
        if item.get("kind") == "patient":
            st.markdown(f"**From the patient record:** {html.escape(plain(item.get('text', '')))}", unsafe_allow_html=True)
            for k, sp in enumerate(item.get("spans", [])[:3]):
                label = f"📄 {DOC_NAMES.get(sp['field'], sp['field'])}: “{sp['quote'][:60]}”"
                if st.button(label, key=f"{key}-{cid}-{k}", use_container_width=True):
                    name = re.search(r"'([^']+)' is (\w+)", item.get("text", ""))
                    ss.evidence = {"field": sp["field"], "start": sp["start"],
                                   "title": f"{name.group(1)} ({name.group(2)})" if name else plain(item.get("text", ""))}
                    go("evidence")
            if not item.get("spans"):
                st.caption("Entered directly as a finding.")
        elif item.get("source") == "ranker contributions":
            st.markdown(f"**Calculated by the ranker** from the HPO annotations of both diseases: "
                        f"{html.escape(plain(item.get('text', '')))}", unsafe_allow_html=True)
        else:
            src = item.get("source") or ""
            where = ("**From a retrieved PubMed passage:**" if src == "PubMed" else
                     "**From the HPO annotations** (disease features curated from the papers listed):")
            st.markdown(f"{where} {html.escape(plain(item.get('text', '')))}", unsafe_allow_html=True)
            name = re.search(r"'([^']+)'", item.get("text", ""))
            for ref in [r for r in item.get("refs", []) if r.startswith("PMID")][:4]:
                st.link_button(f"📚 Open {ref} on PubMed", literature_link(ref, name.group(1) if name else ""),
                               use_container_width=True)


def page_results():
    stepper("results")
    data = load_json(RESULT, None)
    if not data:
        st.info("No results yet.")
        if st.button("Start"):
            go("patient")
        return
    case_key = data["case_path"]
    all_actions = load_json(ACTIONS, {})
    actions = all_actions.get(case_key, {"picked": None, "rejected": []})
    entered = load_json(RESULTS, {}).get(case_key, [])
    shown = {d["id"]: d for lst in data["lists"].values() for d in lst}
    fu = data.get("followup")
    confirmed_ids = (fu or {}).get("confirmed", [])

    def save_actions():
        all_actions[case_key] = actions
        ACTIONS.write_text(json.dumps(all_actions, indent=1))

    top1, top2 = st.columns([5, 1])
    top1.title("Diagnosis")
    if top2.button("← Edit findings", use_container_width=True):
        go("clinical")

    pt = data.get("patient") or {}
    who = " ".join(x for x in (f"{pt['age']}-year-old" if pt.get("age") else "", pt.get("sex") if pt.get("sex") not in (None, "unknown") else "") if x)
    present = [f for f in data["findings"] if f["status"] == "present"]
    absent = [f for f in data["findings"] if f["status"] == "absent"]
    chips = "".join(f"<span class='chip'>{html.escape(f['name'])}</span>" for f in present) + \
        "".join(f"<span class='chip no'>no {html.escape(f['name'].lower())}</span>" for f in absent)
    st.markdown(f"<div class='patient'><b>Patient {html.escape(data['case'])}</b> <span class='muted'>"
                f"{('· ' + who) if who else ''} · {len(present)} findings present, {len(absent)} absent</span><br>{chips}</div>",
                unsafe_allow_html=True)
    if data.get("images"):
        with st.expander(f"Images ({len(data['images'])})", expanded=False):
            if data.get("images_described_by"):
                st.caption(f"Described by {data['images_described_by']}; the description is part of the imaging report.")
            elif data.get("images_note"):
                st.warning(data["images_note"])
            cols = st.columns(4)
            for k, im in enumerate(data["images"]):
                with cols[k % 4]:
                    if Path(im["path"]).exists():
                        st.image(im["path"], use_container_width=True)
                    st.caption(f"{im['modality']} · {im['name']}")
    with st.expander(f"Check what was read from the record ({len(data['findings'])} findings)"
                     + (f" · {len(data['dropped'])} removed by you" if data["dropped"] else "")):
        st.caption("If something was read wrongly, remove it and the diagnosis updates.")
        cols = st.columns(2)
        for k, f in enumerate(data["findings"]):
            with cols[k % 2]:
                a, b = st.columns([5, 1])
                mark = {"present": "✅", "absent": "⛔"}.get(f["status"], "👪" if f["status"] == "family" else "•")
                label = {"present": "", "absent": "absent", "family": "in the family"}.get(f["status"], f["status"])
                words = f"“{f['words']}”" if f["words"] and f["words"].lower() != f["name"].lower() else ""
                a.markdown(f"{mark} {f['name']} <span class='muted'>{' · '.join(x for x in (label, words) if x)}</span>",
                           unsafe_allow_html=True)
                if b.button("Remove", key=f"drop-{f['hpo_id']}"):
                    if run(case_key, data["dropped"] + [f["hpo_id"]]):
                        st.rerun()
        if data["dropped"] and st.button("Restore all findings"):
            if run(case_key):
                st.rerun()

    if fu:
        lines = [f"{shown[d]['name'] if d in shown else d}: "
                 f"{('%.0f%%' % (100 * fu['before'][d])) if fu['before'].get(d) is not None else 'new'} → {p:.0%}"
                 for d, p in fu["after"].items()]
        conf = [shown[d]["name"] for d in confirmed_ids if d in shown]
        st.success(("**Confirmed by genetic test: " + ", ".join(conf) + ".**  \n" if conf else
                    "**Updated with your test results.**  \n") + " · ".join(lines))

    def card(n, d):
        pack = {i["id"]: i for i in d["pack"]}
        picked, rejected = actions["picked"] == d["id"], d["id"] in actions["rejected"]
        conf, same = d["id"] in confirmed_ids, d["id"] in (fu or {}).get("same_gene", [])
        badge = ("<span class='pill' style='color:#fff;background:#047857'>Confirmed</span>" if conf else
                 "<span class='pill' style='color:#047857;background:#D1FAE5'>Your pick</span>" if picked else
                 "<span class='pill' style='color:#64748B;background:#E2E8F0'>Ruled out</span>" if rejected else
                 "<span class='pill' style='color:#64748B;background:#E2E8F0'>Same gene, fits less well</span>" if same else "")
        with st.container(border=True):
            st.markdown(f"<div class='{'rejected' if rejected else ''}'><span class='pct'>{d['probability']:.0%}</span>"
                        f"<span class='rank'>{n}</span><span class='dname'>{html.escape(d['name'])}</span> {badge}"
                        f"<div class='bar'><div style='width:{max(2, d['probability'] * 100):.0f}%'></div></div></div>",
                        unsafe_allow_html=True)
            if rejected:
                if st.button("Undo rule-out", key=f"undo-{d['id']}"):
                    actions["rejected"].remove(d["id"]); save_actions(); st.rerun()
                return
            st.markdown("<span class='muted'>Why it fits: click a finding to see its evidence</span>", unsafe_allow_html=True)
            for k, x in enumerate(d["fits"][:3]):
                freq = ("" if x["freq"] is None else
                        f" · {x['freq']}% of people with it have '{x['via']}' (the broader HPO term)" if x.get("via") else
                        f" · in {x['freq']}% of people with it")
                with st.popover(f"{x['finding']}{freq} · {x['strength']} evidence", use_container_width=True):
                    evidence_links(x["ids"], pack, f"{d['id']}-fit{k}")
            if d.get("against"):
                st.markdown("<span class='muted'>Counts against: " + ", ".join(
                    (f"no {html.escape(x['finding'].lower())} (usually present, {x['freq']}%)" if x["absent"] and x["freq"]
                     else html.escape(x["finding"].lower())) for x in d["against"][:3]) + "</span>", unsafe_allow_html=True)
            if d["missing"]:
                st.markdown("<span class='muted'>Usually seen but not recorded: " +
                            ", ".join(f"{html.escape(x['finding'].lower())} ({x['freq']}%)" for x in d["missing"][:3]) +
                            "</span>", unsafe_allow_html=True)
            b1, b2, b3 = st.columns([1.3, 1, 2])
            if picked:
                if b1.button("Unpick", key=f"unpick-{d['id']}", use_container_width=True):
                    actions["picked"] = None; save_actions(); st.rerun()
            elif b1.button("Pick as most likely", key=f"pick-{d['id']}", type="primary", use_container_width=True):
                actions["picked"] = d["id"]; save_actions(); st.rerun()
            if b2.button("Rule out", key=f"rej-{d['id']}", use_container_width=True):
                actions["rejected"].append(d["id"])
                if picked:
                    actions["picked"] = None
                save_actions(); st.rerun()
            omim = d["id"].split(":")[-1] if d["id"].startswith("OMIM") else None
            b3.markdown("<span class='muted'>Learn more and management: " +
                        (f"<a href='https://omim.org/entry/{omim}' target='_blank'>OMIM</a> · " if omim else "") +
                        f"<a href='https://www.ncbi.nlm.nih.gov/books/NBK1116/?term={d['name'].replace(' ', '+')}' "
                        f"target='_blank'>GeneReviews</a></span>", unsafe_allow_html=True)
            with st.expander("Full explanation"):
                src = "written by a local language model" if d["explanation_source"] == "model" else "written from the evidence"
                st.caption(f"Every statement was checked against its evidence ({src}); {len(d['removed'])} failed and "
                           "are hidden. \"Evidence-linked\" means each statement cites evidence that names the same finding "
                           "and numbers; the tags say where that evidence comes from.")
                for sec, title in SECTIONS.items():
                    sents = d["explanation"].get(sec, [])
                    if not sents:
                        continue
                    st.markdown(f"**{title}**")
                    for j, x in enumerate(sents):
                        s = x["text"]
                        srcs = x.get("sources") or OLD_GRADES.get(x["grade"], [])
                        a, b = st.columns([6, 1])
                        a.markdown(f"{html.escape(plain(s))}<br><span class='muted'>Evidence-linked · </span>"
                                   f"{source_chips(srcs)}", unsafe_allow_html=True)
                        with b.popover("Evidence"):
                            evidence_links([c.strip() for c in s[s.rfind("[") + 1:s.rfind("]")].split(",")], pack,
                                           f"{d['id']}-{sec}{j}")
                if d["removed"]:
                    with st.expander(f"Hidden by the checks ({len(d['removed'])})"):
                        for x in d["removed"]:
                            st.caption(f"{x['text']}  →  {x['reason']}")

    def test_line(x):
        return (f"**{x['test']}**  \n<span class='muted'>if positive → {x['post_test_if_positive']:.0%} · "
                f"if negative → {x['post_test_if_negative']:.1%} · effort {x['burden']}/5</span>")

    left, right = st.columns([3, 2], gap="large")
    with left:
        has_common = bool(data["lists"]["common"])
        tab_rare, tab_common = st.tabs(["Rare diseases", "Common diseases" + ("" if has_common else " (not loaded)")])
        with tab_rare:
            for n, d in enumerate(data["lists"]["rare"], 1):
                card(n, d)
        with tab_common:
            if not has_common:
                st.info("Common-disease profiles are not loaded yet, so this list is empty.")
            for n, d in enumerate(data["lists"]["common"], 1):
                card(n, d)
    with right:
        st.subheader("Next step")
        picked = shown.get(actions["picked"])
        with st.container(border=True):
            if confirmed_ids and any(d in shown for d in confirmed_ids):
                st.markdown(f"**{shown[confirmed_ids[0]]['name']}** is confirmed by genetic testing. No further test is "
                            "needed to confirm it.")
            elif picked:
                st.markdown(f"Tests to confirm or rule out **{picked['name']}**:")
                t = picked.get("tests") or {}
                for x in (([t["gene_test"]] if t.get("gene_test") else []) + t.get("tests", []))[:3]:
                    st.markdown(test_line(x), unsafe_allow_html=True)
                for x in t.get("review", [])[:2]:
                    st.markdown(f"Review the existing **{x['test']}** <span class='muted'>(already done)</span>",
                                unsafe_allow_html=True)
            else:
                st.markdown("No disease picked yet. These tests best tell the top diseases apart:")
                for x in data.get("separate", [])[:3]:
                    parts = " · ".join(f"{shown[d]['name'].split(',')[0]} {p:.0%}" for d, p in x["p_pos"].items() if d in shown)
                    st.markdown(f"**{x['test']}**  \n<span class='muted'>chance of a positive result: {parts} · "
                                f"effort {x['burden']}/5</span>", unsafe_allow_html=True)
                st.caption("Pick a disease on the left to see the tests that confirm it.")
        if data.get("can_enter_results"):
            with st.form("result", clear_on_submit=True):
                st.markdown("**Enter a test result**")
                ordered = ([picked] if picked else []) + [d for d in shown.values() if d is not picked]
                options = list(dict.fromkeys(
                    [x["test"] for d in ordered for x in (([d["tests"]["gene_test"]] if (d.get("tests") or {}).get("gene_test")
                                                          else []) + (d.get("tests") or {}).get("tests", []))]
                    + [x["test"] for x in data.get("separate", [])]))
                test = st.selectbox("Test", options)
                res = st.radio("Result", ["positive", "negative"], horizontal=True)
                det = st.text_input("Details (optional)", placeholder="e.g. pathogenic variant c.4082G>A")
                if st.form_submit_button("Update the diagnosis", type="primary", use_container_width=True):
                    entry = {"test": test, "result": res, "details": det}
                    if "gene" in test.lower() and res == "positive":
                        entry["genes"] = re.findall(r"\b([A-Z][A-Z0-9]{2,})\b", test)[:1]
                        low = det.lower()
                        entry["classification"] = "likely pathogenic" if "likely pathogenic" in low else \
                            "pathogenic" if "pathogenic" in low else "uncertain"
                    allr = load_json(RESULTS, {})
                    allr.setdefault(case_key, []).append(entry)
                    RESULTS.write_text(json.dumps(allr, indent=1))
                    if run(case_key, data["dropped"]):
                        st.rerun()
            if entered:
                st.markdown("**Results entered**")
                for x in entered:
                    st.markdown(f"- {x['test']}: {x['result']}" + (f" ({x['details']})" if x.get("details") else ""))
                if st.button("Clear results", use_container_width=True):
                    allr = load_json(RESULTS, {})
                    allr.pop(case_key, None)
                    RESULTS.write_text(json.dumps(allr, indent=1))
                    if run(case_key, data["dropped"], with_results=False):
                        st.rerun()
        st.divider()
        if st.button("Start a new patient", use_container_width=True):
            ss.form = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
                       for k, v in FORM.items()}
            ss.pop("lab_msg", None)
            go("patient")


# ====================================================================== page 4: evidence in its source document
def page_evidence():
    data = load_json(RESULT, {})
    ev = ss.get("evidence")
    if st.button("← Back to the diagnosis"):
        go("results")
    if not ev:
        st.info("Choose a piece of evidence on the diagnosis page.")
        return
    text = data["source"]["texts"].get(ev["field"], "")
    st.title(DOC_NAMES.get(ev["field"], "Source document"))
    st.markdown(f"Finding: **{html.escape(ev['title'])}**. The highlighted words are where it was read from.")
    spans = [sp for lst in data["lists"].values() for d in lst for i in d["pack"] if i.get("kind") == "patient"
             for sp in i.get("spans", []) if sp["field"] == ev["field"] and sp["start"] == ev["start"]][:1]
    if ev["field"] == "image_context" and data.get("images"):
        cols = st.columns(6)
        for k, im in enumerate(data["images"]):
            if Path(im["path"]).exists():
                cols[k % 6].image(im["path"], caption=im["modality"], use_container_width=True)
    body = highlight(text, spans, target=ev["start"])
    components.html(f"""
      <style>
        body{{margin:0;font-family:ui-monospace,Menlo,monospace}}
        .doc{{font-size:14px;line-height:1.7;white-space:pre-wrap;background:#F8FAFC;border:1px solid #E2E8F0;
              border-radius:8px;padding:14px 16px;color:#1E293B}}
        mark{{background:#FDE68A;padding:1px 3px;border-radius:3px}}
        mark.hit{{background:#FACC15;outline:2px solid #CA8A04}}
        mark.cue{{background:#FECACA}}
      </style>
      <div class='doc'>{body}</div>
      <script>var h=document.getElementById('hit'); if(h) h.scrollIntoView({{block:'center'}});</script>
    """, height=520, scrolling=True)
    st.caption("Yellow: the words the system read this finding from · red: a word saying the finding is absent.")


{"patient": page_patient, "clinical": page_clinical, "results": page_results,
 "evidence": page_evidence}.get(ss.page, page_patient)()

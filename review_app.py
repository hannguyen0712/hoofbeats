"""Diagnosis support screen (Part 4).   streamlit run review_app.py

Page 1: patient details -> page 2: the clinician enters the findings -> page 3: the diagnosis lists, with the
evidence behind each suggestion -> page 4: the source document with the evidence highlighted.
"""
import csv, html, io, json, os, re, shutil, subprocess, sys, time, uuid
from pathlib import Path
from urllib.parse import quote as urlquote
import streamlit as st
import streamlit.components.v1 as components

SESSIONS = Path("results/sessions")                  # one private folder per browser session
SESSION_TTL_HOURS = float(os.environ.get("SESSION_TTL_HOURS", "12"))   # idle sessions older than this are deleted
EXAMPLES = {"": "(start empty)",
            "examples/marfan_note.json": "Genetics: tall teenager with lens problems",
            "examples/tsc_imaging.json": "Neurology: child with seizures and brain MRI",
            "examples/wilson_labs.json": "Hepatology: young adult with liver problems"}
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
DOTS = {"very strong": "●●●", "strong": "●●○", "moderate": "●○○", "weak": "○○○"}
MODALITIES = ["X-ray", "CT", "MRI", "Ultrasound", "Photo", "Other"]
FORM = {"case_id": "", "age": None, "sex": "unknown", "setting": "", "example": "", "note": "", "present": [],
        "absent": [], "other_terms": [], "labs": [], "vitals": {}, "genes": "", "image_report": "", "images": [],
        "image_findings": []}
DOC_NAMES = {"note_text": "Clinical note", "image_context": "Imaging report", "structured_data": "Lab results and other entered values"}

st.set_page_config(page_title="Hoofbeats · diagnosis support", page_icon="🦓", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&display=swap');
:root{--ink:#1B1D33;--hema:#3B3486;--hema-wash:#EEEDF8;--eosin:#A93A63;--eosin-wash:#F8EAF0;--ok:#1C6B4F;
      --ok-wash:#E3F2EB;--line:#E3E4EC;--muted:#5F6380;--paper:#FFFFFF;--panel:#F6F6FA}
html, body, p, li, label, input, textarea, button, h1, h2, h3, h4,
[data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"], [data-baseweb="tab"] {
  font-family:'Atkinson Hyperlegible', system-ui, -apple-system, 'Segoe UI', sans-serif !important}
.block-container{padding-top:2.2rem;max-width:1280px}
h1{font-weight:700;letter-spacing:-0.01em;color:var(--ink)}
.brand{display:flex;align-items:center;gap:10px;margin:0 0 6px 0}
.brand b{font-size:1.2rem;color:var(--ink)}
.brand span{color:var(--muted);font-size:0.9rem;margin-left:4px}
.mark{width:20px;height:20px;border-radius:5px;flex:none;
      background:repeating-linear-gradient(135deg,var(--ink) 0 3px,var(--paper) 3px 6px);border:1px solid var(--ink)}
.stepper{display:flex;margin:0 0 22px 0;border-bottom:1px solid var(--line)}
.step{padding:10px 0 9px 0;margin-right:30px;color:var(--muted);font-size:0.92rem;border-bottom:3px solid transparent;
      margin-bottom:-1px}
.step b{font-weight:700}
.step.on{color:var(--ink);border-bottom-color:var(--hema)}
.step.done{color:var(--ok)}
.pt{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.pt h2{font-size:1.6rem;margin:0;padding:0;color:var(--ink);font-weight:700}
.chips{margin:2px 0 10px 0}
.chip{display:inline-block;border-radius:6px;padding:2px 9px;margin:4px 5px 0 0;font-size:0.84rem;
      background:var(--panel);border:1px solid var(--line);color:var(--ink)}
.chip.no{background:var(--eosin-wash);border-color:transparent;color:var(--eosin)}
.chip.more{background:transparent;border-style:dashed;color:var(--muted)}
[class*="st-key-hero"]{position:relative;padding-left:30px !important;overflow:hidden}
[class*="st-key-hero"]::before{content:'';position:absolute;left:0;top:0;bottom:0;width:12px;
      background:repeating-linear-gradient(135deg,var(--hema) 0 4px,var(--paper) 4px 8px)}
.hero .lbl{color:var(--muted);font-size:0.9rem}
.hero .name{font-size:1.65rem;font-weight:700;color:var(--ink);line-height:1.2;margin:2px 0 6px 0}
.hero .num{float:right;text-align:right;margin:0 0 6px 18px;max-width:40%}
.hero .num b{display:block;font-size:2.3rem;line-height:1;color:var(--hema);font-variant-numeric:tabular-nums}
.hero .num span{color:var(--muted);font-size:0.8rem}
.bar{height:8px;background:var(--hema-wash);border-radius:4px;overflow:hidden;margin:10px 0 10px 0;clear:both}
.bar > div{height:100%;background:var(--hema)}
.ground{display:inline-block;border-radius:6px;padding:3px 10px;font-size:0.82rem;font-weight:700;color:var(--ok);
        background:var(--ok-wash);margin:0 12px 6px 0}
.ratio{color:var(--hema);font-size:0.88rem;font-weight:700}
.sub{font-weight:700;color:var(--ink);margin:12px 0 4px 0}
.sub .muted{font-weight:400}
.sect{font-weight:700;color:var(--ink);font-size:1.05rem;margin:22px 0 6px 2px}
.lit{border-left:3px solid var(--hema);padding:4px 0 4px 10px;margin:6px 0 4px 0;font-size:0.92rem;line-height:1.5}
.lit mark{background:#FDE68A;color:var(--ink);padding:0 2px;border-radius:3px}
.how{color:var(--muted);font-size:0.82rem;line-height:1.5;margin:20px 0 0 2px}
.rail{font-weight:700;color:var(--ink);font-size:1.15rem;margin:6px 0 8px 2px}
.notes{color:var(--muted);font-size:0.88rem;line-height:1.55;margin:8px 0 10px 0}
.notes b{color:var(--ink);font-weight:700}
.cmp{line-height:1.6;margin:4px 0 8px 0}
.muted{color:var(--muted);font-size:0.86rem}
.pill{display:inline-block;border-radius:6px;padding:2px 9px;font-size:0.78rem;font-weight:700;margin:2px 6px 4px 0}
.pill.ok{color:var(--paper);background:var(--ok)}
.pill.pick{color:var(--hema);background:var(--hema-wash)}
.pill.out{color:var(--muted);background:var(--panel);border:1px solid var(--line)}
.rejected{opacity:0.45}
.doc{font-family:ui-monospace,Menlo,monospace;font-size:0.84rem;line-height:1.6;white-space:pre-wrap;
     background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;color:var(--ink)}
.doc mark{background:#FDE68A;color:var(--ink);padding:0 2px;border-radius:3px}
.doc mark.cue{background:var(--eosin-wash)}
.doc .dim{color:#94A3B8}
[data-testid='stAppDeployButton']{display:none}
[data-testid='stPopoverButton'] > div{width:100%;display:flex;justify-content:space-between;align-items:center;gap:8px;text-align:left}
[data-testid='stPopoverButton'] > div > div:first-child{flex:1;display:flex;justify-content:flex-start}
[data-testid='stPopoverButton'] [data-testid='stMarkdownContainer'] p{text-align:left}
</style>""", unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("page", "patient")
ss.setdefault("form", dict(FORM))
for _k, _v in FORM.items():
    ss.form.setdefault(_k, _v)
for _k in ("rep_v", "img_v", "lab_v"):    # bumped after an upload is read, to clear that upload box
    ss.setdefault(_k, 0)


def clean_old_sessions(keep):
    # delete session folders with no file changed in the last SESSION_TTL_HOURS
    if not SESSIONS.exists():
        return
    cutoff = time.time() - SESSION_TTL_HOURS * 3600
    for d in SESSIONS.iterdir():
        if not d.is_dir() or d.name == keep:
            continue
        try:
            newest = max((p.stat().st_mtime for p in d.rglob("*")), default=d.stat().st_mtime)
            if newest < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


if "sid" not in ss:
    ss.sid = uuid.uuid4().hex                 # random, never shown or put in the URL
    clean_old_sessions(ss.sid)
WORK = SESSIONS / ss.sid
WORK.mkdir(parents=True, exist_ok=True)
RESULT = WORK / "part4_demo.json"
ACTIONS = WORK / "part4_actions.json"        # pick / rule out, per case
RESULTS = WORK / "part4_results.json"        # test results entered, per case
RESULTS_NOW = WORK / "part4_results_now.json"
CASE_DIR = WORK / "cases"                     # case files written from the forms
UPLOADS = WORK / "uploads"                    # images uploaded with a case


def pct(p):
    # a disease's percentage for the screen: under 0.5% shows as <1%, not 0%
    return "<1%" if 0 < p < 0.005 else f"{p:.0%}"


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
    labels = [("1", "Patient"), ("2", "Findings"), ("3", "Diagnosis")]
    idx = order.index(active) if active in order else 2
    st.markdown("<div class='brand'><span class='mark'></span><b>Hoofbeats</b><span>Rare-disease diagnosis "
                "support, with every explanation checked against its evidence</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='stepper'>" + "".join(
        f"<div class='step {'on' if i == idx else 'done' if i < idx else ''}'>{'✓' if i < idx else a}&nbsp; <b>{b}</b></div>"
        for i, (a, b) in enumerate(labels)) + "</div>", unsafe_allow_html=True)


def run(case, drop=(), with_results=True):
    # run the pipeline (saved model replies only; a new question falls back to the template explanation)
    cmd = [sys.executable, "demo_part4.py", str(case), "--out", str(RESULT)] + (["--drop", *drop] if drop else [])
    ss.last_case = str(case)
    if with_results and load_json(RESULTS, {}).get(str(case)):
        RESULTS_NOW.write_text(json.dumps(load_json(RESULTS, {})[str(case)]))
        cmd += ["--results", str(RESULTS_NOW)]
    with st.spinner("Reading the findings and ranking the diseases..."):
        res = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "LLM_OFFLINE": "1", "VISION_OFFLINE": "0"})
    ss.stale = bool(res.returncode)
    if res.returncode:
        st.error("Something went wrong while analysing this record, so the diagnosis was not updated. "
                 "Check the findings and try again.")
        with st.expander("Technical details"):
            st.code(res.stderr[-2000:])
        return False
    return True


# ====================================================================== page 1: patient
def page_patient():
    stepper("patient")
    st.title("New patient")
    st.markdown("<span class='muted'>Enter who the patient is and what you found. Hoofbeats lists the likely "
                "diseases, the evidence for each and the test to do next.</span>", unsafe_allow_html=True)
    f = ss.form
    st.markdown("<div class='sub'>Try an example</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for col, (ex, label) in zip(cols, [(k, v) for k, v in EXAMPLES.items() if k]):
        if col.button(label, key=f"ex-{ex}", use_container_width=True):
            f["example"] = ex
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
            go("clinical")
    st.markdown("<div class='sub'>Or enter a new patient</div>", unsafe_allow_html=True)
    with st.form("patient"):
        c1, c2, c3 = st.columns(3)
        case_id = c1.text_input("Patient ID (no names)", value=f["case_id"], placeholder="e.g. clinic-0042")
        age = c2.number_input("Age (years)", min_value=0, max_value=120, value=int(f["age"]) if f["age"] is not None else None,
                              placeholder="e.g. 17")
        sexes = ["unknown", "female", "male", "other"]
        sex = c3.selectbox("Sex", sexes, index=sexes.index(f["sex"]) if f["sex"] in sexes else 0)
        setting = st.text_input("Clinic or setting (optional)", value=f["setting"], placeholder="e.g. genetics clinic")
        if st.form_submit_button("Next: enter the findings", type="primary"):
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
    who = " ".join(x for x in (f"{f['age']}-year-old" if f["age"] is not None else "", f["sex"] if f["sex"] != "unknown" else "") if x)
    st.title("Findings")
    st.caption(f"Patient {f['case_id']}" + (f" · {who}" if who else "") + (f" · {f['setting']}" if f["setting"] else ""))
    names = hpo_terms()

    with st.container(border=True):
        st.markdown("**Clinical note** <span class='muted'>· write as you normally would; you can check and correct what was read on the next page</span>", unsafe_allow_html=True)
        note = st.text_area("Clinical note", value=f["note"], height=200, label_visibility="collapsed",
                            placeholder="History, examination and anything else, in your own words. "
                                        "Negations such as 'no chest pain' are understood.")
        c1, c2 = st.columns(2)
        present = c1.multiselect("Symptoms or signs present", list(names), default=f["present"], placeholder="Type to search")
        absent = c2.multiselect("Checked and absent", list(names), default=f["absent"], placeholder="Type to search")
        if f["other_terms"]:
            st.caption(f"Also kept from the example: {len(f['other_terms'])} finding(s) marked as family history or uncertain.")

    with st.expander("Imaging (optional)" + (" · entered" if f['image_report'] or f['images'] else ""), expanded=bool(f['image_report'] or f['images'])):
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

    with st.expander("Lab results (optional)" + (" · entered" if f['labs'] else ""), expanded=bool(f['labs'])):
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

    with st.expander("Vital signs and genes (optional)" + (" · entered" if f['vitals'] or f['genes'] else ""), expanded=bool(f['vitals'] or f['genes'])):
        v = f["vitals"] or {}
        v1, v2, v3, v4, v5 = st.columns(5)
        bp = v1.text_input("Blood pressure", value=v.get("bp") or "", placeholder="120/80")
        temp = v2.number_input("Temperature °C", value=v.get("temp_c"), placeholder="37.0")
        hr = v3.number_input("Heart rate", value=v.get("hr"), placeholder="72")
        rr = v4.number_input("Breathing rate", value=v.get("rr"), placeholder="16")
        spo2 = v5.number_input("SpO2 %", value=v.get("spo2"), placeholder="98")
        genes = st.text_input("Known gene results", value=f["genes"], placeholder="e.g. FBN1")

    b2, b1, _gap = st.columns([1.1, 0.5, 5])
    back = b1.button("Back")
    submit = b2.button("Find likely diagnoses", type="primary")
    if back:
        f.update(note=note, present=present, absent=absent, image_report=report, genes=genes)
        go("patient")
    if submit:
        if not (note.strip() or report.strip() or present or absent or f["images"] or any(r.get("name") for r in labs)):
            st.error("Please enter a note, an imaging report, some findings, an image or a lab result.")
            return
        both = sorted(set(present) & set(absent))
        if both:
            st.error("Marked both present and absent: " + ", ".join(both) + ". Keep each finding in one list.")
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
            lit = item.get("lit") or []
            if lit:
                st.markdown("**Where the cited papers report it** (abstracts and open-access full texts)")
            for k, h in enumerate(lit):
                s, parts, pos = h["sentence"], [], 0
                for a, b in sorted(h["marks"]):
                    parts += [html.escape(s[pos:a]), "<mark>", html.escape(s[a:b]), "</mark>"]
                    pos = b
                parts.append(html.escape(s[pos:]))
                year = f", {h['year']}" if h.get("year") else ""
                where = h.get("where") or "abstract"
                st.markdown(f"<div class='lit'>{''.join(parts)}<br><span class='muted'>{h['pmid']}{year}, {where}: "
                            f"{html.escape(h['title'][:120])}</span></div>", unsafe_allow_html=True)
                a, b = h["marks"][0]
                if h.get("pmc") and where not in ("title", "abstract"):
                    st.link_button(f"📖 Open the full text ({h['pmc']}) with these words highlighted",
                                   f"https://pmc.ncbi.nlm.nih.gov/articles/{h['pmc']}/#:~:text={urlquote(s[a:b])}",
                                   use_container_width=True)
                else:
                    st.link_button(f"📚 Open {h['pmid']} on PubMed with these words highlighted",
                                   literature_link(h["pmid"], s[a:b]), use_container_width=True)
            if "lit" in item and not lit and item.get("lit_papers"):
                st.caption("The cited papers do not name this finding in their abstract or open-access full text; it "
                           "may be in a table or in a paper that is not open access.")
            done = {h["pmid"] for h in lit}
            for ref in [r for r in item.get("refs", []) if r.startswith("PMID") and r not in done][:max(0, 4 - len(lit))]:
                flag = " (retracted or withdrawn in PubMed)" if ref in (item.get("lit_retracted") or []) else ""
                st.link_button(f"📚 Open {ref} on PubMed{flag}", literature_link(ref, name.group(1) if name else ""),
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
    if ss.get("last_case") and ss.last_case != case_key:
        st.warning("The saved results belong to another case, so they are not shown. Run the analysis again "
                   "for this patient.")
        if st.button("Run again for this patient", type="primary"):
            if run(ss.last_case):
                st.rerun()
        return
    all_actions = load_json(ACTIONS, {})
    actions = all_actions.get(case_key, {"picked": None, "rejected": []})
    entered = load_json(RESULTS, {}).get(case_key, [])
    shown = {d["id"]: d for lst in data["lists"].values() for d in lst}
    fu = data.get("followup")
    confirmed_ids = (fu or {}).get("confirmed", [])

    def save_actions():
        all_actions[case_key] = actions
        ACTIONS.write_text(json.dumps(all_actions, indent=1))

    pt = data.get("patient") or {}
    who = ", ".join(x for x in (f"{pt['age']}-year-old" if pt.get("age") is not None else "",
                                pt.get("sex") if pt.get("sex") not in (None, "unknown") else "") if x)
    who = who.replace("-year-old, ", "-year-old ")
    present = [f for f in data["findings"] if f["status"] == "present"]
    absent = [f for f in data["findings"] if f["status"] == "absent"]

    h1, h2, h3 = st.columns([5, 1.5, 1.1], vertical_alignment="center")
    h1.markdown(f"<div class='pt'><h2>{html.escape(who.capitalize() or 'Patient')}</h2>"
                f"<span class='muted'>Patient {html.escape(data['case'])}: {len(present)} finding{'' if len(present) == 1 else 's'} present, "
                f"{len(absent)} checked and absent</span></div>", unsafe_allow_html=True)
    with h2.popover("How to read this page", use_container_width=True):
        st.markdown("- **The %** is the model's estimate for each disease, given only the findings entered, "
                    "calibrated on held-out rare-disease cases. It ranks the diseases; it is not a clinical "
                    "probability. Look-alike diseases share the chance, so compare the diseases with each other.\n"
                    "- **●●● ●●○ ●○○** show how strongly a finding points to the disease "
                    "(likelihood ratio of at least 100, 10 or 3).\n"
                    "- **✓ statements checked**: every sentence of the explanation was checked against the evidence "
                    "it cites. Sentences that failed are hidden and listed under *Hidden by the checks*.\n"
                    "- **Click a finding** to see its evidence and where it was read in the record.\n"
                    "- **📄** marks a finding that a cited paper reports for this disease (abstract or open-access full text); "
                    "the sentence is "
                    "shown with the words highlighted. It shows the paper reports it, not that the diagnosis is right.\n"
                    "- **Pick as most likely** and **Rule out** record your judgement. **Enter a test result** "
                    "updates the list.\n"
                    "- Decision support only: it does not replace clinical judgement.")
    if h3.button("Edit findings", use_container_width=True):
        go("clinical")

    shown_chips = present[:10]
    chips = "".join(f"<span class='chip'>{html.escape(f['name'])}</span>" for f in shown_chips) + \
        "".join(f"<span class='chip no'>no {html.escape(f['name'].lower())}</span>" for f in absent[:4])
    more = len(present) - len(shown_chips) + max(0, len(absent) - 4)
    if more:
        chips += f"<span class='chip more'>{more} more</span>"
    st.markdown(f"<div class='chips'>{chips}</div>", unsafe_allow_html=True)
    if ss.get("stale"):
        st.warning("The last update failed, so these results are from before it. Fix the problem and run it again.")
    if len(present) < 3:
        st.warning(f"Insufficient evidence: only {len(present)} finding{'' if len(present) == 1 else 's'} present. "
                   "The list below is a weak starting point, not a differential; add more findings.")

    e1, e2 = st.columns([1, 1]) if data.get("images") else (st.container(), None)
    with e1.expander(f"Check and correct what was read ({len(data['findings'])} finding{'' if len(data['findings']) == 1 else 's'}"
                     + (f", {len(data['dropped'])} removed by you)" if data["dropped"] else ")")):
        st.caption("If something was read wrongly, remove it and the list of diseases updates.")
        for f in data["findings"]:
            a, b = st.columns([7, 1], vertical_alignment="center")
            mark = {"present": "✓", "absent": "✕"}.get(f["status"], "♢")
            label = {"present": "", "absent": "absent", "family": "in the family"}.get(f["status"], f["status"])
            words = f"“{f['words']}”" if f["words"] and f["words"].lower() != f["name"].lower() else ""
            a.markdown(f"{mark} {html.escape(f['name'])} <span class='muted'>{', '.join(x for x in (label, words) if x)}"
                       "</span>", unsafe_allow_html=True)
            if b.button("Remove", key=f"drop-{f['hpo_id']}", use_container_width=True):
                if run(case_key, data["dropped"] + [f["hpo_id"]]):
                    st.rerun()
        if data["dropped"] and st.button("Restore all findings"):
            if run(case_key):
                st.rerun()
    if e2 is not None:
        with e2.expander(f"Images ({len(data['images'])})"):
            if data.get("images_described_by"):
                st.caption(f"Described by {data['images_described_by']}; the description is part of the imaging report.")
            elif data.get("images_note"):
                st.warning(data["images_note"])
            cols = st.columns(3)
            for k, im in enumerate(data["images"]):
                with cols[k % 3]:
                    if Path(im["path"]).exists():
                        st.image(im["path"], use_container_width=True)
                    st.caption(f"{im['modality']}: {im['name']}")

    if fu:
        lines = [f"{shown[d]['name'] if d in shown else d}: "
                 f"{pct(fu['before'][d]) if fu['before'].get(d) is not None else 'new'} to {pct(p)}"
                 for d, p in fu["after"].items()]
        conf = [shown[d]["name"] for d in confirmed_ids if d in shown]
        st.success(("**Molecular evidence reported for " + ", ".join(conf) + ".** It needs clinical review (the "
                    "report and the findings) before the diagnosis is confirmed.  \n" if conf else
                    "**Updated with your test results.**  \n") + "; ".join(lines))
        for w in [x.get("warning") for x in fu.get("results") or [] if x.get("warning")]:
            st.warning(w[0].upper() + w[1:] + ".")

    def boxed(key):
        try:
            return st.container(border=True, key=key)
        except TypeError:
            return st.container(border=True)

    def short(name, n=48):
        return name if len(name) <= n else name[:n - 1].rstrip(" ,") + "…"

    def status(d):
        if d["id"] in confirmed_ids:
            return "<span class='pill ok'>Molecular evidence reported: needs clinical review</span>"
        if actions["picked"] == d["id"]:
            return "<span class='pill pick'>Your pick</span>"
        if d["id"] in actions["rejected"]:
            return "<span class='pill out'>Ruled out</span>"
        if d["id"] in (fu or {}).get("same_gene", []):
            return "<span class='pill out'>Same gene, fits less well</span>"
        return ""

    def details(d, nxt=None, first=False):
        pack = {i["id"]: i for i in d["pack"]}
        picked, rejected = actions["picked"] == d["id"], d["id"] in actions["rejected"]
        if rejected:
            if st.button("Undo rule-out", key=f"undo-{d['id']}"):
                actions["rejected"].remove(d["id"]); save_actions(); st.rerun()
            return
        n_kept = sum(len(v) for v in d["explanation"].values())
        ground = (f"<span class='ground'>✓ {n_kept} statement{'' if n_kept == 1 else 's'} checked against the evidence"
                  + (f", {len(d['removed'])} hidden" if d["removed"] else "") + "</span>")
        ratio = (f"<span class='ratio'>{d['probability'] / nxt:.1f} times as likely as the next disease</span>"
                 if first and nxt and d["probability"] >= nxt else "")
        st.markdown(ground + ratio, unsafe_allow_html=True)
        st.markdown("<div class='sub'>Why it fits <span class='muted'>(click a finding to see its evidence)</span></div>",
                    unsafe_allow_html=True)
        for k, x in enumerate(d["fits"][:3]):
            freq = ("" if x["freq"] is None else
                    f", {x['freq']}% have '{x['via']}' (broader term)" if x.get("via") else
                    f", in {x['freq']}% of patients")
            paper = "  📄" if any(pack.get(c, {}).get("lit") for c in x["ids"]) else ""
            with st.popover(f"{DOTS.get(x['strength'], '')}  {x['finding']}{freq}{paper}", use_container_width=True):
                st.caption(f"{x['strength'].capitalize()} evidence")
                evidence_links(x["ids"], pack, f"{d['id']}-fit{k}")
        notes = []
        if d.get("against"):
            notes.append("<b>Counts against:</b> " + ", ".join(
                (f"no {html.escape(x['finding'].lower())} (usually present, {x['freq']}%)" if x["absent"] and x["freq"]
                 else html.escape(x["finding"].lower())) for x in d["against"][:3]))
        if d["missing"]:
            notes.append("<b>Usually seen but not recorded:</b> " +
                         ", ".join(f"{html.escape(x['finding'].lower())} ({x['freq']}%)" for x in d["missing"][:3]))
        if notes:
            st.markdown("<div class='notes'>" + "<br>".join(notes) + "</div>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns([1.3, 1, 1.6], vertical_alignment="center")
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
        b3.markdown("<span class='muted'>Read more: " +
                    (f"<a href='https://omim.org/entry/{omim}' target='_blank'>OMIM</a>, " if omim else "") +
                    f"<a href='https://www.ncbi.nlm.nih.gov/books/NBK1116/?term={d['name'].replace(' ', '+')}' "
                    f"target='_blank'>GeneReviews</a></span>", unsafe_allow_html=True)
        with st.expander("Full explanation"):
            src = "written by a local language model" if d["explanation_source"] == "model" else "written from the evidence"
            k_rm = len(d["removed"])
            st.caption(f"Every statement was checked against its evidence ({src}); {k_rm} failed and "
                       f"{'is' if k_rm == 1 else 'are'} hidden. The tags say where the evidence comes from.")
            for sec, title in SECTIONS.items():
                sents = d["explanation"].get(sec, [])
                if not sents:
                    continue
                st.markdown(f"<div class='sub'>{title}</div>", unsafe_allow_html=True)
                for j, x in enumerate(sents):
                    s = x["text"]
                    srcs = x.get("sources") or OLD_GRADES.get(x["grade"], [])
                    a, b = st.columns([5, 1])
                    a.markdown(f"{html.escape(plain(s))}<br>{source_chips(srcs)}", unsafe_allow_html=True)
                    with b.popover("Evidence"):
                        evidence_links([c.strip() for c in s[s.rfind("[") + 1:s.rfind("]")].split(",")], pack,
                                       f"{d['id']}-{sec}{j}")
            if d["removed"]:
                with st.expander(f"Hidden by the checks ({k_rm})"):
                    for x in d["removed"]:
                        st.caption(f"{x['text']}  ({x['reason']})")

    def compare(a, b):
        with st.expander(f"Compare with number 2, {short(b['name'], 40)}"):
            c1, c2 = st.columns(2)
            for col, d in ((c1, a), (c2, b)):
                rows = [f"{DOTS.get(x['strength'], '')} {html.escape(x['finding'])}"
                        + (f" <span class='muted'>({x['freq']}%)</span>" if x["freq"] is not None else "")
                        for x in d["fits"][:4]]
                miss = ", ".join(html.escape(x["finding"].lower()) for x in d["missing"][:2])
                col.markdown(f"<b>{html.escape(d['name'])}</b>, {pct(d['probability'])}"
                             "<div class='cmp'>" + "<br>".join(rows) + "</div>"
                             + (f"<span class='muted'>Usually seen but not recorded: {miss}</span>" if miss else ""),
                             unsafe_allow_html=True)
            why = a["explanation"].get("vs_runner_up", [])
            if why:
                st.markdown(f"<div class='sub'>Checked statements comparing it with "
                            f"{html.escape(a['runner_up_name'])}</div>", unsafe_allow_html=True)
                for x in why:
                    st.markdown(f"- {html.escape(plain(x['text']))}")

    def disease_list(kind):
        lst = data["lists"][kind]
        n_all = data.get("n_rare" if kind == "rare" else "n_common")
        pool = f"{n_all:,} {kind} diseases" if n_all else f"all {kind} diseases in the knowledge base"
        scale = (f"model estimate among {pool}" if kind == "rare" and data.get("calibrated", True) else
                 f"model score among {pool}, not calibrated")
        top = lst[0]
        nxt = lst[1]["probability"] if len(lst) > 1 else None
        with boxed(f"hero_{kind}"):
            st.markdown(f"<div class='hero {'rejected' if top['id'] in actions['rejected'] else ''}'>"
                        f"<div class='num'><b>{pct(top['probability'])}</b><span>{scale}</span></div>"
                        f"<div class='lbl'>Most likely</div>"
                        f"<div class='name'>{html.escape(top['name'])}</div>{status(top)}"
                        f"<div class='bar'><div style='width:{max(2, top['probability'] * 100):.0f}%'></div></div></div>",
                        unsafe_allow_html=True)
            details(top, nxt, first=True)
            if len(lst) > 1 and top["id"] not in actions["rejected"]:
                compare(top, lst[1])
        if len(lst) > 1:
            st.markdown("<div class='sect'>Other possibilities</div>", unsafe_allow_html=True)
            for n, d in enumerate(lst[1:], 2):
                tag = {"<span class='pill pick'>Your pick</span>": "  (your pick)",
                       "<span class='pill out'>Ruled out</span>": "  (ruled out)"}.get(status(d), "")
                with st.expander(f"{n}.  {short(d['name'], 60)}  **{pct(d['probability'])}**{tag}"):
                    st.markdown(status(d) + f"<div class='bar'><div style='width:{max(2, d['probability'] * 100):.0f}%'>"
                                "</div></div>", unsafe_allow_html=True)
                    details(d)

    def test_line(x):
        return (f"**{x['test']}**  \n<span class='muted'>If positive: {x['post_test_if_positive']:.0%}. "
                f"If negative: {x['post_test_if_negative']:.1%}. Effort {x['burden']} of 5.</span>")

    def summary_text():
        out = ["Hoofbeats case summary (decision support only, not a diagnosis)", "",
               f"Patient {data['case']}" + (f", {who}" if who else ""),
               "Findings present: " + (", ".join(x["name"] for x in present) or "none")]
        if absent:
            out.append("Checked and absent: " + ", ".join(x["name"] for x in absent))
        for kind in ("rare", "common"):
            if not data["lists"][kind]:
                continue
            out += ["", f"Most likely {kind} diseases ("
                    + ("model estimate, calibrated on held-out rare-disease cases" if kind == "rare" and data.get("calibrated", True)
                       else "model score, not calibrated") + "):"]
            for n, d in enumerate(data["lists"][kind], 1):
                tag = (" [your pick]" if actions["picked"] == d["id"] else
                       " [ruled out]" if d["id"] in actions["rejected"] else "")
                out.append(f"{n}. {d['name']}: {pct(d['probability'])}{tag}")
                if d["fits"]:
                    out.append("   Fits: " + "; ".join(x["finding"] + (f" ({x['freq']}% of people with it)"
                               if x["freq"] is not None else "") for x in d["fits"][:3]))
                if d["missing"]:
                    out.append("   Usually seen but not recorded: " + "; ".join(x["finding"] for x in d["missing"][:3]))
        picked_d = shown.get(actions["picked"])
        if picked_d:
            t = picked_d.get("tests") or {}
            tests = [x["test"] for x in ([t["gene_test"]] if t.get("gene_test") else []) + t.get("tests", [])][:3]
            out += ["", f"Suggested tests for {picked_d['name']}: " + ("; ".join(tests) or "none")]
        elif data.get("separate"):
            out += ["", "Tests that best tell the top diseases apart: " + "; ".join(x["test"] for x in data["separate"][:3])]
        if entered:
            out += ["", "Test results entered: " + "; ".join(f"{x['test']}: {x['result']}" for x in entered)]
        hidden = sum(len(d["removed"]) for d in shown.values())
        out += ["", f"Every explanation statement was checked against the evidence it cites; {hidden} that failed "
                "the checks were hidden.", time.strftime("Created %Y-%m-%d %H:%M")]
        return "\n".join(out) + "\n"

    def pipeline_text():
        pl = data.get("pipeline") or {}
        srcs = [d["explanation_source"] for d in shown.values()]
        expl = ("saved local-model replies" if srcs and all(s == "model" for s in srcs) else
                "templates written from the evidence" if not any(s == "model" for s in srcs) else
                "saved local-model replies, and templates where none was saved")
        parts = [f"findings read by {pl.get('reader', 'rules and number parsing')}", f"explanations from {expl}",
                 "rare-disease estimates calibrated on held-out cases" if data.get("calibrated", True) else
                 "scores not calibrated"]
        if data.get("images"):
            parts.append(f"images described by {data['images_described_by']}" if data.get("images_described_by")
                         else "images not described")
        lit = data.get("literature")
        if lit:
            parts.append(f"paper highlights from {lit['papers']} PubMed abstracts and {lit.get('full_text', 0)} "
                         "open-access full texts" + (" (NCBI not reachable: saved copies only)" if lit.get("note") else ""))
        if pl.get("kb"):
            parts.append(pl["kb"])
        return "How this was produced: " + "; ".join(parts) + "."

    left, right = st.columns([3, 2], gap="large")
    with left:
        has_common = bool(data["lists"]["common"])
        loaded = data.get("common_loaded", has_common)
        tab_rare, tab_common = st.tabs(["Rare diseases", "Common diseases" + ("" if loaded else " (not loaded)")])
        with tab_rare:
            if data["lists"]["rare"]:
                disease_list("rare")
            else:
                st.info("No rare disease matches these findings. Add more findings and try again.")
        with tab_common:
            if not has_common:
                st.info("No common disease matches this record. Common-disease profiles are learned from lab values, "
                        "so a record without lab results has nothing to match." if loaded else
                        "Common-disease profiles are not loaded yet, so this list is empty.")
            else:
                disease_list("common")
        st.markdown(f"<div class='how'>{html.escape(pipeline_text())}</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='rail'>Next step</div>", unsafe_allow_html=True)
        picked = shown.get(actions["picked"])
        with st.container(border=True):
            if confirmed_ids and any(d in shown for d in confirmed_ids):
                gene_rows = [x for x in entered if x.get("classification") in ("pathogenic", "likely pathogenic")]
                genes_txt = ", ".join(dict.fromkeys(g for x in gene_rows for g in x.get("genes") or [])) or "a gene of this disease"
                cls_txt = " or ".join(dict.fromkeys(x["classification"] for x in gene_rows)) or "disease-causing"
                st.markdown(f"A {cls_txt} variant was reported in **{genes_txt}**. This supports "
                            f"**{shown[confirmed_ids[0]]['name']}**, but the ranking cannot confirm a diagnosis.")
                st.caption("Check the report (variant, zygosity, inheritance) against the findings, then record your "
                           "decision with Pick as most likely.")
            elif picked:
                st.markdown(f"Tests to confirm or rule out **{picked['name']}**")
                t = picked.get("tests") or {}
                for x in (([t["gene_test"]] if t.get("gene_test") else []) + t.get("tests", []))[:3]:
                    st.markdown(test_line(x), unsafe_allow_html=True)
                for x in t.get("review", [])[:2]:
                    st.markdown(f"Review the existing **{x['test']}** <span class='muted'>(already done)</span>",
                                unsafe_allow_html=True)
            else:
                st.markdown("Tests that best tell the top diseases apart")
                for x in data.get("separate", [])[:3]:
                    parts = ", ".join(f"{short(shown[d]['name'].split(',')[0], 28)} {pct(p)}"
                                      for d, p in x["p_pos"].items() if d in shown)
                    st.markdown(f"**{x['test']}**  \n<span class='muted'>Positive in: {parts}. "
                                f"Effort {x['burden']} of 5.</span>", unsafe_allow_html=True)
                st.caption("Pick a disease to see the tests that confirm it.")
        if data.get("can_enter_results"):
            with st.form("result", clear_on_submit=True):
                st.markdown("**Enter a test result**")
                ordered = ([picked] if picked else []) + [d for d in shown.values() if d is not picked]
                options = list(dict.fromkeys(
                    [x["test"] for d in ordered for x in (([d["tests"]["gene_test"]] if (d.get("tests") or {}).get("gene_test")
                                                          else []) + (d.get("tests") or {}).get("tests", []))]
                    + [x["test"] for x in data.get("separate", [])]))
                test = st.selectbox("Test", options, index=None, placeholder="Choose a test")
                res = st.radio("Result", ["positive", "negative"], index=None, horizontal=True)
                cls = st.selectbox("Variant classification (gene tests only)",
                                   ["not stated", "pathogenic", "likely pathogenic", "uncertain significance (VUS)",
                                    "likely benign", "benign"])
                det = st.text_input("Details (optional)", placeholder="e.g. c.4082G>A, heterozygous")
                if st.form_submit_button("Update the diagnosis", type="primary", use_container_width=True):
                    if not test or not res:
                        st.error("Choose a test and its result first; nothing was changed.")
                    else:
                        entry = {"test": test, "result": res, "details": det}
                        if "gene" in test.lower() and res == "positive":
                            entry["genes"] = re.findall(r"\b([A-Z][A-Z0-9]{2,})\b", test)[:1]
                            entry["classification"] = {"not stated": "uncertain",
                                                       "uncertain significance (VUS)": "uncertain"}.get(cls, cls)
                        allr = load_json(RESULTS, {})
                        allr.setdefault(case_key, []).append(entry)
                        RESULTS.write_text(json.dumps(allr, indent=1))
                        if run(case_key, data["dropped"]):
                            st.rerun()
            if entered:
                st.markdown("**Results entered**")
                for x in entered:
                    extra = ", ".join(v for v in (x.get("classification") if x.get("genes") else "", x.get("details")) if v)
                    st.markdown(f"- {x['test']}: {x['result']}" + (f" ({extra})" if extra else ""))
                if st.button("Clear results", use_container_width=True):
                    allr = load_json(RESULTS, {})
                    allr.pop(case_key, None)
                    RESULTS.write_text(json.dumps(allr, indent=1))
                    if run(case_key, data["dropped"], with_results=False):
                        st.rerun()
        st.download_button("Download a summary for the chart", summary_text(), use_container_width=True,
                           file_name=f"hoofbeats_{data['case']}.txt", mime="text/plain")
        if st.button("Start a new patient", use_container_width=True):
            ss.form = {k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v)
                       for k, v in FORM.items()}
            ss.pop("lab_msg", None)
            go("patient")


# ====================================================================== page 4: evidence in its source document
def page_evidence():
    data = load_json(RESULT, {})
    ev = ss.get("evidence")
    if st.button("Back to the diagnosis"):
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
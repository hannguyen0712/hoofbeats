"""Clinician review screen for Part 4.   streamlit run review_app.py"""
import html, json, os, subprocess, sys
from pathlib import Path
import streamlit as st

RESULT = Path("results/part4_demo.json")
CASES = ["examples/marfan_note.json"] + sorted(str(p) for p in Path("examples").glob("*.json")
                                              if p.name != "marfan_note.json" and "intended" not in p.name) + \
        ["data/phenopacket-store/0.1.27/FBN1/PMID_12203992_Patient_B1.json"]
TITLES = {"supports": "Supports", "against": "Counts against",
          "expected_but_missing": "Expected but not recorded", "vs_runner_up": "Why above the runner-up"}
BADGE = {"Verified": ":green-badge[Verified]", "KB only": ":blue-badge[KB only]",
         "Patient only": ":orange-badge[Patient only]"}

st.set_page_config(page_title="Clinician review", layout="wide")
st.markdown("""<style>
.doc{font-family:ui-monospace,Menlo,monospace;font-size:0.82rem;line-height:1.55;white-space:pre-wrap;
     background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 12px;color:#1E293B}
.doc mark{background:#FDE68A;color:#1E293B;padding:0 2px;border-radius:3px}
.doc mark.cue{background:#FECACA}
.doc .dim{color:#94A3B8}
</style>""", unsafe_allow_html=True)


def run(case, drop=()):
    # re-run the pipeline (saved model replies only; a new question falls back to the template explanation)
    cmd = [sys.executable, "demo_part4.py", case] + (["--drop", *drop] if drop else [])
    with st.spinner("Re-scoring and re-explaining..."):
        res = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "LLM_OFFLINE": "1"})
    if res.returncode:
        st.error(res.stderr[-2000:])


if not RESULT.exists():
    run(CASES[0])
data = json.loads(RESULT.read_text())
texts = data["source"]["texts"]


def highlight(text, spans, start=0, end=None):
    end = len(text) if end is None else end
    marks = []
    for s in spans:
        marks.append((s["start"], s["end"], ""))
        if s.get("cue"):
            marks.append((s["cue"]["start"], s["cue"]["end"], "cue"))
    marks = sorted(m for m in marks if m[0] >= start and m[1] <= end)
    out, pos = [], start
    for a, b, cls in marks:
        if a < pos:
            continue
        out.append(html.escape(text[pos:a]))
        out.append(f"<mark class='{cls}'>{html.escape(text[a:b])}</mark>")
        pos = b
    out.append(html.escape(text[pos:end]))
    return "".join(out)


def glimpse(span, width=110):
    text = texts[span["field"]]
    a, b = max(0, span["start"] - width), min(len(text), span["end"] + width)
    pre = "<span class='dim'>… </span>" if a > 0 else ""
    post = "<span class='dim'> …</span>" if b < len(text) else ""
    return f"<div class='doc'>{pre}{highlight(text, [span], a, b)}{post}</div>"


def show_item(cid, item):
    st.markdown(f"**{cid}** · {item.get('kind', '')}: {item.get('text', 'not found')}")
    if item.get("kind") == "patient":
        for sp in item.get("spans", [])[:2]:
            st.caption(f"From the {data['source']['kind']} ({sp['field']}), grounding: {sp.get('grounding')}")
            st.markdown(glimpse(sp), unsafe_allow_html=True)
        if not item.get("spans"):
            st.caption(f"From the {data['source']['kind']}: {data['case_path']}")
    if item.get("refs"):
        links = [f"[{r}](https://pubmed.ncbi.nlm.nih.gov/{r.split(':')[-1]}/)" for r in item["refs"][:5]]
        st.caption("Knowledge base sources: " + ", ".join(links))


# ---------------- sidebar: case and findings
with st.sidebar:
    st.header("Case")
    pick = st.selectbox("Choose a case", CASES, index=CASES.index(data["case_path"]) if data["case_path"] in CASES else 0)
    if pick != data["case_path"]:
        run(pick)
        st.rerun()
    st.caption(f"{data['case']} · {data['source']['kind']}")
    st.subheader("Findings")
    st.caption("Remove a finding that was read wrongly: the case is re-scored and re-explained.")
    for f in data["findings"]:
        c1, c2 = st.columns([5, 1])
        mark = "✅" if f["status"] == "present" else "⛔" if f["status"] == "absent" else "•"
        words = f" — “{f['words']}”" if f["words"] and f["words"].lower() != f["name"].lower() else ""
        c1.markdown(f"{mark} {f['name']}<br><span style='color:#64748B;font-size:0.8rem'>{f['status']}{words}</span>",
                    unsafe_allow_html=True)
        if c2.button("x", key=f"drop-{f['hpo_id']}", help=f"Remove {f['name']} and re-score"):
            run(data["case_path"], data["dropped"] + [f["hpo_id"]])
            st.rerun()
    if data["dropped"]:
        st.warning(f"{len(data['dropped'])} finding(s) removed by you")
        if st.button("Restore all findings"):
            run(data["case_path"])
            st.rerun()
    with st.expander("Evaluation only: the true diagnosis"):
        st.write(", ".join(data.get("truth") or []) or "not given")

# ---------------- main: two lists
st.title("Clinician review")
st.caption("Every sentence cites numbered evidence (P = patient, K = knowledge base) and passed checks V1-V4: "
           "the evidence exists, it matches the record, the sentence names the cited finding, the numbers match.")


def disease_panel(n, d):
    pack = {i["id"]: i for i in d["pack"]}
    kept = sum(len(v) for v in d["explanation"].values())
    with st.expander(f"{n}. {d['name']} ({d['id']}): {d['probability']:.1%}", expanded=(n == 1)):
        st.caption(f"Explanation by {d['explanation_source']} · {kept} sentences kept, {len(d['removed'])} removed "
                   f"by the checks · runner-up: {d['runner_up_name']}")
        t1, t2, t3 = st.tabs(["Explanation", "Source document", "Suggested tests"])
        with t1:
            for sec, title in TITLES.items():
                sents = d["explanation"].get(sec, [])
                if not sents:
                    continue
                st.markdown(f"**{title}**")
                for j, x in enumerate(sents):
                    s = x["text"]
                    left, right = st.columns([5, 1])
                    left.markdown(f"{BADGE[x['grade']]} {s}")
                    with right.popover("Why?", use_container_width=True):
                        for cid in [c.strip() for c in s[s.rfind("[") + 1:s.rfind("]")].split(",")]:
                            show_item(cid, pack.get(cid, {}))
            if d["removed"]:
                with st.expander(f"Removed by the checks ({len(d['removed'])})"):
                    for x in d["removed"]:
                        st.caption(f"{x['text']} → {x['reason']}")
        with t2:
            if not texts:
                st.info("This case is a structured phenopacket, so there is no free text to highlight.")
            for field, text in texts.items():
                spans = [sp for i in pack.values() if i.get("kind") == "patient" for sp in i.get("spans", [])
                         if sp["field"] == field]
                st.caption(f"{field}: yellow = words the evidence was read from, red = negation word")
                st.markdown(f"<div class='doc'>{highlight(text, spans)}</div>", unsafe_allow_html=True)
        with t3:
            t = d.get("tests")
            if not t:
                st.info("Test suggestions need the test data: python -m hoofbeats download-tests")
            else:
                st.caption(f"Probability before testing: {t['pre_test']:.1%} (from Part 3, the workup)")
                rows = ([t["gene_test"]] if t.get("gene_test") else []) + t.get("tests", [])
                for x in rows:
                    looks = ", ".join(s["name"] for s in x.get("shows", [])[:4])
                    st.markdown(f"- **{x['test']}** ({x['role']}, burden {x['burden']}/5): positive → "
                                f"{x['post_test_if_positive']:.0%}, negative → {x['post_test_if_negative']:.1%}"
                                + (f"  \n  <span style='color:#64748B;font-size:0.85rem'>looks for: {looks}</span>"
                                   if looks else ""), unsafe_allow_html=True)
                    for m in x.get("may_already_be_answered") or []:
                        st.caption(f"The note may already answer this ({m.get('finding')}): “{m.get('sentence')}”")
                for x in t.get("review", []):
                    st.markdown(f"- Review existing **{x['test']}**")


tab_rare, tab_common = st.tabs(["Top 3 rare diseases", "Top 3 common diseases"])
with tab_rare:
    for n, d in enumerate(data["lists"]["rare"], 1):
        disease_panel(n, d)
with tab_common:
    if not data["lists"]["common"]:
        st.info("No common-disease profiles are loaded yet (needs Part 2's learn-common), so this list is empty.")
    for n, d in enumerate(data["lists"]["common"], 1):
        disease_panel(n, d)

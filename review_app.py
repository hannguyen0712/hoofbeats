import html
import json
from pathlib import Path
import streamlit as st

data = json.loads(Path("results/part4_demo.json").read_text())
texts = data["source"]["texts"]
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


def highlight(text, spans, start=0, end=None):
    # text[start:end] as HTML, with every quote in <mark> and every negation cue in <mark class=cue>
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
            continue                                     # overlapping span: skip
        out.append(html.escape(text[pos:a]))
        out.append(f"<mark class='{cls}'>{html.escape(text[a:b])}</mark>")
        pos = b
    out.append(html.escape(text[pos:end]))
    return "".join(out)


def glimpse(span, width=110):
    text = texts[span["field"]]
    a = max(0, span["start"] - width)
    b = min(len(text), span["end"] + width)
    pre = "<span class='dim'>… </span>" if a > 0 else ""
    post = "<span class='dim'> …</span>" if b < len(text) else ""
    return f"<div class='doc'>{pre}{highlight(text, [span], a, b)}{post}</div>"


def show_item(cid, item):
    st.markdown(f"**{cid}** · {item.get('kind', '')}: {item.get('text', 'not found')}")
    if item.get("kind") == "patient":
        if item.get("spans"):
            for sp in item["spans"][:2]:
                st.caption(f"From the {data['source']['kind']} ({sp['field']}), grounding: {sp.get('grounding')}")
                st.markdown(glimpse(sp), unsafe_allow_html=True)
        else:
            st.caption(f"From the {data['source']['kind']}: {data['source']['path']}")
    if item.get("refs"):
        links = [f"[{r}](https://pubmed.ncbi.nlm.nih.gov/{r.split(':')[-1]}/)" for r in item["refs"][:5]]
        st.caption("Knowledge base sources: " + ", ".join(links))


st.title("Clinician review")
st.caption(f"Case: {data['case']} · every sentence cites numbered evidence (P = patient, K = knowledge base) "
           "and passed checks V1-V4")

for n, d in enumerate(data["diseases"], 1):
    pack = {i["id"]: i for i in d["pack"]}
    with st.expander(f"{n}. {d['name']} ({d['id']}): {d['probability']:.1%}", expanded=(n == 1)):
        tab_expl, tab_doc = st.tabs(["Explanation", "Source document"])
        with tab_expl:
            for sec, title in TITLES.items():
                sents = d["explanation"].get(sec, [])
                if not sents:
                    continue
                st.markdown(f"**{title}**")
                for x in sents:
                    s = x["text"]
                    left, right = st.columns([5, 1])
                    left.markdown(f"{BADGE[x['grade']]} {s}")
                    with right.popover("Why?", use_container_width=True):
                        for cid in [c.strip() for c in s[s.rfind("[") + 1:s.rfind("]")].split(",")]:
                            show_item(cid, pack.get(cid, {}))
            if d.get("removed"):
                with st.expander(f"Removed by the checks ({len(d['removed'])})"):
                    for x in d["removed"]:
                        st.caption(f"{x['text']} → {x['reason']}")
        with tab_doc:
            if not texts:
                st.info("This case is a structured phenopacket, so there is no free text to highlight.")
            for field, text in texts.items():
                spans = [sp for i in pack.values() if i.get("kind") == "patient" for sp in i.get("spans", [])
                         if sp["field"] == field]
                st.caption(f"{field}: yellow = words the evidence was read from, red = negation word")
                st.markdown(f"<div class='doc'>{highlight(text, spans)}</div>", unsafe_allow_html=True)

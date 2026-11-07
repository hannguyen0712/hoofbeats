import json
from pathlib import Path
import streamlit as st

data = json.loads(Path("results/part4_demo.json").read_text())
TITLES = {"supports": "Supports", "against": "Counts against",
          "expected_but_missing": "Expected but not recorded", "vs_runner_up": "Why above the runner-up"}
BADGE = {"Verified": ":green-badge[Verified]", "KB only": ":blue-badge[KB only]",
         "Patient only": ":orange-badge[Patient only]"}

st.set_page_config(page_title="Clinician review", layout="wide")
st.title("Clinician review")
st.caption(f"Case: {data['case']} · every sentence cites numbered evidence (P = patient, K = knowledge base) "
           "and passed checks V1-V4")

for n, d in enumerate(data["diseases"], 1):
    pack = {i["id"]: i for i in d["pack"]}
    with st.expander(f"{n}. {d['name']} ({d['id']}): {d['probability']:.1%}", expanded=(n == 1)):
        for sec, title in TITLES.items():
            sents = d["explanation"].get(sec, [])
            if not sents:
                continue
            st.markdown(f"**{title}**")
            for x in sents:
                s = x["text"]
                left, right = st.columns([5, 1])
                left.markdown(f"{BADGE[x['grade']]} {s}")
                with right.popover("Why?"):
                    for cid in [c.strip() for c in s[s.rfind("[") + 1:s.rfind("]")].split(",")]:
                        item = pack.get(cid, {})
                        st.markdown(f"**{cid}**: {item.get('text', 'not found')}")
                        if item.get("words"):
                            st.caption(f"Record says: \"{item['words']}\"")
                        if item.get("refs"):
                            st.caption(", ".join(item["refs"][:5]))
        if d.get("removed"):
            with st.expander(f"Removed by the checks ({len(d['removed'])})"):
                for x in d["removed"]:
                    st.caption(f"{x['text']} → {x['reason']}")

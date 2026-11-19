"""Inspect a claim: one plain sentence, the photo, up to three reasons, five checks. Detail is one click away."""
import streamlit as st

import backend as B
import ui
from inspect_view import render

ui.css()
claims = B.claims()
runs = B.runs()
prod = runs[runs.config_name == "Production"].set_index("claim_id")


def _label(cid):
    c = claims.set_index("claim_id").loc[cid]
    v = prod.loc[cid, "verdict"] if cid in prod.index else ""
    mark = {"likely_fraud": "🔴", "needs_review": "🟠", "clear": "🟢"}.get(v, "")
    return f"{mark} {cid} · {c.device_claimed} · ${float(c.claim_value):,.0f}"


ids = claims["claim_id"].tolist()
if st.session_state.get("insp_claim") not in ids:       # first visit: start from the last claim seen anywhere
    st.session_state["insp_claim"] = st.session_state.get("claim_id") if st.session_state.get("claim_id") in ids else "CLM-1054"
cid = st.selectbox("Claim", ids, format_func=_label, key="insp_claim")   # fixed key: the choice survives reruns
st.session_state["claim_id"] = cid

claim = ui.claim_row(cid)
cfg = ui.get_cfg("a")
with st.spinner("Checking the claim…"):
    r = ui.run(cid, cfg)

st.caption(f"{claim['device_claimed']} · ${claim['claim_value']:,.0f} · filed {claim['claim_date']} · incident "
           f"{claim['incident_date']} · “{claim['damage_described']}”")
render(r, claim, B.image(claim["image_path"]), cid,
       setup_text=f"**Setup in use:** {ui.cfg_name('a')}  \n<span class='wb-small'>Change it under Compare methods.</span>")

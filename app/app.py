"""Fraud Defence Workbench — Databricks App (Streamlit).

insurance_firm device-claim fraud: deepfake & shallowfake detection with every stage
switchable, every verdict explained, everything observable, every human decision
fed back to the gold set.
"""
import streamlit as st

st.set_page_config(page_title="Fraud Defence Workbench", page_icon="🛡️", layout="wide")

import backend as B  # noqa: E402
import ui  # noqa: E402
from image_authenticity.workbench import PRESETS, preset_config  # noqa: E402
from image_authenticity.workbench.catalog import PRESET_NOTES  # noqa: E402

PAGES = {
    "Claims": st.Page("views/claims.py", title="Claims", icon="🔎", default=True),
    "Business view": st.Page("views/business.py", title="Business view", icon="📈"),
    "Ask the data": st.Page("views/ask.py", title="Ask the data", icon="💬"),
    "Compare methods": st.Page("views/methods.py", title="Compare methods", icon="🎛️"),
    "Admin": st.Page("views/governance.py", title="Admin", icon="🏛️"),
}

with st.sidebar:
    st.markdown("### 🛡️ Fraud Defence Workbench")
    st.caption("insurance_firm × Databricks · prototype")
    role = st.selectbox("I am a…", ["Analyst", "Data scientist", "Manager / auditor"], key="role",
                        help="Changes which screen opens first and nothing else. Nothing is hidden.")

order = (["Claims", "Ask the data", "Business view", "Compare methods", "Admin"] if role == "Analyst"
         else ["Business view", "Ask the data", "Compare methods", "Claims", "Admin"] if role == "Manager / auditor"
         else ["Compare methods", "Claims", "Ask the data", "Business view", "Admin"])
nav = st.navigation([PAGES[k] for k in order])

with st.sidebar:
    with st.expander("Settings"):
        st.caption("Setup in use")
        p = st.selectbox("Setup", list(PRESETS), index=list(PRESETS).index(st.session_state.get("cfg_a_name", "Production"))
                         if st.session_state.get("cfg_a_name", "Production") in PRESETS else 0, key="sb_preset",
                         label_visibility="collapsed", format_func=lambda c: {"Production": "Current setup"}.get(c, c))
        if p != st.session_state.get("cfg_a_name", "Production"):
            ui.set_cfg("a", preset_config(p), p)
            B.log("model_switch", "Config A", f"Preset → {p}")
            st.rerun()
        st.caption(PRESET_NOTES.get(p, ""))
        st.caption(f"Signed in as {B.current_user()}")
        st.caption("Test set: 66 claims built from public device photos and made-up claims. Data in "
                   f"`{B.SCHEMA}` (Unity Catalog).")
        if st.button("Refresh the “seen before” index", use_container_width=True,
                     help="Recomputes the fingerprints for every claim photo from the claims table."):
            with st.spinner("Refreshing…"):
                B.rebuild_index()
            st.session_state.pop("_runs", None)
            B.log("index_rebuild", "hash_index", "Rebuilt from claims table")
            st.success("Index refreshed.")

nav.run()

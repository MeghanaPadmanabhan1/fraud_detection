"""Claims: inspect one claim, or work the review queue. One page, two views."""
import runpy

import streamlit as st

import ui

ui.css()
st.markdown("## Claims")
if "_claims_goto" in st.session_state:                     # set by "Send to review queue"
    st.session_state["claims_mode"] = st.session_state.pop("_claims_goto")
VIEWS = {"Inspect a claim": "views/inspector.py", "Check a new photo": "views/upload.py",
         "Review queue": "views/review.py"}
mode = st.radio("View", list(VIEWS), horizontal=True, key="claims_mode", label_visibility="collapsed")
runpy.run_path(VIEWS[mode])

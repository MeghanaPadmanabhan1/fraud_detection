"""Compare methods: try a different way of running a check, see the results, or open the expert switchboard."""
import runpy

import streamlit as st

import ui

ui.css()
st.markdown("## Compare methods")
VIEWS = {"Try a different method": "views/try_methods.py", "Results": "views/observability.py",
         "Expert switchboard": "views/lab.py"}
names = list(VIEWS)
mode = st.radio("View", names, horizontal=True, key="methods_mode", label_visibility="collapsed",
                index=1 if st.session_state.get("role") == "Manager / auditor" else 0)
runpy.run_path(VIEWS[mode])

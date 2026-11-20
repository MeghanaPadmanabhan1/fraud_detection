"""Ask the data: plain-English questions answered by a Genie space over the Lakeflow gold tables."""
import os

import pandas as pd
import streamlit as st

import backend as B
import genie_client
import ui

ui.css()
SPACE = os.getenv("GENIE_SPACE_ID", "")
st.markdown("## Ask the data")
st.caption("Ask about the claims in plain English. Answers come from the checked results (Lakeflow gold tables) and "
           "show the query that produced them.")
if not SPACE:
    st.info("No Genie space is configured for this app.")
    st.stop()

SAMPLES = ["How many claims did we flag as likely fraud, and how much money is that?",
           "Which fraud types do we catch least often?",
           "Which genuine claims were flagged by mistake?",
           "What are the top 5 claims in the review queue?",
           "How much claim value is in claims we cleared?"]
st.markdown("**Try one**")
cols = st.columns(len(SAMPLES))
picked = None
for c, s in zip(cols, SAMPLES):
    if c.button(s, use_container_width=True, key=f"ask_{hash(s)}"):
        picked = s
typed = st.chat_input("Ask a question about the claims")
question = typed or picked

hist = st.session_state.setdefault("ask_history", [])
if question:
    with st.spinner("Thinking…"):
        try:
            res = genie_client.ask(B.client(), SPACE, question, st.session_state.get("ask_conv"))
            st.session_state["ask_conv"] = res["conversation_id"]
            hist.append({"q": question, **res})
            B.log("ask_data", "genie", question[:200])
        except Exception as e:                       # noqa: BLE001
            hist.append({"q": question, "text": f"Sorry, that did not work: {str(e)[:200]}", "sql": "",
                         "columns": [], "rows": []})
for h in reversed(hist):
    with st.chat_message("user"):
        st.write(h["q"])
    with st.chat_message("assistant"):
        st.write(h.get("text") or "Here is the result.")
        if h.get("rows"):
            st.dataframe(pd.DataFrame(h["rows"], columns=h["columns"]), hide_index=True, use_container_width=True)
        if h.get("sql"):
            with st.expander("See the query"):
                st.code(h["sql"], language="sql")
if hist and st.button("Start a new conversation"):
    st.session_state.pop("ask_conv", None); st.session_state["ask_history"] = []; st.rerun()

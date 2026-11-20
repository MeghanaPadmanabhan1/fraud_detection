"""Business view: the outcome in the buyer's terms, read from the Lakeflow gold tables."""
import pandas as pd
import streamlit as st

import backend as B
import ui

ui.css()
st.markdown("## Business view")
st.caption("What the checks caught, what it cost to review, and what is still open. Test data: 66 claims built from "
           "public device photos and made-up claim records, so dollar figures are illustrative.")


@st.cache_data(ttl=120, show_spinner=False)
def load():
    d = B.sql(f"SELECT * FROM {B.LAKEFLOW}.gold_claim_decisions")
    s = B.sql(f"SELECT * FROM {B.LAKEFLOW}.gold_fraud_summary ORDER BY family, fraud_type")
    return d, s


try:
    d, s = load()
except Exception as e:                                   # noqa: BLE001
    st.error(f"The Lakeflow tables could not be read: {str(e)[:200]}")
    st.stop()
if d.empty:
    st.info("The pipeline has not produced results yet.")
    st.stop()

fraud, genuine = d[d.is_fraud == True], d[d.is_fraud == False]          # noqa: E712
caught = fraud[fraud.flagged == True]                                   # noqa: E712
firm = fraud[fraud.firm_call == True]                                   # noqa: E712
false_alarm = genuine[genuine.flagged == True]                          # noqa: E712

a, b, c, e = st.columns(4)
a.metric("Fraud claims flagged", f"{len(caught)} of {len(fraud)}", f"{len(caught) / max(len(fraud), 1):.0%}")
b.metric("Money at stake flagged", f"${caught.claim_value.sum():,.0f}", f"of ${fraud.claim_value.sum():,.0f} fraudulent")
c.metric("Firm “likely fraud” calls", f"{len(firm)}", f"{len(firm)} with no genuine claim among them"
         if int(genuine.firm_call.sum()) == 0 else f"{int(genuine.firm_call.sum())} on genuine claims", delta_color="off")
e.metric("Genuine claims sent to a person", f"{len(false_alarm)} of {len(genuine)}", f"{len(false_alarm) / max(len(genuine), 1):.0%}",
         delta_color="inverse")

st.markdown("#### Where the money is")
chart = s[s.family != "genuine"].set_index("fraud_type")[["claim_value_total", "flagged_value"]].rename(
    columns={"claim_value_total": "Claim value", "flagged_value": "Flagged"})
st.bar_chart(chart, horizontal=True)

st.markdown("#### By fraud type")
view = s.rename(columns={"family": "Family", "fraud_type": "Type", "claims": "Claims", "claim_value_total": "Value $",
                         "flagged_claims": "Flagged", "firm_calls": "Firm calls", "flagged_value": "Flagged $",
                         "flag_rate": "Flag rate"})
st.dataframe(view, hide_index=True, use_container_width=True,
             column_config={"Flag rate": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.0f%%")})

st.markdown("#### Review queue (served from Lakebase)")
st.caption("The strongest cases first. This table is synced from the Lakeflow gold table into Lakebase for the live queue.")
try:
    q = B.sql("SELECT queue_rank, claim_id, device_claimed, claim_value, verdict, confidence, top_check "
              "FROM fraud_lakebase.public.review_queue ORDER BY queue_rank LIMIT 10")
    st.dataframe(q.rename(columns={"queue_rank": "#", "claim_id": "Claim", "device_claimed": "Device",
                                   "claim_value": "Value $", "verdict": "Verdict", "confidence": "Conf.",
                                   "top_check": "Top check"}), hide_index=True, use_container_width=True)
except Exception as e:                                   # noqa: BLE001
    st.caption(f"Live queue not available right now ({str(e)[:100]}).")

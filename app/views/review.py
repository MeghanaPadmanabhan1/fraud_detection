"""Screen 4 — Review Queue & Labelling: every human decision lands in Delta as gold-set signal."""
import cv2
import pandas as pd
import streamlit as st

import backend as B
import lakebase
import plain
import ui

ui.css()
st.caption("Claims the checks flagged, waiting for a person. Every decision is saved with who made it and when, and "
           "feeds the next round of testing.")

claims = B.claims(); runs = B.runs(); labels = B.labels()
cfg_name = st.selectbox("Queue from configuration", ["Production", "Challenger A", "Full forensic", "Fast path only"])
R = runs[runs.config_name == cfg_name].merge(claims, on="claim_id")
first_red = (B.sql(f"SELECT claim_id, first(text) top_reason FROM (SELECT claim_id, text, severity FROM "
                   f"{B.T('findings')} WHERE config_name = :c AND severity IN ('red','amber') AND sid != 's6' "
                   f"ORDER BY claim_id, CASE severity WHEN 'red' THEN 0 ELSE 1 END, score DESC) GROUP BY claim_id",
                   {"c": cfg_name}) if len(R) else pd.DataFrame(columns=["claim_id", "top_reason"]))
Q = R[R.verdict != "clear"].merge(first_red, on="claim_id", how="left")
Q["labeled"] = Q.claim_id.isin(set(labels.claim_id))
Q["age_days"] = (pd.Timestamp("2026-10-27") - pd.to_datetime(Q.claim_date)).dt.days

f1, f2, f3 = st.columns([1, 1, 1.4])
show_done = f1.toggle("Include already-labelled", value=False)
sort = f2.selectbox("Sort by", ["confidence", "claim value", "age"])
verd = f3.multiselect("Verdict", ["likely_fraud", "needs_review"], default=["likely_fraud", "needs_review"])
q = Q[Q.verdict.isin(verd) & (show_done | ~Q.labeled)]
q = q.sort_values({"confidence": "confidence", "claim value": "claim_value", "age": "age_days"}[sort], ascending=False)

c_list, c_detail = st.columns([1.35, 1.65], gap="large")
with c_list:
    st.markdown(f"**{len(q)} in queue**")
    view = q[["claim_id", "verdict", "confidence", "claim_value", "age_days", "top_reason", "labeled"]].rename(
        columns={"claim_id": "Claim", "verdict": "Verdict", "confidence": "Conf.", "claim_value": "Value $",
                 "age_days": "Age (d)", "top_reason": "Top reason", "labeled": "Labelled"})
    view["Conf."] = (view["Conf."] * 100).round()
    ev = st.dataframe(view, hide_index=True, use_container_width=True, height=560, on_select="rerun",
                      selection_mode="single-row",
                      column_config={"Conf.": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
                                     "Top reason": st.column_config.TextColumn(width="large")})
    sel = ev.selection.rows
    if sel:
        st.session_state["review_claim"] = q.iloc[sel[0]].claim_id

with c_detail:
    cid = st.session_state.get("review_claim") or (q.iloc[0].claim_id if len(q) else None)
    if cid is None:
        st.success("Queue is empty.")
    else:
        claim = ui.claim_row(cid)
        r = ui.run(cid, ui.get_cfg("a"), overlays=False)
        st.markdown(f"### {cid} · {claim['device_claimed']}")
        i1, i2 = st.columns([1.2, 1])
        i1.image(cv2.cvtColor(B.image(claim["image_path"]), cv2.COLOR_BGR2RGB), use_container_width=True)
        with i2:
            ui.plain_summary(r)
            for f in plain.top_reasons(r.reasons, 3):
                st.markdown(f"- **{plain.CHECK_NAME.get(plain.plain_stage(f), f.stage)}:** {plain.reason_text(f)}")
        st.markdown("#### Label this claim")
        with st.form(f"label_{cid}"):
            lab = st.radio("Decision", ["confirm_fraud", "confirm_genuine", "overturn_needs_more_info"], horizontal=True,
                           format_func={"confirm_fraud": "Confirm fraud", "confirm_genuine": "Confirm genuine",
                                        "overturn_needs_more_info": "Overturn — needs more info"}.get)
            reason = st.selectbox("Reason code", ["reused_photo", "metadata_mismatch", "ai_altered", "ocr_mismatch",
                                                  "screenshot_or_recapture", "other"])
            note = st.text_input("Note (optional)")
            if st.form_submit_button("Save label to Delta", type="primary"):
                B.add_label(cid, lab, reason, note, r.verdict, ui.cfg_name("a"))
                st.success(f"Labelled {cid} as {lab} by {B.current_user()} — written to "
                           f"{B.T('labels')} and the audit log.")
                st.session_state["review_claim"] = None
                st.rerun()
        prior = labels[labels.claim_id == cid]
        if len(prior):
            st.markdown("**Earlier labels on this claim**")
            st.dataframe(prior[["label", "reason_code", "reviewer", "labeled_at", "note"]], hide_index=True,
                         use_container_width=True)

st.divider()
st.markdown("### Reviewer agreement")
st.caption("Where humans and the model disagreed — raw material for calibration.")
L = labels.merge(claims[["claim_id", "family", "fraud_class"]], on="claim_id", how="left")
if len(L):
    L["agrees_with_model"] = ((L.label == "confirm_fraud") & (L.model_verdict == "likely_fraud")) | \
                             ((L.label == "confirm_genuine") & (L.model_verdict == "clear"))
    L["matches_gold_set"] = (L.label == "confirm_fraud") == (L.family != "genuine")
    st.dataframe(L[["claim_id", "label", "reason_code", "model_verdict", "agrees_with_model", "fraud_class",
                    "matches_gold_set", "reviewer", "labeled_at"]], hide_index=True, use_container_width=True)
    a, b = st.columns(2)
    a.metric("Human ↔ model agreement", f"{L.agrees_with_model.mean():.0%}")
    b.metric("Human ↔ gold-set agreement", f"{L.matches_gold_set.mean():.0%}")


st.divider()
st.markdown("### Operational store (Lakebase)")
st.caption("Each decision is also written to a managed Postgres database (Lakebase) in milliseconds, next to the "
           "Delta record, so the live queue never waits on a warehouse.")
_s = lakebase.status()
if _s["ok"]:
    m1, m2, m3 = st.columns(3)
    m1.metric("Decisions stored", _s["decisions"]); m2.metric("Audit entries", _s["audit"])
    m3.metric("Round trip", f"{_s['ms']} ms")
    _ok, _rows = lakebase.recent_decisions(10)
    if _rows:
        st.dataframe(pd.DataFrame(_rows, columns=["When", "Claim", "Decision", "Reason", "Reviewer", "Model said"]),
                     hide_index=True, use_container_width=True)
    else:
        st.caption("No decisions saved yet. Label a claim above and it will appear here.")
else:
    st.info("Lakebase is not reachable right now, so decisions are being saved to Delta only. "
            f"({_s['error'][:120]})")

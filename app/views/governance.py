"""Screen 5 — Governance & Configuration: routing rules + what-would-change, retention, legal, audit."""
import copy

import pandas as pd
import streamlit as st

import backend as B
import ui
from image_authenticity.workbench import preset_config

ui.css()
st.markdown("## Admin")
st.caption("Governance is a feature, not a gate: routing rules, retention, legal dependencies and the audit trail "
           "are visible and controllable here.")
claims = B.claims()
t1, t2, t3, t4, t5 = st.tabs(["Routing rules", "Legal tracker", "Retention", "Audit log", "Dataset card"])

with t1:
    st.markdown("**Routing rules** — applied to Config A")
    cfg = copy.deepcopy(ui.get_cfg("a"))
    p6, p1 = cfg["s6"]["params"], cfg["s1"]["params"]
    c = st.columns(4)
    review = c[0].slider("Review threshold (confidence)", 0.0, 1.0, float(p6["review"]), 0.05)
    fraud = c[1].slider("Investigator threshold (confidence)", 0.0, 1.0, float(p6["fraud"]), 0.05)
    mm = c[2].selectbox("Missing metadata →", ["amber", "info"], index=["amber", "info"].index(p1["missing_metadata"]),
                        help="No settled rule for screenshot vs deliberately stripped EXIF; 'info' stops routing on it.")
    age = c[3].slider("Max photo age before incident (days)", 1, 365, int(p1["max_photo_age_days"]))
    st.markdown("Fixed rules: any **red** finding → investigator queue · **2+ red findings in different stages or "
                "≥99% confidence** → priority investigator queue (99%-confidence route) · any **amber** → analyst "
                "review · **nothing is auto-denied**.")
    prop = copy.deepcopy(cfg)
    prop["s6"]["params"].update(review=review, fraud=fraud)
    prop["s1"]["params"].update(missing_metadata=mm, max_photo_age_days=age)
    if st.button("Preview: what would change on the gold set?", type="primary"):
        ids = claims.claim_id.tolist()
        B.preload_media(claims["image_path"].tolist() + claims["receipt_path"].dropna().tolist())
        with st.spinner("Re-deciding every gold-set claim under both rule sets…"):
            A, Bp = ui.run_many(ids, cfg), ui.run_many(ids, prop)
            rows = [{"claim_id": cid, "current": A[cid].verdict, "proposed": Bp[cid].verdict} for cid in ids]
        D = pd.DataFrame(rows).merge(claims[["claim_id", "family", "fraud_class"]], on="claim_id")
        moved = D[D.current != D.proposed]
        st.markdown(f"**{len(moved)} of {len(D)} claims would move queues.**")
        st.dataframe(pd.crosstab(D.current, D.proposed), use_container_width=True)
        if len(moved):
            st.dataframe(moved, hide_index=True, use_container_width=True)
        st.session_state["proposed_rules"] = prop
    if st.session_state.get("proposed_rules") and st.button("Apply proposed rules to Config A"):
        ui.set_cfg("a", st.session_state.pop("proposed_rules"), ui.cfg_name("a"))
        B.log("rule_change", "Config A", f"review={review}, fraud={fraud}, missing_metadata={mm}, max_age={age}")
        st.success("Applied and written to the audit log.")

with t2:
    st.markdown("**Legal & vendor dependencies** — visible here, not discovered in email")
    g = B.governance()
    st.dataframe(g, hide_index=True, use_container_width=True,
                 column_config={"notes": st.column_config.TextColumn(width="large")})

with t3:
    st.markdown("**Retention policies**")
    st.dataframe(pd.DataFrame([
        {"data": "IP addresses (claim intake)", "retention": "90 days", "mechanism": "Scheduled Lakeflow job; each run "
         "writes an audit row", "status": "active"},
        {"data": "Claimant PII columns", "retention": "Policy term + 7 years", "mechanism": "Unity Catalog column masks for "
         "roles that do not need them", "status": "active"},
        {"data": "Claim photos & receipts", "retention": "Policy term + 7 years", "mechanism": "UC volume, access by role",
         "status": "active"},
        {"data": "Investigator labels", "retention": "Indefinite (gold set)", "mechanism": "Delta table with time travel",
         "status": "active"}]), hide_index=True, use_container_width=True)
    purges = B.audit()
    st.dataframe(purges[purges.action == "retention_purge"], hide_index=True, use_container_width=True)

with t4:
    a = B.audit()
    f = st.text_input("Search the audit log", placeholder="claim id, actor, action…")
    if f:
        a = a[a.apply(lambda r: f.lower() in " ".join(map(str, r.values)).lower(), axis=1)]
    st.dataframe(a, hide_index=True, use_container_width=True, height=420)
    st.download_button("Export CSV", a.to_csv(index=False).encode(), "audit_log.csv", "text/csv")

with t5:
    st.markdown("""**Gold set v0 — dataset card**

- **Purpose:** prototype evaluation set for the workbench; not a production gold set.
- **Composition:** 66 claims — 14 genuine, 5 genuine-photo/mismatched-receipt (document), 22 shallowfake (reuse exact /
  recompressed / cropped, screenshot, re-photographed screen, EXIF date contradictions), 25 deepfake (AI-generated,
  AI-edited damage, splice, copy-move, upscale).
- **Sources:** 27 genuine damaged-device photos from Wikimedia Commons (CC BY / BY-SA / CC0 / PD; photos with faces
  excluded); AI images and AI edits from the `databricks-gemini-3-pro-image` endpoint; all manipulations scripted with
  OpenCV / Pillow; ground-truth masks for splice, copy-move and AI edits.
- **Synthetic:** claimants, claim dates, receipts and capture EXIF timelines are synthetic. Half of the AI outputs are
  "laundered" (provenance stripped, fake camera EXIF) to test detection without metadata.
- **Known limits:** small; phone-heavy; no real adjudicated claims; no held-out device/tool split yet. The real gold
  set must come from Asurion's adjudicated claims (see the companion dataset prompt).
""")
    st.dataframe(claims[["claim_id", "family", "fraud_class", "source_title", "licence", "artist"]],
                 hide_index=True, use_container_width=True, height=300)

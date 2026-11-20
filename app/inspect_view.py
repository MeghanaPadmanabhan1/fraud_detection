"""The result layout shared by 'Inspect a claim' and 'Check a new photo': banner, photo, reasons, five checks, detail."""
import cv2
import streamlit as st

import backend as B
import plain
import ui


def render(r, claim, img, cid, *, photo_bytes=None, receipt_bytes=None, uploaded=False, setup_text=""):
    """photo_bytes / receipt_bytes are given for uploads (nothing is stored anywhere); filed claims read them from the volume."""
    ui.banner(r, claim)
    match = next((f for f in r.stage("s2").findings if f.evidence.get("match_claim_id")), None)

    left, right = st.columns([1.5, 1.2], gap="large")
    with left:
        shown = {k: v for k, v in plain.OVERLAY_PLAIN.items() if k in ui.overlay_modes(r, claim)}
        options = [shown[k][0] for k in shown]
        back = {shown[k][0]: k for k in shown}
        if match:
            options.insert(1, "Same as an earlier claim")
        mode = st.radio("Show", options, index=0, horizontal=True, label_visibility="collapsed", key=f"show_{cid}")
        if mode == "Same as an earlier claim":
            mc = ui.claim_row(match.evidence["match_claim_id"])
            m1, m2 = st.columns(2)
            m1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"This {'photo' if uploaded else 'claim'} · {cid}",
                     use_container_width=True)
            m2.image(cv2.cvtColor(B.image(mc["image_path"]), cv2.COLOR_BGR2RGB), use_container_width=True,
                     caption=f"Earlier claim · {mc['claim_id']} · filed {mc['claim_date']}")
        else:
            key = back[mode]
            st.image(ui.overlay_image(img, r, key, 0.6, claim.get("mask_path")), use_container_width=True)
            if shown[key][1]:
                st.caption(shown[key][1])

    def _evidence(f):
        """The proof behind one reason, in the form a person can check."""
        k = plain.plain_stage(f)
        if f.evidence.get("match_claim_id"):
            mc = ui.claim_row(f.evidence["match_claim_id"])
            e1, e2 = st.columns(2)
            e1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"This {'photo' if uploaded else 'claim'} · {cid}",
                     use_container_width=True)
            e2.image(cv2.cvtColor(B.image(mc["image_path"]), cv2.COLOR_BGR2RGB), use_container_width=True,
                     caption=f"Earlier claim · {mc['claim_id']} · {mc['claimant_id']} · filed {mc['claim_date']}")
        elif k == "s1":
            ui.metadata_card(claim, r, photo_bytes)
        elif k == "s5":
            ui.receipt_card(claim, r, receipt_bytes)
        elif k == "s3":
            name = next((o for o in ("ELA heatmap", "JPEG ghost map", "Noise map", "Copy-move pairs")
                         if {"ELA heatmap": "ELA", "JPEG ghost map": "JPEG ghost", "Noise map": "Noise",
                             "Copy-move pairs": "Copy-move"}[o] in f.check and o in r.overlays), None)
            if name:
                st.image(ui.overlay_image(img, r, name, 0.6), use_container_width=True)
                st.caption(plain.OVERLAY_PLAIN[name][1])
        st.caption(f"Found by: {f.check}")

    with right:
        st.markdown("**Why we think so**" if r.verdict != "clear" else "**What we found**")
        reasons = plain.top_reasons(r.reasons, 3)
        if not reasons:
            st.markdown('<div class="wb-reasoncard" style="border-left-color:#2E9E5B"><div class="k">All checks</div>'
                        '<div class="x">Nothing unusual found in the photo, its details or the receipt.</div></div>',
                        unsafe_allow_html=True)
        for f in reasons:
            color = ui.SEV_COLOR[f.severity]
            st.markdown(f'<div class="wb-reasoncard" style="border-left-color:{color}">'
                        f'<div class="k">{ui.esc(plain.CHECK_NAME.get(plain.plain_stage(f), f.stage))}</div>'
                        f'<div class="x">{ui.esc(plain.reason_text(f))}</div></div>', unsafe_allow_html=True)
            with st.expander("See the evidence"):
                _evidence(f)
        if uploaded:
            st.button("Send to review queue →", use_container_width=True, disabled=True,
                      help="Filing new claims is not switched on in this prototype. This check saved nothing.")
        elif st.button("Send to review queue →", use_container_width=True, disabled=r.verdict == "clear"):
            st.session_state["review_claim"] = cid
            st.session_state["_claims_goto"] = "Review queue"
            st.rerun()

    st.markdown("##### The five checks")
    ui.checks_strip(r)

    with st.expander("More detail: how this was worked out"):
        c1, c2 = st.columns([1.4, 1])
        with c1:
            st.markdown(setup_text, unsafe_allow_html=True)
        with c2:
            st.toggle("Call AI models live if not already stored", key="live_models",
                      value=st.session_state.get("live_models", True),
                      help="Off = only use results already stored in the Gold tables.")
        t1, t2, t3, t4 = st.tabs(["Photo details", "Receipt", "Speed", "Every finding"])
        with t1:
            ui.metadata_card(claim, r, photo_bytes)
        with t2:
            ui.receipt_card(claim, r, receipt_bytes)
        with t3:
            ui.timeline(r)
        with t4:
            st.caption("Every check that ran, with the method used. Scores and thresholds are in Compare methods → "
                       "Expert switchboard.")
            for s in r.stages:
                st.markdown(f"**{s.num} · {s.name}** — {', '.join(s.techs) if s.status != 'skipped' else 'skipped'}  \n"
                            f"<span class='wb-small'>{s.status} · {s.path} · {s.latency_ms / 1000:.2f}s"
                            + (" · stored result" if s.cached else "") + "</span>", unsafe_allow_html=True)
                ui.findings_list(s.findings) if s.findings else st.caption("No findings.")
        if not uploaded:
            st.markdown(f"**Test-set answer** (for evaluating the checks, not shown to investigators): "
                        f"{ui.FAMILY_LABEL.get(claim['family'], claim['family'])} · `{claim['fraud_class']}`  \n"
                        f"{claim.get('notes') or ''}")
            st.caption(f"Source: {claim.get('source_title')} · {claim.get('licence')} · {claim.get('artist')}")

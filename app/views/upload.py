"""Check a new photo: upload a photo (and receipt), get the same checks as a filed claim. Nothing is saved."""
import datetime as dt

import cv2
import numpy as np
import streamlit as st

import backend as B
import ui
import uploads
from inspect_view import render

ui.css()
ss = st.session_state
today = dt.date.today()
st.caption("Upload a photo of a damaged device, and the receipt if you have one, to get the same checks a filed claim "
           "gets. **Nothing is saved.** The files stay in this browser session, and “Clear” removes them.")

DEFAULTS = {"up_device": "", "up_damage": "Cracked screen after a drop", "up_value": 300.0, "up_claimant": "CUST-NEW",
            "up_incident": today - dt.timedelta(days=2), "up_purchase": today - dt.timedelta(days=365),
            "up_sscd": True, "up_n": 0}
for k, v in DEFAULTS.items():
    ss.setdefault(k, v)


def use_sample(key):
    """Button callback: load an example photo built from an earlier claim, and fill in matching details."""
    try:
        raw, src = uploads.sample_bytes(key), uploads.source_claim(key)
    except Exception as e:                                   # volume not reachable, etc.
        ss["up_error"] = f"Could not load the example: {e!r}"[:200]
        return
    ss.pop("up_error", None); ss.pop("up_result", None)
    ss["up_n"] += 1                                          # new uploader widgets: drops any file chosen before
    ss["up_sample"] = {"key": key, "bytes": raw}
    ss.update(up_device=src["device_claimed"], up_damage="Cracked screen after a drop",
              up_value=float(src["claim_value"]), up_claimant="CUST-NEW",
              up_incident=today - dt.timedelta(days=2), up_purchase=today - dt.timedelta(days=365))


def clear_all():
    for k in ("up_sample", "up_result", "up_error", "_upload_live"):
        ss.pop(k, None)
    ss["up_n"] += 1
    for k in ("up_device", "up_damage", "up_value", "up_claimant", "up_incident", "up_purchase"):
        ss[k] = DEFAULTS[k]


left, right = st.columns([1.15, 1], gap="large")
with left:
    up = st.file_uploader("Photo of the damaged device (required)", type=["jpg", "jpeg", "png"],
                          key=f"up_photo_{ss['up_n']}", help="JPEG or PNG, up to 15 MB. Upload the original file, not a "
                          "screenshot or a camera-roll export if you can, because the camera details are part of the check.")
    rc = st.file_uploader("Receipt (optional)", type=["jpg", "jpeg", "png"], key=f"up_receipt_{ss['up_n']}",
                          help="A photo or scan of the receipt, JPEG or PNG. PDF receipts are not supported yet.")
    sample = ss.get("up_sample")
    photo = up.getvalue() if up is not None else (sample["bytes"] if sample else None)
    receipt = rc.getvalue() if rc is not None else None

    ok_photo = ok_receipt = True
    if photo is not None:
        ok_photo, msg = uploads.check_image(photo, "photo")
        if not ok_photo:
            st.error(msg)
        else:
            st.image(cv2.cvtColor(cv2.imdecode(np.frombuffer(photo, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
                     caption=(f"Example: {uploads.SAMPLE[sample['key']]['label']}" if up is None and sample
                              else up.name) + f" · {msg}", width=360)
            if up is None and sample:
                st.info(uploads.SAMPLE[sample["key"]]["note"] + " The match may name another claim that used "
                        "the same source photo.")
            elif uploads.has_face(photo):
                st.warning("This photo seems to contain a person's face. Please check you are allowed to share it. "
                           "Photos of the device alone are best.")
    if receipt is not None:
        ok_receipt, rmsg = uploads.check_image(receipt, "receipt")
        st.error(rmsg) if not ok_receipt else st.caption(f"Receipt: {rmsg}")
    if ss.get("up_error"):
        st.error(ss["up_error"])

    st.markdown("**No photo handy? Try an example** <span class='wb-small'>(built from photos already in the test claims)"
                "</span>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, s in enumerate(uploads.SAMPLES):
        cols[i % 2].button(s["label"], key=f"up_ex_{s['key']}", on_click=use_sample, args=(s["key"],),
                           use_container_width=True)

with right:
    st.markdown("**About the claim**")
    device = st.text_input("Device being claimed", key="up_device", placeholder="e.g. Apple iPhone 14")
    damage = st.text_input("What happened", key="up_damage")
    value = st.number_input("Claim value ($)", min_value=0.0, step=10.0, key="up_value")
    d1, d2 = st.columns(2)
    incident = d1.date_input("Incident date", key="up_incident", max_value=today)
    purchase = d2.date_input("Purchase date", key="up_purchase", max_value=today,
                             help="Compared with the date on the receipt, if you upload one.")
    claimant = st.text_input("Customer ID", key="up_claimant",
                             help="Used to tell “the same customer reusing a photo” from “a different customer”.")
    st.checkbox("Also look for cropped or screenshotted copies of earlier photos", key="up_sscd",
                help="Uses the learned look-alike search (SSCD). The quick fingerprint misses crops and screenshots.")

details = {"device": device, "damage": damage, "value": value, "incident_date": incident, "purchase_date": purchase,
           "claimant_id": claimant}
ready = photo is not None and ok_photo and ok_receipt and bool(device.strip())
b1, b2, b3 = st.columns([1.2, 0.8, 3])
go = b1.button("Check this claim", type="primary", disabled=not ready, use_container_width=True,
               help=None if ready else "Add a photo and the device being claimed first.")
b2.button("Clear", on_click=clear_all, use_container_width=True, help="Remove the files and the result from this session.")

if go:
    with st.spinner("Checking the photo… this takes a few seconds (the AI vision model and receipt reader are live calls)."):
        claim, r, cfg = uploads.run_upload(photo, receipt, details, ss["up_sscd"], ss.get("live_models", True))
    ss["up_result"] = {"claim": claim, "r": r, "photo": photo, "receipt": receipt, "sscd": ss["up_sscd"]}
    try:                                                     # the audit trail records THAT a check ran, never the files
        import hashlib
        B.log("upload_check", claim["claim_id"], f"photo sha256 {hashlib.sha256(photo).hexdigest()[:16]}, "
              f"{len(photo)} bytes, receipt {'yes' if receipt else 'no'}, verdict {r.verdict}")
    except Exception:
        st.caption("(The audit entry could not be written; the check itself ran.)")

res = ss.get("up_result")
if res:
    st.divider()
    stale = ready and uploads.build_claim(photo, receipt, details)["claim_id"] != res["claim"]["claim_id"]
    if stale:
        st.info("You changed something since this check. Press “Check this claim” again to update the result.")
    claim = res["claim"]
    st.caption(f"{claim['device_claimed']} · ${claim['claim_value']:,.0f} · incident {claim['incident_date']} · "
               f"customer {claim['claimant_id']} · “{claim['damage_described']}”")
    render(res["r"], claim, cv2.imdecode(np.frombuffer(res["photo"], np.uint8), cv2.IMREAD_COLOR), claim["claim_id"],
           photo_bytes=res["photo"], receipt_bytes=res["receipt"], uploaded=True,
           setup_text=f"**Setup:** {ui.cfg_name('a')}" + (", with the learned look-alike search" if res["sscd"] else "")
           + (", receipt read by a vision model" if res["receipt"] else "")
           + "  \n<span class='wb-small'>This check was not saved or added to the “seen before” index.</span>")

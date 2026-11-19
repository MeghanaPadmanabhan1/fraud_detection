"""Screen 2 — Pipeline Lab: switch any stage's technology, A/B two configs, batch-run the gold set."""
import copy
import json
import time

import numpy as np
import pandas as pd
import streamlit as st

import backend as B
import ui
from image_authenticity.workbench import PRESETS, preset_config
from image_authenticity.workbench.catalog import PRESET_NOTES, STAGES

ui.css()
st.caption("The expert switchboard. Pick the technology for each stage, re-run a claim, compare two configurations, "
           "and batch-run the gold set — the same evaluation harness the data-science team uses.")

saved = B.saved_configs()
choices = list(PRESETS) + [f"saved: {n}" for n in saved["name"].tolist()] if len(saved) else list(PRESETS)


def load_choice(choice):
    if choice.startswith("saved: "):
        row = saved[saved.name == choice[7:]].iloc[0]
        return json.loads(row.config_json)
    return preset_config(choice)


h = st.columns([1.3, 1.3, 1.3, 1.6])
with h[0]:
    pa = st.selectbox("Config A preset", choices, index=0)
    if st.button("Load into A", use_container_width=True):
        ui.set_cfg("a", load_choice(pa), pa); B.log("model_switch", "Config A", f"Loaded preset {pa}"); st.rerun()
with h[1]:
    pb = st.selectbox("Config B preset", choices, index=1)
    if st.button("Load into B", use_container_width=True):
        ui.set_cfg("b", load_choice(pb), pb); st.rerun()
with h[2]:
    editing = st.radio("Editing", ["Config A", "Config B"], horizontal=True)
    st.caption(f"A = **{ui.cfg_name('a')}** · B = **{ui.cfg_name('b')}**")
with h[3]:
    with st.popover("Save config A…", use_container_width=True):
        nm = st.text_input("Name", value=f"{ui.cfg_name('a')} v2")
        notes = st.text_input("Notes")
        if st.button("Save to Delta"):
            B.save_config(nm, ui.get_cfg("a"), notes); st.success("Saved and audited."); st.rerun()
    st.caption(PRESET_NOTES.get(st.session_state.get("cfg_a_name"), ""))

slot = "a" if editing == "Config A" else "b"
cfg = copy.deepcopy(ui.get_cfg(slot))
changed = False


def opt_list(stage):
    return [o for o in stage["options"] if o["available"]], [o for o in stage["options"] if not o["available"]]


cards = st.columns(4)
for i, stage in enumerate(STAGES):
    sid = stage["id"]; sc = cfg[sid]
    with cards[i % 4]:
        with st.container(border=True):
            st.markdown(f"**{stage['num']} · {stage['name']}**  \n<span class='wb-small'>{stage['question']}</span>",
                        unsafe_allow_html=True)
            st.markdown(ui.chip(stage["path"]) + ui.chip(stage["budget"]), unsafe_allow_html=True)
            avail, unavail = opt_list(stage)
            keys = [o["key"] for o in avail]; labels = {o["key"]: o["label"] for o in avail}
            en = st.toggle("Enabled", value=sc["enabled"], key=f"{slot}_{sid}_en",
                           disabled=sid in ("s0", "s6", "s7"))
            if stage["mode"] == "multi":
                cur = [t for t in sc["tech"] if t in keys]
                val = st.multiselect("Technologies", keys, default=cur, format_func=labels.get,
                                     key=f"{slot}_{sid}_tech", disabled=not en)
            else:
                cur = sc["tech"] if sc["tech"] in keys else keys[0]
                val = st.selectbox("Technology", keys, index=keys.index(cur), format_func=labels.get,
                                   key=f"{slot}_{sid}_tech", disabled=not en or len(keys) == 1)
            if sid == "s6":
                ex = stage["explainers"]; ek = [o["key"] for o in ex]
                e = st.selectbox("Explainer", ek, index=ek.index(sc["params"].get("explainer", "template")),
                                 format_func={o["key"]: o["label"] for o in ex}.get, key=f"{slot}_s6_expl")
                if e != sc["params"].get("explainer"):
                    sc["params"]["explainer"] = e; changed = True
            with st.expander("Thresholds"):
                p = sc["params"]
                if sid == "s1":
                    p["max_photo_age_days"] = st.slider("Max photo age before incident (days)", 1, 365,
                                                        p["max_photo_age_days"], key=f"{slot}_s1_age")
                    p["missing_metadata"] = st.selectbox("Missing-metadata policy", ["amber", "info"],
                                                         index=["amber", "info"].index(p["missing_metadata"]),
                                                         key=f"{slot}_s1_mm")
                elif sid == "s2":
                    p["phash_max"] = st.slider("pHash max distance (of 64)", 0, 24, p["phash_max"], key=f"{slot}_s2_ph")
                    p["dhash_max"] = st.slider("dHash max distance (of 64)", 0, 24, p["dhash_max"], key=f"{slot}_s2_dh")
                    p["orb_min_inliers"] = st.slider("ORB min inliers", 5, 120, p["orb_min_inliers"], key=f"{slot}_s2_orb")
                    p["embed_min_cos"] = st.slider("Embedding min cosine", 0.80, 1.0, float(p["embed_min_cos"]),
                                                   0.005, key=f"{slot}_s2_emb")
                    p["sscd_min_cos"] = st.slider("SSCD min cosine", 0.30, 0.95, float(p.get("sscd_min_cos", 0.60)),
                                                  0.01, key=f"{slot}_s2_sscd")
                elif sid in ("s3", "s4"):
                    if sid == "s4":
                        p["amber"], p["red"] = st.slider("Vision-model amber / red", 0.0, 1.0,
                                                         (float(p["amber"]), float(p["red"])), 0.05, key=f"{slot}_s4_v")
                    for t in (val or []):
                        if t in p.get("thresholds", {}):
                            a, r_ = p["thresholds"][t]
                            p["thresholds"][t] = list(st.slider(labels[t], 0.0, 1.0, (float(a), float(r_)), 0.005,
                                                                key=f"{slot}_{sid}_{t}"))
                elif sid == "s6":
                    p["review"] = st.slider("Review threshold", 0.0, 1.0, float(p["review"]), 0.05, key=f"{slot}_s6_r")
                    p["fraud"] = st.slider("Fraud threshold", 0.0, 1.0, float(p["fraud"]), 0.05, key=f"{slot}_s6_f")
                else:
                    st.caption("No thresholds for this stage.")
            if unavail:
                st.caption("Also in the switchboard: " + "; ".join(f"*{o['label']}* — {o['note']}" for o in unavail))
            if en != sc["enabled"] or val != sc["tech"]:
                changed = True
            sc["enabled"], sc["tech"] = en, val
if cfg != ui.get_cfg(slot):
    if changed:
        B.log("model_switch", f"Config {slot.upper()}", "; ".join(
            f"{s['id']}={cfg[s['id']]['tech']}" for s in STAGES if cfg[s["id"]] != ui.get_cfg(slot)[s["id"]]))
    st.session_state[f"cfg_{slot}"] = cfg

st.divider()
st.markdown("### Re-run a claim — A vs B")
claims = B.claims()
ids = claims["claim_id"].tolist()
cid = st.selectbox("Claim", ids, index=ids.index(st.session_state.get("claim_id", "CLM-1054"))
                   if st.session_state.get("claim_id", "CLM-1054") in ids else 0,
                   format_func=lambda c: f"{c} · {claims.set_index('claim_id').loc[c, 'device_claimed']} · "
                                         f"{claims.set_index('claim_id').loc[c, 'fraud_class']}")
st.session_state["claim_id"] = cid
with st.spinner("Running both configurations…"):
    ra, rb = ui.run(cid, ui.get_cfg("a"), overlays=False), ui.run(cid, ui.get_cfg("b"), overlays=False)
if ra.verdict != rb.verdict:
    st.warning(f"**Verdict flips:** A says **{ui.VERDICT[ra.verdict][0]}**, B says **{ui.VERDICT[rb.verdict][0]}**.")
else:
    st.info(f"Both configurations agree: **{ui.VERDICT[ra.verdict][0]}** (A {ra.confidence:.0%} · B {rb.confidence:.0%}).")
ca, cd, cb = st.columns([1.3, 1.6, 1.3])
with ca:
    st.markdown(f"**A · {ui.cfg_name('a')}**"); ui.verdict_panel(ra)
with cb:
    st.markdown(f"**B · {ui.cfg_name('b')}**"); ui.verdict_panel(rb)
with cd:
    st.markdown("**What changed, stage by stage**")
    rows = []
    for sa, sb in zip(ra.stages, rb.stages):
        rows.append({"Stage": f"{sa.num} {sa.name}", "A": f"{sa.severity if sa.status != 'skipped' else 'skipped'}",
                     "B": f"{sb.severity if sb.status != 'skipped' else 'skipped'}",
                     "A tech": ", ".join(sa.techs), "B tech": ", ".join(sb.techs),
                     "changed": "◆" if (sa.severity, sa.status) != (sb.severity, sb.status) else ""})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.apply(lambda r: ["background-color:#FFF4D6" if r["changed"] else "" for _ in r], axis=1),
                 hide_index=True, use_container_width=True, height=330)
    st.caption(f"Fast path A {ra.sync_latency_ms / 1000:.2f}s vs B {rb.sync_latency_ms / 1000:.2f}s")

st.divider()
st.markdown("### Batch run on the gold set")
st.caption("Runs both configurations over every gold-set claim (66) and reports what matters: recall per fraud "
           "family (measured separately), false positives per 1,000 genuine claims, review load and latency.")


def evaluate(results, claims_df):
    df = pd.DataFrame([{"claim_id": c, "verdict": r.verdict, "sync": r.sync_latency_ms, "tokens": r.tokens}
                       for c, r in results.items()]).merge(claims_df[["claim_id", "family"]], on="claim_id")
    df["flagged"] = df.verdict != "clear"; df["fraud"] = df.verdict == "likely_fraud"
    out = {}
    for fam in ["deepfake", "shallowfake", "document"]:
        d = df[df.family == fam]
        out[f"{fam} recall (flagged)"] = d.flagged.mean()
        out[f"{fam} recall (likely fraud)"] = d.fraud.mean()
    g = df[df.family == "genuine"]
    out["genuine flagged per 1,000"] = g.flagged.mean() * 1000
    out["genuine 'likely fraud' per 1,000"] = g.fraud.mean() * 1000
    flagged = df[df.flagged]
    out["precision of flags"] = (flagged.family != "genuine").mean() if len(flagged) else np.nan
    out["median fast-path latency (s)"] = df.sync.median() / 1000
    out["p95 fast-path latency (s)"] = df.sync.quantile(0.95) / 1000
    out["model tokens per claim"] = df.tokens.mean()
    return out, df


if st.button("▶ Batch run A and B on the gold set", type="primary"):
    B.preload_media(claims["image_path"].tolist() + claims["receipt_path"].dropna().tolist())
    prog = st.progress(0.0, "Running…")
    t0 = time.time()
    res = {s: ui.run_many(ids, ui.get_cfg(s), prog, f"Config {s.upper()}") for s in ("a", "b")}
    prog.empty()
    st.session_state["batch"] = {"a": evaluate(res["a"], claims), "b": evaluate(res["b"], claims),
                                 "names": (ui.cfg_name("a"), ui.cfg_name("b")), "secs": time.time() - t0}
    B.log("batch_run", f"{ui.cfg_name('a')} vs {ui.cfg_name('b')}", f"{len(ids)} gold-set claims")

if "batch" in st.session_state:
    bt = st.session_state["batch"]
    (ma, da), (mb, db) = bt["a"], bt["b"]
    better_low = {"genuine flagged per 1,000", "genuine 'likely fraud' per 1,000", "median fast-path latency (s)",
                  "p95 fast-path latency (s)", "model tokens per claim"}
    rows = []
    for k in ma:
        a, b = ma[k], mb[k]
        win = "" if pd.isna(a) or pd.isna(b) or abs(a - b) < 1e-9 else ("B" if (b < a) == (k in better_low) else "A")
        fmt = (lambda v: f"{v:.0%}") if ("recall" in k or "precision" in k) else (lambda v: f"{v:,.2f}" if v < 100 else f"{v:,.0f}")
        rows.append({"Metric": k, f"A · {bt['names'][0]}": fmt(a), f"B · {bt['names'][1]}": fmt(b), "Better": win})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    m1, m2 = st.columns(2)
    for col, (name, d) in zip((m1, m2), ((bt["names"][0], da), (bt["names"][1], db))):
        col.markdown(f"**Verdicts by true family — {name}**")
        col.dataframe(pd.crosstab(d.family, d.verdict), use_container_width=True)
    st.caption(f"Batch completed in {bt['secs']:.0f}s. Fast-path checks re-ran live; slow-path results came from "
               "Gold (or live calls where not yet cached). Gold set v0: CC/PD device photos + synthetic claims.")

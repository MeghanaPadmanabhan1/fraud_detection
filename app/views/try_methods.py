"""Try a different method: change how a check works, see one claim change, then test on all test claims."""
import copy

import streamlit as st

import backend as B
import plain
import ui
from image_authenticity.workbench.catalog import STAGE_BY_ID

st.caption("Pick how each check works. We compare your trial with the setup in use, first on one claim, then on all "
           "test claims.")
claims = B.claims()
ids = claims["claim_id"].tolist()

if "trial_init" not in st.session_state:
    ui.set_cfg("b", ui.get_cfg("a"), "Your trial")
    st.session_state["trial_init"] = True
cfg = copy.deepcopy(ui.get_cfg("b"))

top = st.columns([4, 1])
top[0].markdown("##### How should each check work?")
if top[1].button("Reset to current", use_container_width=True):
    ui.set_cfg("b", ui.get_cfg("a"), "Your trial")
    for k in [k for k in st.session_state if str(k).startswith("try_")]:
        del st.session_state[k]
    st.rerun()

for sid, name, question in plain.CHECKS:
    stage = STAGE_BY_ID[sid]
    avail = [o for o in stage["options"] if o["available"]]
    keys = [o["key"] for o in avail]
    lab = {o["key"]: plain.option_label(sid, o["key"], o["label"]) for o in avail}
    sc = cfg[sid]
    c = st.columns([1.5, 0.7, 3])
    c[0].markdown(f"**{name}**  \n<span class='wb-small'>{question}</span>", unsafe_allow_html=True)
    en = c[1].toggle("On", value=sc["enabled"], key=f"try_{sid}_en")
    with c[2]:
        if stage["mode"] == "multi":
            cur = [t for t in sc["tech"] if t in keys]
            val = st.multiselect("Method", keys, default=cur, format_func=lambda k: lab[k][0],
                                 key=f"try_{sid}_m", disabled=not en, label_visibility="collapsed")
        else:
            cur = sc["tech"] if sc["tech"] in keys else keys[0]
            val = st.selectbox("Method", keys, index=keys.index(cur), format_func=lambda k: lab[k][0],
                               key=f"try_{sid}_m", disabled=not en or len(keys) == 1, label_visibility="collapsed")
            if lab[val][1]:
                st.caption(lab[val][1])
    sc["enabled"], sc["tech"] = en, val
if cfg != ui.get_cfg("b"):
    st.session_state["cfg_b"] = cfg
    B.log("model_switch", "Trial", "; ".join(f"{k}={cfg[k]['tech']}" for k in ("s1", "s2", "s3", "s4", "s5")))

st.divider()
st.markdown("##### What changes for one claim?")
cid = st.selectbox("Claim", ids, index=ids.index(st.session_state.get("claim_id", "CLM-1054"))
                   if st.session_state.get("claim_id", "CLM-1054") in ids else 0, key="try_claim",
                   format_func=lambda c_: f"{c_} · {claims.set_index('claim_id').loc[c_, 'device_claimed']}")
st.session_state["claim_id"] = cid
with st.spinner("Running both setups…"):
    ra, rb = ui.run(cid, ui.get_cfg("a"), overlays=False), ui.run(cid, ui.get_cfg("b"), overlays=False)
ca, cb = st.columns(2)
with ca:
    st.markdown("**Setup in use**")
    ui.plain_summary(ra)
    ui.checks_strip(ra)
with cb:
    st.markdown("**Your trial**")
    ui.plain_summary(rb)
    ui.checks_strip(rb)
diffs = [(n, a[3], b[3]) for a, b, n in zip(plain.check_states(ra), plain.check_states(rb),
                                            [x[1] for x in plain.CHECKS]) if a[3] != b[3]]
if diffs:
    st.info("**What changed:** " + "; ".join(f"{n}: {a} → {b}" for n, a, b in diffs))
elif ra.verdict == rb.verdict:
    st.info("No difference for this claim.")

st.divider()
st.markdown("##### Test it on all test claims")
st.caption(f"Runs both setups on all {len(ids)} test claims, 66 photos built from public images and made-up claims. "
           "About half are fraud of different kinds, so it shows which kinds each setup misses.")
if st.button("▶ Run on all test claims", type="primary"):
    B.preload_media(claims["image_path"].tolist() + claims["receipt_path"].dropna().tolist())
    prog = st.progress(0.0, "Running…")
    res = {s: ui.run_many(ids, ui.get_cfg(s), prog, "Setup in use" if s == "a" else "Your trial") for s in ("a", "b")}
    prog.empty()
    st.session_state["try_batch"] = {s: ui.evaluate_plain(res[s], claims) for s in ("a", "b")}
    B.log("batch_run", "Trial vs setup in use", f"{len(ids)} test claims")

if "try_batch" in st.session_state:
    a, b = st.session_state["try_batch"]["a"], st.session_state["try_batch"]["b"]
    dc, dw = b["caught"] - a["caught"], b["wrong"] - a["wrong"]
    df_, dfw = b["firm"] - a["firm"], b["firm_wrong"] - a["firm_wrong"]
    if (dc, dw, df_, dfw) == (0, 0, 0, 0):
        st.success("Same result: the trial makes the same calls on every test claim.")
    else:
        st.success(f"Compared with the setup in use, your trial: flags {abs(dc)} {'more' if dc >= 0 else 'fewer'} fraud "
                   f"claims, makes {abs(df_)} {'more' if df_ >= 0 else 'fewer'} firm “likely fraud” calls on real fraud, "
                   f"and flags {abs(dw)} {'more' if dw >= 0 else 'fewer'} genuine claims.")
    m = st.columns(4)
    m[0].metric("Fraud flagged", f"{b['caught']} of {b['fraud_n']}", f"{dc:+d} vs setup in use",
                help="Sent to a person for any reason: review or investigation.")
    m[1].metric("Firm “likely fraud” calls", f"{b['firm']} of {b['fraud_n']}", f"{df_:+d} vs setup in use",
                help="Fraud claims sent straight to an investigator rather than just flagged for a look.")
    m[2].metric("Genuine claims wrongly flagged", f"{b['wrong']} of {b['genuine_n']}", f"{dw:+d} vs setup in use",
                delta_color="inverse")
    m[3].metric("Time per claim", f"{b['secs']:.1f} s", f"{b['secs'] - a['secs']:+.1f} s", delta_color="inverse")
    st.markdown("**By kind of fraud** (caught of total: setup in use → trial)")
    for k in a["by_type"]:
        (ca_, n), (cb_, _) = a["by_type"][k], b["by_type"][k]
        st.markdown(f"- {k}: {ca_} of {n} → **{cb_} of {n}**")
    st.caption("Small test set: a difference of one or two claims can be luck. Treat this as a first look.")
st.caption("Need thresholds, side-by-side tables or the full list of methods? Open the Expert switchboard.")

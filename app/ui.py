"""Shared UI components: pipeline rail, verdict panel, findings, overlays, cards."""
from __future__ import annotations

import copy
import hashlib
import html
import json

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backend as B
import plain
from image_authenticity.workbench import PRESETS, preset_config, run_claim
from image_authenticity.workbench.catalog import STAGES
from image_authenticity.workbench.checks import read_metadata

SEV_COLOR = {"red": "#D7263D", "amber": "#E89F00", "green": "#2E9E5B", "info": "#8A94A6"}
SEV_ICON = {"red": "●", "amber": "●", "green": "●", "info": "○"}
VERDICT = {"likely_fraud": ("Likely fraud", "#D7263D"), "needs_review": ("Needs review", "#E89F00"),
           "clear": ("Clear", "#2E9E5B")}
STATUS_DOT = {"ran": None, "skipped": "#C5CBD6", "pending": "#5B8DEF", "failed": "#7B2CBF"}
FAMILY_LABEL = {"deepfake": "Deepfake — pixels generated or altered",
                "shallowfake": "Shallowfake — real pixels, false context",
                "document": "Document — receipt doesn't support the claim", "genuine": "Genuine"}

CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
.wb-card {border: 1px solid #E3E7EE; border-radius: 10px; padding: 12px 14px; background: #fff; margin-bottom: 10px;}
.wb-rail-row {display:flex; align-items:flex-start; gap:8px; padding:7px 8px; border-radius:8px; margin-bottom:4px;
  border:1px solid #EEF1F5; background:#FAFBFC;}
.wb-dot {width:11px; height:11px; border-radius:50%; margin-top:5px; flex:0 0 11px;}
.wb-rail-title {font-weight:600; font-size:0.86rem; line-height:1.15;}
.wb-rail-sub {font-size:0.74rem; color:#5A6475; line-height:1.2;}
.wb-verdict {border-radius:12px; padding:14px 16px; color:#fff; margin-bottom:10px;}
.wb-verdict h2 {color:#fff; margin:0; font-size:1.5rem;}
.wb-verdict .sub {opacity:0.95; font-size:0.85rem;}
.wb-reason {border-left:4px solid; padding:6px 10px; margin:6px 0; background:#FAFBFC; border-radius:0 8px 8px 0;
  font-size:0.86rem;}
.wb-chip {display:inline-block; font-size:0.7rem; padding:1px 7px; border-radius:10px; background:#EEF1F5;
  color:#3A4556; margin-right:4px; font-weight:600;}
.wb-kv {display:flex; justify-content:space-between; gap:8px; font-size:0.84rem; padding:3px 0;
  border-bottom:1px dashed #EEF1F5;}
.wb-kv span:first-child {color:#5A6475;}
.wb-small {font-size:0.78rem; color:#5A6475;}
.wb-banner {border-radius:12px; padding:16px 20px; color:#fff; margin:4px 0 12px 0;}
.wb-banner .t {font-size:0.85rem; font-weight:700; letter-spacing:0.04em; text-transform:uppercase; opacity:0.95;}
.wb-banner .h {font-size:1.35rem; font-weight:600; line-height:1.3; margin:4px 0 6px 0;}
.wb-banner .s {font-size:0.88rem; opacity:0.95;}
.wb-strip {display:flex; gap:8px; margin:8px 0 4px 0; flex-wrap:wrap;}
.wb-check {flex:1 1 140px; border:1px solid #E3E7EE; border-radius:10px; padding:8px 12px; background:#fff;}
.wb-check .n {font-size:0.8rem; color:#5A6475;}
.wb-check .w {font-size:1rem; font-weight:600;}
.wb-reasoncard {border:1px solid #E3E7EE; border-left:6px solid; border-radius:10px; padding:10px 14px;
  background:#fff; margin-bottom:6px;}
.wb-reasoncard .k {font-size:0.78rem; font-weight:700; color:#5A6475; text-transform:uppercase; letter-spacing:0.04em;}
.wb-reasoncard .x {font-size:0.95rem; margin-top:2px;}
.wb-badge {display:inline-block; padding:2px 9px; border-radius:10px; color:#fff; font-size:0.75rem; font-weight:600;}
</style>
"""


def css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(s):
    return html.escape(str(s or ""))


def badge(text, color):
    return f'<span class="wb-badge" style="background:{color}">{esc(text)}</span>'


def chip(text):
    return f'<span class="wb-chip">{esc(text)}</span>'


# ------------------------------------------------------------------ config + runs
def cfg_key(cfg) -> str:
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def get_cfg(slot="a"):
    k = f"cfg_{slot}"
    if k not in st.session_state:
        st.session_state[k] = preset_config("Production" if slot == "a" else "Challenger A")
        st.session_state[f"{k}_name"] = "Production" if slot == "a" else "Challenger A"
    return st.session_state[k]


def set_cfg(slot, cfg, name):
    st.session_state[f"cfg_{slot}"] = copy.deepcopy(cfg)
    st.session_state[f"cfg_{slot}_name"] = name


def cfg_name(slot="a"):
    get_cfg(slot)
    name = st.session_state[f"cfg_{slot}_name"]
    if name in PRESETS and cfg_key(PRESETS[name]) != cfg_key(get_cfg(slot)):
        return f"{name} (modified)"
    return name


def claim_row(claim_id) -> dict:
    c = B.claims().set_index("claim_id").loc[claim_id].to_dict()
    c = {k: (None if isinstance(v, float) and v != v else v) for k, v in c.items()}  # NaN -> None
    c["claim_id"] = claim_id
    c["claim_date"] = str(c["claim_date"])[:10]
    for k in ("incident_date", "purchase_date"):
        c[k] = str(c[k])[:10]
    c["claim_value"] = float(c["claim_value"])
    return c


def run(claim_id, cfg, overlays=True):
    live = st.session_state.get("live_models", True)
    key = (claim_id, cfg_key(cfg), overlays, live)
    store = st.session_state.setdefault("_runs", {})
    if key not in store:
        c = claim_row(claim_id)
        rec = B.media(c["receipt_path"]) if c.get("receipt_path") else None
        store[key] = run_claim(c, B.media(c["image_path"]), rec, cfg, B.context(live), want_overlays=overlays)
    return store[key]


def run_many(claim_ids, cfg, progress=None, label=""):
    """Gold-set batch: same engine, claims in parallel threads (OpenCV releases the GIL).
    Explanations do not affect metrics, so batch evaluation uses the template explainer."""
    import concurrent.futures as cf
    cfg = copy.deepcopy(cfg)
    cfg["s6"]["params"]["explainer"] = "template"
    live = st.session_state.get("live_models", True)
    ctx = B.context(live)
    store = st.session_state.setdefault("_runs", {})
    rows = {cid: claim_row(cid) for cid in claim_ids}
    B.preload_media([r["image_path"] for r in rows.values()] + [r["receipt_path"] for r in rows.values()])
    for e in ctx.index.entries:                     # warm ORB / SSCD features once, before threading
        if cfg["s2"]["tech"] == "orb":
            ctx.index.orb_for(e, ctx.index_loader)
        elif cfg["s2"]["tech"] == "sscd":
            ctx.index.sscd_for(e, ctx.index_loader)
    k = cfg_key(cfg)

    def one(cid):
        key = (cid, k, False, live)
        if key in store:
            return cid, store[key]
        c = rows[cid]
        rec = B.media(c["receipt_path"]) if c.get("receipt_path") else None
        return cid, run_claim(c, B.media(c["image_path"]), rec, cfg, ctx)

    out = {}
    with cf.ThreadPoolExecutor(4) as ex:
        for i, (cid, r) in enumerate(ex.map(one, claim_ids)):
            out[cid] = r; store[(cid, k, False, live)] = r
            if progress is not None:
                progress.progress((i + 1) / len(claim_ids), f"{label} · {cid}")
    return out


# ------------------------------------------------------------------ pipeline rail
def pipeline_rail(r, cfg, compact=False):
    st.markdown("**Pipeline**", help="Eight switchable stages. Dot colour = worst finding. Click a stage below "
                                     "the image for its evidence; switch technologies in the Pipeline Lab.")
    for s in r.stages:
        color = STATUS_DOT.get(s.status) or SEV_COLOR[s.severity]
        tech = ", ".join(s.techs) if s.status != "skipped" else "skipped"
        sc = "" if s.score is None or s.sid == "s6" else f" · score {s.score:.2f}"
        lat = "" if s.status == "skipped" else f" · {s.latency_ms / 1000:.1f}s"
        status = {"pending": " · pending", "failed": " · failed"}.get(s.status, "")
        st.markdown(f'<div class="wb-rail-row"><div class="wb-dot" style="background:{color}"></div><div>'
                    f'<div class="wb-rail-title">{s.num} · {esc(s.name)}</div>'
                    f'<div class="wb-rail-sub">{esc(tech)}{sc}{lat}{status}</div></div></div>',
                    unsafe_allow_html=True)


# ------------------------------------------------------------------ verdict
def verdict_panel(r, claim=None):
    label, color = VERDICT[r.verdict]
    fam = " · ".join(FAMILY_LABEL[f].split(" — ")[0] for f in r.families) or ("No fraud indicated"
                                                                             if r.verdict == "clear" else "—")
    st.markdown(f'<div class="wb-verdict" style="background:{color}"><h2>{label}</h2>'
                f'<div class="sub">Confidence {r.confidence:.0%} · {esc(fam)}</div>'
                f'<div class="sub">→ {esc(r.route)}</div></div>', unsafe_allow_html=True)
    st.markdown(f"**Why** <span class='wb-small'>({esc(r.explainer.replace('vlm:', '').replace('template', 'template explainer'))})</span>",
                unsafe_allow_html=True)
    st.markdown(f'<div class="wb-card" style="font-size:0.88rem">{esc(r.explanation)}</div>', unsafe_allow_html=True)
    st.markdown("**Reasons** — each linked to the check that produced it")
    if not r.reasons:
        st.markdown('<div class="wb-reason" style="border-color:#2E9E5B">No check crossed its threshold.</div>',
                    unsafe_allow_html=True)
    for f in r.reasons[:7]:
        st.markdown(f'<div class="wb-reason" style="border-color:{SEV_COLOR[f.severity]}">'
                    f'{chip("Stage " + f.stage[1:])}{chip(f.check)}<br>{esc(f.text)}</div>', unsafe_allow_html=True)
    st.caption("The rules engine decides; the explainer only words the decision. Nothing is auto-denied.")


def findings_list(findings):
    for f in findings:
        st.markdown(f'<div class="wb-reason" style="border-color:{SEV_COLOR[f.severity]}">{chip(f.check)} '
                    f'{esc(f.text)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ image + overlays
def _heat_blend(img, heat, alpha):
    h = cv2.resize(heat.astype(np.float32), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    h = np.clip(h, 0, 1)
    color = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    a = (alpha * np.clip(h * 1.6, 0.15, 1.0))[..., None]
    return (img * (1 - a) + color * a).astype(np.uint8)


def overlay_image(img, r, mode, alpha, mask_path=None):
    out = img.copy()
    if mode in r.overlays and mode != "Copy-move pairs":
        out = _heat_blend(out, r.overlays[mode], alpha)
    elif mode == "Copy-move pairs":
        pairs = r.overlays.get("Copy-move pairs") or []
        for p, q in pairs[:200]:   # pair coordinates are in full-image pixels
            p, q = tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int))
            cv2.line(out, p, q, (60, 60, 255), 2, cv2.LINE_AA)
            cv2.circle(out, p, 5, (0, 220, 0), -1); cv2.circle(out, q, 5, (0, 220, 0), -1)
    elif mode == "Ground-truth mask (gold set)" and mask_path:
        m = cv2.imdecode(np.frombuffer(B.media(mask_path), np.uint8), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (img.shape[1], img.shape[0])) > 127
        tint = out.copy(); tint[m] = (0.45 * tint[m] + 0.55 * np.array([60, 60, 230])).astype(np.uint8)
        out = tint
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (40, 40, 255), 3)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def overlay_modes(r, claim):
    modes = ["Photo only"] + [k for k in ("ELA heatmap", "Noise map", "JPEG ghost map", "Copy-move pairs")
                              if k in r.overlays]
    if claim.get("mask_path"):
        modes.append("Ground-truth mask (gold set)")
    return modes


# ------------------------------------------------------------------ cards
def metadata_card(claim, r, photo_bytes=None):
    meta = read_metadata(photo_bytes if photo_bytes is not None else B.media(claim["image_path"]))
    s1 = r.stage("s1")
    rows = []
    def row(k, v, sev="info"):
        rows.append(f'<div class="wb-kv"><span>{esc(k)}</span><span><span style="color:{SEV_COLOR[sev]}">'
                    f'{SEV_ICON[sev]}</span> {esc(v)}</span></div>')
    taken = meta.get("taken")
    date_sev = "info"
    for f in s1.findings:
        if "taken" in f.text or "dated" in f.text:
            date_sev = f.severity
    row("Capture date (EXIF)", taken.strftime("%d %b %Y %H:%M") if taken else "missing",
        date_sev if taken else ("amber" if not meta.get("make") else "info"))
    row("Capturing device", f"{meta.get('make') or ''} {meta.get('model') or ''}".strip() or "missing",
        "green" if meta.get("make") else "amber")
    sw = (meta.get("software") or "").strip("\x00")
    row("Editing software", sw or "none recorded", "amber" if sw else "green")
    row("C2PA manifest", "present (signature check pending legal)" if meta.get("c2pa") else "not present", "info")
    row("IPTC AI-generated tag", "PRESENT — declares AI origin" if meta.get("ai_provenance") else "not present",
        "red" if meta.get("ai_provenance") else "green")
    row("Format · resolution", f"{meta.get('format')} · {meta.get('width')}×{meta.get('height')}")
    row("IP / share indicators", "not captured in the prototype", "info")
    st.markdown('<div class="wb-card"><b>Metadata</b>' + "".join(rows) + "</div>", unsafe_allow_html=True)


def receipt_card(claim, r, receipt_bytes=None):
    s5 = r.stage("s5")
    if not claim.get("receipt_path") and receipt_bytes is None:
        st.markdown('<div class="wb-card"><b>Receipt</b><div class="wb-small">No receipt submitted.</div></div>',
                    unsafe_allow_html=True)
        return
    c1, c2 = st.columns([1, 1.6])
    with c1:
        rimg = (cv2.imdecode(np.frombuffer(receipt_bytes, np.uint8), cv2.IMREAD_COLOR) if receipt_bytes is not None
                else B.image(claim["receipt_path"]))
        st.image(cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB), caption="Receipt as submitted")
    with c2:
        st.markdown(f"**Receipt check** <span class='wb-small'>({esc(', '.join(s5.techs))})</span>",
                    unsafe_allow_html=True)
        if s5.status == "skipped":
            st.caption("Stage 5 is switched off in this configuration.")
        findings_list(s5.findings)
        st.markdown(f'<div class="wb-kv"><span>Claimed value</span><span>${claim["claim_value"]:,.2f}</span></div>'
                    f'<div class="wb-kv"><span>Claimed purchase date</span><span>{esc(claim["purchase_date"])}</span></div>'
                    f'<div class="wb-kv"><span>Claimed device</span><span>{esc(claim["device_claimed"])}</span></div>',
                    unsafe_allow_html=True)


def timeline(r, height=230):
    names = [f"{s.num} · {s.name}" for s in r.stages][::-1]
    vals = [s.latency_ms / 1000 for s in r.stages][::-1]
    colors = [STATUS_DOT.get(s.status) or SEV_COLOR[s.severity] for s in r.stages][::-1]
    text = [("skipped" if s.status == "skipped" else f"{s.latency_ms / 1000:.2f}s"
             + (" · cached slow path" if s.cached else "") + (" · pending" if s.status == "pending" else ""))
            for s in r.stages][::-1]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colors, text=text,
                           textposition="outside", cliponaxis=False))
    fig.update_layout(height=height, margin=dict(l=0, r=60, t=10, b=10), xaxis_title="seconds",
                      plot_bgcolor="#fff", xaxis=dict(gridcolor="#EEF1F5"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Fast path (stages 0–3, synchronous): **{r.sync_latency_ms / 1000:.2f}s** · "
               f"Slow path (stages 4–6, async): **{r.async_latency_ms / 1000:.1f}s** · "
               f"{r.tokens:,} model tokens. The 30 s gateway limit only applies to the synchronous path.")


# ------------------------------------------------------------------ plain-language components
def banner(r, claim=None):
    label, color = plain.verdict_title(r)
    more = plain.more_signs(r)
    sub = (f"{more} other sign{'s' if more != 1 else ''} below · " if more and r.verdict != "clear" else "") \
        + plain.next_step(r)
    fam = plain.families_plain(r) if r.verdict != "clear" else ""
    tag = "nothing flagged" if r.verdict == "clear" else f"confidence {plain.confidence_word(r.confidence).lower()}"
    st.markdown(f'<div class="wb-banner" style="background:{color}"><div class="t">{esc(label)} · {tag}'
                f'{" · " + esc(fam) if fam else ""}</div>'
                f'<div class="h">{esc(plain.headline(r))}</div><div class="s">{esc(sub)}</div></div>',
                unsafe_allow_html=True)


def plain_summary(r):
    """Compact verdict + reasons for side-by-side views."""
    label, color = plain.verdict_title(r)
    st.markdown(f'<div class="wb-banner" style="background:{color};padding:10px 14px"><div class="t">{esc(label)} · '
                f'{"nothing flagged" if r.verdict == "clear" else "confidence " + plain.confidence_word(r.confidence).lower()}</div>'
                f'<div class="s" style="font-size:0.9rem">{esc(plain.headline(r))}</div></div>', unsafe_allow_html=True)


def checks_strip(r):
    cells = []
    for _sid, name, sym, word, sev in plain.check_states(r):
        color = SEV_COLOR.get(sev) or STATUS_DOT.get(sev) or "#8A94A6"
        cells.append(f'<div class="wb-check"><div class="n">{esc(name)}</div>'
                     f'<div class="w" style="color:{color}">{sym} {esc(word)}</div></div>')
    st.markdown('<div class="wb-strip">' + "".join(cells) + "</div>", unsafe_allow_html=True)


def evaluate_plain(results, claims_df):
    """Investigator-level numbers for a batch: fraud caught, genuine wrongly flagged, time, by fraud type."""
    df = pd.DataFrame([{"claim_id": c, "verdict": r.verdict, "sync": r.sync_latency_ms}
                       for c, r in results.items()]).merge(claims_df[["claim_id", "family"]], on="claim_id")
    df["flagged"] = df.verdict != "clear"
    fraud, gen = df[df.family != "genuine"], df[df.family == "genuine"]
    firm = df.verdict == "likely_fraud"
    return {"caught": int(fraud.flagged.sum()), "fraud_n": len(fraud),
            "firm": int(firm[fraud.index].sum()), "firm_wrong": int(firm[gen.index].sum()),
            "wrong": int(gen.flagged.sum()), "genuine_n": len(gen),
            "secs": df.sync.median() / 1000,
            "by_type": {FAMILY_PLAIN_SHORT[f]: (int(d.flagged.sum()), len(d))
                        for f, d in df[df.family != "genuine"].groupby("family")}}


FAMILY_PLAIN_SHORT = {"shallowfake": "Reused or misrepresented photos", "deepfake": "Edited or AI-made photos",
                      "document": "Receipts that don't match"}

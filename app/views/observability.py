"""Screen 3 — Observability: KPIs, signal heatmap, latency heatmap, device × model, leaderboard."""
import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import backend as B
import ui
from image_authenticity.workbench import PRESETS
from image_authenticity.workbench.checks import brand_of

ui.css()
st.caption("Results from running each setup on all test claims. Same Gold tables the pipeline writes, no separate copy.")

claims = B.claims(); runs = B.runs(); stages = B.stage_results(); ts = B.tech_scores(); labels = B.labels()
cfgs = [c for c in PRESETS if c in set(runs.config_name)]     # presets with a stored gold-set batch run
NICE = {"Production": "Current setup"}
cfg = st.selectbox("Setup", cfgs, index=0, format_func=lambda c: NICE.get(c, c),
                   help="Each setup was run on all test claims.")
R = runs[runs.config_name == cfg].merge(claims[["claim_id", "family", "fraud_class", "device_claimed",
                                                "claim_date", "claim_value"]], on="claim_id")

# ------------------------------------------------------------------ three numbers
_fraud = R[R.family != "genuine"]
h = st.columns(3)
h[0].metric("Fraud caught", f"{(_fraud.verdict != 'clear').mean():.0%}",
            f"{int((_fraud.verdict != 'clear').sum())} of {len(_fraud)} fraud claims", delta_color="off")
h[1].metric("Genuine claims wrongly flagged", f"{(R[R.family == 'genuine'].verdict != 'clear').mean():.0%}",
            f"{int((R[R.family == 'genuine'].verdict != 'clear').sum())} of {int((R.family == 'genuine').sum())} genuine",
            delta_color="off")
h[2].metric("Time per claim", f"{R.sync_latency_ms.astype(float).median() / 1000:.1f} s",
            "median, photo checks", delta_color="off")
_t = (_fraud.assign(caught=_fraud.verdict != "clear").groupby("family").caught.agg(["sum", "count"]))
st.markdown("**By kind of fraud:** " + " · ".join(
    f"{ui.FAMILY_PLAIN_SHORT[f]} {int(r_['sum'])} of {int(r_['count'])}" for f, r_ in _t.iterrows()))
if not st.toggle("Show the detailed charts and measures", value=False):
    st.stop()
st.divider()

# ------------------------------------------------------------------ KPIs
photo_genuine = R[R.family.isin(["genuine"])]
flag = R.verdict != "clear"
labeled = set(labels.claim_id)
queue = R[flag & ~R.claim_id.isin(labeled)]
over = labels[(labels.label == "confirm_genuine") & (labels.model_verdict != "clear")]
age_days = (pd.Timestamp("2026-10-27") - pd.to_datetime(queue.claim_date)).dt.days
k = st.columns(6)
k[0].metric("Verdict mix", f"{(R.verdict == 'likely_fraud').sum()} / {(R.verdict == 'needs_review').sum()} / "
                           f"{(R.verdict == 'clear').sum()}", help="likely fraud / needs review / clear")
k[1].metric("Genuine flagged per 1,000", f"{(photo_genuine.verdict != 'clear').mean() * 1000:,.0f}",
            help="Analyst-load metric on the gold set (genuine claims sent to review or investigation).")
k[2].metric("Reviewer overturn rate", f"{len(over) / max(1, len(labels)):.0%}",
            help="Share of human labels that overturned a model flag.")
k[3].metric("Review queue", f"{len(queue)} claims", f"median age {age_days.median():.0f} d" if len(queue) else None,
            delta_color="off")
idx = B.reuse_index()
k[4].metric("Reuse index freshness", f"{len(idx.entries)}/{len(claims)} indexed",
            f"built {max(0, int((dt.datetime.now() - idx.built_at).total_seconds() // 60))} min ago", delta_color="off",
            help="The index is rebuilt from the claims Delta table — it cannot silently go stale.")
price = st.session_state.get("usd_per_mtok", 3.0)
k[5].metric("Model tokens / claim", f"{R.tokens.mean():,.0f}", f"≈ ${R.tokens.mean() * price / 1e6:.4f} at "
                                                                 f"${price}/M tok", delta_color="off")

tabs = st.tabs(["Signal heatmap", "Latency heatmap", "Device × detector", "Model leaderboard"])

PRETTY = (ts[["sid", "technique", "label"]].drop_duplicates()
          .assign(col=lambda d: d.sid.str[1] + " · " + d.label.str.replace(r" \(.*\)$", "", regex=True).str[:34]))
order = PRETTY.sort_values(["sid", "technique"])

with tabs[0]:
    st.caption("Rows = claims, columns = every check run alone on the gold set. Green → red by score. A column red "
               "for everything is a broken check; a row red everywhere is confident fraud; a row red in one column is "
               "a case for human review.")
    fams = st.multiselect("Families", ["genuine", "document", "shallowfake", "deepfake"],
                          default=["genuine", "document", "shallowfake", "deepfake"])
    t = ts[ts.family.isin(fams)].merge(order[["sid", "technique", "col"]], on=["sid", "technique"])
    t["score"] = pd.to_numeric(t.score).fillna(0)
    piv = t.pivot_table(index="claim_id", columns="col", values="score", aggfunc="max")
    meta = claims.set_index("claim_id").loc[piv.index]
    fam_rank = {"genuine": 0, "document": 1, "shallowfake": 2, "deepfake": 3}
    piv = piv.loc[sorted(piv.index, key=lambda c: (fam_rank[meta.loc[c, "family"]], meta.loc[c, "fraud_class"], c))]
    piv = piv[[c for c in order.col if c in piv.columns]]
    ylab = [f"{c} · {claims.set_index('claim_id').loc[c, 'fraud_class']}" for c in piv.index]
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=ylab, zmin=0, zmax=1,
                               colorscale=[[0, "#2E9E5B"], [0.5, "#F2C14E"], [1, "#D7263D"]],
                               hovertemplate="%{y}<br>%{x}<br>score %{z:.2f}<extra></extra>"))
    fig.update_layout(height=max(500, 15 * len(piv)), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(side="top", tickangle=-40), yaxis=dict(autorange="reversed", tickfont=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.caption("Rows = pipeline stages, columns = week of claim. Cell = p95 latency. Fast-path cells approaching the "
               "synchronous budget would glow — this is the 30-second problem made visible.")
    S = stages[stages.config_name == cfg].merge(claims[["claim_id", "claim_date"]], on="claim_id")
    S = S[S.status != "skipped"]
    S["week"] = pd.to_datetime(S.claim_date).dt.to_period("W").dt.start_time.dt.strftime("%d %b")
    p = S.groupby(["stage_num", "stage_name", "week"]).latency_ms.quantile(0.95).reset_index()
    p["stage"] = p.stage_num.astype(int).astype(str) + " · " + p.stage_name
    pv = p.pivot(index="stage", columns="week", values="latency_ms") / 1000
    weeks = sorted(pv.columns, key=lambda w: dt.datetime.strptime(w + " 2026", "%d %b %Y"))
    pv = pv[weeks]
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, text=np.round(pv.values, 2),
                               texttemplate="%{text}s", colorscale="YlOrRd", zmin=0,
                               hovertemplate="%{y}<br>week of %{x}<br>p95 %{z:.2f}s<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    fast = S[S.sid.isin(["s0", "s1", "s2", "s3"])].groupby("claim_id").latency_ms.sum() / 1000
    st.caption(f"Fast path per claim: median {fast.median():.2f}s, p95 {fast.quantile(.95):.2f}s, max {fast.max():.2f}s "
               f"— against a 10 s self-imposed budget (alert at 80%). Slow-path latencies are the recorded "
               f"API times of the cached model calls.")

TARGET = {"s1": {"shallowfake", "deepfake"}, "s2": {"shallowfake"}, "s3": {"deepfake"}, "s4": {"deepfake"},
          "s5": {"document"}}


def target_of(sid, tech):
    if tech == "recapture":
        return {"shallowfake"}
    return TARGET[sid]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


lb = []
champ = {sid: (PRESETS["Production"][sid]["tech"]) for sid in ("s1", "s2", "s3", "s4", "s5")}
for (sid, tech, label), g in ts.groupby(["sid", "technique", "label"]):
    g = g.assign(score=pd.to_numeric(g.score).fillna(0))
    tgt = target_of(sid, tech)
    neg_fams = {"genuine"} if sid == "s5" else {"genuine", "document"}
    if sid == "s5":
        g = g[g.claim_id.isin(claims[claims.receipt_path.notna()].claim_id)]
    pos, neg = g[g.family.isin(tgt)], g[g.family.isin(neg_fams)]
    c = champ[sid]
    is_champ = tech in c if isinstance(c, list) else tech == c
    lb.append({"Stage": int(sid[1]), "Technology": label, "Badge": "🏆 champion" if is_champ else "challenger",
               "Targets": "+".join(sorted(tgt)), "AUC": auc(pos.score, neg.score), "Recall": pos.fired.mean(),
               "False-positive rate": neg.fired.mean(), "FP per 1,000 genuine": neg.fired.mean() * 1000,
               "Median latency (s)": g.latency_ms.median() / 1000, "Tokens / claim": g.tokens.mean(),
               "_tech": tech, "_sid": sid})
LB = pd.DataFrame(lb).sort_values(["Stage", "AUC"], ascending=[True, False])

with tabs[2]:
    st.caption("Rows = claimed device brand, columns = detectors. Cell = accuracy on the gold set for that "
               "detector's target fraud family. Answers “does this detector work on this device?”")
    t = ts.merge(claims[["claim_id", "device_claimed"]], on="claim_id")
    t["brand"] = t.device_claimed.map(lambda d: (brand_of(d) or "other").title())
    acc = []
    for (sid, tech, label, brand), g in t.groupby(["sid", "technique", "label", "brand"]):
        tgt = target_of(sid, tech)
        g = g[g.family.isin(tgt | {"genuine", "document"} if sid != "s5" else tgt | {"genuine"})]
        if sid == "s5":
            g = g[g.claim_id.isin(claims[claims.receipt_path.notna()].claim_id)]
        if len(g) < 2:
            continue
        acc.append({"brand": brand, "col": f"{sid[1]} · {label[:30]}", "acc": (g.fired == g.family.isin(tgt)).mean(),
                    "n": len(g)})
    A = pd.DataFrame(acc)
    keep = [c for c in A.col.unique() if not c.startswith("5")]
    pv = A[A.col.isin(keep)].pivot(index="brand", columns="col", values="acc")
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, zmin=0.3, zmax=1,
                               colorscale=[[0, "#D7263D"], [0.5, "#F2C14E"], [1, "#2E9E5B"]],
                               text=np.round(pv.values * 100), texttemplate="%{text:.0f}%",
                               hovertemplate="%{y}<br>%{x}<br>accuracy %{z:.0%}<extra></extra>"))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(side="top", tickangle=-40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Gold set v0 is small (66 claims, mostly phones) — cells rest on a handful of claims each.")

with tabs[3]:
    st.caption("Every technology ranked on the same gold set, alone. AUC is threshold-free (how well scores separate "
               "fraud from genuine); recall and false-positive rate are at the calibrated threshold. A challenger "
               "becomes champion only by beating the champion here.")
    show = LB.drop(columns=["_tech", "_sid"]).copy()
    st.dataframe(show.style.format({"AUC": "{:.2f}", "Recall": "{:.0%}", "False-positive rate": "{:.0%}",
                                    "FP per 1,000 genuine": "{:,.0f}", "Median latency (s)": "{:.2f}",
                                    "Tokens / claim": "{:,.0f}"}, na_rep="—")
                 .background_gradient(subset=["AUC"], cmap="RdYlGn", vmin=0.4, vmax=1.0),
                 hide_index=True, use_container_width=True, height=820)
    st.caption("OCR row: accuracy of the receipt cross-check on claims that carry a receipt. Positives for photo "
               "checks exclude receipt-only (document) fraud, whose photos are genuine.")
    st.number_input("Assumed blended model price ($ per 1M tokens) for cost estimates", 0.0, 100.0,
                    float(st.session_state.get("usd_per_mtok", 3.0)), 0.5, key="usd_per_mtok")

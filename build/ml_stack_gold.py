"""Option C: pool EVERY check's score into one learned fraud / not-fraud answer, and compare it with the hand-written
rules engine on gold set v0 (66 claims: 52 fraud, 14 genuine).

Pre-specified (nothing tuned on these results):
  features   the 24 stored per-technique scores (tech_scores.parquet; receipt checks with no receipt = 0.03, i.e. 'nothing found')
             + SSCD cosine to the closest EARLIER claim (the engine's own 'prior claims only' rule)
  models     logistic regression (standardised, L2, C=0.3, balanced) and random forest (200 trees, min leaf 2, balanced)
  evaluation repeated stratified 5-fold CV, claims that share a source photo ALWAYS on the same side of a split (grouped by source_id)
  baseline   the rules engine's own confidence (Production preset; noisy-OR over fired findings, 0 for 'clear')
Caveats built into the design: rules thresholds were calibrated on these same genuine photos (favours the rules);
only 14 genuine claims (one false alarm = 7%); repeat-to-repeat spread reflects split randomness, not sampling uncertainty.

    ml_env/bin/python -I build/ml_stack_gold.py
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(ROOT, "wb_data", "tables") + "/"
c = pd.read_parquet(T + "claims.parquet").reset_index(drop=True)
y = (c.family != "genuine").astype(int).values
groups, fclass, family = c.source_id.values, c.fraud_class.values, c.family.values

# ---------------------------------------------------------------- features
W = (pd.read_parquet(T + "tech_scores.parquet").pivot_table(index="claim_id", columns="technique", values="score", aggfunc="first")
     .reindex(c.claim_id).fillna(0.03))
sv = {r.claim_id: np.array(json.loads(r.sscd), float) for r in pd.read_parquet(T + "sscd_index.parquet").itertuples()}
order = c.sort_values(["claim_date", "claim_id"]).claim_id.tolist()
W["sscd_maxcos_prior"] = [max([sv[cid] @ sv[p] for p in order[:order.index(cid)]], default=0.0) for cid in c.claim_id]
ALL = list(W.columns)
VARIANTS = {"all signals": ALL,
            "without metadata checks (s1)": [k for k in ALL if not k.startswith("exif")],
            "without vision models": [k for k in ALL if not k.startswith("vlm:")]}

# ---------------------------------------------------------------- helpers
def make(kind, seed):
    if kind == "logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))
    return RandomForestClassifier(200, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=1)


def oof(X, kind, seed, drop_class=None):
    s = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        if drop_class is not None:
            tr = tr[fclass[tr] != drop_class]                  # the model never sees this fraud type
        s[te] = make(kind, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return s


def recall_at(s, k, mask=None):
    """Recall on fraud when at most k genuine claims score higher (k=1 -> 1/14 = 7% false alarms)."""
    t = np.sort(s[y == 0])[::-1][k]
    sel = (y == 1) if mask is None else mask
    return float((s[sel] > t).mean())


KS = (0, 1, 2, 6)
def summarise(scores_list):
    out = {"AUC": [roc_auc_score(y, s) for s in scores_list]}
    for k in KS:
        out[f"recall @ {k}/14 false alarms"] = [recall_at(s, k) for s in scores_list]
    return {k: (np.mean(v), np.std(v)) for k, v in out.items()}


md = ["# Option C: one learned score from all signals vs the rules engine (gold set v0)\n",
      "66 claims (52 fraud, 14 genuine). Cross-validated, claims from the same source photo never split across train/test. "
      "Rules engine = its own confidence score (thresholds were calibrated on these genuine photos, so it is favoured).\n"]

# ---------------------------------------------------------------- baseline: rules engine
r = pd.read_parquet(T + "runs.parquet")
rules = r[r.config_name == "Production"].set_index("claim_id").reindex(c.claim_id).confidence.astype(float).values
base = {"AUC": roc_auc_score(y, rules), **{f"recall @ {k}/14 false alarms": recall_at(rules, k) for k in KS}}
md.append("## Result\n\n| Method | AUC | recall @ 0 false alarms | @ 1 (7%) | @ 2 (14%) | @ 6 (43%) |\n|---|---|---|---|---|---|")
md.append("| Rules engine (hand-written, Production) | %.2f | %.0f%% | %.0f%% | %.0f%% | %.0f%% |" % (
    base["AUC"], *(100 * base[f"recall @ {k}/14 false alarms"] for k in KS)))
REPS = 30
res = {}
for vname, cols in VARIANTS.items():
    X = W[cols].values
    for kind in ("logistic regression", "random forest"):
        sl = [oof(X, kind, rep) for rep in range(REPS)]
        res[(vname, kind)] = (sl, summarise(sl))
        m = res[(vname, kind)][1]
        md.append(f"| Learned, {kind}, {vname} | {m['AUC'][0]:.2f} ± {m['AUC'][1]:.2f} | " + " | ".join(
            f"{100 * m[f'recall @ {k}/14 false alarms'][0]:.0f}% ± {100 * m[f'recall @ {k}/14 false alarms'][1]:.0f}" for k in KS) + " |")
md.append("\n± is the spread over %d repeats of the CV split (not sampling uncertainty: with 14 genuine claims, one claim moves a rate by 7 points).\n" % REPS)

# ---------------------------------------------------------------- per fraud class: seen vs never seen in training
Xa = W[ALL].values
md.append("## By fraud type: recall at 1/14 false alarms, when the model HAS seen that type vs has NEVER seen it\n")
md.append("| Fraud type | n | Rules engine | Logistic, seen | Logistic, unseen | Forest, seen | Forest, unseen |\n|---|---|---|---|---|---|---|")
classes = [k for k in pd.unique(fclass) if k != "genuine"]
seen = {kind: res[("all signals", kind)][0] for kind in ("logistic regression", "random forest")}
UREPS = 8
unseen = {kind: {k: [] for k in classes} for kind in seen}
for kind in seen:
    for k in classes:
        for rep in range(UREPS):
            s = oof(Xa, kind, rep, drop_class=k)
            unseen[kind][k].append(recall_at(s, 1, mask=(fclass == k)))
tot = {"rules": [], **{f"{kd}_{w}": [] for kd in seen for w in ("seen", "unseen")}}
for k in classes:
    msk = fclass == k
    row = [recall_at(rules, 1, msk)]
    for kind in seen:
        row += [np.mean([recall_at(s, 1, msk) for s in seen[kind]]), np.mean(unseen[kind][k])]
    md.append(f"| {k} | {msk.sum()} | " + " | ".join(f"{100 * v:.0f}%" for v in [row[0], row[1], row[2], row[3], row[4]]) + " |")
    tot["rules"].append((row[0], msk.sum()))
    for i, kd in enumerate(seen):
        tot[f"{kd}_seen"].append((row[1 + 2 * i], msk.sum())); tot[f"{kd}_unseen"].append((row[2 + 2 * i], msk.sum()))
avg = lambda v: sum(a * n for a, n in v) / sum(n for _, n in v)
md.append("| **all fraud (claim-weighted)** | 52 | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %.0f%% |" % (
    100 * avg(tot["rules"]), 100 * avg(tot["logistic regression_seen"]), 100 * avg(tot["logistic regression_unseen"]),
    100 * avg(tot["random forest_seen"]), 100 * avg(tot["random forest_unseen"])))

# ---------------------------------------------------------------- what the models lean on
lr = make("logistic regression", 0).fit(Xa, y)
co = pd.Series(lr[-1].coef_[0], index=ALL).sort_values(key=abs, ascending=False)
rf = make("random forest", 0).fit(Xa, y)
im = pd.Series(rf.feature_importances_, index=ALL).sort_values(ascending=False)
md.append("\n## What they lean on (fit on all 66 claims; for reading only)\n")
md.append("Logistic, largest standardised weights: " + ", ".join(f"{k} {v:+.2f}" for k, v in co.head(8).items()))
md.append("\nForest, largest importances: " + ", ".join(f"{k} {v:.2f}" for k, v in im.head(8).items()))
open(os.path.join(ROOT, "wb_data", "ml_exp", "stack_results.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))

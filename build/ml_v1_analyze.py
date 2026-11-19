"""Analysis of the v1 held-out test. Pre-specified: AUC (AI vs real) with stratified bootstrap 95% CI per detector and
view; operating point at the detector's DEFAULT threshold (CLIP LLR>0, SPAI>0.5, app detectors' amber thresholds from the
catalog); recall by generator group; confound probes; reuse recall / false hits at frozen thresholds.

    ml_env/bin/python -I build/ml_v1_analyze.py
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from image_authenticity.workbench.catalog import STAGE_BY_ID   # noqa: E402

OUT = os.path.join(ROOT, "wb_data", "ml_v1")
items = json.load(open(os.path.join(OUT, "items.json")))
ids = [i["id"] for i in items]
y = np.array([i["label"] for i in items])
grp = np.array([i["group"] for i in items])
rng = np.random.default_rng(0)


def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None, :]).sum() + .5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def boot(s, y, B=1000):
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    v = []
    for _ in range(B):
        idx = np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]
        v.append(auc(s[idx], y[idx]))
    return np.percentile(v, [2.5, 97.5])


def spai(variant):
    p = os.path.join(OUT, f"spai_{variant}", f"{variant}.csv")        # SPAI names the file after the input folder
    if not os.path.exists(p):
        return None
    d = {r["image"].rsplit(".", 1)[0]: float(r["spai"]) for r in csv.DictReader(open(p))}
    return np.array([d[i] for i in ids])


amber = {k: v[0] for k, v in STAGE_BY_ID["s3"]["params"]["thresholds"].items()}
amber.update({k: v[0] for k, v in STAGE_BY_ID["s4"]["params"]["thresholds"].items()})
NAME = {"clipdet_latent10k_plus": "CLIP detector (Cozzolino, latent10k+)", "clipdet_latent10k": "CLIP detector (latent10k)",
        "spai": "SPAI", "ela": "ELA", "jpeg_ghost": "JPEG ghost", "noise": "Noise residual", "copy_move_orb": "Copy-move ORB",
        "copy_move_sift": "Copy-move SIFT", "spectral_fft": "Spectral FFT", "spectral_dct": "Spectral DCT",
        "ela_uniformity": "ELA uniformity", "upscale": "Upscale", "recapture": "Recapture",
        "aspect_probe": "PROBE: aspect ratio only", "bytes_per_pixel_probe": "PROBE: bytes per pixel only"}
md = ["# ML vs pixel checks: held-out test v1\n",
      f"{int(y.sum())} AI-made images ({', '.join(f'{g} {int((grp == g).sum())}' for g in sorted(set(grp[y == 1])))}) vs "
      f"{int((y == 0).sum())} real photos ({', '.join(f'{g} {int((grp == g).sum())}' for g in sorted(set(grp[y == 0])))}). "
      "Thresholds frozen from gold set v0.\n"]
table = {}
for variant in ("orig", "social", "norm"):
    f = os.path.join(OUT, f"scores_{variant}.json")
    if not os.path.exists(f):
        continue
    S = json.load(open(f))
    sc = {k: np.array(S[k]) for k in NAME if k in S}
    sp = spai(variant)
    if sp is not None:
        sc["spai"] = sp
    table[variant] = sc
md.append("## AUC, AI vs real (95% bootstrap CI)\n")
md.append("| Detector | " + " | ".join(table) + " |\n|---|" + "---|" * len(table))
for k in NAME:
    cells = []
    for v, sc in table.items():
        if k in sc:
            a, (lo, hi) = auc(sc[k], y), boot(sc[k], y)
            cells.append(f"{a:.2f} [{lo:.2f}-{hi:.2f}]")
        else:
            cells.append("n/a")
    md.append(f"| {NAME[k]} | " + " | ".join(cells) + " |")
md.append("\n## At the default threshold (recall on AI / false alarms on real)\n")
md.append("| Detector | view | recall | false alarms |\n|---|---|---|---|")
DEF = {"clipdet_latent10k_plus": 0.0, "clipdet_latent10k": 0.0, "spai": 0.5, **{k: v for k, v in amber.items()}}
for k, thr in DEF.items():
    for v, sc in table.items():
        if k in sc:
            f = sc[k] > thr if k != "spai" else sc[k] >= thr
            md.append(f"| {NAME.get(k, k)} | {v} | {f[y == 1].mean():.0%} ({int(f[y == 1].sum())}/{int((y == 1).sum())}) | "
                      f"{f[y == 0].mean():.0%} ({int(f[y == 0].sum())}/{int((y == 0).sum())}) |")
md.append("\n## Recall by generator group at default thresholds (norm view if present, else orig)\n")
v = "norm" if "norm" in table else "orig"
gs = sorted(set(grp[y == 1]))
md.append("| Detector | " + " | ".join(gs) + " |\n|---|" + "---|" * len(gs))
for k in ("clipdet_latent10k_plus", "spai", "upscale"):
    if k in table[v]:
        f = table[v][k] >= DEF.get(k, 0.5) if k == "spai" else table[v][k] > DEF.get(k, 0.5)
        md.append(f"| {NAME[k]} | " + " | ".join(f"{f[grp == g].mean():.0%}" for g in gs) + " |")
# ---- reuse
rf = os.path.join(OUT, "reuse_results.json")
if os.path.exists(rf):
    R = json.load(open(rf))
    rows = R["rows"]
    md.append(f"\n## Reuse (index of {R['index_size']} photos; frozen thresholds {R['thresholds']})\n")
    md.append(f"SSCD vectors came from the deployed Model Serving endpoint `{R.get('sscd_source', '?')}` "
              f"({R.get('endpoint_calls', 0)} calls, median {R.get('endpoint_median_ms', 0):.0f} ms per call; lowest cosine to "
              f"the local weights {R.get('endpoint_parity_min_cosine', 0):.4f}). pHash and ORB ran locally.\n")
    md.append("| Method | recall (hit on the right source) | by transform: crop / screenshot / recompress / relight | "
              "false hits on never-indexed photos | ms per query |\n|---|---|---|---|---|")
    for m, hi in (("phash", False), ("orb", True), ("sscd", True)):
        th = R["thresholds"][m]
        def hit(r):
            return (r[f"{m}_score"] >= th) if hi else (r[f"{m}_score"] <= th)
        pos = [r for r in rows if r["kind"] == "positive"]
        neg = [r for r in rows if r["kind"] == "negative"]
        ok = [hit(r) and r[f"{m}_top"] == r["source"] for r in pos]
        by = {t: np.mean([o for o, r in zip(ok, pos) if r["transform"] == t]) for t in ("crop", "screenshot", "recompress", "relight")}
        md.append(f"| {m} | {np.mean(ok):.0%} ({sum(ok)}/{len(pos)}) | " + " / ".join(f"{by[t]:.0%}" for t in by) +
                  f" | {sum(hit(r) for r in neg)}/{len(neg)} | {R['seconds_per_query'][m] * 1000:.0f} |")
open(os.path.join(OUT, "results.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))

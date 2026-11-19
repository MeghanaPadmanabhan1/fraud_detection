"""Analysis: learned detectors vs the classical OpenCV checks on gold set v0 (66 claims).

Metrics were fixed before looking at results:
  * AUC (threshold-free) with a stratified bootstrap 95% CI (2000 resamples, seed 7);
  * recall at ZERO false positives (threshold just above the highest-scoring genuine photo; in-sample, so
    optimistic in absolute terms but identical for every detector);
  * false-positive rate / recall at each detector's OWN default threshold (CLIP LLR>0, SPAI p>0.5, app
    thresholds for the classical checks).
Negatives = genuine photos (genuine + document claims, 19). Positives are evaluated per fraud group.

    python -I build/ml_exp_analyze.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_exp_common import EXP, WB, load_claims, provenance_state

rng = np.random.default_rng(7)
claims = load_claims()
N = len(claims)
ids = [c["claim_id"] for c in claims]
cls = np.array([c["fraud_class"] for c in claims])
fam = np.array([c["family"] for c in claims])
prov = np.array([provenance_state(c) or "" for c in claims])
NEG = np.isin(fam, ["genuine", "document"])
GROUPS = {
    "AI-made (generated + AI-edited)": np.isin(cls, ["ai_generated", "ai_edited_damage"]),
    "  AI, provenance tag intact": np.isin(cls, ["ai_generated", "ai_edited_damage"]) & (prov == "intact"),
    "  AI, provenance stripped": np.isin(cls, ["ai_generated", "ai_edited_damage"]) & (prov == "laundered"),
    "conventional edits (splice / copy-move / upscale)": np.isin(cls, ["splice", "copy_move", "upscale"]),
    "screen recapture (screenshot / re-photo)": np.isin(cls, ["screenshot", "rephotograph"]),
}
PRIMARY = ["AI-made (generated + AI-edited)", "conventional edits (splice / copy-move / upscale)",
           "screen recapture (screenshot / re-photo)"]

APP_THR = {"ela": .855, "jpeg_ghost": .961, "noise": .99, "copy_move_orb": .5, "copy_move_sift": .922,
           "recapture": .5, "upscale": .5, "spectral_fft": .5, "spectral_dct": .5, "ela_uniformity": .5}
ML_THR = {"clipdet_latent10k_plus": 0.0, "clipdet_latent10k": 0.0, "spai": 0.5}
KIND = {**{k: "OpenCV / pixel" for k in APP_THR}, "clipdet_latent10k_plus": "learned (CLIP)",
        "clipdet_latent10k": "learned (CLIP)", "spai": "learned (spectral ViT)",
        "aspect_probe": "confound probe", "bytes_per_pixel_probe": "confound probe"}
PRETTY = {"ela": "ELA hotspots", "jpeg_ghost": "JPEG ghost", "noise": "Noise residual", "copy_move_orb": "Copy-move ORB",
          "copy_move_sift": "Copy-move SIFT", "spectral_fft": "Spectral FFT", "spectral_dct": "Spectral DCT",
          "ela_uniformity": "ELA uniformity", "upscale": "Upscale detector", "recapture": "Recapture (screen)",
          "clipdet_latent10k_plus": "CLIP detector (latent10k_plus)", "clipdet_latent10k": "CLIP detector (latent10k)",
          "spai": "SPAI", "aspect_probe": "probe: aspect ratio", "bytes_per_pixel_probe": "probe: bytes per pixel"}


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def auc_ci(pos, neg, B=2000):
    v = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(B)]
    return tuple(np.percentile(v, [2.5, 97.5]))


def fmt_auc(a, ci):
    return f"{a:.2f} [{ci[0]:.2f}–{ci[1]:.2f}]"


# ------------------------------------------------------------------ load scores
def load_scores(variant):
    S = {}
    cj = json.load(open(os.path.join(EXP, f"classical_{variant}.json")))
    assert cj["claim_ids"] == ids
    S.update({k: np.array(v, float) for k, v in cj["scores"].items()})
    z = np.load(os.path.join(EXP, f"ml_{variant}.npz"))
    assert list(z["claim_ids"]) == ids
    S["clipdet_latent10k_plus"], S["clipdet_latent10k"] = z["clipdet_latent10k_plus"], z["clipdet_latent10k"]
    for f in glob.glob(os.path.join(EXP, f"spai_{variant}", "**", "*.csv"), recursive=True):
        df = pd.read_csv(f)
        if "spai" in df.columns:
            col = next(c for c in df.columns if c in ("image", "filename", "path"))
            m = {os.path.splitext(os.path.basename(str(p)))[0]: s for p, s in zip(df[col], df["spai"])}
            if all(i in m for i in ids):
                S["spai"] = np.array([m[i] for i in ids], float)
    return S, cj, z


def vlm_scores():
    p = os.path.join(WB, "cache", "vlm.json")
    out = {}
    for r in json.load(open(p)):
        if r.get("error"):
            continue
        out.setdefault(r["endpoint"], {})[r["claim_id"]] = max(float(r.get("ai_generated_prob") or 0),
                                                              float(r.get("manipulated_prob") or 0))
    return {f"vlm:{ep.replace('databricks-', '')}": np.array([d.get(i, 0.0) for i in ids]) for ep, d in out.items()}


# ------------------------------------------------------------------ tables
def detection_tables(variant, S, md):
    neg = S_neg = None
    rows = []
    for name, sc in S.items():
        negs = sc[NEG]
        r = {"detector": PRETTY.get(name, name), "type": KIND.get(name, "vision LLM (reference)")}
        for g, mask in GROUPS.items():
            pos = sc[mask]
            a = auc(pos, negs); lo, hi = auc_ci(pos, negs)
            r[g] = (a, lo, hi, float((pos > negs.max()).mean()), len(pos))
        thr = APP_THR.get(name, ML_THR.get(name))
        if thr is not None:
            r["fpr_default"] = float((negs > thr).mean())
            r["thr"] = thr
            r["recall_default"] = {g: float((sc[m] > thr).mean()) for g, m in GROUPS.items()}
        rows.append(r)
    md.append(f"\n### Image-level detection — `{variant}` images ({NEG.sum()} genuine photos as negatives)\n")
    md.append("AUC with 95% bootstrap CI · recall at zero false positives (R@0FP). Higher is better; 0.50 = chance.\n")
    head = "| Detector | Type | " + " | ".join(f"{g.strip()} (n={GROUPS[g].sum()})" for g in PRIMARY) + " |"
    md += [head, "|" + "---|" * (2 + len(PRIMARY))]
    order = sorted(rows, key=lambda r: -r[PRIMARY[0]][0])
    for r in order:
        cells = [f"{fmt_auc(*r[g][:1], r[g][1:3])} · {r[g][3]:.0%}" for g in PRIMARY]
        md.append(f"| {r['detector']} | {r['type']} | " + " | ".join(cells) + " |")
    md.append("\n**AI-made claims split by whether the generator's provenance tag survived** "
              "(AUC [CI] · R@0FP) — the stripped rows are the real test of the pixel/learned detectors:\n")
    sub = [g for g in GROUPS if g.startswith("  ")]
    md += ["| Detector | " + " | ".join(f"{g.strip()} (n={GROUPS[g].sum()})" for g in sub) + " |", "|" + "---|" * (1 + len(sub))]
    for r in order:
        md.append(f"| {r['detector']} | " + " | ".join(f"{fmt_auc(*r[g][:1], r[g][1:3])} · {r[g][3]:.0%}" for g in sub) + " |")
    probes = [r for r in rows if r["type"] == "confound probe"]
    if probes:
        md.append("\n**Confound probes, best direction** (max of AUC and 1−AUC; what a trivially confounded feature "
                  "achieves with no forensic signal at all):\n")
        md += ["| Probe | " + " | ".join(g.strip() for g in PRIMARY) + " |", "|" + "---|" * (1 + len(PRIMARY))]
        for r in probes:
            md.append(f"| {r['detector']} | " + " | ".join(f"{max(r[g][0], 1 - r[g][0]):.2f}" for g in PRIMARY) + " |")
    dflt = [r for r in order if "thr" in r]
    md.append("\n**Operating point at each detector's own default threshold** "
              f"(app thresholds for OpenCV checks; CLIP LLR>0; SPAI p>0.5). FPR is on the {NEG.sum()} genuine photos:\n")
    md += ["| Detector | threshold | false-positive rate | recall: AI-made | recall: conventional edits | recall: recapture |",
           "|---|---|---|---|---|---|"]
    for r in dflt:
        rd = r["recall_default"]
        md.append(f"| {r['detector']} | {r['thr']:g} | {r['fpr_default']:.0%} ({int(round(r['fpr_default'] * NEG.sum()))}/{NEG.sum()}) | "
                  f"{rd[PRIMARY[0]]:.0%} | {rd[PRIMARY[1]]:.0%} | {rd[PRIMARY[2]]:.0%} |")
    return rows


def per_class_table(variant, S, md):
    kinds = ["ai_generated", "ai_edited_damage", "splice", "copy_move", "upscale", "screenshot", "rephotograph"]
    md.append(f"\n**Per-fraud-class AUC, `{variant}` images** (exploratory: only 4-8 claims per class, so differences "
              f"smaller than ~0.2 are noise; 0.50 = chance, shown without CIs):\n")
    md += ["| Detector | " + " | ".join(f"{k} (n={int((cls == k).sum())})" for k in kinds) + " |", "|" + "---|" * (1 + len(kinds))]
    for name, sc in sorted(S.items(), key=lambda kv: PRETTY.get(kv[0], kv[0])):
        md.append(f"| {PRETTY.get(name, name)} | " + " | ".join(f"{auc(sc[cls == k], sc[NEG]):.2f}" for k in kinds) + " |")


def reuse_tables(variant, md):
    cj = json.load(open(os.path.join(EXP, f"classical_{variant}.json")))
    P = dict(np.load(os.path.join(EXP, f"reuse_pairs_classical_{variant}.npz")))
    z = np.load(os.path.join(EXP, f"ml_{variant}.npz"))
    P["sscd_disc_mixup"], P["sscd_disc_large"] = z["sscd_disc_mixup"] @ z["sscd_disc_mixup"].T, z["sscd_disc_large"] @ z["sscd_disc_large"].T
    order = sorted(range(N), key=lambda i: (claims[i]["claim_date"], claims[i]["claim_id"]))
    rank = {i: r for r, i in enumerate(order)}
    src = [c["source_id"] for c in claims]
    earlier = [[j for j in range(N) if rank[j] < rank[i]] for i in range(N)]
    pos = np.array([any(src[j] == src[i] for j in earlier[i]) for i in range(N)])
    DEFAULT = {"phash": 1 - 10 / 64, "dhash": 1 - 10 / 64, "orb": 25.0, "thumb": 0.96,
               "sscd_disc_mixup": 0.75, "sscd_disc_large": 0.75}
    LABEL = {"phash": "pHash (app default)", "dhash": "dHash", "orb": "ORB + RANSAC", "thumb": "Thumbnail cosine (app 'embedding')",
             "sscd_disc_mixup": "SSCD mixup (learned)", "sscd_disc_large": "SSCD large (learned)"}
    sc = {k: np.array([max([M[i, j] for j in earlier[i]], default=M.min()) for i in range(N)]) for k, M in P.items()}
    md.append(f"\n### Reuse / near-duplicate retrieval — `{variant}` images\n")
    md.append(f"Ground truth: a claim is a reuse if an *earlier* claim was built from the same source photo "
              f"(any manipulation: exact copy, recompression, crop, screenshot, re-photo, splice, AI edit…). "
              f"{pos.sum()} reuse claims vs {(~pos).sum()} first uses. Score = highest similarity to any earlier claim.\n")
    md += ["| Method | AUC [95% CI] | recall at 0 false positives | at default threshold: recall / false alarms | ms per comparison |",
           "|---|---|---|---|---|"]
    res = {}
    for k in ("phash", "dhash", "thumb", "orb", "sscd_disc_mixup", "sscd_disc_large"):
        a = auc(sc[k][pos], sc[k][~pos]); lo, hi = auc_ci(sc[k][pos], sc[k][~pos])
        r0 = float((sc[k][pos] > sc[k][~pos].max()).mean())
        rd = float((sc[k][pos] >= DEFAULT[k]).mean()); fd = int((sc[k][~pos] >= DEFAULT[k]).sum())
        res[k] = (a, lo, hi, r0, rd, fd)
        ms = cj["median_seconds"].get("_pair_seconds", float("nan")) * 1000 if k == "orb" else 0.0
        md.append(f"| {LABEL[k]} | {fmt_auc(a, (lo, hi))} | {r0:.0%} | {rd:.0%} / {fd} of {(~pos).sum()} | {'~' + format(ms, '.1f') if k == 'orb' else '<0.01 (vector dot product)'} |")
    md.append("\n**Recall at the zero-false-positive threshold, by how the reused photo was altered** "
              "(n = reuse claims of that kind):\n")
    kinds = [k for k, _ in sorted(((c, int((pos & (cls == c)).sum())) for c in set(cls)), key=lambda t: -t[1]) if _ > 0]
    md += ["| Method | " + " | ".join(f"{k} (n={int((pos & (cls == k)).sum())})" for k in kinds) + " |", "|" + "---|" * (1 + len(kinds))]
    for k in ("phash", "dhash", "thumb", "orb", "sscd_disc_mixup", "sscd_disc_large"):
        thr = sc[k][~pos].max()
        md.append(f"| {LABEL[k]} | " + " | ".join(f"{(sc[k][pos & (cls == kk)] > thr).mean():.0%}" for kk in kinds) + " |")
    return res


def main():
    md = ["# ML detectors vs OpenCV / pixel-level checks — experiment results (gold set v0, 66 claims)\n"]
    all_rows = {}
    V = vlm_scores()
    for variant in ("orig", "social"):
        S, cj, z = load_scores(variant)
        if variant == "orig":
            S.update(V)
        all_rows[variant] = detection_tables(variant, S, md)
        per_class_table(variant, S, md)
        reuse_tables(variant, md)
    timing = {v: json.load(open(os.path.join(EXP, f"ml_{v}_timing.json"))) for v in ("orig",)}
    cj = json.load(open(os.path.join(EXP, "classical_orig.json")))
    md.append("\n### Cost on this laptop (Apple M2 Pro, CPU only, median per image)\n")
    md += ["| Method | time per image |", "|---|---|"]
    for k, v in sorted(cj["median_seconds"].items()):
        if not k.startswith("_") and "probe" not in k:
            md.append(f"| {PRETTY.get(k, k)} | {v * 1000:.0f} ms |")
    for k, v in timing["orig"].items():
        md.append(f"| {k} | {v * 1000:.0f} ms |")
    sp = os.path.join(EXP, "spai_timing.json")
    if os.path.exists(sp):
        md.append(f"| SPAI | {json.load(open(sp))['s_per_img']:.1f} s |")
    open(os.path.join(EXP, "results.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()

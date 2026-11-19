"""Classical (OpenCV / NumPy) baselines for one image variant, computed with the SAME code the Workbench app
runs: per-image forensic scores, confound probes, and pairwise reuse similarities (pHash, dHash, ORB+RANSAC,
thumbnail cosine).

    python -I build/ml_exp_classical.py <orig|social>
"""
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_exp_common import EXP, load_claims, variant_path

from image_authenticity import detectors  # noqa: F401  (registers detectors)
from image_authenticity.core import build
from image_authenticity.detectors import _util
from image_authenticity.workbench import checks

variant = sys.argv[1]
claims = load_claims()
N = len(claims)
DET = {"ela": ("ela_local", "ela_heatmap_v1"), "jpeg_ghost": ("jpeg_ghost", "jpeg_ghost_v1"),
       "noise": ("noise", "noise_residual_v1"), "copy_move_orb": ("copy_move", "orb_v1"),
       "copy_move_sift": ("copy_move", "sift_v1"), "spectral_fft": ("spectral", "fft_radial_v1"),
       "spectral_dct": ("spectral", "dct_stats_v1"), "ela_uniformity": ("ela_global", "ela_uniformity_v1")}
dets = {k: build(*v) for k, v in DET.items()}

scores = {k: [] for k in list(DET) + ["upscale", "recapture", "aspect_probe", "bytes_per_pixel_probe"]}
secs = {k: [] for k in scores}
imgs, hashes = [], []
for c in claims:
    p = variant_path(c, variant)
    raw = open(p, "rb").read()
    img = _util.decode_bgr(raw)
    imgs.append(img)
    for k, d in dets.items():
        t0 = time.perf_counter(); s = d.analyze(img, raw, {}).score; secs[k].append(time.perf_counter() - t0)
        scores[k].append(float(s) if s is not None else 0.0)
    meta = checks.read_metadata(raw)
    t0 = time.perf_counter(); scores["upscale"].append(checks.upscale_check(img)["score"]); secs["upscale"].append(time.perf_counter() - t0)
    t0 = time.perf_counter(); scores["recapture"].append(checks.recapture_check(img, meta)["score"]); secs["recapture"].append(time.perf_counter() - t0)
    h, w = img.shape[:2]
    scores["aspect_probe"].append(abs(float(np.log(w / h))))                 # confound probe, not a detector
    scores["bytes_per_pixel_probe"].append(len(raw) / (w * h))               # confound probe, not a detector
    hashes.append((checks.phash(img), checks.dhash(img), checks.embedding(img)))
    print(".", end="", flush=True)
print()

# ---- pairwise reuse similarity (higher = more similar); symmetric, diagonal unused
P = {k: np.zeros((N, N)) for k in ("phash", "dhash", "orb", "thumb")}
orb = [checks.orb_features(im) for im in imgs]
t0 = time.perf_counter()
for i in range(N):
    for j in range(i + 1, N):
        P["phash"][i, j] = 1 - checks.hamming(hashes[i][0], hashes[j][0]) / 64
        P["dhash"][i, j] = 1 - checks.hamming(hashes[i][1], hashes[j][1]) / 64
        P["thumb"][i, j] = float(np.dot(hashes[i][2], hashes[j][2]))
        P["orb"][i, j] = checks.orb_inliers(orb[i], orb[j])
    # mirror at the end
for k in P:
    P[k] = P[k] + P[k].T
pair_secs = (time.perf_counter() - t0) / (N * (N - 1) / 2)
secs["_pair_seconds"] = [pair_secs]

os.makedirs(EXP, exist_ok=True)
json.dump({"claim_ids": [c["claim_id"] for c in claims], "scores": scores,
           "median_seconds": {k: float(np.median(v)) for k, v in secs.items()}},
          open(os.path.join(EXP, f"classical_{variant}.json"), "w"))
np.savez(os.path.join(EXP, f"reuse_pairs_classical_{variant}.npz"), **P)
print(f"saved classical_{variant}.json; pairwise reuse comparisons: {pair_secs * 1000:.2f} ms/pair")

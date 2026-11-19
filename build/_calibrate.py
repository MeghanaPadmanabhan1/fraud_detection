"""Per-technique thresholds from the genuine-photo score distribution (FP budget)."""
import collections, json, os, sys
sys.path.insert(0, sys.argv[1])
import cv2, numpy as np
from image_authenticity.core import build
from image_authenticity import detectors
from image_authenticity.workbench import checks
D = sys.argv[2]; C = json.load(open(os.path.join(D, "claims.json")))
loc = lambda p: os.path.join(D, p.split("claims_media/")[1])
T = {"ela": ("ela_local", "ela_heatmap_v1"), "jpeg_ghost": ("jpeg_ghost", "jpeg_ghost_v1"),
     "noise": ("noise", "noise_residual_v1"), "copy_move_orb": ("copy_move", "orb_v1"),
     "copy_move_sift": ("copy_move", "sift_v1"), "spectral_fft": ("spectral", "fft_radial_v1"),
     "spectral_dct": ("spectral", "dct_stats_v1"), "ela_uniformity": ("ela_global", "ela_uniformity_v1")}
S = collections.defaultdict(list)
for c in C:
    if c["family"] not in ("genuine", "document"): continue
    raw = open(loc(c["image_path"]), "rb").read(); img = cv2.imdecode(np.frombuffer(raw, np.uint8), 1)
    for t, (a, b) in T.items(): S[t].append(build(a, b).analyze(img, raw, {}).score)
    S["recapture"].append(checks.recapture_check(img, checks.read_metadata(raw))["score"])
    S["upscale"].append(checks.upscale_check(img)["score"])
out = {}
for t, v in S.items():
    v = np.array(v); amber = float(min(0.99, max(0.5, np.percentile(v, 95) + 0.02))); red = float(min(0.995, max(0.8, v.max() + 0.02, amber + 0.05)))
    out[t] = [round(amber, 3), round(red, 3)]
    print(f"{t:16s} genuine p95={np.percentile(v,95):.3f} max={v.max():.3f} -> amber {amber:.3f} red {red:.3f}")
print(json.dumps(out))

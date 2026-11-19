import collections, json, os, sys
sys.path.insert(0, sys.argv[1])
import cv2, numpy as np
from image_authenticity.core import build
from image_authenticity import detectors
from image_authenticity.workbench import checks
D = sys.argv[2]; C = json.load(open(os.path.join(D, "claims.json")))
loc = lambda p: os.path.join(D, p.split("claims_media/")[1])
M = collections.defaultdict(lambda: collections.defaultdict(list))
dets = {"ela": build("ela_local", "ela_heatmap_v1"), "ghost": build("jpeg_ghost", "jpeg_ghost_v1"),
        "noise": build("noise", "noise_residual_v1"), "cm_orb": build("copy_move", "orb_v1"),
        "cm_sift": build("copy_move", "sift_v1"), "fft": build("spectral", "fft_radial_v1"),
        "dct": build("spectral", "dct_stats_v1"), "elaU": build("ela_global", "ela_uniformity_v1")}
for c in C:
    raw = open(loc(c["image_path"]), "rb").read(); img = cv2.imdecode(np.frombuffer(raw, np.uint8), 1)
    meta = checks.read_metadata(raw); k = c["fraud_class"] if c["family"] != "document" else "genuine"
    r = checks.recapture_check(img, meta); M["recap"][k].append(r["score"]); M["moire"][k].append(r["moire_peak"])
    M["upscale"][k].append(checks.upscale_check(img)["score"])
    for n, d in dets.items():
        e = d.analyze(img, raw, {})
        val = {"ela": e.evidence.get("ela_peak_z"), "ghost": e.score, "noise": e.evidence.get("noise_outlier_frac"),
               "cm_orb": e.evidence.get("clone_pairs"), "cm_sift": e.evidence.get("clone_pairs")}.get(n, e.score)
        M[n][k].append(val if val is not None else 0)
classes = ["genuine", "reuse_exact", "screenshot", "rephotograph", "splice", "copy_move", "upscale", "ai_edited_damage", "ai_generated"]
print(f"{'metric':8s} " + " ".join(f"{c[:9]:>10s}" for c in classes))
for m, byc in M.items():
    row = []
    for c in classes:
        v = byc.get(c, [])
        row.append(f"{np.median(v):5.2f}/{max(v):5.2f}" if v else "    -     ")
    print(f"{m:8s} " + " ".join(f"{x:>10s}" for x in row))

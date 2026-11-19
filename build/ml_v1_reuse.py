"""Reuse test on v1: pHash (<=10/64), ORB+RANSAC (>=25 inliers), SSCD mixup (cosine >= 0.60), thresholds FROZEN from
gold set v0 / the app defaults. A query is a hit if its best-scoring indexed photo clears the threshold.
Positives must hit THEIR source photo; negatives (photos never indexed) must not hit anything.

    ml_env/bin/python -I build/ml_v1_reuse.py
"""
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from image_authenticity.workbench import checks, models   # noqa: E402

ENDPOINT = os.environ.get("SSCD_ENDPOINT", "image-authenticity-sscd")   # SSCD vectors come from the deployed endpoint
PARITY, ENDPOINT_SECS, RETRIES = [], [], []

R = json.load(open(os.path.join(ROOT, "wb_data", "ml_v1", "reuse", "reuse.json")))
TH = {"phash": 10, "orb": 25, "sscd": 0.60}


def prep(path):
    im = cv2.imread(path)
    f = 1600 / max(im.shape[:2])
    im = cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else im
    t = time.perf_counter()
    remote = None
    for attempt in range(5):                                                   # the deployed Model Serving path
        remote = models.sscd_remote(ENDPOINT, im, timeout=60, cooldown=0)
        if remote is not None:
            break
        time.sleep(5 * (attempt + 1))
    RETRIES.append(attempt)
    if remote is None:
        raise RuntimeError(f"SSCD endpoint {ENDPOINT} did not answer for {path}")
    ENDPOINT_SECS.append(time.perf_counter() - t)
    PARITY.append(float(remote @ checks.sscd_embedding(im)))                   # same vector as the local weights?
    return dict(ph=checks.phash(im), orb=checks.orb_features(im), sscd=remote)


t0 = time.time()
IDX = [(e, prep(e["path"])) for e in R["index"]]
print("indexed", len(IDX), f"in {time.time() - t0:.0f}s", flush=True)
rows, secs = [], {"phash": 0.0, "orb": 0.0, "sscd": 0.0}
for q in R["queries"]:
    f = prep(q["path"])
    best = {}
    for name, fn, hi in (("phash", lambda a, b: checks.hamming(a["ph"], b["ph"]), False),
                         ("orb", lambda a, b: checks.orb_inliers(a["orb"], b["orb"]), True),
                         ("sscd", lambda a, b: float(a["sscd"] @ b["sscd"]), True)):
        t = time.perf_counter()
        sc = [(fn(f, v), e["id"]) for e, v in IDX]
        secs[name] += time.perf_counter() - t
        s, who = (max(sc) if hi else min(sc))
        best[name] = (s, who)
    rows.append(dict(q, **{f"{n}_score": float(best[n][0]) for n in best}, **{f"{n}_top": best[n][1] for n in best}))
out = dict(rows=rows, thresholds=TH, sscd_source=f"endpoint:{ENDPOINT}", endpoint_calls=len(PARITY), endpoint_retried_calls=int(sum(r > 0 for r in RETRIES)),
           endpoint_parity_min_cosine=float(min(PARITY)), endpoint_median_ms=float(np.median(ENDPOINT_SECS) * 1000), seconds_per_query={k: v / len(R["queries"]) for k, v in secs.items()},
           index_size=len(IDX))
json.dump(out, open(os.path.join(ROOT, "wb_data", "ml_v1", "reuse_results.json"), "w"), indent=1)
print("saved reuse_results.json")

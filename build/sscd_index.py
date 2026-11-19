"""Compute SSCD vectors for every claim in the gold set and write wb_data/tables/sscd_index.parquet
(claim_id, sscd = JSON list). Also prints the zero-false-positive cosine threshold on the gold set.

    ml_env/bin/python -I build/sscd_index.py
"""
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from image_authenticity.workbench import checks   # noqa: E402

C = pd.read_parquet(os.path.join(ROOT, "wb_data", "tables", "claims.parquet"))
rows, E = [], {}
for r in C.itertuples(index=False):
    img = cv2.imread(os.path.join(ROOT, "wb_data", "dataset", "images", os.path.basename(r.image_path)))
    E[r.claim_id] = v = checks.sscd_embedding(img)
    rows.append({"claim_id": r.claim_id, "sscd": json.dumps([round(float(x), 5) for x in v])})
pd.DataFrame(rows).to_parquet(os.path.join(ROOT, "wb_data", "tables", "sscd_index.parquet"), index=False)
print(len(rows), "vectors written")

# zero-false-positive threshold: reuse = an earlier claim was built from the same source photo (source_id)
order = C.sort_values(["claim_date", "claim_id"]).reset_index(drop=True)
pos, sc = [], []
for i, r in order.iterrows():
    prior = order.iloc[:i]
    sc.append(max([float(E[r.claim_id] @ E[p]) for p in prior.claim_id], default=-1))
    pos.append(bool((prior.source_id == r.source_id).any()))
pos, sc = np.array(pos), np.array(sc)
print(f"reuse n={pos.sum()}  non-reuse n={(~pos).sum()}")
print(f"max score among non-reuse (zero-FP threshold) = {sc[~pos].max():.3f}; p95 non-reuse = {np.percentile(sc[~pos], 95):.3f}")
for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75):
    print(f"  threshold {t:.2f}: recall {(sc[pos] >= t).mean():.0%}, false alarms {(sc[~pos] >= t).sum()}/{(~pos).sum()}")

import collections, datetime as dt, json, os, sys
sys.path.insert(0, sys.argv[1])
import cv2, numpy as np
from image_authenticity.workbench import preset_config, run_claim, checks
from image_authenticity.workbench.engine import EngineContext
D = sys.argv[2]; C = json.load(open(os.path.join(D, "claims.json")))
loc = lambda p: os.path.join(D, p.split("claims_media/")[1])
raw = {c["claim_id"]: open(loc(c["image_path"]), "rb").read() for c in C}
idx = checks.ReuseIndex(built_at=dt.datetime.now())
for c in C: idx.entries.append(checks.make_entry(c, cv2.imdecode(np.frombuffer(raw[c["claim_id"]], np.uint8), 1)))
ctx = EngineContext(index=idx)
cnt = collections.Counter(); scores = collections.defaultdict(list)
for c in C:
    run = run_claim(c, raw[c["claim_id"]], None, preset_config("Production"), ctx)
    for s in run.stages:
        for f in s.findings:
            if f.score is not None: scores[(f.check[:28], c["family"])].append(round(f.score, 2))
    if c["family"] == "genuine":
        for r in run.reasons: cnt[(r.severity, r.check[:40], r.text[:70])] += 1
for k, v in cnt.most_common(12): print(v, k)
print()
for (chk, fam), v in sorted(scores.items()):
    print(f"{chk:30s} {fam:12s} median={np.median(v):.2f} max={max(v):.2f} n={len(v)}")

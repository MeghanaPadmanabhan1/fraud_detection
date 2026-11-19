"""Smoke-test the workbench engine locally on the gold set (no VLM / OCR).
Usage: python -I build/local_engine_test.py <repo> <dataset_dir>"""
import collections, datetime as dt, json, os, sys, time
sys.path.insert(0, sys.argv[1])
import cv2
from image_authenticity.workbench import preset_config, run_claim
from image_authenticity.workbench import checks
from image_authenticity.workbench.engine import EngineContext

D = sys.argv[2]
claims = json.load(open(os.path.join(D, "claims.json")))
loc = lambda p: os.path.join(D, p.split("claims_media/")[1])
raw = {c["claim_id"]: open(loc(c["image_path"]), "rb").read() for c in claims}
idx = checks.ReuseIndex(built_at=dt.datetime.now())
for c in claims:
    idx.entries.append(checks.make_entry(c, cv2.imdecode(__import__("numpy").frombuffer(raw[c["claim_id"]], "uint8"), 1)))
ctx = EngineContext(index=idx, index_loader=lambda e: cv2.imread(loc(e.image_path)))

# raw calibration metrics
print("class                 recapture  moire  upscale_hf")
for c in claims:
    img = cv2.imread(loc(c["image_path"]))
    r = checks.recapture_check(img, checks.read_metadata(raw[c["claim_id"]]))
    u = checks.upscale_check(img)
    print(f"{c['fraud_class']:22s} {r['score']:.2f}      {r['moire_peak']:5.1f}  {u['hf_log_ratio']:5.2f}")

for preset in ["Production", "Challenger A"]:
    cfg = preset_config(preset)
    tally = collections.defaultdict(collections.Counter)
    t0 = time.time()
    for c in claims:
        rec = open(loc(c["receipt_path"]), "rb").read() if c.get("receipt_path") else None
        run = run_claim(c, raw[c["claim_id"]], rec, cfg, ctx)
        tally[c["fraud_class"]][run.verdict] += 1
    print(f"\n=== {preset} ({time.time() - t0:.1f}s for {len(claims)} claims) ===")
    for k, v in sorted(tally.items()):
        print(f"  {k:22s} {dict(v)}")

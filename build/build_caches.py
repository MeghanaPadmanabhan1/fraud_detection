"""Slow-path caches: vision-model detection (4 VLMs), receipt OCR (3 VLMs + ai_parse_document).
Usage: DATABRICKS_CONFIG_PROFILE=e2 python -I build/build_caches.py <repo> <wb_data>"""
import concurrent.futures as cf, json, os, sys
sys.path.insert(0, sys.argv[1])
import cv2, numpy as np
from image_authenticity.workbench.catalog import VLMS
from image_authenticity.workbench import models
WB = sys.argv[2]; D = os.path.join(WB, "dataset"); C = json.load(open(os.path.join(D, "claims.json")))
loc = lambda p: os.path.join(D, p.split("claims_media/")[1])
os.makedirs(os.path.join(WB, "cache"), exist_ok=True)
jobs = [("vlm", c, ep) for c in C for ep in VLMS]
jobs += [("ocr", c, ep) for c in C if c.get("receipt_path") for ep in list(VLMS)[:3]]
def work(j):
    kind, c, ep = j
    if kind == "vlm":
        return {"claim_id": c["claim_id"], "endpoint": ep, **models.vlm_detect(ep, cv2.imread(loc(c["image_path"])), c)}
    r = models.vlm_ocr(ep, open(loc(c["receipt_path"]), "rb").read())
    return {"claim_id": c["claim_id"], "engine": f"vlm_ocr:{ep}", **r}
vlm, ocr = [], []
with cf.ThreadPoolExecutor(10) as ex:
    for i, r in enumerate(ex.map(work, jobs)):
        (vlm if "endpoint" in r else ocr).append(r)
        if i % 40 == 0: print(i, "/", len(jobs), flush=True)
# ai_parse_document results (batch SQL, one statement for all receipts)
elapsed = float(open(os.path.join(WB, "ai_parse_elapsed.txt")).read())
rows = json.load(open(os.path.join(WB, "ai_parse.json")))
for r in rows:
    f, text = models.parsed_document_fields(r["parsed"])
    ocr.append({"claim_id": r["claim_id"], "engine": "ai_parse_document", "fields": f, "raw_text": text[:2000],
                "latency_ms": elapsed * 1000 / len(rows), "tokens": 0, "error": None})
json.dump(vlm, open(os.path.join(WB, "cache/vlm.json"), "w"))
json.dump(ocr, open(os.path.join(WB, "cache/ocr.json"), "w"))
errs = [r for r in vlm + ocr if r.get("error")]
print(len(vlm), "vlm rows,", len(ocr), "ocr rows,", len(errs), "errors")
for e in errs[:5]: print("  ", e.get("endpoint") or e.get("engine"), e["error"][:150])

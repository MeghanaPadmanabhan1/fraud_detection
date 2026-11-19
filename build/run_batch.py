"""Batch evaluation: run every preset over the gold set, score every technique for
the leaderboard, and write the workbench Gold tables as parquet for loading to Delta.

Usage: DATABRICKS_CONFIG_PROFILE=e2 python -I build/run_batch.py <repo> <wb_data>
"""
import concurrent.futures as cf, copy, datetime as dt, json, os, sys, uuid
sys.path.insert(0, sys.argv[1])
import cv2, numpy as np, pandas as pd
from image_authenticity.workbench import PRESETS, run_claim, checks, models
from image_authenticity.workbench.catalog import STAGE_BY_ID, VLMS, option_label, preset_config
from image_authenticity.workbench.engine import EngineContext

WB = sys.argv[2]; D = os.path.join(WB, "dataset"); OUT = os.path.join(WB, "tables"); os.makedirs(OUT, exist_ok=True)
C = json.load(open(os.path.join(D, "claims.json")))
loc = lambda p: os.path.join(D, p.split("claims_media/")[1])
RAW = {c["claim_id"]: open(loc(c["image_path"]), "rb").read() for c in C}
REC = {c["claim_id"]: open(loc(c["receipt_path"]), "rb").read() for c in C if c.get("receipt_path")}
VLM = {(r["claim_id"], r["endpoint"]): r for r in json.load(open(os.path.join(WB, "cache/vlm.json")))}
OCR = {(r["claim_id"], r["engine"]): r for r in json.load(open(os.path.join(WB, "cache/ocr.json")))}
NOW = dt.datetime(2026, 10, 27, 9, 0, 0)

idx = checks.ReuseIndex(built_at=NOW)
for c in C:
    idx.entries.append(checks.make_entry(c, cv2.imdecode(np.frombuffer(RAW[c["claim_id"]], np.uint8), 1)))
EXPL = {}


def vlm_lookup(cid, ep, img):
    r = VLM.get((cid, ep)); return dict(r, cached=True) if r else None


def ocr_lookup(cid, engine, rec):
    r = OCR.get((cid, engine)); return dict(r, cached=True) if r else None


def explain_lookup(ep, prompt, cid):
    k = (cid, ep, prompt)
    if k not in EXPL:
        EXPL[k] = models.explain(ep, prompt)
    return dict(EXPL[k], cached=True)


ctx = EngineContext(index=idx, index_loader=lambda e: cv2.imread(loc(e.image_path)),
                    vlm=vlm_lookup, ocr=ocr_lookup, explain=explain_lookup)

runs, stage_rows, finding_rows = [], [], []
batch_id = uuid.uuid4().hex[:8]
for name in PRESETS:
    cfg = preset_config(name)
    run_id = f"{name.lower().replace(' ', '_')}-{batch_id}"
    print("running", name, flush=True)

    def one(c):
        return c, run_claim(c, RAW[c["claim_id"]], REC.get(c["claim_id"]), cfg, ctx)

    with cf.ThreadPoolExecutor(6) as ex:
        results = list(ex.map(one, C))
    for c, r in results:
        runs.append({"run_id": run_id, "config_name": name, "claim_id": c["claim_id"], "verdict": r.verdict,
                     "confidence": r.confidence, "families": ",".join(r.families), "route": r.route,
                     "explanation": r.explanation, "explainer": r.explainer,
                     "sync_latency_ms": r.sync_latency_ms, "async_latency_ms": r.async_latency_ms,
                     "tokens": r.tokens, "run_ts": NOW.isoformat(), "config_json": json.dumps(cfg)})
        for s in r.stages:
            stage_rows.append({"run_id": run_id, "config_name": name, "claim_id": c["claim_id"], "sid": s.sid,
                               "stage_num": s.num, "stage_name": s.name, "techs": " + ".join(s.techs),
                               "status": s.status, "score": s.score, "severity": s.severity,
                               "latency_ms": round(s.latency_ms, 1), "tokens": s.tokens, "cached": s.cached})
            for f in s.findings:
                finding_rows.append({"run_id": run_id, "config_name": name, "claim_id": c["claim_id"],
                                     "sid": f.stage, "check_name": f.check, "severity": f.severity, "text": f.text,
                                     "family": f.family, "score": f.score,
                                     "evidence_json": json.dumps(f.evidence, default=str)})

# ---- per-technique scores for the leaderboard (each technique alone, every claim) ----
tech_rows = []
single = {"s1": ["exif_rules", "exif_full"], "s2": ["phash", "dhash", "orb", "embed"],
          "s5": ["ai_parse_document"] + [f"vlm_ocr:{e}" for e in list(VLMS)[:3]]}
multi = {"s3": ["ela", "jpeg_ghost", "noise", "copy_move_orb", "copy_move_sift", "recapture", "upscale"],
         "s4": [f"vlm:{e}" for e in VLMS] + ["spectral_fft", "spectral_dct", "ela_uniformity"]}
base = preset_config("Production")
for sid, techs in list(single.items()) + list(multi.items()):
    for t in techs:
        cfg = copy.deepcopy(base)
        for s2 in ("s1", "s2", "s3", "s4", "s5"):
            cfg[s2]["enabled"] = s2 == sid
        cfg[sid]["tech"] = [t] if sid in multi else t
        cfg["s6"]["params"]["explainer"] = "template"
        print("  technique", sid, t, flush=True)
        for c in C:
            r = run_claim(c, RAW[c["claim_id"]], REC.get(c["claim_id"]), cfg, ctx)
            s = r.stage(sid)
            fired = [f for f in s.findings if f.severity in ("red", "amber")]
            tech_rows.append({"claim_id": c["claim_id"], "family": c["family"], "fraud_class": c["fraud_class"],
                              "sid": sid, "technique": t, "label": option_label(sid, t), "status": s.status,
                              "score": s.score, "severity": s.severity, "fired": bool(fired),
                              "latency_ms": round(s.latency_ms, 1), "tokens": s.tokens})

# ---- tables ----
claims_df = pd.DataFrame(C)
claims_df["claim_date"] = pd.to_datetime(claims_df["claim_date"]).dt.date
pd.DataFrame(runs).to_parquet(os.path.join(OUT, "runs.parquet"), index=False)
pd.DataFrame(stage_rows).to_parquet(os.path.join(OUT, "stage_results.parquet"), index=False)
pd.DataFrame(finding_rows).to_parquet(os.path.join(OUT, "findings.parquet"), index=False)
pd.DataFrame(tech_rows).to_parquet(os.path.join(OUT, "tech_scores.parquet"), index=False)
claims_df.to_parquet(os.path.join(OUT, "claims.parquet"), index=False)
pd.DataFrame([{"claim_id": e.claim_id, "claimant_id": e.claimant_id, "claim_date": e.claim_date,
               "image_path": e.image_path, "phash": str(e.phash), "dhash": str(e.dhash),
               "embedding": json.dumps([round(float(x), 5) for x in e.emb]), "indexed_at": NOW.isoformat()}
              for e in idx.entries]).to_parquet(os.path.join(OUT, "hash_index.parquet"), index=False)
vlm_df = pd.DataFrame([dict(v, anomalies=json.dumps(v.get("anomalies") or [])) for v in VLM.values()])
vlm_df.to_parquet(os.path.join(OUT, "vlm_cache.parquet"), index=False)
pd.DataFrame([dict(o, fields=json.dumps(o.get("fields"))) for o in OCR.values()]).to_parquet(
    os.path.join(OUT, "ocr_cache.parquet"), index=False)
pd.DataFrame([{"claim_id": k[0], "endpoint": k[1], "prompt": k[2], "text": v.get("text"),
               "latency_ms": v.get("latency_ms"), "tokens": v.get("tokens")} for k, v in EXPL.items()]).to_parquet(
    os.path.join(OUT, "explain_cache.parquet"), index=False)
print("rows: runs", len(runs), "stages", len(stage_rows), "findings", len(finding_rows), "tech", len(tech_rows))
summary = pd.DataFrame(runs).merge(claims_df[["claim_id", "family"]], on="claim_id")
print(summary.groupby(["config_name", "family", "verdict"]).size().unstack(fill_value=0))

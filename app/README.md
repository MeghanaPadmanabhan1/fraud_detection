# Fraud Defence Workbench (prototype)

Databricks App (Streamlit) implementing the *Fraud Defence Workbench — Architecture, Technology & Experience Design*.
URL: https://fraud-defence-workbench-1444828305810485.aws.databricksapps.com  (workspace e2-demo-field-eng)

## Screens (simplified)
Plain-language wording lives in `plain.py`; the engine keeps its technical names.
1. **Claims** — *Inspect a claim*: one-sentence verdict banner, the photo with a "Show" switch (edited areas, same as an
   earlier claim…), up to three reasons each with "See the evidence", a strip of five checks (Photo details, Seen before?,
   Signs of editing, Made by AI?, Receipt) and a collapsed "More detail" (metadata, receipt, speed, every finding, test-set
   answer). *Review queue*: flagged claims, one-click labels + reason codes → `labels` table + audit log.
   *Check a new photo*: upload a photo (and optional receipt), fill in the claim, get the same banner / reasons / five
   checks. Nothing is saved (files live in the browser session; "Clear" removes them; the audit log records only a hash,
   size and verdict). Four example photos built from earlier claims (cropped, screenshot, forwarded, AI-tagged) work
   without a file. Uploads are validated before decoding (JPEG/PNG, 15 MB, 40 MP). New photos are embedded by the SSCD
   endpoint; the receipt is read by a vision model (the document reader only has stored results for test claims).
2. **Compare methods** — *Try a different method*: plain dropdown per check, one claim before/after, then all test claims
   (fraud flagged, firm "likely fraud" calls, genuine wrongly flagged, time, by kind of fraud). *Results*: three headline
   numbers, detailed charts behind a toggle. *Expert switchboard*: the original Pipeline Lab (thresholds, A/B, batch).
3. **Admin** — routing rules, legal tracker, retention, audit log, dataset card.
Sidebar: role picker (changes which page opens first) and a Settings drawer (setup in use, index refresh).

## Architecture
- Engine: `image_authenticity/workbench/` (shared by batch build and app). Rules engine decides; explainer words it.
- Stage 2 reuse has a selector: pHash / dHash / ORB / thumbnail embedding / **SSCD** (Meta's learned copy-detection
  descriptor, `sscd_min_cos` 0.60). SSCD is served by the Model Serving endpoint `image-authenticity-sscd` (UC model `image_authenticity.workbench.sscd`,
  registered by `build/register_sscd.py`, scale-to-zero off). Claims already indexed use vectors precomputed into
  `image_authenticity.workbench.sscd_index` (`build/sscd_index.py`); new photos call the endpoint (8 s timeout, 60 s
  circuit breaker). If it does not answer, the check says "vector unavailable" instead of guessing. No torch or
  weights in the app.
- Fast path (stages 0–3) runs live in the app (~0.1–0.4 s). Slow path (vision models, OCR, VLM explanations) is
  precomputed into Gold caches; the app calls models live only for configs/claims not yet cached.
- Data: `image_authenticity.workbench.*` (claims, runs, stage_results, findings, tech_scores, hash_index,
  vlm_cache, ocr_cache, explain_cache, labels, audit_log, governance_items, saved_configs) + volume `claims_media`.
- Gold set v0 rebuild: `build/fetch_pool.py` → `build/build_dataset.py` → `build/build_caches.py` → `build/run_batch.py`
  (run locally against the workspace; then load parquet to Delta).

## 20-minute demo script (from the design doc)
1. Inspector: open CLM-1054 (AI-edited damage that also reuses an earlier claim's photo) — verdict, reasons, overlays.
2. Lab: load Challenger A into B, A/B the cropped reuse CLM-1029 — pHash misses, ORB catches; batch-run both.
3. Observability: signal heatmap — the noise-residual column is red for everything (a broken check); leaderboard.
4. Review Queue: label a claim; show it in the audit log and reviewer-agreement view.

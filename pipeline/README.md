# Lakeflow pipeline: raw claims to decisions

**Raw data.** `landing/claims_batch_{1,2,3}.jsonl` are the raw claim feed (30, 22 and 14 synthetic claim records, 66 in all).
`make_landing_feed.py` regenerates them from the claims table. The photos and receipts they point to are public
Wikimedia Commons images in the `claims_media` volume.

**Run it.**
1. Copy `landing/*.jsonl` to `/Volumes/<catalog>/workbench/claims_media/landing/claims/`.
2. Create a serverless Lakeflow declarative pipeline with `claims_pipeline.sql` as its source, catalog `image_authenticity`, schema `lakeflow`.
3. Start an update. It builds two bronze streaming tables (Auto Loader), `silver_claims` (quality rules), and three gold views.

**What a real run produced:** `../evidence/01_lakeflow_pipeline_output.md` (row counts, three landed files, decisions vs the known label) and `../evidence/02_lakeflow_event_log.txt`.

The gold tables join the detection engine's stored verdicts (`workbench.runs`, `findings`), so the pipeline's output is
what Genie, Lakebase and the app read.

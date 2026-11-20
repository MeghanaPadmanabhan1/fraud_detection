# Image Authenticity POC — deepfake / shallow-fake / provenance detection

## End-to-end build: raw claims to a business app (Databricks)

**Problem:** device-protection claims arrive with a photo and a receipt. Fraud is a reused photo, an AI-made or edited
photo, or a receipt that does not support the claim. This build catches them, explains why, and puts a ranked queue in
front of investigators.

| Stage | What was built | Code | Output as text |
|---|---|---|---|
| Lakeflow | Auto Loader pipeline: raw JSON claim feed and photo files to bronze, silver (quality rules) and gold | `pipeline/claims_pipeline.sql` | `evidence/01_*`, `evidence/02_*` |
| Unity Catalog | Catalog, schemas, volume, registered SSCD model, tags, comments, least-privilege grants | `uc/governance.sql` | `evidence/README.md` |
| Lakebase | Review queue synced to Postgres; reviewer decisions and audit log written to Postgres | `lakebase/`, `app/lakebase.py` | `evidence/04_*` |
| ML and Gen AI | SSCD copy detection on Model Serving; vision models for scene and receipt; rules engine; explainer | `image_authenticity/workbench/` | `ML_EXPERIMENT.md` |
| Genie | A space over the gold tables, also used inside the app | `genie/`, `app/genie_client.py` | `evidence/03_*` |
| Databricks app | Inspector, photo upload, business view, ask-the-data, compare methods, admin | `app/` | `evidence/05_*` |

Deploy notes: replace the `<genie-space-id>` placeholder in `app/app.yaml` with your Genie space id; set `LAKEBASE_HOST` there only if the app identity cannot read instance metadata. Replace `<app-service-principal-client-id>` in `uc/governance.sql` with the app's client ID.

Business deck: `deck/DECK.md`. Evidence index: `evidence/README.md`.
Data is synthetic claim records over public Wikimedia Commons photos. No customer data is used.

---

## Original modular POC (detectors)

Baseline, **modular** image-fraud detection for the insurance_firm <> Databricks POC.
Built on `<workspace>`. Every detection technique is a swappable *slot*:
changing which model fills a slot is a one-line config edit, not a code change.

## Why modular
Mandela wants to experiment and A/B models per step and eventually combine them
into one composite fraud score. So each step is a `slot`, each slot has one or
more registered implementations, and `config.py` picks the active one.

```
image bytes
   │  Pipeline.run()
   ├─ DEEPFAKE arm ─────────────────────────────┐
   │   spectral   (fft_radial_v1)   FFT power spectrum
   │   ela_global (ela_uniformity_v1) uniform ELA => synthetic
   │   learned    (vlm_zeroshot_v1)  FMAPI vision LLM, zero-shot + context
   ├─ SHALLOW-FAKE arm ─────────────────────────┤
   │   ela_local  (ela_heatmap_v1)   localized ELA hotspots + heatmap
   │   copy_move  (orb_v1)           cloned-region keypoint matching
   │   noise      (noise_residual_v1) clustered noise inconsistency (splice)
   │   jpeg_ghost (jpeg_ghost_v1)    prior-edit compression ghost
   ├─ PROVENANCE arm ───────────────────────────┤
   │   watermark  (c2pa_synthid_v1)  C2PA + EXIF + SynthID slot (stubbed)
   └─ composite.py → fraud_score, per-arm confidence, fired signals, narrative
```

## Swap a model for a slot
Edit `CONFIG_OVERRIDES` in `notebooks/02_run_pipeline.py` (or call
`get_config(overrides)`):

```python
get_config({"slots": {
    "learned":  {"params": {"endpoint": "databricks-gpt-5"}},  # try a different VLM
    "spectral": {"impl": "fft_radial_v1", "weight": 0.4},       # retune weight
}})
```

`image_authenticity.core.available()` lists every registered impl per slot.

## Add a new model to a slot
1. Add a module under `image_authenticity/detectors/` with a `Detector` subclass
   (set `slot`, `name`, `version`; implement `analyze`) decorated with `@register`.
2. Import it in `detectors/__init__.py`.
3. Point the slot at it in config. Done.

## On Databricks (`image_authenticity` catalog)
- `notebooks/01_generate_samples.py` — builds a labeled sample set:
  real photos (`flower_photos`), programmatic shallow-fakes, and **real AI images
  generated via the Gemini image FMAPI endpoint** (they carry SynthID + an IPTC
  `trainedAlgorithmicMedia` provenance tag). → `raw.sample_manifest`
- `notebooks/02_run_pipeline.py` — runs the pipeline, writes `results.detections`
  (per-signal) and `results.scores` (composite) + heatmaps, prints a calibration
  summary (mean score per slot × true label) for threshold tuning.
- AI/BI dashboard reads `results.scores` / `results.detections`.

## What the baseline run showed (42 images: 18 real, 9 Gemini, 15 edited)
| true label | mean fraud score | verdict outcome |
|---|---|---|
| authentic | **0.10** | 18/18 `likely_authentic` (zero false positives) |
| deepfake | **0.80** | 9/9 `likely_fraudulent` |
| shallow_fake | 0.24 | 4/15 flagged (`needs_review`) |

**Headline finding for the POC:** against a SOTA generator (Gemini), the lightweight
*spectral* and *ELA-uniformity* heuristics barely separate real from synthetic
(hf-ratio 0.80 vs 0.76; ELA spread 0.33 vs 0.33), and even a strong **VLM can't
reliably call a photorealistic AI image fake from pixels alone** (some scored
ai-prob 0.05 — "consistent lighting, realistic reflections"). What *does* catch
every Gemini image is **verifiable provenance**: the IPTC `trainedAlgorithmicMedia`
tag (and C2PA). So the composite treats provenance **asymmetrically** — a present
AI-provenance tag dominates (→ fraudulent), but its *absence* never implies
authenticity (fraudsters strip metadata). This is the concrete case for
"detection beyond C2PA / SynthID / watermarks" that Mandela asked for.

Shallow-fakes are the harder half: splice 3/5, copy-move 1/5, local-recompress
0/5 caught. The forensic detectors need per-slot tuning (and better still,
trained models like Noiseprint) — exactly the iteration the slot design enables.
Heatmaps are written for every image regardless of score, so an adjuster can see
*where* a region looks edited even when the composite stays under threshold.

## Honest limitations (baseline)
- **SynthID**: no public verifier API exists; detection is Google-proprietary.
  The `watermark` slot reads C2PA/IPTC/EXIF provenance and *stubs* SynthID
  (returns `indeterminate` + explanation), wired to accept a real verifier later.
- The CPU detectors are transparent heuristics, not trained SOTA models. The
  point of the slot design is to swap trained models (DIRE, UnivFD, Noiseprint,
  a frequency-domain classifier) in per slot as they're validated.
- Thresholds are first-pass; calibrate against real suspect images once shared.

## Local dev
`python -m venv .venv && .venv/bin/pip install numpy opencv-python-headless pillow scipy`
then `.venv/bin/python local_smoketest.py` (runs the CPU detectors; the `learned`
slot needs Databricks auth).

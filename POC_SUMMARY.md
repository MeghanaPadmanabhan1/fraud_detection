# Image Authenticity POC — Implementation, Status & Performance

**Asurion × Databricks** · AI-generated-image fraud detection · built on `e2-demo-field-eng`
Last updated: 2026-10-01

This POC delivers the capability Asurion's current pipeline lacks: detecting
**AI-generated (deepfake)** and **manipulated (shallow-fake)** claim images, plus
**provenance / watermark** checks — as a **modular, swappable framework** where each
detection step can host multiple interchangeable models for experimentation.

> Code links point to the deployed workspace copy under
> `/Workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/`.

---

## 1. How it's structured

```
ARM (the question)  →  FUNCTIONALITY (a capability)  →  MODEL (swappable implementation)
```

- **Arm** — the question being asked: *is the whole image synthetic?* (deepfake),
  *was a real photo edited?* (shallow-fake), *does the file declare its origin?* (provenance).
- **Functionality** — one detection capability (e.g. Spectral, Copy-move, Learned/VLM).
- **Model** — the concrete implementation plugged into a functionality. Swapping a model
  is a one-line config/parameter change; every model returns the same
  `DetectionResult{score, label, evidence, heatmap}`.

Core framework code:

| File | Role |
|---|---|
| [`core.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/core.py) | `Detector` interface + registry (`@register`, `build`, `available`) |
| [`config.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/config.py) | which model fills each functionality + arm weights |
| [`pipeline.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/pipeline.py) | runs the selected models over an image |
| [`composite.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/composite.py) | blends scores → fraud score + narrative (asymmetric provenance override) |
| [`fmapi.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/fmapi.py) | version-independent Foundation Model API calls |

---

## 2. Functionalities implemented — what we chose, and the libraries used

| Arm | Functionality | Casts net on | Model(s) implemented | Library / service | Code |
|---|---|---|---|---|---|
| Deepfake | Spectral | Whole-image AI via frequency artifacts | FFT radial power spectrum; DCT high-freq energy | `numpy.fft`, `opencv` (`cv2.dct`) | [`spectral.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/detectors/spectral.py) |
| Deepfake | ELA uniformity | Suspiciously uniform recompression error | ELA uniformity (q90) | `opencv`, `numpy` | [`ela.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/detectors/ela.py) |
| Deepfake | **Learned (VLM)** | AI generation + edits + device ID + context | Claude Sonnet 5, GPT-5, Gemini 3.1 Pro, Llama-4 Maverick | Databricks **Foundation Model API** via [`fmapi.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/fmapi.py) | [`vlm.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/detectors/vlm.py) |
| Shallow-fake | ELA heatmap | Localized edited region (+ adjuster heatmap) | ELA localized hotspots (q90) | `opencv`, `numpy` | [`ela.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/detectors/ela.py) |
| Shallow-fake | Copy-move | Region cloned within same image | ORB keypoints; SIFT keypoints | `opencv` (`ORB`, `SIFT`, `BFMatcher`) | [`copy_move.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/detectors/copy_move.py) |
| Shallow-fake | Noise / splice | Region spliced from another source | Noise-residual inconsistency | `opencv`, `numpy` | [`noise.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/detectors/noise.py) |
| Shallow-fake | JPEG ghost | Region with different prior compression | JPEG ghost (quality sweep) | `opencv`, `numpy` | [`jpeg_ghost.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/detectors/jpeg_ghost.py) |
| Provenance | Watermark / C2PA | Verifiable AI-origin declaration | C2PA + IPTC `trainedAlgorithmicMedia` + SynthID stub | `Pillow` (EXIF) + byte-scan | [`watermark.py`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/image_authenticity/detectors/watermark.py) |

**Design intent:** the baseline deliberately uses lightweight, dependency-free
techniques (OpenCV/NumPy/Pillow) + managed Foundation Models — no training, no model
weights to download, runs immediately. Sophistication is additive (see §5).

---

## 3. Current performance

Test set: **54 labeled images** — 22 authentic (18 `flower_photos` + **4 real phone
photos** from Wikimedia), 23 shallow-fakes (15 edited flowers + **8 edited real phones**),
9 AI images generated via the Gemini image endpoint (carry real provenance tags). Score
0–1, higher = more suspicious. **Separation** = mean score on a functionality's target
fake class − mean on authentic (higher = better discriminator). Default Learned model =
**Llama-4 Maverick** (see §4).

### Per-functionality (default models, 54 images)

| Functionality | Model | authentic | target class | Read |
|---|---|---|---|---|
| Provenance · Watermark/C2PA | C2PA+IPTC+SynthID stub | 0.50* | deepfake **0.90** | ✅ Decisive (9/9 Gemini) |
| Deepfake · Learned (VLM) | **Llama-4 Maverick** | 0.09 | deepfake **0.54**, shallow **0.58** | ✅ Strong wide net (both arms) |
| Shallow-fake · Noise/splice | Noise-residual | 0.30 | shallow 0.43 | ⚠️ noisy, but **deepfake 0.84** |
| Shallow-fake · Copy-move | ORB keypoints | 0.10 | shallow 0.24 | 🟡 Modest (+0.14) |
| Shallow-fake · ELA heatmap | ELA localized | 0.25 | shallow 0.31 | 🟡 Weak (+0.06) |
| Shallow-fake · JPEG ghost | JPEG ghost | 0.20 | shallow 0.20 | ❌ Flat |
| Deepfake · Spectral | FFT radial | 0.06 | deepfake 0.07 | ❌ Flat vs Gemini |
| Deepfake · ELA uniformity | ELA uniformity | 0.10 | deepfake 0.08 | ❌ Flat |

\*`0.50` = `indeterminate` (no provenance to verify) → excluded from the composite.

### Composite outcome (full pipeline)

| True class | Verdict outcome | Mean fraud score |
|---|---|---|
| authentic | 21/22 `likely_authentic`, 1 `needs_review` | 0.12 |
| deepfake | **9/9 `likely_fraudulent`** | 0.80 |
| shallow_fake | **10/23 flagged** (`needs_review`), 13 `likely_authentic` | 0.26 |

**Shallow-fake detection is much stronger on real device photos** (the realistic case):
**phones 7/8 flagged vs flowers 3/15** — device photos have more structure, so cloned/
spliced regions produce stronger signals, and the VLM reads device edits well.

**Headline finding:** against a SOTA generator (Gemini) the classic spectral/ELA
heuristics are flat; what reliably catches synthetic images is **provenance** (0.90), the
**VLM** (semantic), and **noise** (0.84). The composite treats provenance
*asymmetrically* — a present AI-provenance tag dominates, but its *absence* never implies
authenticity (fraudsters strip metadata). Swapping the default Learned model to Llama-4
Maverick raised recall on both fake types, at the cost of one authentic image tipping to
`needs_review` (a recall/false-positive trade-off worth tuning).

---

## 4. VLM model A/B (strict — identical 54-image set)

The **Learned** functionality runs on any of four vision models; each run appends to
**`image_authenticity.results.model_eval`**, charted on the dashboard's **Model A/B**
page. Head-to-head on the same 54 images:

| VLM | Separation | authentic (false-positive) | deepfake |
|---|---|---|---|
| **Llama-4 Maverick** (default) | **0.328** | 0.105 | 0.433 |
| Claude Sonnet 5 | 0.288 | **0.045** (cleanest) | 0.333 |
| Gemini 3.1 Pro | 0.255 | 0.073 | 0.328 |
| GPT-5 | 0.225 | 0.173 (noisiest) | 0.398 |

**Llama-4 Maverick** has the best separation and is the new default; **Claude Sonnet 5**
has the lowest false-positive rate — the better pick if minimizing false positives is the
priority. (Gemini initially returned nulls because it emits list-structured message
content vs. a string; fixed in the FMAPI helper.)

---

## 5. Implemented vs. TBD, and what we could add next

### ✅ Implemented (this POC)
- Modular arm → functionality → model framework with config/widget/job-parameter model switching.
- 8 functionalities, 11 model implementations (incl. 4 VLMs).
- Labeled data generation: real flowers, real Gemini deepfakes, and **real phone photos** (authentic + shallow-faked).
- Composite scoring + explainable narrative + per-image ELA/noise/copy-move **heatmaps**.
- Delta result tables, per-functionality evaluation (`model_eval`), AI/BI dashboard (Overview + Model A/B).
- **Completed strict four-way VLM A/B** on the 54-image set → Llama-4 Maverick set as default.

### 🟡 Partial / known weak spots
- Shallow-fake detection is strong on real phones (7/8) but weak on flowers and on the
  **local-recompress** tactic — the forensic ELA/JPEG-ghost detectors need tuning or trained models.
- Thresholds calibrated on 54 synthetic/curated images — recalibrate on Asurion's real suspect set.

### ⬜ TBD — more functionalities & model options we could add

**Deepfake (stronger models for the Learned/Spectral functionalities)**
- **DIRE** (diffusion reconstruction error) — strong vs diffusion/Gemini.
- **UnivFD / Ojha CLIP-linear-probe** — generalizes across generators.
- **Wang-ResNet** (CNN trained on GAN images) — for GAN-origin content.
- **GAN fingerprinting / source attribution** — "which generator made this" (Phase 2).
- Off-the-shelf HuggingFace AI-image classifiers.

**Shallow-fake (localization)**
- **Noiseprint** (trained camera-noise CNN) — SOTA splice localization.
- **ManTraNet / learned forgery-localization** models.

**Provenance**
- **Real SynthID verifier** (Google-proprietary; stubbed today, drop-in ready).
- Full **C2PA signature verification** via the `c2pa` library (vs. today's tag scan).

**Beyond this POC's scope — the rest of Asurion's 8-model vision**
- Reverse-image / external velocity, internal velocity (perceptual-hash / vector dedup),
  OCR (`ai_extract` / Textract), image recognition & contextual analysis (partially
  covered today by the VLM's device-ID + context output).

**Delivery mechanics for heavy models (recommended):** serve big GPU models as
**Databricks Model Serving endpoints** (as the VLM does today) and call them from a
detector — keeps the pipeline light, versions/governs the model in Unity Catalog, scales
independently, and mirrors Asurion's API-gateway architecture. Register models in UC;
declare heavy dependencies in a serving container or defined environment (not ad-hoc
`%pip`).

---

## 6. Latency & the 30-second inference timeout

Asurion's current constraint is a **30-second timeout at the API gateway** fronting
inference, while some models need up to ~60s — which forces their interim workaround of
running models in the background and surfacing results at a later checkpoint in the claim
flow. (That ceiling is a *gateway/platform* limit, not a model limit; if the gateway is
Azure — e.g. API Management or Application Gateway — its default request/response and idle
timeouts are a common source of exactly this behavior.)

How this maps on Databricks:

- **Batch / async scoring — what this POC already does, and the robust answer.** The
  pipeline runs as a **job** that scores images and writes results to Delta
  (`results.scores`). There is no synchronous request/response gateway in the loop, so the
  30-second ceiling never applies; the claim app reads the scored result at the next
  checkpoint. This is exactly Asurion's stated interim pattern, but native to the
  lakehouse — and the results table can be mirrored to **Lakebase/Postgres** for
  low-latency app reads. `ai_query` can batch-score many images in a single Spark job.
- **Real-time Model Serving when a synchronous answer is required.** A Databricks Model
  Serving endpoint's request timeout is configurable and well beyond a 30-second edge
  gateway. The forensic CPU detectors (spectral, ELA, copy-move, noise, JPEG-ghost) each
  return in well under a second, so a "fast-signals" endpoint can answer comfortably
  inside 30s.
- **Split fast vs. slow signals.** Return the sub-second forensic + provenance signals
  synchronously for an immediate partial fraud score, and run the heavier VLM / future GPU
  models asynchronously, merging at the next checkpoint. Because each functionality is an
  independent detector, they can be fanned out concurrently — wall-clock ≈ the slowest
  active model, not the sum.

**Honest status:** this POC runs in **batch/async** mode (no synchronous gateway in the
loop), which is itself the circumvention and matches Asurion's interim plan. We have
**not** yet built a sub-30s synchronous serving endpoint — that's a straightforward next
step (serve the fast signals; keep the VLM async).

## 7. Where it lives / how to run

- **Catalog:** `image_authenticity` — `raw` (images + manifest), `results` (`scores`,
  `detections`, `model_eval`), volumes (`images`, `heatmaps`).
- **Notebooks:** [`01_generate_samples`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/notebooks/01_generate_samples) (build labeled data) ·
  [`02_run_pipeline`](https://e2-demo-field-eng.cloud.databricks.com/#workspace/Users/meghana.padmanabhan@databricks.com/asurion-image-authenticity/notebooks/02_run_pipeline) (select models, run, evaluate).
- **Job:** [Run Detection Pipeline (02)](https://e2-demo-field-eng.cloud.databricks.com/jobs/491613969987603) — model picks exposed as base parameters (headless-overridable). Pinned serverless environment version 2.
- **Dashboard:** [Image Authenticity POC](https://e2-demo-field-eng.cloud.databricks.com/sql/dashboardsv3/01f1bd4b648d1aaaa33097df07693a04/published) — Overview + Model A/B pages.

---

## 8. Honest limitations
- **SynthID** invisible pixel watermark has no public verifier (Google-proprietary); the
  slot verifies C2PA/IPTC provenance instead and is stubbed for a real verifier.
- CPU heuristics are transparent baselines, not SOTA; the point of the slot design is to
  swap trained models in per functionality.
- Thresholds are first-pass, calibrated on 42 synthetic samples — recalibrate against
  Asurion's real suspect images.
- Shallow-fake detection is the weak arm and the first place to invest.

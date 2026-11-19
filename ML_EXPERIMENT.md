# Do learned detectors beat the OpenCV / pixel-level checks? — experiment report

Gold set v0 (66 claims), run 2026-10-29. Full tables: `wb_data/ml_exp/results.md`. Scripts: `build/ml_exp_*.py`, `build/ml_spai_run.py`.

## Bottom line

| Stage | Question | Verdict |
|---|---|---|
| **2 · Reuse** | Does a learned copy-detector beat the app's pHash? | **Yes, clearly.** SSCD catches 85–94% of reuse claims at zero false alarms vs 65% for pHash. It matches, but does not beat, ORB+RANSAC on accuracy; its advantage is that it scales. |
| **4 · AI-made images** | Do the CLIP detector or SPAI beat the OpenCV checks on Gemini images? | **No — neither works.** After removing file-format cues, both are at or below chance, as are the OpenCV checks. A trivial aspect-ratio probe (AUC 0.74) outscores all of them. |
| **3 · Edits (splice / copy-move)** | Do they beat OpenCV? | **No.** Nothing here exceeds ~0.7 AUC on these. Dedicated forensic-ML localizers (TruFor etc.) were **not tested**. |

What the learned detectors *do* detect is resampling: upscaled photos and re-photographed screens (CLIP AUC 0.92–0.99, SPAI 0.80–0.93) — but the app's own upscale detector already does that (0.97–1.00).

## What was tested

| Model | Source (commit / licence) | Weights (SHA-256) | Used as |
|---|---|---|---|
| SSCD mixup / large (Meta, CVPR 2022) | facebookresearch/sscd-copy-detection, MIT | `9f26bd4c…` / `16a19635…` (dl.fbaipublicfiles.com) | Reuse: cosine to earlier claims |
| CLIP detector (Cozzolino et al., CVPRW 2024) | grip-unina/ClipBased-SyntheticImageDetection @ `c76ef7f`, Apache-2.0 | heads verified against LFS SHA; backbone `laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K` safetensors `947d3262…` | AI-image score (LLR > 0 = synthetic), both released heads |
| SPAI (Karageorgiou et al., CVPR 2025) | mever-team/spai @ `8ff7b3b`, Apache-2.0 | `24159f27…` (Google Drive link from the README) | AI-image score (sigmoid, p > 0.5) |

Baselines: the Workbench's own checks (ELA, JPEG ghost, noise residual, copy-move ORB/SIFT, FFT/DCT spectral, ELA uniformity, upscale, recapture; pHash, dHash, ORB+RANSAC, thumbnail cosine), run through the same code as the app. Reference: the four vision LLMs from the app's cache (original images only).

**Protocol (fixed before looking at results).** Negatives: the 19 genuine photos. Metrics: AUC with stratified bootstrap 95% CI (2,000 resamples); recall at zero false positives (in-sample threshold, optimistic but identical for all); false-positive rate at each detector's own default threshold. Two image sets: `orig`, and `social` (longest side ≤ 1024 px, JPEG q70, metadata stripped — a messaging-app forward). Reuse ground truth: a claim is a reuse if an earlier claim was built from the same source photo (34 reuse vs 32 first uses).

**Why `social` matters.** In `orig`, AI images differ from the genuine photos in file size (~1 MB vs ~270 KB), format (50–60% PNG vs 0%) and aspect ratio. A "bytes per pixel" probe alone scores AUC 0.83 on AI images. Several OpenCV scores that look good in `orig` (ELA uniformity 0.75, recapture 0.77) are format cues, not forensics: they drop to 0.60 and 0.50 in `social`. Aspect ratio (0.74) cannot be removed.

## Results

### Reuse (Stage 2) — AUC [95% CI] · recall at zero false positives

| Method | `orig` | `social` |
|---|---|---|
| pHash (app default) | 0.87 [0.78–0.95] · 65% | 0.87 [0.77–0.95] · 65% |
| dHash | 0.95 [0.90–0.99] · 65% | 0.95 [0.89–0.99] · 65% |
| Thumbnail cosine (app "embedding") | 0.82 [0.71–0.92] · 68% | 0.82 [0.71–0.92] · 68% |
| ORB + RANSAC | 0.94 [0.86–1.00] · 91% | 0.95 [0.89–1.00] · 91% |
| **SSCD mixup** | 0.94 [0.86–1.00] · 85% | 0.95 [0.87–1.00] · 85% |
| **SSCD large** | 0.96 [0.89–1.00] · 88% | 0.96 [0.90–1.00] · 85% |

- SSCD vs ORB: confidence intervals overlap almost completely — **no accuracy difference can be claimed**. ORB compares pairs (1.4 ms per pair, a linear scan of the index); SSCD gives a 512-d vector (51 ms to embed once) that fits Vector Search.
- Hashes fail on crops (pHash 0%, dHash 33%) and screenshots (0%). SSCD and ORB catch all crops.
- **Preprocessing caveat.** My pre-specified run squashed images to 320×320, which hurt tall screenshots (SSCD recall 25–50%). A **post-hoc** check with the other resize the SSCD README allows (short edge 288) raised recall at zero false positives to **94%** for both models, screenshots to 100%. I picked that after seeing the result, so treat 94% as optimistic; the pre-specified figure is 85–88%.
- SSCD's published threshold (cosine > 0.75) is too conservative here (recall 47–68% depending on model and image set); it must be recalibrated on real data.

### AI-made images (Stage 4) — AUC [95% CI], n = 13 AI claims vs 19 genuine

| Detector | `orig` | `social` (strict) | AI with provenance stripped (`orig` / `social`, n=6) |
|---|---|---|---|
| SPAI | 0.70 [0.50–0.86] | 0.48 [0.27–0.69] | 0.57 / 0.42 |
| CLIP detector (best head) | 0.53 [0.32–0.72] | 0.43 [0.22–0.64] | 0.55 / 0.50 |
| Best OpenCV check (ELA uniformity) | 0.75 [0.55–0.91] | 0.60 [0.38–0.79] | 0.54 / 0.59 |
| Spectral FFT / DCT | 0.40 / 0.42 | 0.28 / 0.42 | FFT 0.31, DCT 0.29 (`orig`) |
| *Probe: aspect ratio* | *0.74* | *0.74* | *0.77* |
| *Probe: bytes per pixel* | *0.83* | *0.65* | *0.64 / 0.70* |
| Vision LLMs, for reference (`orig` only) | Llama-4 0.82, GPT-5 0.80, Claude 0.65, Gemini 0.35 | not re-run | Llama-4 0.79, GPT-5 0.78, Claude 0.81 |

At each detector's own default threshold, on genuine photos: **SPAI flags 16% (`orig`) and 26% (`social`)** — too noisy for a team whose main pain is false positives — and catches 31% of AI images. The CLIP detector flags 5% and catches 0%.

The CLIP detector's failure is not a pipeline bug: all 295 backbone tensors equal the downloaded file, the head matches, and zero-shot CLIP correctly labels our phone photos (`build/ml_exp_verify_clip.py`). It also behaves sensibly elsewhere (AUC 0.99 on upscaled photos).

### Other classes (exploratory — only 4 claims each)

- Upscale / re-photograph, AUC `orig` / `social`: CLIP 0.99 / 0.92 and 0.96 / 0.87; SPAI 0.93 / 0.80 and 0.83 / 0.91; the app's upscale detector 0.97 / 1.00 and 1.00 / 0.97.
- Splice and copy-move: no OpenCV or learned-pixel detector exceeds 0.74; the dedicated copy-move checks score 0.64–0.74 on copy-move. Vision LLMs are best (GPT-5 0.84 splice, 0.91 copy-move; `orig`).
- The app's ELA-hotspot and JPEG-ghost checks sit at or below chance on AI and splice claims, and the noise-residual check flags 32% (`orig`) to 58% (`social`) of genuine photos at its production threshold.

### Cost (Apple M2 Pro, CPU)

SSCD 51 / 150 ms per image · CLIP detector 243 ms · SPAI 4.5 s (2.2 s on social-size images) · OpenCV checks 2–79 ms.

## Caveats — read before quoting any number

1. **Small, synthetic set.** 66 claims, 13 AI, 19 genuine photos. CIs are wide: differences under ~0.2 AUC are noise. Per-class tables have n = 4–8.
2. **One generator.** All AI images come from Gemini (3 Pro Image) — "no detector works" means *on this generator*. Both ML detectors were published before it. I had no in-distribution positive control for them.
3. **Thresholds are in-sample** (set on the same 19 genuine photos), so recall-at-zero-false-positives is optimistic in absolute terms.
4. **Confounds** (format, size, aspect ratio) are reduced by `social` but not removed.
5. **Models used as released**, no fine-tuning. Deepfake-Eval-2024 reports that off-the-shelf image detectors lose ~45% AUC on in-the-wild content and that fine-tuning helps — so a fine-tuned result could differ.
6. **Not tested:** TruFor (licence is nonprofit-only), IML-ViT, FakeShield, Noiseprint, AIDE, UnivFD, DIRE, and any GPU model. Vision-LLM numbers come from the app's earlier cache and were not re-run on `social`.
7. **Safety handling.** Weights were downloaded into separate directories, hashes recorded, and Python ran in isolated mode. SPAI's checkpoint uses pickle: I scanned its imports statically and loaded it through an unpickler allowing only six harmless globals (it would have raised on anything else). The CLIP backbone was loaded from safetensors, not the pickle `.bin`. Shims for SPAI: `.cuda()` mapped to CPU, unused `clip`/`neptune`/`tsnecuda` imports stubbed, data loaded in-process.

## Recommendations

1. **Add SSCD as a Stage 2 challenger**, backed by Vector Search; keep ORB+RANSAC to verify the top candidates (retrieve with SSCD, confirm with ORB). Recalibrate the threshold on adjudicated claims. Use the aspect-preserving resize.
2. **Do not add the CLIP detector or SPAI to Stage 4 as-is.** For AI-made photos, lean on the provenance tag (when intact) and the vision models; if a learned detector is wanted, fine-tune on Asurion's own labelled photos and re-test across several generators.
3. **Demote** ELA-hotspot, JPEG-ghost and noise-residual in the default configuration — they add false positives without measurable recall on this data.
4. Before trusting any AI-image detector's headline number, run it on a compression-normalised set alongside a trivial metadata probe, as done here.

---

# Addendum: held-out test v1 (2026-11-02)

Second run on data that played no part in choosing any threshold (thresholds frozen from gold set v0).
Full tables: `wb_data/ml_v1/results.md`. Code: `build/ml_v1_{build,features,reuse,analyze}.py`, `build/fetch_ai_web.py`,
`build/fetch_phone_real.py`, `build/gen_ai_v1.py`; SPAI via `build/ml_spai_run.py dir`.

**Data.** 52 AI-made images: 30 from three Gemini image models (3 Pro, 3.1 Flash, 3.1 Flash-Lite; 10 scenes each, all 1408x768 PNG)
plus 22 Commons images from DALL-E, Bing Image Creator, Adobe Firefly, Ideogram, Midjourney (Commons held no usable images for
the other generators queried; a CLIP zero-shot "photo vs artwork" score kept the most photo-like). 32 real photos: 29 phone-camera
photos from Commons "Taken with <phone>" categories (faces dropped) and 3 damaged-device photos. Three views: `orig`, `social`
(max side 1024, JPEG q70, no metadata) and `norm` (centre square crop, 768 px, JPEG q75, no metadata; removes format, size and aspect cues).

## AI detection (AUC, AI vs real; 95% bootstrap CI)
| Detector | orig | social | norm |
|---|---|---|---|
| CLIP detector (Cozzolino, latent10k+) | 0.79 [0.70-0.88] | 0.71 [0.59-0.81] | 0.77 [0.67-0.86] |
| SPAI | 0.73 [0.61-0.83] | 0.55 [0.42-0.68] | 0.60 [0.46-0.71] |
| Best OpenCV check (upscale) | 0.61 | 0.68 | 0.74 |
| Every other OpenCV/NumPy check | 0.37-0.64 | 0.43-0.63 | 0.37-0.57 |

- **The CLIP detector does better here than on gold set v0, but unevenly.** Gemini only: 0.72 / 0.59 / 0.69 (orig/social/norm).
  Other generators: 0.89 / 0.87 / 0.89. That fits what it was trained on (diffusion and GAN-era generators) and is consistent with
  the first result (0.53/0.43 on Gemini 3 Pro Image alone).
- **At its default threshold (LLR > 0) it is conservative:** 29-38% recall, 0-3% false alarms. Gemini 3 Pro Image: 10% recall.
  Its second head (latent10k) flags nothing at its default threshold on any view.
- **SPAI is not usable as released:** 28-31% of real photos flagged at its default threshold, and AUC is 0.55-0.60 once format cues
  are removed. It also took 10.9 s/img on the original view (full-resolution images) against 2.0 s/img on the others.
- **The format confound is visible again.** An aspect-ratio-only probe scores 0.97 on Gemini vs real on `orig`/`social` and 0.50 on `norm`.
- **No OpenCV check separates AI from real on these photos** (best usable one, upscale, 0.74 on `norm`, 3-8% recall at its default
  threshold). The app's spectral and ELA-uniformity checks flag nothing at their default thresholds.

## Reuse (88 indexed photos; 80 altered copies to find, 40 altered never-indexed photos that must not match)
SSCD vectors came from the deployed Model Serving endpoint (208 calls, median 468 ms, cosine 1.0000 to the local weights).
| Method (frozen threshold) | Found the right source | crop / screenshot / recompress / relight | False hits |
|---|---|---|---|
| pHash (<= 10 of 64) | 34% (27/80) | 0% / 0% / 100% / 35% | 0/40 |
| ORB + RANSAC (>= 25 inliers) | 99% (79/80) | 100% / 95% / 100% / 100% | 3/40 |
| SSCD (cosine >= 0.60) | 100% (80/80) | 100% / 100% / 100% / 100% | 0/40 |

The 0.60 SSCD threshold was set on gold set v0, so this is a held-out confirmation of it, on a different photo population.

## Caveats
1. **Content confound.** The Gemini images show broken devices; the real photos are ordinary Commons phone photos (landscapes, objects)
   and only 3 show damaged devices. The CLIP detector reads CLIP features, so part of its separation may be scene content, not generation artefacts.
2. **Small and uneven.** 32 real photos, 2 Ideogram and 2 Midjourney images. CIs are wide; per-generator figures are indicative only.
3. **Real photos came mostly as 1600 px renditions** from Commons (EXIF stripped); only the first 5 are untouched originals. This affects `orig`; `norm` removes it.
4. **Models as released**, no fine-tuning. Localisation models (TruFor and similar) still untested.
5. **Reuse transforms are synthetic** (crop, screenshot-in-viewer, heavy recompress, rotate + relight), not real fraudsters' edits.
6. The first reuse run died on one failed endpoint call (the endpoint had scaled to zero); it was rerun with retries.

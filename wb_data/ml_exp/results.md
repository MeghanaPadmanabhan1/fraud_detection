# ML detectors vs OpenCV / pixel-level checks — experiment results (gold set v0, 66 claims)


### Image-level detection — `orig` images (19 genuine photos as negatives)

AUC with 95% bootstrap CI · recall at zero false positives (R@0FP). Higher is better; 0.50 = chance.

| Detector | Type | AI-made (generated + AI-edited) (n=13) | conventional edits (splice / copy-move / upscale) (n=12) | screen recapture (screenshot / re-photo) (n=8) |
|---|---|---|---|---|
| probe: bytes per pixel | confound probe | 0.83 [0.68–0.96] · 54% | 0.35 [0.16–0.55] · 0% | 0.58 [0.31–0.82] · 25% |
| vlm:llama-4-maverick | vision LLM (reference) | 0.82 [0.67–0.95] · 15% | 0.81 [0.65–0.95] · 0% | 0.85 [0.68–0.97] · 12% |
| vlm:gpt-5 | vision LLM (reference) | 0.80 [0.63–0.95] · 62% | 0.87 [0.72–0.98] · 58% | 0.90 [0.74–1.00] · 62% |
| Recapture (screen) | OpenCV / pixel | 0.77 [0.62–0.88] · 54% | 0.62 [0.50–0.75] · 25% | 0.75 [0.56–0.94] · 50% |
| ELA uniformity | OpenCV / pixel | 0.75 [0.55–0.91] · 46% | 0.30 [0.13–0.50] · 0% | 0.28 [0.09–0.49] · 0% |
| probe: aspect ratio | confound probe | 0.74 [0.55–0.91] · 0% | 0.47 [0.26–0.67] · 0% | 0.74 [0.47–0.93] · 50% |
| SPAI | learned (spectral ViT) | 0.70 [0.50–0.86] · 31% | 0.61 [0.38–0.81] · 25% | 0.65 [0.41–0.88] · 12% |
| vlm:claude-sonnet-5 | vision LLM (reference) | 0.65 [0.45–0.84] · 31% | 0.72 [0.52–0.89] · 0% | 0.69 [0.40–0.91] · 38% |
| Noise residual | OpenCV / pixel | 0.62 [0.40–0.83] · 0% | 0.48 [0.26–0.69] · 0% | 0.61 [0.30–0.88] · 38% |
| Copy-move SIFT | OpenCV / pixel | 0.59 [0.38–0.79] · 23% | 0.55 [0.35–0.76] · 0% | 0.62 [0.39–0.82] · 0% |
| Upscale detector | OpenCV / pixel | 0.55 [0.35–0.76] · 15% | 0.57 [0.34–0.80] · 25% | 0.94 [0.82–1.00] · 75% |
| Copy-move ORB | OpenCV / pixel | 0.53 [0.36–0.69] · 8% | 0.53 [0.35–0.71] · 8% | 0.62 [0.39–0.84] · 12% |
| CLIP detector (latent10k_plus) | learned (CLIP) | 0.53 [0.32–0.72] · 0% | 0.66 [0.44–0.86] · 33% | 0.85 [0.68–0.97] · 25% |
| Spectral DCT | OpenCV / pixel | 0.42 [0.22–0.62] · 8% | 0.68 [0.44–0.87] · 42% | 0.62 [0.36–0.84] · 12% |
| Spectral FFT | OpenCV / pixel | 0.40 [0.21–0.60] · 0% | 0.67 [0.43–0.87] · 33% | 0.81 [0.58–0.98] · 38% |
| vlm:gemini-3-1-pro | vision LLM (reference) | 0.35 [0.18–0.53] · 0% | 0.61 [0.43–0.78] · 0% | 0.37 [0.18–0.55] · 0% |
| CLIP detector (latent10k) | learned (CLIP) | 0.34 [0.15–0.54] · 0% | 0.57 [0.36–0.80] · 25% | 0.68 [0.45–0.89] · 12% |
| JPEG ghost | OpenCV / pixel | 0.25 [0.10–0.44] · 0% | 0.57 [0.35–0.77] · 8% | 0.77 [0.57–0.93] · 12% |
| ELA hotspots | OpenCV / pixel | 0.25 [0.09–0.45] · 8% | 0.38 [0.19–0.59] · 0% | 0.46 [0.24–0.68] · 0% |

**AI-made claims split by whether the generator's provenance tag survived** (AUC [CI] · R@0FP) — the stripped rows are the real test of the pixel/learned detectors:

| Detector | AI, provenance tag intact (n=7) | AI, provenance stripped (n=6) |
|---|---|---|
| probe: bytes per pixel | 1.00 [1.00–1.00] · 100% | 0.64 [0.41–0.86] · 0% |
| vlm:llama-4-maverick | 0.84 [0.65–0.98] · 14% | 0.79 [0.59–0.96] · 17% |
| vlm:gpt-5 | 0.83 [0.61–1.00] · 71% | 0.78 [0.54–0.99] · 50% |
| Recapture (screen) | 1.00 [1.00–1.00] · 100% | 0.50 [0.50–0.50] · 0% |
| ELA uniformity | 0.94 [0.80–1.00] · 71% | 0.54 [0.25–0.79] · 17% |
| probe: aspect ratio | 0.72 [0.47–0.93] · 0% | 0.77 [0.51–1.00] · 0% |
| SPAI | 0.80 [0.60–0.96] · 43% | 0.57 [0.30–0.82] · 17% |
| vlm:claude-sonnet-5 | 0.52 [0.26–0.77] · 14% | 0.81 [0.58–1.00] · 50% |
| Noise residual | 0.61 [0.32–0.89] · 0% | 0.62 [0.31–0.91] · 0% |
| Copy-move SIFT | 0.58 [0.33–0.80] · 14% | 0.61 [0.29–0.87] · 33% |
| Upscale detector | 0.59 [0.32–0.83] · 14% | 0.50 [0.22–0.77] · 17% |
| Copy-move ORB | 0.49 [0.29–0.71] · 0% | 0.57 [0.34–0.81] · 17% |
| CLIP detector (latent10k_plus) | 0.51 [0.26–0.77] · 0% | 0.55 [0.26–0.80] · 0% |
| Spectral DCT | 0.53 [0.26–0.79] · 14% | 0.29 [0.10–0.53] · 0% |
| Spectral FFT | 0.47 [0.24–0.71] · 0% | 0.31 [0.12–0.52] · 0% |
| vlm:gemini-3-1-pro | 0.39 [0.17–0.65] · 0% | 0.30 [0.12–0.51] · 0% |
| CLIP detector (latent10k) | 0.35 [0.13–0.59] · 0% | 0.32 [0.11–0.59] · 0% |
| JPEG ghost | 0.14 [0.00–0.32] · 0% | 0.38 [0.17–0.62] · 0% |
| ELA hotspots | 0.26 [0.09–0.49] · 0% | 0.23 [0.01–0.54] · 17% |

**Confound probes, best direction** (max of AUC and 1−AUC; what a trivially confounded feature achieves with no forensic signal at all):

| Probe | AI-made (generated + AI-edited) | conventional edits (splice / copy-move / upscale) | screen recapture (screenshot / re-photo) |
|---|---|---|---|
| probe: aspect ratio | 0.74 | 0.53 | 0.74 |
| probe: bytes per pixel | 0.83 | 0.65 | 0.58 |

**Operating point at each detector's own default threshold** (app thresholds for OpenCV checks; CLIP LLR>0; SPAI p>0.5). FPR is on the 19 genuine photos:

| Detector | threshold | false-positive rate | recall: AI-made | recall: conventional edits | recall: recapture |
|---|---|---|---|---|---|
| Recapture (screen) | 0.5 | 0% (0/19) | 0% | 0% | 50% |
| ELA uniformity | 0.5 | 0% (0/19) | 0% | 0% | 0% |
| SPAI | 0.5 | 16% (3/19) | 31% | 33% | 38% |
| Noise residual | 0.99 | 32% (6/19) | 62% | 25% | 50% |
| Copy-move SIFT | 0.922 | 5% (1/19) | 23% | 0% | 0% |
| Upscale detector | 0.5 | 0% (0/19) | 0% | 25% | 50% |
| Copy-move ORB | 0.5 | 0% (0/19) | 8% | 0% | 0% |
| CLIP detector (latent10k_plus) | 0 | 5% (1/19) | 0% | 42% | 25% |
| Spectral DCT | 0.5 | 0% (0/19) | 0% | 17% | 12% |
| Spectral FFT | 0.5 | 0% (0/19) | 0% | 0% | 0% |
| CLIP detector (latent10k) | 0 | 0% (0/19) | 0% | 8% | 12% |
| JPEG ghost | 0.961 | 5% (1/19) | 0% | 8% | 38% |
| ELA hotspots | 0.855 | 5% (1/19) | 8% | 0% | 0% |

**Per-fraud-class AUC, `orig` images** (exploratory: only 4-8 claims per class, so differences smaller than ~0.2 are noise; 0.50 = chance, shown without CIs):

| Detector | ai_generated (n=8) | ai_edited_damage (n=5) | splice (n=4) | copy_move (n=4) | upscale (n=4) | screenshot (n=4) | rephotograph (n=4) |
|---|---|---|---|---|---|---|---|
| CLIP detector (latent10k) | 0.31 | 0.39 | 0.30 | 0.54 | 0.88 | 0.64 | 0.72 |
| CLIP detector (latent10k_plus) | 0.49 | 0.59 | 0.41 | 0.59 | 0.99 | 0.74 | 0.96 |
| Copy-move ORB | 0.58 | 0.46 | 0.58 | 0.66 | 0.34 | 0.61 | 0.62 |
| Copy-move SIFT | 0.65 | 0.49 | 0.56 | 0.66 | 0.45 | 0.70 | 0.54 |
| ELA hotspots | 0.26 | 0.23 | 0.34 | 0.51 | 0.30 | 0.34 | 0.58 |
| ELA uniformity | 0.73 | 0.79 | 0.37 | 0.26 | 0.28 | 0.09 | 0.46 |
| JPEG ghost | 0.28 | 0.20 | 0.41 | 0.55 | 0.74 | 0.95 | 0.59 |
| Noise residual | 0.50 | 0.80 | 0.45 | 0.72 | 0.28 | 0.82 | 0.39 |
| Recapture (screen) | 0.75 | 0.80 | 0.62 | 0.62 | 0.62 | 1.00 | 0.50 |
| SPAI | 0.70 | 0.69 | 0.39 | 0.50 | 0.93 | 0.47 | 0.83 |
| Spectral DCT | 0.36 | 0.51 | 0.41 | 0.67 | 0.95 | 0.46 | 0.79 |
| Spectral FFT | 0.37 | 0.44 | 0.33 | 0.67 | 1.00 | 0.74 | 0.88 |
| Upscale detector | 0.42 | 0.76 | 0.36 | 0.39 | 0.97 | 0.88 | 1.00 |
| probe: aspect ratio | 0.95 | 0.42 | 0.48 | 0.45 | 0.48 | 1.00 | 0.48 |
| probe: bytes per pixel | 0.84 | 0.83 | 0.59 | 0.30 | 0.16 | 0.83 | 0.33 |
| vlm:claude-sonnet-5 | 0.87 | 0.30 | 0.80 | 0.58 | 0.78 | 0.74 | 0.63 |
| vlm:gemini-3-1-pro | 0.31 | 0.41 | 0.67 | 0.58 | 0.58 | 0.26 | 0.47 |
| vlm:gpt-5 | 0.83 | 0.76 | 0.84 | 0.91 | 0.85 | 0.97 | 0.84 |
| vlm:llama-4-maverick | 0.85 | 0.77 | 0.81 | 0.81 | 0.81 | 0.92 | 0.77 |

### Reuse / near-duplicate retrieval — `orig` images

Ground truth: a claim is a reuse if an *earlier* claim was built from the same source photo (any manipulation: exact copy, recompression, crop, screenshot, re-photo, splice, AI edit…). 34 reuse claims vs 32 first uses. Score = highest similarity to any earlier claim.

| Method | AUC [95% CI] | recall at 0 false positives | at default threshold: recall / false alarms | ms per comparison |
|---|---|---|---|---|
| pHash (app default) | 0.87 [0.78–0.95] | 65% | 62% / 0 of 32 | <0.01 (vector dot product) |
| dHash | 0.95 [0.90–0.99] | 65% | 62% / 0 of 32 | <0.01 (vector dot product) |
| Thumbnail cosine (app 'embedding') | 0.82 [0.71–0.92] | 68% | 44% / 0 of 32 | <0.01 (vector dot product) |
| ORB + RANSAC | 0.94 [0.86–1.00] | 91% | 91% / 1 of 32 | ~1.4 |
| SSCD mixup (learned) | 0.94 [0.85–1.00] | 85% | 62% / 0 of 32 | <0.01 (vector dot product) |
| SSCD large (learned) | 0.96 [0.89–1.00] | 88% | 47% / 0 of 32 | <0.01 (vector dot product) |

**Recall at the zero-false-positive threshold, by how the reused photo was altered** (n = reuse claims of that kind):

| Method | ai_edited_damage (n=5) | splice (n=4) | reuse_exact (n=4) | screenshot (n=4) | rephotograph (n=4) | reuse_recompressed (n=3) | copy_move (n=3) | reuse_cropped (n=3) | metadata_future_date (n=2) | upscale (n=2) |
|---|---|---|---|---|---|---|---|---|---|---|
| pHash (app default) | 100% | 50% | 100% | 0% | 50% | 100% | 67% | 0% | 100% | 100% |
| dHash | 100% | 75% | 100% | 0% | 0% | 100% | 67% | 33% | 100% | 100% |
| Thumbnail cosine (app 'embedding') | 100% | 50% | 100% | 50% | 25% | 100% | 67% | 0% | 100% | 100% |
| ORB + RANSAC | 80% | 75% | 100% | 100% | 75% | 100% | 100% | 100% | 100% | 100% |
| SSCD mixup (learned) | 100% | 75% | 100% | 25% | 75% | 100% | 100% | 100% | 100% | 100% |
| SSCD large (learned) | 100% | 75% | 100% | 50% | 75% | 100% | 100% | 100% | 100% | 100% |

### Image-level detection — `social` images (19 genuine photos as negatives)

AUC with 95% bootstrap CI · recall at zero false positives (R@0FP). Higher is better; 0.50 = chance.

| Detector | Type | AI-made (generated + AI-edited) (n=13) | conventional edits (splice / copy-move / upscale) (n=12) | screen recapture (screenshot / re-photo) (n=8) |
|---|---|---|---|---|
| probe: aspect ratio | confound probe | 0.74 [0.55–0.90] · 0% | 0.47 [0.26–0.68] · 0% | 0.74 [0.49–0.96] · 50% |
| probe: bytes per pixel | confound probe | 0.65 [0.46–0.83] · 8% | 0.39 [0.20–0.60] · 0% | 0.31 [0.11–0.53] · 0% |
| ELA uniformity | OpenCV / pixel | 0.60 [0.38–0.79] · 8% | 0.46 [0.25–0.67] · 0% | 0.26 [0.09–0.48] · 0% |
| Copy-move SIFT | OpenCV / pixel | 0.58 [0.36–0.78] · 15% | 0.57 [0.36–0.76] · 0% | 0.43 [0.18–0.68] · 0% |
| JPEG ghost | OpenCV / pixel | 0.55 [0.35–0.77] · 0% | 0.64 [0.45–0.84] · 0% | 0.95 [0.86–1.00] · 38% |
| Noise residual | OpenCV / pixel | 0.51 [0.29–0.72] · 0% | 0.36 [0.16–0.58] · 0% | 0.65 [0.34–0.91] · 38% |
| Copy-move ORB | OpenCV / pixel | 0.51 [0.36–0.66] · 0% | 0.59 [0.43–0.76] · 0% | 0.58 [0.40–0.78] · 0% |
| Recapture (screen) | OpenCV / pixel | 0.50 [0.50–0.50] · 0% | 0.50 [0.50–0.50] · 0% | 0.75 [0.56–0.94] · 50% |
| ELA hotspots | OpenCV / pixel | 0.48 [0.29–0.68] · 8% | 0.32 [0.14–0.51] · 0% | 0.28 [0.09–0.49] · 0% |
| SPAI | learned (spectral ViT) | 0.48 [0.27–0.69] · 0% | 0.55 [0.33–0.76] · 0% | 0.79 [0.60–0.95] · 12% |
| CLIP detector (latent10k_plus) | learned (CLIP) | 0.43 [0.22–0.64] · 0% | 0.68 [0.46–0.88] · 8% | 0.66 [0.42–0.88] · 0% |
| Spectral DCT | OpenCV / pixel | 0.42 [0.21–0.63] · 0% | 0.68 [0.45–0.88] · 33% | 0.67 [0.43–0.86] · 12% |
| Upscale detector | OpenCV / pixel | 0.39 [0.19–0.60] · 0% | 0.57 [0.33–0.79] · 33% | 0.66 [0.39–0.89] · 25% |
| Spectral FFT | OpenCV / pixel | 0.28 [0.10–0.47] · 8% | 0.71 [0.48–0.90] · 33% | 0.81 [0.61–0.97] · 38% |
| CLIP detector (latent10k) | learned (CLIP) | 0.27 [0.10–0.47] · 0% | 0.58 [0.33–0.83] · 50% | 0.54 [0.25–0.82] · 12% |

**AI-made claims split by whether the generator's provenance tag survived** (AUC [CI] · R@0FP) — the stripped rows are the real test of the pixel/learned detectors:

| Detector | AI, provenance tag intact (n=7) | AI, provenance stripped (n=6) |
|---|---|---|
| probe: aspect ratio | 0.72 [0.47–0.93] · 0% | 0.77 [0.50–1.00] · 0% |
| probe: bytes per pixel | 0.61 [0.37–0.83] · 0% | 0.70 [0.48–0.89] · 17% |
| ELA uniformity | 0.62 [0.38–0.85] · 0% | 0.59 [0.33–0.84] · 17% |
| Copy-move SIFT | 0.65 [0.41–0.88] · 14% | 0.50 [0.21–0.79] · 17% |
| JPEG ghost | 0.56 [0.35–0.77] · 0% | 0.54 [0.22–0.87] · 0% |
| Noise residual | 0.53 [0.23–0.83] · 0% | 0.48 [0.22–0.75] · 0% |
| Copy-move ORB | 0.47 [0.32–0.65] · 0% | 0.55 [0.37–0.77] · 0% |
| Recapture (screen) | 0.50 [0.50–0.50] · 0% | 0.50 [0.50–0.50] · 0% |
| ELA hotspots | 0.48 [0.23–0.75] · 14% | 0.48 [0.23–0.75] · 0% |
| SPAI | 0.53 [0.27–0.79] · 0% | 0.42 [0.18–0.68] · 0% |
| CLIP detector (latent10k_plus) | 0.37 [0.14–0.62] · 0% | 0.50 [0.19–0.80] · 0% |
| Spectral DCT | 0.51 [0.23–0.77] · 0% | 0.31 [0.11–0.53] · 0% |
| Upscale detector | 0.42 [0.17–0.69] · 0% | 0.36 [0.11–0.63] · 0% |
| Spectral FFT | 0.40 [0.15–0.68] · 14% | 0.15 [0.01–0.32] · 0% |
| CLIP detector (latent10k) | 0.29 [0.08–0.55] · 0% | 0.25 [0.04–0.53] · 0% |

**Confound probes, best direction** (max of AUC and 1−AUC; what a trivially confounded feature achieves with no forensic signal at all):

| Probe | AI-made (generated + AI-edited) | conventional edits (splice / copy-move / upscale) | screen recapture (screenshot / re-photo) |
|---|---|---|---|
| probe: aspect ratio | 0.74 | 0.53 | 0.74 |
| probe: bytes per pixel | 0.65 | 0.61 | 0.69 |

**Operating point at each detector's own default threshold** (app thresholds for OpenCV checks; CLIP LLR>0; SPAI p>0.5). FPR is on the 19 genuine photos:

| Detector | threshold | false-positive rate | recall: AI-made | recall: conventional edits | recall: recapture |
|---|---|---|---|---|---|
| ELA uniformity | 0.5 | 0% (0/19) | 0% | 0% | 0% |
| Copy-move SIFT | 0.922 | 0% (0/19) | 15% | 0% | 0% |
| JPEG ghost | 0.961 | 0% (0/19) | 0% | 0% | 38% |
| Noise residual | 0.99 | 58% (11/19) | 54% | 33% | 62% |
| Copy-move ORB | 0.5 | 0% (0/19) | 0% | 0% | 0% |
| Recapture (screen) | 0.5 | 0% (0/19) | 0% | 0% | 50% |
| ELA hotspots | 0.855 | 21% (4/19) | 15% | 0% | 0% |
| SPAI | 0.5 | 26% (5/19) | 31% | 33% | 62% |
| CLIP detector (latent10k_plus) | 0 | 5% (1/19) | 0% | 17% | 12% |
| Spectral DCT | 0.5 | 0% (0/19) | 0% | 17% | 12% |
| Upscale detector | 0.5 | 0% (0/19) | 0% | 33% | 12% |
| Spectral FFT | 0.5 | 0% (0/19) | 0% | 0% | 0% |
| CLIP detector (latent10k) | 0 | 0% (0/19) | 0% | 0% | 0% |

**Per-fraud-class AUC, `social` images** (exploratory: only 4-8 claims per class, so differences smaller than ~0.2 are noise; 0.50 = chance, shown without CIs):

| Detector | ai_generated (n=8) | ai_edited_damage (n=5) | splice (n=4) | copy_move (n=4) | upscale (n=4) | screenshot (n=4) | rephotograph (n=4) |
|---|---|---|---|---|---|---|---|
| CLIP detector (latent10k) | 0.25 | 0.31 | 0.33 | 0.41 | 1.00 | 0.32 | 0.76 |
| CLIP detector (latent10k_plus) | 0.37 | 0.53 | 0.45 | 0.66 | 0.92 | 0.45 | 0.87 |
| Copy-move ORB | 0.51 | 0.49 | 0.64 | 0.74 | 0.39 | 0.52 | 0.64 |
| Copy-move SIFT | 0.70 | 0.38 | 0.66 | 0.64 | 0.40 | 0.32 | 0.55 |
| ELA hotspots | 0.51 | 0.43 | 0.45 | 0.26 | 0.24 | 0.13 | 0.42 |
| ELA uniformity | 0.66 | 0.52 | 0.54 | 0.32 | 0.53 | 0.08 | 0.43 |
| JPEG ghost | 0.61 | 0.46 | 0.57 | 0.67 | 0.70 | 0.99 | 0.92 |
| Noise residual | 0.39 | 0.69 | 0.33 | 0.54 | 0.22 | 0.99 | 0.32 |
| Recapture (screen) | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 1.00 | 0.50 |
| SPAI | 0.44 | 0.55 | 0.43 | 0.41 | 0.80 | 0.67 | 0.91 |
| Spectral DCT | 0.40 | 0.44 | 0.43 | 0.68 | 0.92 | 0.58 | 0.76 |
| Spectral FFT | 0.21 | 0.40 | 0.45 | 0.70 | 0.99 | 0.86 | 0.76 |
| Upscale detector | 0.34 | 0.48 | 0.36 | 0.36 | 1.00 | 0.36 | 0.97 |
| probe: aspect ratio | 0.95 | 0.42 | 0.48 | 0.45 | 0.48 | 1.00 | 0.48 |
| probe: bytes per pixel | 0.76 | 0.48 | 0.59 | 0.29 | 0.30 | 0.16 | 0.46 |

### Reuse / near-duplicate retrieval — `social` images

Ground truth: a claim is a reuse if an *earlier* claim was built from the same source photo (any manipulation: exact copy, recompression, crop, screenshot, re-photo, splice, AI edit…). 34 reuse claims vs 32 first uses. Score = highest similarity to any earlier claim.

| Method | AUC [95% CI] | recall at 0 false positives | at default threshold: recall / false alarms | ms per comparison |
|---|---|---|---|---|
| pHash (app default) | 0.87 [0.77–0.95] | 65% | 62% / 0 of 32 | <0.01 (vector dot product) |
| dHash | 0.95 [0.90–0.99] | 65% | 62% / 0 of 32 | <0.01 (vector dot product) |
| Thumbnail cosine (app 'embedding') | 0.82 [0.72–0.92] | 68% | 44% / 0 of 32 | <0.01 (vector dot product) |
| ORB + RANSAC | 0.95 [0.89–1.00] | 91% | 91% / 0 of 32 | ~1.4 |
| SSCD mixup (learned) | 0.95 [0.87–1.00] | 85% | 68% / 0 of 32 | <0.01 (vector dot product) |
| SSCD large (learned) | 0.96 [0.89–1.00] | 85% | 62% / 0 of 32 | <0.01 (vector dot product) |

**Recall at the zero-false-positive threshold, by how the reused photo was altered** (n = reuse claims of that kind):

| Method | ai_edited_damage (n=5) | splice (n=4) | reuse_exact (n=4) | screenshot (n=4) | rephotograph (n=4) | reuse_recompressed (n=3) | copy_move (n=3) | reuse_cropped (n=3) | metadata_future_date (n=2) | upscale (n=2) |
|---|---|---|---|---|---|---|---|---|---|---|
| pHash (app default) | 100% | 50% | 100% | 0% | 50% | 100% | 67% | 0% | 100% | 100% |
| dHash | 100% | 75% | 100% | 0% | 0% | 100% | 67% | 33% | 100% | 100% |
| Thumbnail cosine (app 'embedding') | 100% | 50% | 100% | 50% | 25% | 100% | 67% | 0% | 100% | 100% |
| ORB + RANSAC | 80% | 75% | 100% | 100% | 75% | 100% | 100% | 100% | 100% | 100% |
| SSCD mixup (learned) | 100% | 75% | 100% | 25% | 75% | 100% | 100% | 100% | 100% | 100% |
| SSCD large (learned) | 100% | 75% | 100% | 25% | 75% | 100% | 100% | 100% | 100% | 100% |

### Cost on this laptop (Apple M2 Pro, CPU only, median per image)

| Method | time per image |
|---|---|
| Copy-move ORB | 18 ms |
| Copy-move SIFT | 65 ms |
| ELA hotspots | 24 ms |
| ELA uniformity | 23 ms |
| JPEG ghost | 79 ms |
| Noise residual | 7 ms |
| Recapture (screen) | 43 ms |
| Spectral DCT | 2 ms |
| Spectral FFT | 55 ms |
| Upscale detector | 48 ms |
| sscd_disc_mixup | 51 ms |
| sscd_disc_large | 150 ms |
| clip_backbone | 243 ms |
| SPAI | 4.5 s |
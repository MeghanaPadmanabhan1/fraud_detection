# ML vs pixel checks: held-out test v1

52 AI-made images (gemini_3_1_flash 10, gemini_3_1_flash_lite 10, gemini_3_pro 10, web:Adobe Firefly 6, web:Bing Image Creator 6, web:DALL-E 6, web:Ideogram 2, web:Midjourney 2) vs 32 real photos (real_damaged 3, real_phone 29). Thresholds frozen from gold set v0.

## AUC, AI vs real (95% bootstrap CI)

| Detector | orig | social | norm |
|---|---|---|---|
| CLIP detector (Cozzolino, latent10k+) | 0.79 [0.70-0.88] | 0.71 [0.59-0.81] | 0.77 [0.67-0.86] |
| CLIP detector (latent10k) | 0.73 [0.61-0.85] | 0.69 [0.57-0.81] | 0.73 [0.62-0.83] |
| SPAI | 0.73 [0.61-0.83] | 0.55 [0.42-0.68] | 0.60 [0.46-0.71] |
| ELA | 0.41 [0.29-0.55] | 0.44 [0.33-0.57] | 0.37 [0.24-0.51] |
| JPEG ghost | 0.61 [0.49-0.73] | 0.46 [0.33-0.59] | 0.40 [0.28-0.52] |
| Noise residual | 0.61 [0.48-0.74] | 0.49 [0.34-0.63] | 0.55 [0.41-0.68] |
| Copy-move ORB | 0.46 [0.35-0.56] | 0.45 [0.36-0.53] | 0.48 [0.38-0.57] |
| Copy-move SIFT | 0.47 [0.34-0.59] | 0.51 [0.38-0.63] | 0.50 [0.38-0.62] |
| Spectral FFT | 0.64 [0.51-0.78] | 0.43 [0.28-0.58] | 0.45 [0.32-0.58] |
| Spectral DCT | 0.64 [0.50-0.77] | 0.58 [0.45-0.72] | 0.56 [0.43-0.70] |
| ELA uniformity | 0.60 [0.48-0.72] | 0.63 [0.49-0.75] | 0.57 [0.42-0.70] |
| Upscale | 0.61 [0.47-0.74] | 0.68 [0.54-0.79] | 0.74 [0.62-0.85] |
| Recapture | 0.76 [0.68-0.84] | 0.50 [0.50-0.50] | 0.50 [0.50-0.50] |
| PROBE: aspect ratio only | 0.63 [0.50-0.75] | 0.63 [0.51-0.74] | 0.50 [0.50-0.50] |
| PROBE: bytes per pixel only | 0.74 [0.63-0.83] | 0.52 [0.38-0.66] | 0.48 [0.34-0.62] |

## At the default threshold (recall on AI / false alarms on real)

| Detector | view | recall | false alarms |
|---|---|---|---|
| CLIP detector (Cozzolino, latent10k+) | orig | 29% (15/52) | 0% (0/32) |
| CLIP detector (Cozzolino, latent10k+) | social | 37% (19/52) | 3% (1/32) |
| CLIP detector (Cozzolino, latent10k+) | norm | 38% (20/52) | 0% (0/32) |
| CLIP detector (latent10k) | orig | 0% (0/52) | 0% (0/32) |
| CLIP detector (latent10k) | social | 0% (0/52) | 0% (0/32) |
| CLIP detector (latent10k) | norm | 0% (0/52) | 0% (0/32) |
| SPAI | orig | 58% (30/52) | 28% (9/32) |
| SPAI | social | 29% (15/52) | 28% (9/32) |
| SPAI | norm | 42% (22/52) | 31% (10/32) |
| ELA | orig | 4% (2/52) | 16% (5/32) |
| ELA | social | 23% (12/52) | 22% (7/32) |
| ELA | norm | 0% (0/52) | 3% (1/32) |
| JPEG ghost | orig | 2% (1/52) | 0% (0/32) |
| JPEG ghost | social | 4% (2/52) | 6% (2/32) |
| JPEG ghost | norm | 6% (3/52) | 0% (0/32) |
| Noise residual | orig | 35% (18/52) | 28% (9/32) |
| Noise residual | social | 21% (11/52) | 38% (12/32) |
| Noise residual | norm | 23% (12/52) | 34% (11/32) |
| Copy-move ORB | orig | 4% (2/52) | 9% (3/32) |
| Copy-move ORB | social | 2% (1/52) | 3% (1/32) |
| Copy-move ORB | norm | 4% (2/52) | 0% (0/32) |
| Copy-move SIFT | orig | 15% (8/52) | 12% (4/32) |
| Copy-move SIFT | social | 12% (6/52) | 12% (4/32) |
| Copy-move SIFT | norm | 10% (5/52) | 12% (4/32) |
| Recapture | orig | 0% (0/52) | 0% (0/32) |
| Recapture | social | 0% (0/52) | 0% (0/32) |
| Recapture | norm | 0% (0/52) | 0% (0/32) |
| Upscale | orig | 8% (4/52) | 0% (0/32) |
| Upscale | social | 6% (3/52) | 0% (0/32) |
| Upscale | norm | 6% (3/52) | 0% (0/32) |
| Spectral FFT | orig | 0% (0/52) | 0% (0/32) |
| Spectral FFT | social | 0% (0/52) | 0% (0/32) |
| Spectral FFT | norm | 0% (0/52) | 0% (0/32) |
| Spectral DCT | orig | 0% (0/52) | 0% (0/32) |
| Spectral DCT | social | 0% (0/52) | 0% (0/32) |
| Spectral DCT | norm | 0% (0/52) | 0% (0/32) |
| ELA uniformity | orig | 0% (0/52) | 0% (0/32) |
| ELA uniformity | social | 0% (0/52) | 0% (0/32) |
| ELA uniformity | norm | 0% (0/52) | 0% (0/32) |

## Recall by generator group at default thresholds (norm view if present, else orig)

| Detector | gemini_3_1_flash | gemini_3_1_flash_lite | gemini_3_pro | web:Adobe Firefly | web:Bing Image Creator | web:DALL-E | web:Ideogram | web:Midjourney |
|---|---|---|---|---|---|---|---|---|
| CLIP detector (Cozzolino, latent10k+) | 30% | 40% | 10% | 33% | 50% | 67% | 100% | 50% |
| SPAI | 30% | 60% | 10% | 67% | 33% | 83% | 50% | 0% |
| Upscale | 0% | 0% | 0% | 0% | 0% | 50% | 0% | 0% |

## Reuse (index of 88 photos; frozen thresholds {'phash': 10, 'orb': 25, 'sscd': 0.6})

SSCD vectors came from the deployed Model Serving endpoint `endpoint:image-authenticity-sscd` (208 calls, median 468 ms per call; lowest cosine to the local weights 1.0000). pHash and ORB ran locally.

| Method | recall (hit on the right source) | by transform: crop / screenshot / recompress / relight | false hits on never-indexed photos | ms per query |
|---|---|---|---|---|
| phash | 34% (27/80) | 0% / 0% / 100% / 35% | 0/40 | 0 |
| orb | 99% (79/80) | 100% / 95% / 100% / 100% | 3/40 | 147 |
| sscd | 100% (80/80) | 100% / 100% / 100% / 100% | 0/40 | 0 |

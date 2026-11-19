"""Error Level Analysis (ELA) -- used by BOTH arms with different readouts.

Resave the image as JPEG at a fixed quality and diff against the original.
Each 8x8 JPEG block recompresses by an amount tied to how much it was already
compressed.

- Deepfake arm (ela_global): a fully synthetic image was generated at one
  uniform "quality" everywhere, so its ELA is suspiciously flat. We score low
  spatial variance of the ELA map as suspicious.
- Shallow-fake arm (ela_local): a spliced/cloned/painted region was saved a
  different number of times, so it lights up against a quiet background. We
  score the strongest localized hotspot and emit a heatmap for the adjuster.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..core import LABEL_AI, LABEL_AUTHENTIC, LABEL_MANIPULATED, Detector, register
from . import _util


def _ela_map(image_bgr: np.ndarray, quality: int) -> np.ndarray:
    resaved = _util.jpeg_resave(image_bgr, quality)
    diff = cv2.absdiff(image_bgr, resaved).astype(np.float32)
    return diff.max(axis=2)  # per-pixel max channel error


@register
class ELAGlobalDetector(Detector):
    """Deepfake readout: flat, uniform ELA across the whole frame."""

    slot = "ela_global"
    name = "ela_uniformity_v1"
    version = "1.0"

    def analyze(self, image_bgr, image_bytes=None, context=None):
        try:
            quality = int(self.config.get("quality", 90))
            ela = _ela_map(image_bgr, quality)
            # Block-wise mean energy; uniformity == low spread across blocks.
            blocks = _block_reduce(ela, 16, np.mean)
            spread = float(np.std(blocks) / (np.mean(blocks) + 1e-6))
            # Low spread == uniform == synthetic-leaning. Calibrated so real
            # photos (block spread ~0.33) score low; only near-flat ELA
            # (spread < ~0.1) pushes the score up.
            score = _util.squash(-spread, midpoint=-0.07, steepness=10.0)
            label = LABEL_AI if score >= 0.5 else LABEL_AUTHENTIC
            return self.result(
                float(score),
                label,
                evidence={
                    "ela_block_spread": round(spread, 4),
                    "ela_mean": round(float(np.mean(ela)), 3),
                    "quality": quality,
                    "note": "low ela_block_spread => uniform error => AI-leaning.",
                },
            )
        except Exception as e:  # pragma: no cover
            return self.result(None, "indeterminate", error=repr(e))


@register
class ELALocalDetector(Detector):
    """Shallow-fake readout: localized ELA hotspots + heatmap overlay."""

    slot = "ela_local"
    name = "ela_heatmap_v1"
    version = "1.0"

    def analyze(self, image_bgr, image_bytes=None, context=None):
        try:
            quality = int(self.config.get("quality", 90))
            ela = _ela_map(image_bgr, quality)
            blocks = _block_reduce(ela, 16, np.mean)
            bg = np.median(blocks)
            mad = np.median(np.abs(blocks - bg)) + 1e-6
            z = (blocks - bg) / (1.4826 * mad)  # robust z-score per block
            hot_frac = float((z > 4.0).mean())
            peak = float(z.max())
            # Score driven by how strong the strongest localized hotspot is,
            # gated so a globally busy image doesn't trigger it.
            score = _util.squash(peak, midpoint=6.0, steepness=0.7)
            score = float(score * (1.0 - min(hot_frac * 3, 0.6)))
            label = LABEL_MANIPULATED if score >= 0.5 else LABEL_AUTHENTIC

            heatmap_path = None
            out = (context or {}).get("heatmap_out")
            if out:
                heatmap_path = _util.save_heatmap(
                    image_bgr, np.clip(z, 0, None), out, title="ELA"
                )
            return self.result(
                float(score),
                label,
                evidence={
                    "ela_peak_z": round(peak, 3),
                    "ela_hot_block_frac": round(hot_frac, 4),
                    "quality": quality,
                    "note": "localized high-z blocks => a region was edited/pasted.",
                },
                heatmap_path=heatmap_path,
            )
        except Exception as e:  # pragma: no cover
            return self.result(None, "indeterminate", error=repr(e))


def _block_reduce(a: np.ndarray, block: int, fn) -> np.ndarray:
    h, w = a.shape
    hh, ww = h // block, w // block
    if hh == 0 or ww == 0:
        return np.array([[fn(a)]])
    a = a[: hh * block, : ww * block].reshape(hh, block, ww, block)
    return fn(a, axis=(1, 3))

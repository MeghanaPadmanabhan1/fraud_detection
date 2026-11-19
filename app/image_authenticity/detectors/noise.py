"""Shallow-fake arm -- noise-residual inconsistency (splice detection).

Different cameras / generators leave a different sensor-noise fingerprint. When
a region is spliced in from another source, its local noise statistics differ
from the rest of the frame. We extract a high-pass noise residual, measure the
local noise energy per block, and score how inconsistent (multi-modal) that map
is. Full Noiseprint (a trained CNN) is the SOTA swap-in for this slot.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core import LABEL_AUTHENTIC, LABEL_MANIPULATED, Detector, register
from . import _util
from .ela import _block_reduce
from .jpeg_ghost import _spatial_clustering


@register
class NoiseResidualDetector(Detector):
    slot = "noise"
    name = "noise_residual_v1"
    version = "1.0"

    def analyze(self, image_bgr, image_bytes=None, context=None):
        try:
            gray = _util.to_gray(image_bgr).astype(np.float32)
            # High-pass residual: image minus a denoised version.
            denoised = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
            residual = gray - denoised

            # Local noise energy per block (std of the residual).
            energy = _block_reduce(residual, 16, np.std)
            flat = energy.ravel()
            if flat.size < 8:
                return self.result(0.0, LABEL_AUTHENTIC, error="image too small")

            # A spliced region is a CONTIGUOUS cluster of blocks whose noise
            # energy sits far from the median -- not scattered high-variance
            # blocks (which just mean the photo has texture). Requiring spatial
            # clustering keeps real, busy photos from false-positiving.
            med = np.median(flat)
            mad = np.median(np.abs(flat - med)) + 1e-6
            z = np.abs(energy - med) / (1.4826 * mad)
            outlier_mask = (z > 3.5).astype(np.uint8)
            outlier_frac = float(outlier_mask.mean())
            clustering = _spatial_clustering(outlier_mask)

            # Score only when deviating blocks are both present AND clustered.
            score = float(_util.squash(outlier_frac * clustering, 0.02, 90.0))
            label = LABEL_MANIPULATED if score >= 0.5 else LABEL_AUTHENTIC

            heatmap_path = None
            out = (context or {}).get("heatmap_out")
            if out:
                heatmap_path = _util.save_heatmap(
                    image_bgr, energy, out, title="noise"
                )
            return self.result(
                score,
                label,
                evidence={
                    "noise_outlier_frac": round(outlier_frac, 4),
                    "noise_cluster_ratio": round(float(clustering), 4),
                    "note": "a clustered block of deviating noise energy => spliced source.",
                },
                heatmap_path=heatmap_path,
            )
        except Exception as e:  # pragma: no cover
            return self.result(None, "indeterminate", error=repr(e))

"""Shallow-fake arm -- JPEG ghost detection.

When part of a JPEG is edited and the whole thing is resaved, that region
carries a different compression history than its surroundings. Resaving the
image across a sweep of qualities and measuring the per-region error reveals a
"ghost": a region whose error bottoms out at a different quality than the rest
of the frame -- evidence it was originally saved at a different quality.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core import LABEL_AUTHENTIC, LABEL_MANIPULATED, Detector, register
from . import _util
from .ela import _block_reduce


@register
class JPEGGhostDetector(Detector):
    slot = "jpeg_ghost"
    name = "jpeg_ghost_v1"
    version = "1.0"

    def analyze(self, image_bgr, image_bytes=None, context=None):
        try:
            qualities = self.config.get("qualities", list(range(55, 100, 5)))
            block = int(self.config.get("block", 16))
            gray0 = _util.to_gray(image_bgr).astype(np.float32)

            # For each quality, block-mean squared error vs the original.
            stacks = []
            for q in qualities:
                resaved = _util.jpeg_resave(image_bgr, q)
                g = _util.to_gray(resaved).astype(np.float32)
                sq = (gray0 - g) ** 2
                stacks.append(_block_reduce(sq, block, np.mean))
            cube = np.stack(stacks, axis=0)  # (Q, bh, bw)

            # Per block, which quality minimizes the error. A ghost shows up as
            # a cluster of blocks whose argmin differs from the global mode.
            argmin_q = np.argmin(cube, axis=0)
            vals, counts = np.unique(argmin_q, return_counts=True)
            mode_q = vals[np.argmax(counts)]
            ghost_frac = float((argmin_q != mode_q).mean())

            # Only count it as a ghost if those deviating blocks are spatially
            # clustered (an edited region), not scattered noise.
            dev = (argmin_q != mode_q).astype(np.uint8)
            clustering = _spatial_clustering(dev)
            score = _util.squash(ghost_frac * clustering, 0.08, 18.0)
            label = LABEL_MANIPULATED if score >= 0.5 else LABEL_AUTHENTIC

            heatmap_path = None
            out = (context or {}).get("heatmap_out")
            if out:
                heat = np.abs(argmin_q.astype(np.float32) - mode_q)
                heatmap_path = _util.save_heatmap(image_bgr, heat, out, title="jpeg-ghost")

            return self.result(
                float(score),
                label,
                evidence={
                    "ghost_block_frac": round(ghost_frac, 4),
                    "clustering": round(float(clustering), 4),
                    "mode_quality": int(qualities[mode_q]),
                    "note": "clustered blocks minimizing at a different quality => prior edit.",
                },
                heatmap_path=heatmap_path,
            )
        except Exception as e:  # pragma: no cover
            return self.result(None, "indeterminate", error=repr(e))


def _spatial_clustering(mask: np.ndarray) -> float:
    """1.0 if deviating blocks are contiguous, ~0 if scattered."""
    if mask.sum() == 0:
        return 0.0
    n, labels = cv2.connectedComponents(mask)
    if n <= 1:
        return 0.0
    sizes = [int((labels == i).sum()) for i in range(1, n)]
    return max(sizes) / float(mask.sum())

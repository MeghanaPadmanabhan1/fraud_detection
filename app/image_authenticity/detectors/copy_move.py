"""Shallow-fake arm -- copy-move forgery detection via ORB keypoints.

Catches a region cloned within the SAME image: duplicating a crack to make it
look worse, or pasting a nearby texture over a serial number. We match ORB
descriptors against themselves and keep pairs whose descriptors are nearly
identical but whose locations are far apart -- the signature of a clone.

ORB is used (not SIFT) because it's patent-free and ships with opencv-python.
Swap in SIFT/PatchMatch for the slot later if desired.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core import LABEL_AUTHENTIC, LABEL_MANIPULATED, Detector, register
from . import _util


def _clone_pairs(kps, desc, norm, pts, min_dist, ratio):
    """Shared copy-move logic: find descriptor matches far apart in space."""
    bf = cv2.BFMatcher(norm)
    # k=3 so we can skip the trivial self-match (nearest == itself).
    matches = bf.knnMatch(desc, desc, k=3)
    pairs = []
    for m in matches:
        cands = [x for x in m if x.queryIdx != x.trainIdx]
        if len(cands) < 2:
            continue
        best, second = cands[0], cands[1]
        # Lowe-style ratio test: descriptor must be a strong, unique match.
        if best.distance < ratio * (second.distance + 1e-6):
            p, q = pts[best.queryIdx], pts[best.trainIdx]
            if np.linalg.norm(p - q) > min_dist:
                pairs.append((p, q))
    return pairs


def _finish(self, image_bgr, kps, clone_pairs, context):
    n_pairs = len(clone_pairs)
    score = _util.squash(n_pairs, midpoint=12.0, steepness=0.18)
    label = LABEL_MANIPULATED if score >= 0.5 else LABEL_AUTHENTIC
    heatmap_path = None
    out = (context or {}).get("heatmap_out")
    if out and n_pairs:
        vis = image_bgr.copy()
        for p, q in clone_pairs[:200]:
            p = tuple(np.round(p).astype(int))
            q = tuple(np.round(q).astype(int))
            cv2.line(vis, p, q, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(vis, p, 3, (0, 255, 0), -1)
            cv2.circle(vis, q, 3, (0, 255, 0), -1)
        import os
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, vis)
        heatmap_path = out
    return self.result(
        float(score), label,
        evidence={"keypoints": len(kps), "clone_pairs": n_pairs,
                  "note": "matched descriptors at distant locations => cloned region."},
        heatmap_path=heatmap_path,
    )


@register
class ORBCopyMoveDetector(Detector):
    slot = "copy_move"
    name = "orb_v1"
    version = "1.0"

    def analyze(self, image_bgr, image_bytes=None, context=None):
        try:
            gray = _util.to_gray(image_bgr)
            min_dist = float(self.config.get("min_spatial_dist", 40))
            ratio = float(self.config.get("ratio", 0.5))
            orb = cv2.ORB_create(nfeatures=int(self.config.get("nfeatures", 2000)))
            kps, desc = orb.detectAndCompute(gray, None)
            if desc is None or len(kps) < 10:
                return self.result(0.0, LABEL_AUTHENTIC,
                    evidence={"keypoints": 0 if desc is None else len(kps),
                              "note": "too few keypoints to assess."})
            pts = np.array([kp.pt for kp in kps])
            pairs = _clone_pairs(kps, desc, cv2.NORM_HAMMING, pts, min_dist, ratio)
            return _finish(self, image_bgr, kps, pairs, context)
        except Exception as e:  # pragma: no cover
            return self.result(None, "indeterminate", error=repr(e))


@register
class SIFTCopyMoveDetector(Detector):
    """Alternative copy-move model -- SIFT keypoints (float descriptors, L2).

    Same functionality as the ORB model; SIFT keypoints are more distinctive
    and scale-invariant, often catching clones ORB misses, at higher cost.
    Needs opencv's SIFT (patent-free since 2020; bundled in opencv-python).
    """

    slot = "copy_move"
    name = "sift_v1"
    version = "1.0"

    def analyze(self, image_bgr, image_bytes=None, context=None):
        try:
            gray = _util.to_gray(image_bgr)
            min_dist = float(self.config.get("min_spatial_dist", 40))
            ratio = float(self.config.get("ratio", 0.6))
            if not hasattr(cv2, "SIFT_create"):
                return self.result(None, "indeterminate",
                    error="cv2.SIFT_create unavailable in this OpenCV build")
            sift = cv2.SIFT_create(nfeatures=int(self.config.get("nfeatures", 2000)))
            kps, desc = sift.detectAndCompute(gray, None)
            if desc is None or len(kps) < 10:
                return self.result(0.0, LABEL_AUTHENTIC,
                    evidence={"keypoints": 0 if desc is None else len(kps),
                              "note": "too few keypoints to assess."})
            pts = np.array([kp.pt for kp in kps])
            pairs = _clone_pairs(kps, desc, cv2.NORM_L2, pts, min_dist, ratio)
            return _finish(self, image_bgr, kps, pairs, context)
        except Exception as e:  # pragma: no cover
            return self.result(None, "indeterminate", error=repr(e))

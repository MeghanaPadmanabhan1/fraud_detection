"""Shared helpers for detectors: decoding, resaving, heatmap overlays."""
from __future__ import annotations

import io
import os
from typing import Optional

import cv2
import numpy as np


def decode_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image bytes")
    return img


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def jpeg_resave(image_bgr: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode as JPEG at `quality` and decode back (for ELA / JPEG ghost)."""
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("jpeg re-encode failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def squash(value: float, midpoint: float, steepness: float) -> float:
    """Logistic squashing of a raw metric into a 0..1 score.

    midpoint:  raw value that maps to 0.5.
    steepness: larger == sharper transition.
    """
    import math

    try:
        return 1.0 / (1.0 + math.exp(-steepness * (value - midpoint)))
    except OverflowError:
        return 0.0 if value < midpoint else 1.0


def save_heatmap(
    base_bgr: np.ndarray,
    heat01: np.ndarray,
    out_path: str,
    alpha: float = 0.5,
    title: Optional[str] = None,
) -> Optional[str]:
    """Overlay a 0..1 heatmap on the image and write a PNG. Returns path or None."""
    try:
        heat = cv2.resize(heat01.astype(np.float32), (base_bgr.shape[1], base_bgr.shape[0]))
        heat_u8 = (normalize01(heat) * 255).astype(np.uint8)
        color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(base_bgr, 1 - alpha, color, alpha, 0)
        if title:
            cv2.putText(
                overlay, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA,
            )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, overlay)
        return out_path
    except Exception:
        return None

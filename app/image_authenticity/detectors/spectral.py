"""Deepfake arm -- spectral / frequency analysis.

Real camera photos have a natural ~1/f power spectrum and sensor noise across
high frequencies. GAN/diffusion images deviate: diffusion images are often
"too clean" (steep high-frequency roll-off) and GAN images show periodic peaks
(the checkerboard artifact of transposed-convolution upsampling).

This baseline extracts the azimuthally-averaged power spectrum and scores two
heuristics -- high-frequency energy deficit and spectral peakiness. It's a
transparent, CPU-only stand-in; the slot can be swapped for a trained
frequency-domain classifier without touching the pipeline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core import LABEL_AI, LABEL_AUTHENTIC, Detector, register
from . import _util


def _radial_profile(mag: np.ndarray) -> np.ndarray:
    h, w = mag.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.int32)
    rmax = r.max()
    tbin = np.bincount(r.ravel(), mag.ravel(), minlength=rmax + 1)
    nr = np.bincount(r.ravel(), minlength=rmax + 1)
    nr[nr == 0] = 1
    return tbin / nr


@register
class FFTRadialDetector(Detector):
    slot = "spectral"
    name = "fft_radial_v1"
    version = "1.0"

    def analyze(self, image_bgr, image_bytes=None, context=None):
        try:
            gray = _util.to_gray(image_bgr).astype(np.float64)
            # Hann window to suppress edge leakage before the FFT.
            wy = np.hanning(gray.shape[0])[:, None]
            wx = np.hanning(gray.shape[1])[None, :]
            gray = gray * (wy * wx)
            f = np.fft.fftshift(np.fft.fft2(gray))
            mag = np.log1p(np.abs(f))
            prof = _radial_profile(mag)
            prof = prof[: len(prof) // 2]  # drop the noisy far-corner tail
            if len(prof) < 16:
                return self.result(0.0, LABEL_AUTHENTIC, error="image too small")

            # Normalize radius to 0..1 and split low vs high frequency bands.
            n = len(prof)
            lo = prof[: n // 4].mean()
            hi = prof[3 * n // 4 :].mean()
            hf_ratio = float(hi / (lo + 1e-9))  # low == over-smooth high freq

            # Spectral slope: fit log power vs log radius. Natural images sit
            # around a characteristic slope; very steep == synthetic smoothing.
            radii = np.arange(1, n)
            slope = float(np.polyfit(np.log(radii), prof[1:] + 1e-9, 1)[0])

            # Peakiness: residual of the profile vs its smoothed trend. GAN
            # grids create periodic bumps -> high residual energy.
            k = max(3, n // 32)
            kernel = np.ones(k) / k
            smooth = np.convolve(prof, kernel, mode="same")
            resid = prof - smooth
            peakiness = float(np.std(resid) / (np.mean(prof) + 1e-9))

            # Calibrated on real photos vs Gemini images: both sit near
            # hf_ratio~0.8, peakiness~0.08, so midpoints are set past that
            # range -- the slot fires only on strong deviations (over-smooth
            # roll-off or periodic GAN grids), staying conservative otherwise.
            smooth_score = _util.squash(-hf_ratio, midpoint=-0.35, steepness=8.0)
            peak_score = _util.squash(peakiness, midpoint=0.14, steepness=35.0)
            score = float(np.clip(0.6 * smooth_score + 0.4 * peak_score, 0, 1))

            label = LABEL_AI if score >= 0.5 else LABEL_AUTHENTIC
            return self.result(
                score,
                label,
                evidence={
                    "hf_energy_ratio": round(hf_ratio, 4),
                    "spectral_slope": round(slope, 4),
                    "peakiness": round(peakiness, 4),
                    "smooth_score": round(smooth_score, 3),
                    "peak_score": round(peak_score, 3),
                    "note": "low hf_energy_ratio => over-smooth (diffusion); "
                    "high peakiness => periodic grid (GAN).",
                },
            )
        except Exception as e:  # pragma: no cover
            return self.result(None, "indeterminate", error=repr(e))


@register
class DCTStatsDetector(Detector):
    """Alternative spectral model -- DCT high-frequency energy ratio.

    Same functionality as FFTRadialDetector (deepfake spectral analysis), a
    different method: a 2D DCT concentrates energy in low-frequency (top-left)
    coefficients; synthetic images tend to carry proportionally less
    high-frequency energy. Provided so the Spectral functionality has a second
    model to A/B against the FFT one.
    """

    slot = "spectral"
    name = "dct_stats_v1"
    version = "1.0"

    def analyze(self, image_bgr, image_bytes=None, context=None):
        try:
            import cv2

            gray = _util.to_gray(image_bgr).astype(np.float32)
            g = cv2.resize(gray, (256, 256))
            d = np.abs(cv2.dct(g))
            # Diagonal index i+j splits low- vs high-frequency coefficients.
            ij = np.add.outer(np.arange(256), np.arange(256))
            hf = float(d[ij > 256].sum())
            lf = float(d[ij <= 256].sum())
            hf_ratio = hf / (lf + 1e-9)
            # Low high-frequency ratio => over-smooth => synthetic-leaning.
            score = float(_util.squash(-hf_ratio, midpoint=-0.12, steepness=14.0))
            label = LABEL_AI if score >= 0.5 else LABEL_AUTHENTIC
            return self.result(
                score,
                label,
                evidence={
                    "dct_hf_ratio": round(hf_ratio, 4),
                    "note": "low dct_hf_ratio => little high-frequency energy => AI-leaning.",
                },
            )
        except Exception as e:  # pragma: no cover
            return self.result(None, "indeterminate", error=repr(e))

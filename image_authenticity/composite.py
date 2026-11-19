"""Composite scoring + human-readable explanation (the adjuster-facing layer).

Combines the per-signal scores into: a confidence per arm (deepfake / shallow),
one overall fraud score, and a plain-language narrative of which signals fired
and why. No single technique is trusted alone -- a fraudster must defeat the
whole ensemble.
"""
from __future__ import annotations

from typing import List

from .core import LABEL_AI, LABEL_INDETERMINATE, DetectionResult


def _arm_score(results: List[DetectionResult], slots_cfg: dict, arm: str):
    """Weighted mean over valid, decisive detectors in one arm. None if empty."""
    num = den = 0.0
    for r in results:
        meta = slots_cfg.get(r.slot, {})
        if meta.get("arm") != arm:
            continue
        if r.score is None or r.error or r.label == LABEL_INDETERMINATE:
            continue
        w = float(meta.get("weight", 1.0))
        num += w * float(r.score)
        den += w
    return (num / den) if den else None


def composite_score(results: List[DetectionResult], config: dict) -> dict:
    slots_cfg = config["slots"]
    comp = config["composite"]
    fire = float(comp.get("fire_threshold", 0.5))

    deepfake = _arm_score(results, slots_cfg, "deepfake")
    shallow = _arm_score(results, slots_cfg, "shallow")
    watermark = _arm_score(results, slots_cfg, "watermark")

    # Overall: renormalize arm weights over the arms that produced a score.
    arm_w = comp.get("arm_weights", {})
    parts = {"deepfake": deepfake, "shallow": shallow, "watermark": watermark}
    num = den = 0.0
    for arm, val in parts.items():
        if val is None:
            continue
        w = float(arm_w.get(arm, 0.0))
        num += w * val
        den += w
    overall = (num / den) if den else None

    # Asymmetric provenance override. A verifiable C2PA/IPTC AI-provenance tag
    # (or, later, a real SynthID verdict) is near-cryptographic proof the image
    # is synthetic -- it should dominate the soft pixel/semantic signals, which
    # by themselves cannot reliably flag a SOTA generator. The reverse does NOT
    # hold: ABSENCE of provenance is never evidence of authenticity (fraudsters
    # strip metadata), so we only ever escalate here, never discount.
    prov_override = None
    for r in results:
        meta = slots_cfg.get(r.slot, {})
        if (meta.get("arm") == "watermark" and r.label == LABEL_AI
                and r.score is not None and r.score >= 0.85):
            prov_override = max(prov_override or 0.0, float(r.score))
    if prov_override is not None:
        overall = max(overall or 0.0, 0.80)

    # Which individual signals fired, most suspicious first.
    fired = []
    for r in sorted(results, key=lambda x: (x.score is None, -(x.score or 0))):
        if r.score is not None and not r.error and r.score >= fire:
            note = r.evidence.get("note") or r.evidence.get("rationale") or ""
            fired.append(
                {"slot": r.slot, "detector": r.detector, "score": round(r.score, 3),
                 "label": r.label, "why": note}
            )

    narrative = _narrative(overall, deepfake, shallow, fired, results)
    verdict = _verdict(overall)

    return {
        "fraud_score": None if overall is None else round(overall, 4),
        "deepfake_confidence": None if deepfake is None else round(deepfake, 4),
        "shallowfake_confidence": None if shallow is None else round(shallow, 4),
        "watermark_signal": None if watermark is None else round(watermark, 4),
        "verdict": verdict,
        "fired_signals": fired,
        "narrative": narrative,
    }


def _verdict(overall) -> str:
    if overall is None:
        return "indeterminate"
    if overall >= 0.7:
        return "likely_fraudulent"
    if overall >= 0.45:
        return "needs_review"
    return "likely_authentic"


def _narrative(overall, deepfake, shallow, fired, results) -> str:
    if overall is None:
        return "No decisive signals; manual review recommended."
    lead = {
        "likely_fraudulent": "High likelihood of a fake or manipulated image.",
        "needs_review": "Mixed signals; route to a human adjuster.",
        "likely_authentic": "No strong evidence of manipulation.",
    }[_verdict(overall)]
    bits = [lead]
    # Call out verifiable provenance first -- it's the strongest evidence.
    for r in results:
        if r.slot == "watermark" and r.label == LABEL_AI and (r.score or 0) >= 0.85:
            bits.append("Verifiable AI-provenance tag embedded (C2PA/IPTC) "
                        "-- image declares itself AI-generated")
            break
    if deepfake is not None:
        bits.append(f"Deepfake arm {deepfake:.0%}")
    if shallow is not None:
        bits.append(f"shallow-fake arm {shallow:.0%}")
    if fired:
        tops = ", ".join(f"{f['slot']} ({f['score']:.2f})" for f in fired[:3])
        bits.append(f"Signals firing: {tops}")
    else:
        bits.append("No individual signal crossed threshold")
    # Surface the VLM's device/context read if present.
    for r in results:
        if r.slot == "learned" and r.evidence.get("device_guess"):
            bits.append(f"Device read: {r.evidence['device_guess']}")
            break
    return ". ".join(bits) + "."

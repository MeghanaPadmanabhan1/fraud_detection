"""The orchestrator: run the configured detectors over one image and compose.

    from image_authenticity.pipeline import Pipeline
    pipe = Pipeline()                       # uses DEFAULT_CONFIG
    out = pipe.run(open("claim.jpg","rb").read(), image_id="claim1",
                   heatmap_dir="/Volumes/.../heatmaps")
    out["composite"]["fraud_score"]         # 0..1
    out["results"]                          # list[DetectionResult]
"""
from __future__ import annotations

from typing import List, Optional

from . import detectors  # noqa: F401  (import triggers detector registration)
from .composite import composite_score
from .config import get_config
from .core import DetectionResult, build
from .detectors._util import decode_bgr


class Pipeline:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_config()
        self.detectors = {
            slot: build(slot, spec["impl"], **spec.get("params", {}))
            for slot, spec in self.config["slots"].items()
        }

    def describe(self) -> dict:
        """Active impl per slot -- handy to print at the top of a demo."""
        return {slot: d.name for slot, d in self.detectors.items()}

    def run(
        self,
        image_bytes: bytes,
        image_id: str = "image",
        context: Optional[dict] = None,
        heatmap_dir: Optional[str] = None,
    ) -> dict:
        base_ctx = dict(context or {})
        try:
            image_bgr = decode_bgr(image_bytes)
        except Exception as e:
            return {
                "image_id": image_id,
                "results": [],
                "composite": {"fraud_score": None, "verdict": "indeterminate",
                              "narrative": f"decode failed: {e}", "fired_signals": []},
            }

        results: List[DetectionResult] = []
        for slot, det in self.detectors.items():
            ctx = dict(base_ctx)
            if heatmap_dir:
                ctx["heatmap_out"] = f"{heatmap_dir.rstrip('/')}/{image_id}__{slot}.png"
            try:
                results.append(det.analyze(image_bgr, image_bytes, ctx))
            except Exception as e:  # defensive: one bad detector can't sink the run
                results.append(
                    DetectionResult(det.name, slot, None, "indeterminate", error=repr(e))
                )

        comp = composite_score(results, self.config)
        return {"image_id": image_id, "results": results, "composite": comp}

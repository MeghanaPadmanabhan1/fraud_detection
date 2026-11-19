"""Core abstractions for the modular image-authenticity pipeline.

Every detection technique is a plugin that implements a single tiny interface
(`Detector.analyze`) and declares which *slot* it fills. A slot is one step in
the pipeline (e.g. "spectral", "copy_move"). Multiple implementations can
register for the same slot; which one is active is chosen by config, not code.

Swapping a model for a slot == changing one line of config.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Labels a detector may emit. Kept small and shared so the composite layer and
# the dashboard can reason about them uniformly.
LABEL_AI = "ai_generated"
LABEL_MANIPULATED = "manipulated"
LABEL_AUTHENTIC = "authentic"
LABEL_INDETERMINATE = "indeterminate"


@dataclass
class DetectionResult:
    """One detector's verdict for one image.

    score: 0..1, higher == more suspicious (more likely fake/manipulated).
    label: one of the LABEL_* constants above.
    evidence: human-readable signals that explain the score (goes to the UI).
    heatmap_path: optional path to an overlay PNG written to a UC Volume.
    error: populated instead of a score when the detector failed to run.
    """

    detector: str
    slot: str
    score: float
    label: str
    evidence: dict = field(default_factory=dict)
    heatmap_path: Optional[str] = None
    error: Optional[str] = None

    def as_row(self) -> dict:
        """Flatten for writing to a Delta table (evidence -> JSON string)."""
        import json

        return {
            "detector": self.detector,
            "slot": self.slot,
            "score": None if self.score is None else float(self.score),
            "label": self.label,
            "evidence_json": json.dumps(self.evidence, default=_json_default),
            "heatmap_path": self.heatmap_path,
            "error": self.error,
        }


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class Detector(ABC):
    """Base class for every detection plugin.

    Subclasses set `slot`, `name`, `version` and implement `analyze`. Register
    them with the `@register` decorator so the pipeline can build them by name.
    """

    slot: str = "unknown"
    name: str = "base"
    version: str = "v1"

    def __init__(self, **kwargs):
        # Arbitrary per-impl config (thresholds, endpoint names, etc.).
        self.config = kwargs

    @abstractmethod
    def analyze(
        self,
        image_bgr: np.ndarray,
        image_bytes: Optional[bytes] = None,
        context: Optional[dict] = None,
    ) -> DetectionResult:
        """Analyze one image.

        image_bgr:   decoded image as an HxWx3 uint8 BGR array (OpenCV order).
        image_bytes: original file bytes (needed by ELA / JPEG-ghost / C2PA
                     which care about the exact on-disk encoding).
        context:     optional metadata (filename, claimed device, address...).
        """

    # Convenience so subclasses can build a result without repeating the slot.
    def result(self, score, label, **kw) -> DetectionResult:
        return DetectionResult(
            detector=self.name, slot=self.slot, score=score, label=label, **kw
        )


# --- registry -------------------------------------------------------------
# slot -> { impl_name -> Detector subclass }
_REGISTRY: dict[str, dict[str, type[Detector]]] = {}


def register(cls: type[Detector]) -> type[Detector]:
    """Class decorator: make a detector discoverable by (slot, name)."""
    _REGISTRY.setdefault(cls.slot, {})[cls.name] = cls
    return cls


def build(slot: str, name: str, **kwargs) -> Detector:
    """Instantiate the detector registered for (slot, name)."""
    try:
        return _REGISTRY[slot][name](**kwargs)
    except KeyError as e:  # pragma: no cover - config error path
        raise KeyError(
            f"No detector registered for slot={slot!r} name={name!r}. "
            f"Available: {available()}"
        ) from e


def available() -> dict[str, list[str]]:
    """Map of slot -> registered impl names (for introspection / UI)."""
    return {slot: sorted(impls) for slot, impls in _REGISTRY.items()}

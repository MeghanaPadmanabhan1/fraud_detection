"""Modular image-authenticity detection for insurance claim fraud.

Deepfake (fully synthetic) + shallow-fake (edited real photo) + provenance
(C2PA / SynthID) detection as swappable, config-driven slots.
"""
from .config import DEFAULT_CONFIG, get_config
from .core import DetectionResult, Detector, available, build, register
from .pipeline import Pipeline

__all__ = [
    "Pipeline",
    "DEFAULT_CONFIG",
    "get_config",
    "DetectionResult",
    "Detector",
    "available",
    "build",
    "register",
]
__version__ = "0.1.0"

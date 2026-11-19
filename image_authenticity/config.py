"""Pipeline configuration -- the single place you swap models per slot.

Each slot names ONE active implementation plus the params to construct it. To
A/B a different model for a slot, change `impl` to another name registered for
that slot (see `image_authenticity.core.available()` for the menu) -- no code
change anywhere else. `arm` and `weight` feed the composite scorer.
"""
from __future__ import annotations

import copy

DEFAULT_CONFIG = {
    "slots": {
        # ---- Deepfake arm (is the whole image synthetic?) ----
        # Calibration note: against SOTA generators (Gemini) the lightweight
        # frequency/ELA heuristics barely separate real from synthetic, so they
        # are weighted low and kept conservative. The semantic VLM carries the
        # arm; provenance (watermark arm) is the other strong deepfake signal.
        "spectral": {
            "impl": "fft_radial_v1",
            "arm": "deepfake",
            "weight": 0.15,
            "params": {},
        },
        "ela_global": {
            "impl": "ela_uniformity_v1",
            "arm": "deepfake",
            "weight": 0.15,
            "params": {"quality": 90},
        },
        "learned": {
            "impl": "vlm_zeroshot_v1",
            "arm": "deepfake",
            "weight": 0.70,
            # Llama-4 Maverick was the strongest deepfake discriminator in the
            # VLM A/B (separation 0.47 vs Claude 0.29, GPT-5 0.20).
            "params": {"endpoint": "databricks-llama-4-maverick"},
        },
        # ---- Shallow-fake arm (was a real photo edited?) ----
        "ela_local": {
            "impl": "ela_heatmap_v1",
            "arm": "shallow",
            "weight": 0.30,
            "params": {"quality": 90},
        },
        "copy_move": {
            "impl": "orb_v1",
            "arm": "shallow",
            "weight": 0.25,
            "params": {},
        },
        "noise": {
            "impl": "noise_residual_v1",
            "arm": "shallow",
            "weight": 0.25,
            "params": {},
        },
        "jpeg_ghost": {
            "impl": "jpeg_ghost_v1",
            "arm": "shallow",
            "weight": 0.20,
            "params": {},
        },
        # ---- Provenance / watermark arm ----
        "watermark": {
            "impl": "c2pa_synthid_v1",
            "arm": "watermark",
            "weight": 1.0,
            "params": {},
        },
    },
    "composite": {
        # Provenance (C2PA/IPTC) is a strong, verifiable signal, so the
        # watermark arm carries real weight rather than a token share.
        "arm_weights": {"deepfake": 0.45, "shallow": 0.30, "watermark": 0.25},
        "fire_threshold": 0.5,
    },
}


def get_config(overrides: dict | None = None) -> dict:
    """Return a deep copy of the default config, with optional slot overrides.

    overrides example -- swap the learned slot's model and the spectral impl:
        {"slots": {"learned": {"params": {"endpoint": "databricks-gpt-5"}},
                   "spectral": {"impl": "some_other_v2"}}}
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        for slot, patch in overrides.get("slots", {}).items():
            cfg["slots"].setdefault(slot, {}).update(patch)
        if "composite" in overrides:
            cfg["composite"].update(overrides["composite"])
    return cfg

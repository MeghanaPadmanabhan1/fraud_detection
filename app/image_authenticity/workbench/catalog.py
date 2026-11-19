"""Stage catalog: the eight switchable stages, their technology options, presets.

Mirrors §4 of the Fraud Defence Workbench design. Options the prototype cannot
run (vendor APIs not connected, GPU models not deployed, proprietary verifiers)
are listed with available=False and a reason, so the UI shows the full switchboard
honestly instead of hiding the gaps.
"""
import copy

VLMS = {
    "databricks-llama-4-maverick": "Llama-4 Maverick",
    "databricks-claude-sonnet-5": "Claude Sonnet 5",
    "databricks-gemini-3-1-pro": "Gemini 3.1 Pro",
    "databricks-gpt-5": "GPT-5",
}


def _opt(key, label, available=True, note="", path=None):
    return {"key": key, "label": label, "available": available, "note": note, "path": path}


STAGES = [
    {"id": "s0", "num": 0, "name": "Capture", "question": "Was this photo taken cleanly?",
     "path": "sync", "budget": "< 1 s", "mode": "single", "default": "native_upload",
     "options": [_opt("native_upload", "Native upload (web / app)"),
                 _opt("c2pa_capture", "C2PA-signed controlled capture SDK", False,
                      "Requires the capture SDK in the claim channel.")]},
    {"id": "s1", "num": 1, "name": "Provenance & metadata",
     "question": "Does the photo's story match the claim?", "path": "sync", "budget": "1–2 s",
     "mode": "single", "default": "exif_full",
     "options": [_opt("exif_rules", "EXIF rules (date vs claim, editing software, stripped)"),
                 _opt("exif_full", "EXIF rules + C2PA / IPTC provenance"),
                 _opt("synthid", "Google SynthID detector", False,
                      "Google-proprietary; no public verifier API. Drop-in when access is granted.")],
     "params": {"max_photo_age_days": 30, "missing_metadata": "amber"}},
    {"id": "s2", "num": 2, "name": "Reuse & shallowfake", "question": "Has this photo been seen before?",
     "path": "sync (internal) / async (web)", "budget": "1–3 s / 10–60 s", "mode": "single",
     "default": "phash",
     "options": [_opt("phash", "Perceptual hash (pHash) near-match"),
                 _opt("dhash", "Difference hash (dHash) near-match"),
                 _opt("orb", "Keypoint match (ORB + RANSAC) — crop-robust"),
                 _opt("embed", "Perceptual embedding kNN (Vector Search pattern)"),
                 _opt("sscd", "SSCD learned copy-detection embedding (Meta, CVPR 2022)", True,
                      "Neural descriptor; survives crops, screenshots, re-photos. Vectors are precomputed in the index; "
                      "new photos need torch + weights."),
                 _opt("opensearch", "Legacy OpenSearch index", False, "Comparator only — not connected."),
                 _opt("google_web", "Google Web Detection API (web reuse)", False,
                      "External, async path; not connected in the prototype.")],
     "params": {"phash_max": 10, "dhash_max": 10, "orb_min_inliers": 25, "embed_min_cos": 0.96, "sscd_min_cos": 0.60}},
    {"id": "s3", "num": 3, "name": "Pixel forensics", "question": "Was this photo edited?",
     "path": "sync (trio) / async (full)", "budget": "2–5 s / 10–30 s", "mode": "multi",
     "default": ["ela", "jpeg_ghost", "upscale"],
     "options": [_opt("ela", "Error Level Analysis (ELA)"),
                 _opt("jpeg_ghost", "JPEG ghost / double compression"),
                 _opt("noise", "Noise residual (PRNU-lite splice check)"),
                 _opt("copy_move_orb", "Copy-move (ORB)"),
                 _opt("copy_move_sift", "Copy-move (SIFT)"),
                 _opt("recapture", "Recapture: screenshot & re-photographed screen"),
                 _opt("upscale", "Resampling / upscale detector"),
                 _opt("lighting", "Lighting & shadow consistency", False, "Not implemented in the prototype.")],
     # Per-technique [amber, red] thresholds, calibrated on gold-set v0 genuine photos
     # (amber ≈ genuine p95, red just above the genuine max) — a false-positive budget.
     "params": {"thresholds": {"ela": [0.855, 0.905], "jpeg_ghost": [0.961, 0.995], "noise": [0.99, 0.995],
                               "copy_move_orb": [0.5, 0.8], "copy_move_sift": [0.922, 0.975],
                               "recapture": [0.5, 0.8], "upscale": [0.5, 0.8]}}},
    {"id": "s4", "num": 4, "name": "AI deepfake detection", "question": "Is this AI-made or AI-altered?",
     "path": "sync (small) / async (large)", "budget": "2–8 s / 15–60 s", "mode": "multi",
     "default": ["vlm:databricks-llama-4-maverick"],
     "options": [_opt(f"vlm:{ep}", f"{name} (vision model)", path="async") for ep, name in VLMS.items()]
     + [_opt("spectral_fft", "Spectral analysis (FFT)"),
        _opt("spectral_dct", "Spectral analysis (DCT)"),
        _opt("ela_uniformity", "ELA uniformity (whole-image)"),
        _opt("trufor", "TruFor localizer (anomaly mask)", False, "Needs a GPU Model Serving endpoint."),
        _opt("catnet", "CAT-Net v2", False, "Needs a GPU Model Serving endpoint."),
        _opt("inhouse", "In-house models 1–4", False, "Customer's production models — plug in as challengers.")],
     "params": {"amber": 0.5, "red": 0.8,
                "thresholds": {"spectral_fft": [0.5, 0.8], "spectral_dct": [0.5, 0.8], "ela_uniformity": [0.5, 0.8]}}},
    {"id": "s5", "num": 5, "name": "Documents & OCR", "question": "Does the receipt support the claim?",
     "path": "async", "budget": "5–20 s", "mode": "single", "default": "ai_parse_document",
     "options": [_opt("ai_parse_document", "Databricks ai_parse_document")]
     + [_opt(f"vlm_ocr:{ep}", f"{name} OCR (vision model)") for ep, name in list(VLMS.items())[:3]]
     + [_opt("textract", "AWS Textract", False, "External vendor — comparator, not connected."),
        _opt("vision_ai", "Google Vision AI", False, "External vendor — comparator, not connected.")]},
    {"id": "s6", "num": 6, "name": "Reasoning & verdict", "question": "What is the decision, and why?",
     "path": "async", "budget": "5–30 s", "mode": "single", "default": "rules_v1",
     "options": [_opt("rules_v1", "Rules engine (deterministic) + plain-language reasons"),
                 _opt("agent", "Agentic orchestrator (Agent Framework)", False,
                      "Opt-in mode; must beat the rules engine on the gold set first.")],
     "explainers": [_opt("template", "Template explainer (no LLM)")]
     + [_opt(f"vlm:{ep}", f"{name} explainer") for ep, name in VLMS.items()],
     "params": {"review": 0.45, "fraud": 0.75, "explainer": "template"}},
    {"id": "s7", "num": 7, "name": "Human review & feedback", "question": "Did we get it right?",
     "path": "async", "budget": "Human time", "mode": "single", "default": "review_queue",
     "options": [_opt("review_queue", "Review queue + labelling → Delta gold set")]},
]
STAGE_BY_ID = {s["id"]: s for s in STAGES}


def option_label(sid, key):
    st = STAGE_BY_ID[sid]
    for o in st["options"] + st.get("explainers", []):
        if o["key"] == key:
            return o["label"]
    return key


def default_config():
    cfg = {}
    for s in STAGES:
        cfg[s["id"]] = {"enabled": True, "tech": copy.deepcopy(s["default"]),
                        "params": copy.deepcopy(s.get("params", {}))}
    return cfg


def _with(base, **stages):
    cfg = copy.deepcopy(base)
    for sid, patch in stages.items():
        p = dict(patch)
        if "params" in p:
            cfg[sid]["params"].update(p.pop("params"))
        cfg[sid].update(p)
    return cfg


_BASE = default_config()
PRESETS = {
    "Production": _BASE,
    "Challenger A": _with(
        _BASE,
        s2={"tech": "orb"},
        s3={"tech": ["ela", "jpeg_ghost", "upscale", "recapture", "copy_move_sift"]},
        s4={"tech": ["vlm:databricks-claude-sonnet-5", "spectral_dct"]},
        s5={"tech": "vlm_ocr:databricks-llama-4-maverick"},
        s6={"params": {"explainer": "vlm:databricks-llama-4-maverick"}}),
    "SSCD reuse": _with(_BASE, s2={"tech": "sscd"}),
    "Fast path only": _with(_BASE, s4={"enabled": False}, s5={"enabled": False}),
    "Full forensic": _with(
        _BASE,
        s2={"tech": "orb"},
        s3={"tech": ["ela", "jpeg_ghost", "noise", "copy_move_orb", "copy_move_sift", "recapture", "upscale"]},
        s4={"tech": ["vlm:databricks-llama-4-maverick", "spectral_dct", "ela_uniformity"]}),
}
PRESET_NOTES = {
    "Production": "Current default: pHash reuse, ELA + JPEG-ghost + upscale trio, Llama-4 vision detector, ai_parse_document.",
    "Challenger A": "Crop-robust reuse (ORB), recapture + upscale + SIFT copy-move, Claude vision, Llama OCR.",
    "SSCD reuse": "Production, but Stage 2 reuse uses the SSCD learned embedding (served from a Model Serving endpoint) instead of pHash.",
    "Fast path only": "Synchronous stages 0–3 only — fits well inside a 30 s gateway budget.",
    "Full forensic": "Every available check on. Slowest, most thorough.",
}


def preset_config(name):
    return copy.deepcopy(PRESETS[name])

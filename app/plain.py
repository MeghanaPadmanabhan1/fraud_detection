"""Plain-language layer for the Workbench UI. No Streamlit here, so it can be tested on its own.

The engine keeps its technical names (stages, techniques, scores). This module only translates what an investigator
reads: headline, check names, reason wording, method names. Anything not translated falls back to the engine's text.
"""
from __future__ import annotations

import re

# Five checks an investigator cares about (stage 0 capture, 6 verdict and 7 review are not "checks").
CHECKS = [("s1", "Photo details", "Does the photo's story match the claim?"),
          ("s2", "Seen before?", "Has this photo been used in another claim?"),
          ("s3", "Signs of editing", "Does the photo look edited?"),
          ("s4", "Made by AI?", "Does the photo look AI-made or AI-altered?"),
          ("s5", "Receipt", "Does the receipt back up the claim?")]
CHECK_NAME = {sid: name for sid, name, _ in CHECKS}
CHECK_QUESTION = {sid: q for sid, _, q in CHECKS}

SEV_WORD = {"red": ("Problem", "✗"), "amber": ("Take a look", "⚠"), "green": ("OK", "✓"), "info": ("OK", "✓")}

FAMILY_PLAIN = {"shallowfake": "Reused or misrepresented real photo", "deepfake": "Edited or AI-made photo",
                "document": "Receipt doesn't match the claim"}

# Overlay names the engine produces -> what the investigator sees.
OVERLAY_PLAIN = {
    "Photo only": ("Photo", ""),
    "ELA heatmap": ("Where the photo differs most", "A guide, not proof. This map always lights up the parts of a photo that "
                                                 "differ most from the rest (edges and fine detail), even on genuine photos. "
                                                 "Only treat it as a sign when the check result above also flags editing."),
    "Noise map": ("Where the grain differs most", "A guide, not proof. Camera grain is usually even across a photo. "
                                                "A patch that stands out may have been pasted in."),
    "JPEG ghost map": ("Where save history differs most (JPEG ghost)", "A guide, not proof. Warm areas went through a different number of "
                                                              "saves than the rest. Forwarded or re-saved photos show this too."),
    "Copy-move pairs": ("Cloned parts", "Lines join parts of the photo that are near-identical, which suggests "
                                        "one part was copied onto another."),
    "Ground-truth mask (gold set)": ("Known edited area (test data)", "The area the test set says was altered. "
                                                                      "Use it to check the detectors."),
}

# Method names for the "try a different method" screens: key -> (plain label, one-line hint).
OPTION_PLAIN = {
    "s1": {"exif_rules": ("Photo details only", "Camera date, device and editing software."),
           "exif_full": ("Photo details + AI tags", "Adds checks for tags that say an AI made the image.")},
    "s2": {"phash": ("Quick fingerprint", "Fast. Catches exact and re-saved copies, misses crops and screenshots."),
           "dhash": ("Quick fingerprint (alternative)", "Similar to the quick fingerprint."),
           "orb": ("Feature matching", "Finds cropped copies and shows where they match. Compares one pair at a time."),
           "embed": ("Simple look-alike search", "Rough match on overall brightness pattern."),
           "sscd": ("Learned look-alike search", "Finds crops, screenshots and re-photos. Scales to millions of claims.")},
    "s3": {"ela": ("Re-save pattern (ELA)", "Spots areas saved differently from the rest."),
           "jpeg_ghost": ("Save history (JPEG ghost)", "Spots areas that were saved a different number of times. Forwarded photos can trigger it."),
           "noise": ("Grain pattern (noise)", "Spots patches whose camera grain doesn't match."),
           "copy_move_orb": ("Cloned parts (fast)", "Finds parts copied onto other parts."),
           "copy_move_sift": ("Cloned parts (thorough)", "Slower, finds more clones."),
           "recapture": ("Screenshot or photo of a screen", "Spots a photo taken of another screen."),
           "upscale": ("Enlarged photo", "Spots a small image blown up to look larger.")},
    "s4": {"spectral_fft": ("Fine-detail pattern (A)", "Looks at the photo's fine-detail pattern."),
           "spectral_dct": ("Fine-detail pattern (B)", "Looks at the photo's fine-detail pattern."),
           "ela_uniformity": ("Evenness of re-save pattern", "AI images are often unusually even.")},
    "s5": {"ai_parse_document": ("Databricks document reader", "Reads the receipt text and numbers.")},
}
_VLM = re.compile(r"^(vlm|vlm_ocr):(.+)$")


def option_label(sid, key, fallback=None):
    """Plain label + hint for an option key; vision models keep their product name."""
    d = OPTION_PLAIN.get(sid, {})
    if key in d:
        return d[key]
    m = _VLM.match(key)
    if m:
        name = (fallback or m.group(2)).replace(" (vision model)", "").replace(" OCR", "")
        return (f"{name}" + (" reads the receipt" if m.group(1) == "vlm_ocr" else " looks at the photo"),
                "A vision model that describes what it sees. Slower, needs a model call.")
    return (fallback or key, "")


def confidence_word(c: float) -> str:
    return "High" if c >= 0.85 else "Medium" if c >= 0.6 else "Low"


def _clean(text: str) -> str:
    """Drop trailing technique detail and rewrite the few phrases that read like lab output."""
    t = text.strip()
    t = re.sub(r"\s+—\s+(pHash|dHash|SSCD|embedding|\d+ geometrically)[^—]*$", ".", t)
    t = re.sub(r"\.\.$", ".", t)
    m = re.match(r"JPEG ghost: a cluster of blocks carries a different compression history \((\d+)% of the image\)", t)
    if m:
        return (f"About {m.group(1)}% of the photo was saved differently from the rest (JPEG ghost check). "
                "This also happens with forwarded or re-saved photos, so on its own it is a weak sign.")
    if t.startswith("Resampling:"):
        return "The photo looks enlarged from a much smaller original."
    m = re.match(r"(.+?) rates this photo (\d+)% likely digitally altered: (.+)$", t)
    if m:
        return f"{m.group(1)} thinks this photo is {m.group(2)}% likely to be altered: {m.group(3)}"
    if t.startswith("No camera metadata"):
        return ("The camera details were removed from this photo. That happens with screenshots and forwarded "
                "photos, but also when someone strips them on purpose.")
    t = t.replace("The file declares itself AI-generated: IPTC DigitalSourceType = trainedAlgorithmicMedia, "
                  "embedded by the image generator.", "The file itself says it was made by an AI image generator.")
    return t


def reason_text(f) -> str:
    return _clean(f.text)


def plain_stage(f) -> str:
    """Which of the five checks an investigator would file this finding under.
    The engine records the AI-origin tag under provenance (s1) and the vision model's device-mismatch note under AI
    detection (s4); to a reader the first is an 'AI-made?' sign and the second a 'does the story match?' sign."""
    if f.stage == "s1" and f.text.startswith("The file declares itself AI-generated"):
        return "s4"
    if f.stage == "s4" and f.check.endswith("(context)"):
        return "s1"
    return f.stage


# Checks the experiments found unreliable on real phone photos (they fire on forwarded or re-saved genuine photos).
# They still count in the decision; they just never lead the banner when stronger evidence exists.
WEAK_CHECKS = ("JPEG ghost", "Error Level", "ELA", "Noise residual")
_SEV_RANK = {"red": 0, "amber": 1}


def _is_weak(f) -> bool:
    return any(k in f.check for k in WEAK_CHECKS)


def top_reasons(reasons, n=3):
    """Strong evidence before weak, then red before amber, then score. One reason per check before a second from the same check."""
    ordered = sorted(reasons, key=lambda f: (_is_weak(f), _SEV_RANK.get(f.severity, 2), -(f.score or 0)))
    out, seen = [], set()
    for f in ordered:
        if plain_stage(f) not in seen:
            out.append(f); seen.add(plain_stage(f))
    for f in ordered:
        if f not in out:
            out.append(f)
    return out[:n]


def _sentence(t: str) -> str:
    t = t.strip()
    return t if t[-1:] in ".!?”\")" else t + "."


def headline(r) -> str:
    if r.verdict == "clear" or not r.reasons:
        return "No check found a problem with this claim."
    return _sentence(reason_text(top_reasons(r.reasons, 1)[0]))


def more_signs(r) -> int:
    """How many other checks (besides the headline's) also raised something."""
    return max(0, len({plain_stage(f) for f in r.reasons}) - 1)


def verdict_title(r) -> tuple[str, str]:
    """(label, colour)."""
    return {"likely_fraud": ("Likely fraud", "#D7263D"), "needs_review": ("Needs a human look", "#E89F00"),
            "clear": ("Looks fine", "#2E9E5B")}[r.verdict]


def next_step(r) -> str:
    return {"likely_fraud": "Send to an investigator.", "needs_review": "Send to an analyst for review.",
            "clear": "No action needed."}[r.verdict] + " Nothing is denied automatically."


_RANK = {"info": 0, "green": 0, "amber": 1, "red": 2}


def check_states(r):
    """[(sid, name, symbol, word, severity)] for the checks strip."""
    worst = {sid: "green" for sid, _, _ in CHECKS}
    for st_ in r.stages:
        for f in st_.findings:
            k = plain_stage(f)
            if k in worst and _RANK.get(f.severity, 0) > _RANK[worst[k]]:
                worst[k] = f.severity
    could_not = {plain_stage(f) for st_ in r.stages for f in st_.findings if f.evidence.get("unavailable")}
    out = []
    for sid, name, _ in CHECKS:
        s = r.stage(sid)
        if sid in could_not and worst[sid] == "green":
            out.append((sid, name, "!", "Didn't run", "failed"))
        elif worst[sid] != "green":
            word, sym = SEV_WORD[worst[sid]]
            out.append((sid, name, sym, word, worst[sid]))
        elif s.status == "skipped":
            out.append((sid, name, "–", "Off", "skipped"))
        elif s.status == "pending":
            out.append((sid, name, "…", "Waiting", "pending"))
        elif s.status == "failed":
            out.append((sid, name, "!", "Didn't run", "failed"))
        else:
            out.append((sid, name, "✓", "OK", "green"))
    return out


def families_plain(r) -> str:
    return " · ".join(FAMILY_PLAIN[f] for f in r.families if f in FAMILY_PLAIN)

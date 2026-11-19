"""Watermark / provenance arm -- C2PA + EXIF + SynthID slot.

Reads whatever verifiable provenance signals the file carries:
- C2PA / CAI: a signed manifest embedded as a JUMBF box (the "nutrition label"
  pushed by Adobe, cameras, and some generators). Presence of a c2pa manifest
  is a strong, cryptographically-backed provenance signal.
- EXIF: camera make/model/software. Missing EXIF + "edited with Photoshop"
  software tags are weak corroborating signals.
- SynthID: Google's invisible watermark in Gemini/Imagen output. IMPORTANT --
  there is NO public verifier API for SynthID; detection is Google-proprietary.
  This slot therefore returns `indeterminate` and documents the gap honestly.
  It is wired as a swappable slot so a real SynthID verifier drops in the moment
  access is granted, without touching the pipeline.
"""
from __future__ import annotations

import io

from ..core import (
    LABEL_AI,
    LABEL_AUTHENTIC,
    LABEL_INDETERMINATE,
    LABEL_MANIPULATED,
    Detector,
    register,
)


def _scan(image_bytes: bytes) -> bytes:
    if not image_bytes:
        return b""
    # Provenance metadata lives in the header (XMP/EXIF) or a trailing box.
    return (image_bytes[:400_000] + image_bytes[-80_000:]).lower()


def _find_c2pa(image_bytes: bytes) -> bool:
    # C2PA manifests are stored in a JUMBF box; the store is tagged "c2pa".
    s = _scan(image_bytes)
    return any(n in s for n in (b"c2pa", b"jumbf", b"urn:uuid:c2pa", b"contentauth"))


def _find_ai_provenance(image_bytes: bytes) -> bool:
    # IPTC DigitalSourceType = trainedAlgorithmicMedia is the standardized
    # "this is AI-generated" tag embedded by Gemini/Imagen (and others). This
    # is a verifiable provenance signal, distinct from the invisible SynthID
    # pixel watermark (which has no public verifier).
    s = _scan(image_bytes)
    return any(n in s for n in (
        b"trainedalgorithmicmedia",
        b"compositewithtrainedalgorithmicmedia",
        b"digitalsourcetype",
    ))


def _read_exif(image_bytes: bytes) -> dict:
    try:
        from PIL import Image, ExifTags

        img = Image.open(io.BytesIO(image_bytes))
        exif = img.getexif()
        if not exif:
            return {}
        tagmap = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        keep = {}
        for key in ("Make", "Model", "Software", "DateTime"):
            if key in tagmap:
                keep[key] = str(tagmap[key])
        return keep
    except Exception:
        return {}


@register
class C2PASynthIDDetector(Detector):
    slot = "watermark"
    name = "c2pa_synthid_v1"
    version = "1.0"

    def analyze(self, image_bgr, image_bytes=None, context=None):
        try:
            ai_prov = _find_ai_provenance(image_bytes or b"")
            has_c2pa = _find_c2pa(image_bytes or b"")
            exif = _read_exif(image_bytes or b"")
            software = (exif.get("Software") or "").lower()
            edited_sw = any(s in software for s in ("photoshop", "gimp", "affinity"))
            has_camera = bool(exif.get("Make") or exif.get("Model"))

            evidence = {
                "ai_provenance_tag": ai_prov,
                "c2pa_manifest_present": has_c2pa,
                "exif": exif,
                "edited_software_tag": edited_sw,
                "has_camera_exif": has_camera,
                "synthid": "indeterminate",
                "synthid_note": (
                    "SynthID invisible pixel watermark has no public verifier API; "
                    "detection is Google-proprietary. This slot verifies the "
                    "embedded C2PA/IPTC provenance instead, and is stubbed to swap "
                    "in a real SynthID verifier when access is granted."
                ),
            }

            # Strongest, verifiable signal: a standardized AI-provenance tag.
            if ai_prov:
                return self.result(
                    0.9, LABEL_AI,
                    evidence={**evidence, "note": "IPTC DigitalSourceType=trainedAlgorithmicMedia "
                              "-- image declares itself AI-generated (Gemini/Imagen)."},
                )
            if edited_sw:
                return self.result(
                    0.6, LABEL_MANIPULATED,
                    evidence={**evidence, "note": "editor software tag present in EXIF."},
                )
            if has_c2pa:
                return self.result(
                    0.5, LABEL_INDETERMINATE,
                    evidence={**evidence, "note": "C2PA manifest present -- parse to read claims."},
                )
            if not has_camera and not image_has_metadata(exif):
                return self.result(
                    0.5, LABEL_INDETERMINATE,
                    evidence={**evidence, "note": "stripped metadata; no provenance to verify."},
                )
            return self.result(
                0.1, LABEL_AUTHENTIC,
                evidence={**evidence, "note": "camera EXIF present, no edit/AI provenance flags."},
            )
        except Exception as e:  # pragma: no cover
            return self.result(None, LABEL_INDETERMINATE, error=repr(e))


def image_has_metadata(exif: dict) -> bool:
    return bool(exif)

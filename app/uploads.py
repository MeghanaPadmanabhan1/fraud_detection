"""Check a new photo: file validation, the claim record built from the form, example photos, and the run.

Nothing here writes anything: the uploaded files live in the user's session only. Uploads are untrusted input, so
the checks run on the file header first (size, format, pixel count) before a single pixel is decoded.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import io
import json

import cv2
import numpy as np
from PIL import Image

import backend as B
import ui
from image_authenticity.workbench import run_claim

MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 40_000_000          # 40 MP: covers current phones, rejects decompression bombs
MIN_SIDE = 200
OCR_ENGINE = "vlm_ocr:databricks-llama-4-maverick"      # the document reader only has stored results for test claims
_FACE = []


def check_image(raw: bytes, what: str = "photo") -> tuple[bool, str]:
    """(ok, message). Header checks come first, so an oversized image is refused before it is decoded."""
    if not raw:
        return False, f"The {what} is empty."
    if len(raw) > MAX_BYTES:
        return False, f"The {what} is {len(raw) / 1e6:.0f} MB. The limit is {MAX_BYTES // 2 ** 20} MB."
    try:
        im = Image.open(io.BytesIO(raw))
        fmt, (w, h) = im.format, im.size
    except Exception:
        return False, f"The {what} could not be read as an image. Please upload a JPEG or PNG."
    if fmt not in ("JPEG", "PNG"):
        return False, f"The {what} is a {fmt or 'unknown'} file. Please upload a JPEG or PNG."
    if w * h > MAX_PIXELS:
        return False, f"The {what} is {w * h / 1e6:.0f} megapixels. The limit is {MAX_PIXELS // 10 ** 6}."
    if min(w, h) < MIN_SIDE:
        return False, f"The {what} is very small ({w}×{h}). It needs at least {MIN_SIDE} pixels on the short side."
    try:
        im.verify()
        ok = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) is not None
    except Exception:
        ok = False
    if not ok:
        return False, f"The {what} looks damaged. Please try another file."
    return True, f"{fmt} · {w}×{h} · {len(raw) / 1e6:.1f} MB"


def has_face(raw: bytes) -> bool:
    """The test set excludes photos of people; warn if an upload seems to contain a face."""
    try:
        if not _FACE:
            _FACE.append(cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"))
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        f = 800 / max(img.shape)
        if f < 1:
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        return len(_FACE[0].detectMultiScale(img, 1.1, 6, minSize=(50, 50))) > 0
    except Exception:
        return False


def build_claim(photo: bytes, receipt: bytes | None, d: dict) -> dict:
    """The claim record the engine needs. The id is a hash of everything entered, so changing anything is a new check."""
    key = photo + (receipt or b"") + json.dumps(d, sort_keys=True, default=str).encode()
    return {"claim_id": "NEW-" + hashlib.sha256(key).hexdigest()[:8],
            "claimant_id": (d["claimant_id"] or "").strip() or "CUST-NEW",
            "claim_date": dt.date.today().isoformat(), "incident_date": d["incident_date"].isoformat(),
            "device_claimed": d["device"].strip(), "damage_described": (d["damage"] or "").strip(),
            "channel": "web upload", "claim_value": float(d["value"]),
            "purchase_date": d["purchase_date"].isoformat(),
            "image_path": None, "receipt_path": None, "mask_path": None,
            "family": "unknown", "fraud_class": "unknown", "notes": None}


def run_upload(photo: bytes, receipt: bytes | None, d: dict, use_sscd: bool, live: bool):
    claim = build_claim(photo, receipt, d)
    cfg = copy.deepcopy(ui.get_cfg("a"))
    cfg["s6"]["params"]["explainer"] = "template"
    if use_sscd:
        cfg["s2"]["tech"] = "sscd"
    if receipt and cfg["s5"]["tech"] == "ai_parse_document":
        cfg["s5"]["tech"] = OCR_ENGINE
    r = run_claim(claim, photo, receipt, cfg, B.upload_context(claim, live), want_overlays=True)
    for st_ in r.stages:                # an upload has no slow path to wait for: pending means the check did not run
        if st_.status == "pending":
            st_.status = "failed"
            for f in st_.findings:
                if "pending" in f.text:
                    f.text = ("This check did not run: the AI model did not answer, or live AI calls are switched "
                              "off. Press “Check this claim” to try again.")
    return claim, r, cfg


# ------------------------------------------------------------------ example photos (built from earlier claims)
SAMPLES = [
    {"key": "crop", "label": "Cropped copy of an earlier photo", "source": "CLM-1008", "how": "crop",
     "note": "A tighter crop of CLM-1008's photo. Quick fingerprints often miss crops like this; the learned look-alike search is built to find them."},
    {"key": "shot", "label": "Screenshot of an earlier photo", "source": "CLM-1010", "how": "screenshot",
     "note": "CLM-1010's photo as seen on a phone screen, with the status bar and toolbar. Camera details are gone."},
    {"key": "fwd", "label": "Forwarded copy of an earlier photo", "source": "CLM-1012", "how": "forward",
     "note": "CLM-1012's photo shrunk and heavily compressed, like a photo forwarded through a messaging app."},
    {"key": "ai", "label": "Image straight from an AI generator", "source": "CLM-1059", "how": "as_is",
     "note": "CLM-1059's image exactly as the generator made it, so it still carries its AI tag. It also matches that claim."},
]
SAMPLE = {s["key"]: s for s in SAMPLES}


def source_claim(key: str) -> dict:
    return ui.claim_row(SAMPLE[key]["source"])


def sample_bytes(key: str) -> bytes:
    s = SAMPLE[key]
    raw = B.media(source_claim(key)["image_path"])
    if s["how"] == "as_is":
        return raw
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    q = 90
    if s["how"] == "crop":
        img = img[int(h * .125):int(h * .875), int(w * .125):int(w * .875)]
    elif s["how"] == "screenshot":
        W, H, q = 1170, 2532, 85
        r = cv2.resize(img, (W, round(h * W / w)), interpolation=cv2.INTER_AREA)
        canvas = np.full((H, W, 3), 28, np.uint8)
        y = (H - r.shape[0]) // 2
        canvas[y:y + r.shape[0]] = r
        cv2.rectangle(canvas, (0, 0), (W, 110), (40, 40, 40), -1)
        cv2.rectangle(canvas, (0, H - 150), (W, H), (40, 40, 40), -1)
        img = canvas
    elif s["how"] == "forward":
        img, q = cv2.resize(img, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA), 40
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()      # re-encoding drops EXIF

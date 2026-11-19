"""Workbench-specific checks: metadata vs claim, reuse index, recapture,
resampling, receipt cross-check, and forensic overlay maps for the UI."""
from __future__ import annotations

import datetime as dt
import io
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..detectors import _util
from ..detectors.ela import _block_reduce, _ela_map
from ..detectors.watermark import _find_ai_provenance, _find_c2pa

EDITORS = ("photoshop", "lightroom", "gimp", "snapseed", "picsart", "facetune",
           "affinity", "pixelmator", "canva", "remini")


# ----------------------------------------------------------------- metadata
def read_metadata(image_bytes: bytes) -> dict:
    from PIL import Image
    out = {"format": None, "make": None, "model": None, "software": None,
           "taken": None, "width": None, "height": None, "has_exif": False}
    try:
        im = Image.open(io.BytesIO(image_bytes))
        out.update(format=im.format, width=im.width, height=im.height)
        ex = im.getexif()
        if ex:
            out["has_exif"] = len(ex) > 0
            out["make"], out["model"], out["software"] = ex.get(271), ex.get(272), ex.get(305)
            taken = ex.get_ifd(0x8769).get(36867) or ex.get(306)
            if taken:
                try:
                    out["taken"] = dt.datetime.strptime(str(taken).strip("\x00"), "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    pass
    except Exception:
        pass
    out["ai_provenance"] = _find_ai_provenance(image_bytes)
    out["c2pa"] = _find_c2pa(image_bytes)
    return out


# ----------------------------------------------------------------- hashing / reuse
def _gray_small(img_bgr, side=512):
    g = _util.to_gray(img_bgr)
    s = max(g.shape)
    if s > side:
        f = side / s
        g = cv2.resize(g, (int(g.shape[1] * f), int(g.shape[0] * f)), interpolation=cv2.INTER_AREA)
    return g


def phash(img_bgr) -> int:
    g = cv2.resize(_util.to_gray(img_bgr), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    bits = d > np.median(d[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def dhash(img_bgr) -> int:
    g = cv2.resize(_util.to_gray(img_bgr), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def embedding(img_bgr) -> np.ndarray:
    g = cv2.resize(_util.to_gray(img_bgr), (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    v = (g - g.mean()).flatten()
    return v / (np.linalg.norm(v) + 1e-9)


def hamming(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


_ORB = None


def orb_features(img_bgr):
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(nfeatures=1500)
    g = _gray_small(img_bgr, 800)
    kps, desc = _ORB.detectAndCompute(g, None)
    pts = np.float32([k.pt for k in kps]) if kps else np.zeros((0, 2), np.float32)
    return pts, desc


def orb_inliers(fa, fb) -> int:
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None or len(da) < 10 or len(db) < 10:
        return 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    if len(good) < 10:
        return len(good) // 3
    src = pa[[g.queryIdx for g in good]]
    dst = pb[[g.trainIdx for g in good]]
    cv2.setRNGSeed(7)   # deterministic RANSAC → reproducible evidence text and cache keys
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    return int(mask.sum()) if mask is not None else 0


# ----------------------------------------------------------------- SSCD (learned copy-detection descriptor)
# Meta's SSCD (Pizzi et al., CVPR 2022; MIT). torch is imported lazily so the app still starts without it; claims
# whose vectors were precomputed into the index work without torch (lookup mode), new images need torch + weights.
SSCD_MODEL = "sscd_disc_mixup"      # ResNet-50, 512-d descriptor
_SSCD = {}


def _sscd_weights_path():
    name = f"{SSCD_MODEL}.torchscript.pt"
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (os.environ.get("WB_SSCD_DIR"), os.path.join(here, "..", "..", "wb_data", "ml_models", "sscd"),
              os.path.join(here, "..", "..", "app", "models")):
        if d and os.path.exists(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


def sscd_live_available() -> bool:
    if "ok" not in _SSCD:
        try:
            import torch  # noqa: F401
            _SSCD["ok"] = _sscd_weights_path() is not None
        except Exception:
            _SSCD["ok"] = False
    return _SSCD["ok"]


def sscd_embedding(img_bgr) -> np.ndarray:
    """512-d L2-normalised SSCD descriptor. Preprocessing: short edge 288, aspect kept, ImageNet norm."""
    import torch
    if "net" not in _SSCD:
        _SSCD["net"] = torch.jit.load(_sscd_weights_path(), map_location="cpu").eval()
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    h, w = img_bgr.shape[:2]
    k = 288 / min(h, w)
    nw, nh = max(1, round(w * k)), max(1, round(h * k))
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255 - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])
    with torch.no_grad():
        v = _SSCD["net"](torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float())[0]
        v = torch.nn.functional.normalize(v, dim=0)
    return v.numpy().astype(np.float32)


@dataclass
class IndexEntry:
    claim_id: str
    claimant_id: str
    claim_date: str
    image_path: str
    phash: int
    dhash: int
    emb: np.ndarray
    orb: Optional[tuple] = None
    sscd: Optional[np.ndarray] = None


@dataclass
class ReuseIndex:
    """In-memory view of the Delta hash index (Silver). Built from the claims table,
    so it is always as fresh as the table — the OpenSearch staleness lesson."""
    entries: list = field(default_factory=list)
    built_at: Optional[dt.datetime] = None
    images: dict = field(default_factory=dict)     # claim_id -> bgr (for ORB, lazy)

    def prior(self, claim):
        key = (claim["claim_date"], claim["claim_id"])
        return [e for e in self.entries if (e.claim_date, e.claim_id) < key and e.claim_id != claim["claim_id"]]

    def sscd_for(self, e, loader=None):
        """Precomputed vector if the index has one, else compute from the image (needs torch + weights)."""
        if e.sscd is None and loader is not None and sscd_live_available():
            img = self.images.get(e.claim_id)
            if img is None:
                img = loader(e)
                self.images[e.claim_id] = img
            e.sscd = sscd_embedding(img) if img is not None else None
        return e.sscd

    def orb_for(self, e, loader=None):
        if e.orb is None:
            img = self.images.get(e.claim_id)
            if img is None and loader is not None:
                img = loader(e)
                self.images[e.claim_id] = img
            e.orb = orb_features(img) if img is not None else (np.zeros((0, 2)), None)
        return e.orb


def make_entry(claim, img_bgr, with_sscd=False) -> IndexEntry:
    return IndexEntry(claim["claim_id"], claim["claimant_id"], claim["claim_date"], claim["image_path"],
                      phash(img_bgr), dhash(img_bgr), embedding(img_bgr),
                      sscd=sscd_embedding(img_bgr) if with_sscd and sscd_live_available() else None)


# ----------------------------------------------------------------- recapture
SCREEN_SIZES = {(1179, 2556), (1170, 2532), (1290, 2796), (1284, 2778), (1080, 2400), (1080, 2340),
                (1440, 3200), (1125, 2436), (828, 1792), (750, 1334), (1242, 2688)}


def moire_peak(gray) -> float:
    """Peak-to-background ratio of isolated periodic peaks, measured on full-resolution
    crops (downsampling first would alias the screen grid away)."""
    h, w = gray.shape
    s = min(512, h, w)
    peaks = []
    for cy, cx in [(h // 2, w // 2), (h // 3, w // 3), (2 * h // 3, 2 * w // 3)]:
        y0, x0 = max(0, cy - s // 2), max(0, cx - s // 2)
        c = gray[y0:y0 + s, x0:x0 + s].astype(np.float32)
        if c.shape != (s, s):
            continue
        c = (c - c.mean()) * np.outer(np.hanning(s), np.hanning(s))
        L = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(c))))
        yy, xx = np.indices(L.shape)
        r = np.hypot(yy - s / 2, xx - s / 2) / s
        m = (r > 0.15) & (r < 0.49) & (np.abs(yy - s / 2) > 3) & (np.abs(xx - s / 2) > 3)
        v = L[m]
        peaks.append(float((np.percentile(v, 99.99) - np.median(v)) / (v.std() + 1e-6)))
    return max(peaks) if peaks else 0.0


def recapture_check(img_bgr, meta) -> dict:
    """Screenshot cues (strong: status bar, exact screen resolution; weak: PNG, no camera
    EXIF) + moiré peaks from photographing a screen."""
    h, w = img_bgr.shape[:2]
    gray = _util.to_gray(img_bgr)
    strong, weak = [], []
    if (w, h) in SCREEN_SIZES or (h, w) in SCREEN_SIZES:
        strong.append(f"resolution {w}×{h} matches a phone screen exactly")
    band = gray[: max(8, int(h * 0.045))]
    med = np.median(band)
    uniform = float((np.abs(band.astype(np.int16) - med) < 14).mean())
    contrast = float((np.abs(band.astype(np.int16) - med) > 80).mean())
    if uniform > 0.85 and 0.003 < contrast < 0.15 and h > w * 1.6:
        strong.append("a status bar with text along the top edge")
    if (meta.get("format") or "").upper() == "PNG":
        weak.append("saved as PNG (cameras write JPEG/HEIC)")
    if not (meta.get("make") or meta.get("model")):
        weak.append("no camera make/model in the metadata")
    peak = moire_peak(gray)
    score, kind = 0.05, None
    if strong and (weak or len(strong) > 1):
        score, kind = 0.95, "screenshot"
    elif strong:
        score, kind = 0.7, "screenshot"
    elif weak:
        score = 0.15 * len(weak)
    if peak > MOIRE_T:
        s2 = float(min(0.97, 0.6 + 0.08 * (peak - MOIRE_T)))
        if s2 > score:
            score, kind = s2, "re-photographed screen"
        strong.append(f"periodic moiré peaks in the frequency spectrum (strength {peak:.1f})")
    return {"score": float(score), "cues": strong + weak, "moire_peak": round(peak, 2), "kind": kind}


MOIRE_T = 6.0


# ----------------------------------------------------------------- resampling / upscale
UPSCALE_T = 1.95


def upscale_check(img_bgr) -> dict:
    """Upscaled images lack energy near Nyquist relative to their pixel count."""
    g = _util.to_gray(img_bgr).astype(np.float32)
    s = max(g.shape)
    if s > 1600:
        f = 1600 / s
        g = cv2.resize(g, (int(g.shape[1] * f), int(g.shape[0] * f)), interpolation=cv2.INTER_AREA)
    g = g - g.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    cy, cx = np.array(P.shape) // 2
    yy, xx = np.indices(P.shape)
    r = np.sqrt(((yy - cy) / P.shape[0]) ** 2 + ((xx - cx) / P.shape[1]) ** 2)
    hi = P[(r > 0.30) & (r < 0.5)].mean()
    mid = P[(r > 0.08) & (r < 0.2)].mean()
    ratio = float(np.log10(hi / (mid + 1e-9) + 1e-12))
    score = float(_util.squash(-ratio, UPSCALE_T, 6.0))
    return {"score": score, "hf_log_ratio": round(ratio, 2)}


# ----------------------------------------------------------------- receipts
MONEY = re.compile(r"\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2}))")
DATE = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")


def parse_receipt_text(text: str) -> dict:
    """Pull total / date / item out of OCR'd receipt text."""
    out = {"total": None, "date": None, "item": None}
    text = re.sub(r"</tr>|<br\s*/?>", "\n", text or "")
    text = re.sub(r"</td>\s*<td>", "  ", text)
    text = re.sub(r"<[^>]+>", "", text)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for l in lines:
        if re.search(r"\btotal\b", l, re.I) and not re.search(r"sub", l, re.I):
            m = MONEY.findall(l)
            if m:
                out["total"] = float(m[-1].replace(",", ""))
    if out["total"] is None:
        amounts = [float(x.replace(",", "")) for x in MONEY.findall(text or "")]
        out["total"] = max(amounts) if amounts else None
    m = DATE.search(text or "")
    if m:
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y = y + 2000 if y < 100 else y
        try:
            out["date"] = dt.date(y, mo, d).isoformat()
        except ValueError:
            pass
    for i, l in enumerate(lines):
        if re.search(r"\bSN\b|serial", l, re.I) and i > 0:
            out["item"] = lines[i - 1]
            break
    return out


BRANDS = {"apple": ["apple", "iphone", "ipad", "macbook"], "samsung": ["samsung", "galaxy"],
          "google": ["google", "pixel"], "huawei": ["huawei"], "oneplus": ["oneplus"],
          "nokia": ["nokia"], "microsoft": ["microsoft", "lumia"], "motorola": ["motorola", "moto"],
          "xiaomi": ["xiaomi", "redmi"], "lg": [" lg "], "sony": ["sony", "xperia"]}


def brand_of(text: str):
    t = f" {(text or '').lower()} "
    for b, keys in BRANDS.items():
        if any(k in t for k in keys):
            return b
    return None


def item_matches(receipt_item, claimed):
    if not receipt_item:
        return None
    rb, cb = brand_of(receipt_item), brand_of(claimed)
    if rb and cb:
        return rb == cb and (_model_tokens(receipt_item) & _model_tokens(claimed) or not _model_tokens(claimed))
    a = set(re.findall(r"[a-z0-9]+", receipt_item.lower()))
    b = set(re.findall(r"[a-z0-9]+", claimed.lower()))
    return len(a & b) / max(1, len(b)) >= 0.5


def _model_tokens(s):
    return {t for t in re.findall(r"[a-z]*\d+[a-z]*", (s or "").lower())}


# ----------------------------------------------------------------- overlays (UI)
def ela_overlay(img_bgr, quality=90):
    ela = _ela_map(img_bgr, quality)
    blocks = _block_reduce(ela, 8, np.mean)
    bg = np.median(blocks); mad = np.median(np.abs(blocks - bg)) + 1e-6
    z = np.clip((blocks - bg) / (1.4826 * mad), 0, 12) / 12.0
    return z


def noise_overlay(img_bgr):
    gray = _util.to_gray(img_bgr).astype(np.float32)
    resid = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    e = _block_reduce(resid, 16, np.std)
    return _util.normalize01(e)


def ghost_overlay(img_bgr, qualities=range(55, 100, 5)):
    g0 = _util.to_gray(img_bgr).astype(np.float32)
    stack = [_block_reduce((g0 - _util.to_gray(_util.jpeg_resave(img_bgr, q)).astype(np.float32)) ** 2, 16, np.mean)
             for q in qualities]
    arg = np.argmin(np.stack(stack), axis=0)
    vals, counts = np.unique(arg, return_counts=True)
    return (arg != vals[np.argmax(counts)]).astype(np.float32)

"""Local smoke test for the CPU detectors (VLM slot removed -- needs Databricks).

Fabricates a textured "real" image, an over-smoothed "AI-like" image, a
copy-move forgery, and a splice, then checks the pipeline runs end to end and
the relevant signals move in the right direction.
"""
import cv2
import numpy as np

from image_authenticity import get_config
from image_authenticity.core import available
from image_authenticity.pipeline import Pipeline

rng = np.random.default_rng(0)


def real_like(h=256, w=256):
    # Structured content + fine sensor-like noise across all frequencies.
    base = np.zeros((h, w, 3), np.float32)
    for _ in range(40):
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        p1 = tuple(int(x) for x in rng.integers(0, min(h, w), 2))
        p2 = tuple(int(x) for x in rng.integers(0, min(h, w), 2))
        cv2.line(base, p1, p2, c, int(rng.integers(1, 4)))
    base += rng.normal(0, 12, base.shape).astype(np.float32)
    return np.clip(base, 0, 255).astype(np.uint8)


def ai_like(img):
    # Over-smooth: kills high-frequency detail -> steep spectral roll-off,
    # uniform ELA. Mimics the "too clean" diffusion signature.
    sm = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(sm, 0.95, img, 0.05, 0)


def copy_move(img):
    out = img.copy()
    patch = out[40:110, 40:110].copy()
    out[150:220, 150:220] = patch  # clone a block to a distant location
    return out


def splice(img):
    out = img.copy()
    # Paste a differently-noised, differently-compressed block.
    block = rng.normal(128, 40, (60, 60, 3)).astype(np.uint8)
    block = cv2.GaussianBlur(block, (3, 3), 0)
    out[170:230, 30:90] = block
    return out


def to_jpg(img, q=92):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()


print("registered detectors:", available())

# Drop the VLM slot for local testing (no Databricks auth here).
cfg = get_config()
cfg["slots"].pop("learned")
pipe = Pipeline(cfg)
print("active slots:", pipe.describe())

base = real_like()
cases = {
    "real": to_jpg(base),
    "ai_like": to_jpg(ai_like(base)),
    "copy_move": to_jpg(copy_move(base)),
    "splice": to_jpg(splice(base)),
}

for name, b in cases.items():
    out = pipe.run(b, image_id=name, heatmap_dir="/tmp/iaheat")
    c = out["composite"]
    per = {r.slot: (None if r.score is None else round(r.score, 2)) for r in out["results"]}
    print(f"\n== {name} ==")
    print("  fraud_score:", c["fraud_score"], "verdict:", c["verdict"])
    print("  per-slot:", per)
    print("  narrative:", c["narrative"])
    errs = [(r.slot, r.error) for r in out["results"] if r.error]
    if errs:
        print("  ERRORS:", errs)

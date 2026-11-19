"""Shared helpers for the ML-vs-pixel experiment on gold set v0 (66 claims).

Run directly to build the 'social' variant (messaging-app style forward: longest side <= 1024 px,
JPEG q70, no EXIF/ICC/IPTC) and print per-class image statistics, which expose confounds such as
resolution / format differences between genuine and AI images.

    python -I build/ml_exp_common.py <repo_dir>
"""
import io
import json
import os
import sys

import numpy as np
from PIL import Image

REPO = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else os.path.expanduser(
    "~/asurion-image-authenticity")
WB = os.path.join(REPO, "wb_data")
DATASET = os.path.join(WB, "dataset")
EXP = os.path.join(WB, "ml_exp")
SOCIAL_DIR = os.path.join(EXP, "social", "images")
sys.path.insert(0, REPO)

VARIANTS = ("orig", "social")


def load_claims():
    return json.load(open(os.path.join(DATASET, "claims.json")))


def orig_path(claim):
    return os.path.join(DATASET, "images", os.path.basename(claim["image_path"]))


def variant_path(claim, variant):
    if variant == "orig":
        return orig_path(claim)
    return os.path.join(SOCIAL_DIR, claim["claim_id"] + ".jpg")


def make_social(src, dst, max_side=1024, quality=70):
    im = Image.open(src).convert("RGB")
    if max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((round(im.width * f), round(im.height * f)), Image.LANCZOS)
    im.save(dst, "JPEG", quality=quality)          # no exif= argument -> metadata stripped


def build_variants():
    os.makedirs(SOCIAL_DIR, exist_ok=True)
    for c in load_claims():
        make_social(orig_path(c), variant_path(c, "social"))


def provenance_state(claim):
    """AI claims only: 'intact' (generator output kept, PNG) vs 'laundered' (re-encoded JPEG + fake EXIF)."""
    if claim["fraud_class"] not in ("ai_generated", "ai_edited_damage"):
        return None
    return "intact" if orig_path(claim).lower().endswith(".png") else "laundered"


def group_of(claim):
    fc, fam = claim["fraud_class"], claim["family"]
    if fam in ("genuine", "document"):
        return "negative"
    return fc


if __name__ == "__main__":
    build_variants()
    from collections import defaultdict
    rows = defaultdict(list)
    for c in load_claims():
        p = orig_path(c)
        im = Image.open(p)
        rows[c["fraud_class"] if c["family"] != "document" else "genuine"].append(
            (im.width, im.height, im.format, os.path.getsize(p) / 1e3))
    print(f"{'class':20s} {'n':>3s} {'median WxH':>13s} {'MP':>5s} {'PNG%':>5s} {'KB':>6s} | after social: median WxH")
    for k, v in sorted(rows.items()):
        w, h = np.median([x[0] for x in v]), np.median([x[1] for x in v])
        sw = [Image.open(variant_path(c, "social")).size for c in load_claims()
              if (c["fraud_class"] if c["family"] != "document" else "genuine") == k]
        print(f"{k:20s} {len(v):3d} {w:6.0f}x{h:<6.0f} {w * h / 1e6:5.2f} {100 * np.mean([x[2] == 'PNG' for x in v]):5.0f} "
              f"{np.median([x[3] for x in v]):6.0f} | {np.median([s[0] for s in sw]):.0f}x{np.median([s[1] for s in sw]):.0f}")
    print("\nAI claims by provenance state:",
          {s: sum(1 for c in load_claims() if provenance_state(c) == s) for s in ("intact", "laundered")})

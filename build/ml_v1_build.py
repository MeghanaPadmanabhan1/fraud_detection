"""Build held-out test set v1 for the ML-vs-pixel experiment.

Positives (AI-made): 30 Gemini-family generations (3 models x 10 scenes) + 22 Commons images from other generators.
Negatives (real):    real phone-camera originals from Commons 'Taken with <phone>' + 3 damaged-device Commons photos.
None of these images was used to choose any threshold (thresholds are frozen from gold set v0).

Three views of every image, so format / size / aspect cues can be separated from pixel evidence:
  orig   as downloaded / generated
  social messaging-app forward: longest side <= 1024, JPEG q70, no metadata
  norm   centre square crop, 768 px, JPEG q75, no metadata (removes aspect ratio, resolution and container cues)

Also builds a reuse test: 20 'claim' photos are indexed; each is altered 4 ways (crop, screenshot-in-viewer, heavy
recompress, rotate+relight) and queried; 10 held-out photos get the same alterations and must NOT match.

    python -I build/ml_v1_build.py
"""
import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1 = os.path.join(ROOT, "wb_data", "dataset_v1")
OUT = os.path.join(ROOT, "wb_data", "ml_v1")
shutil.rmtree(OUT, ignore_errors=True)
for v in ("orig", "social", "norm", "reuse/queries", "reuse/index"):
    os.makedirs(os.path.join(OUT, v), exist_ok=True)

items = []
short = {"databricks-gemini-3-pro-image": "gemini_3_pro", "databricks-gemini-3-1-flash-image": "gemini_3_1_flash",
         "databricks-gemini-3-1-flash-lite-image": "gemini_3_1_flash_lite"}
for r in json.load(open(os.path.join(V1, "raw_ai", "index.json"))):
    if "file" in r:
        items.append(dict(id=f"ai_{short[r['model']]}_{r['i']}", src=os.path.join(V1, "raw_ai", r["file"]), label=1,
                          group=short[r["model"]], origin="generated here"))
for r in json.load(open(os.path.join(V1, "web_ai", "index.json"))):
    items.append(dict(id="ai_web_" + r["file"].rsplit(".", 1)[0].replace("web_", ""), src=os.path.join(V1, "web_ai", r["file"]),
                      label=1, group="web:" + r["generator"], origin=r["page"], licence=r["licence"]))
for r in json.load(open(os.path.join(V1, "real_phone", "index.json"))):
    items.append(dict(id="real_" + r["file"].rsplit(".", 1)[0].replace("real_", ""), src=os.path.join(V1, "real_phone", r["file"]),
                      label=0, group="real_phone", origin=r["page"], licence=r["licence"]))
for j, r in enumerate(json.load(open(os.path.join(ROOT, "wb_data", "pool_raw3", "pool.json")))):
    items.append(dict(id=f"real_dmg_{j}", src=os.path.join(ROOT, "wb_data", "pool_raw3", r["file"]), label=0,
                      group="real_damaged", origin=r.get("page"), licence=r.get("licence")))


def save_jpeg(im, path, q):
    im.save(path, "JPEG", quality=q)             # no exif= argument -> metadata stripped


for it in items:
    ext = os.path.splitext(it["src"])[1].lower()
    dst = os.path.join(OUT, "orig", it["id"] + ext)
    shutil.copy(it["src"], dst)
    it["orig"] = dst
    im = Image.open(it["src"]).convert("RGB")
    it["w"], it["h"], it["bytes"], it["fmt"] = im.width, im.height, os.path.getsize(it["src"]), ext.strip(".")
    s = im.copy()
    if max(s.size) > 1024:
        f = 1024 / max(s.size)
        s = s.resize((round(s.width * f), round(s.height * f)), Image.LANCZOS)
    it["social"] = os.path.join(OUT, "social", it["id"] + ".jpg"); save_jpeg(s, it["social"], 70)
    side = min(im.size)
    l, t = (im.width - side) // 2, (im.height - side) // 2
    n = im.crop((l, t, l + side, t + side)).resize((768, 768), Image.LANCZOS)
    it["norm"] = os.path.join(OUT, "norm", it["id"] + ".jpg"); save_jpeg(n, it["norm"], 75)
json.dump(items, open(os.path.join(OUT, "items.json"), "w"), indent=1)
print(len(items), "items;", sum(i["label"] for i in items), "AI,", sum(1 - i["label"] for i in items), "real")


# ------------------------------------------------------------------ reuse test
def load(path, max_side=1600):
    im = cv2.imread(path)
    f = max_side / max(im.shape[:2])
    return cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else im


def t_crop(im):
    h, w = im.shape[:2]
    return im[int(h * .125):int(h * .875), int(w * .125):int(w * .875)]


def t_screenshot(im):
    W, H = 1170, 2532
    s = W / im.shape[1]
    r = cv2.resize(im, (W, round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full((H, W, 3), 28, np.uint8)
    y = (H - r.shape[0]) // 2
    canvas[y:y + r.shape[0]] = r
    cv2.rectangle(canvas, (0, 0), (W, 110), (40, 40, 40), -1)               # status bar
    cv2.rectangle(canvas, (0, H - 150), (W, H), (40, 40, 40), -1)           # toolbar
    return canvas


def t_recompress(im):
    return cv2.resize(im, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)


def t_relight(im):
    h, w = im.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 4, 1.08)
    r = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return cv2.convertScaleAbs(r, alpha=1.15, beta=14)


TR = {"crop": (t_crop, 90), "screenshot": (t_screenshot, 85), "recompress": (t_recompress, 40), "relight": (t_relight, 80)}
reals = [i for i in items if i["label"] == 0]
rng = np.random.default_rng(3)
order = rng.permutation(len(reals))
S = [reals[k] for k in order[:20]]
H = [reals[k] for k in order[20:30]]
D = [reals[k] for k in order[30:]]
reuse = {"index": [], "queries": []}
for it in S + D:                       # indexed originals (claims already on file)
    p = os.path.join(OUT, "reuse", "index", it["id"] + ".jpg")
    cv2.imwrite(p, load(it["orig"]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    reuse["index"].append(dict(id=it["id"], path=p, kind="source" if it in S else "distractor"))
gold = json.load(open(os.path.join(ROOT, "wb_data", "dataset", "claims.json")))
for c in gold:                         # the 66 gold-set photos are extra distractors
    reuse["index"].append(dict(id="gold_" + c["claim_id"], path=os.path.join(ROOT, "wb_data", "dataset", "images",
                                                                              os.path.basename(c["image_path"])), kind="gold"))
for kind, group in (("positive", S), ("negative", H)):
    for it in group:
        base = load(it["orig"])
        for tname, (fn, q) in TR.items():
            p = os.path.join(OUT, "reuse", "queries", f"{it['id']}__{tname}.jpg")
            cv2.imwrite(p, fn(base), [cv2.IMWRITE_JPEG_QUALITY, q])
            reuse["queries"].append(dict(id=f"{it['id']}__{tname}", path=p, kind=kind, transform=tname, source=it["id"]))
json.dump(reuse, open(os.path.join(OUT, "reuse", "reuse.json"), "w"), indent=1)
print("reuse test:", len(reuse["index"]), "indexed,", sum(q["kind"] == "positive" for q in reuse["queries"]), "positive queries,",
      sum(q["kind"] == "negative" for q in reuse["queries"]), "negative queries")

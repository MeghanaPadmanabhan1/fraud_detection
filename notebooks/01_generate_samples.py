# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Generate a labeled sample set
# MAGIC
# MAGIC Builds a small, **labeled** image set in a UC Volume so we can exercise and
# MAGIC calibrate the detectors before insurance_firm's real suspect images arrive.
# MAGIC
# MAGIC | label | source | how |
# MAGIC |---|---|---|
# MAGIC | `authentic` | real camera photos | `/databricks-datasets/flower_photos` |
# MAGIC | `shallow_fake` | edited real photos | programmatic copy-move / splice / local recompress |
# MAGIC | `deepfake` | fully synthetic | generated via the **Gemini image** FMAPI endpoint (carries real SynthID + IPTC AI-provenance) |
# MAGIC
# MAGIC Output: images under `/Volumes/image_authenticity/raw/images/<label>/` and a
# MAGIC manifest Delta table `image_authenticity.raw.sample_manifest`.

# COMMAND ----------
# MAGIC %pip install -q --no-deps opencv-python-headless pillow
# MAGIC # --no-deps: do NOT let pip upgrade numpy (breaks the runtime pandas ABI)
# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import io, os, json, base64, datetime, random
import numpy as np
import cv2

CATALOG = "image_authenticity"
VOL = f"/Volumes/{CATALOG}/raw/images"
N_AUTHENTIC = 18          # real photos to sample
N_DEEPFAKE = 10           # images to generate via Gemini
IMAGE_ENDPOINT = "databricks-gemini-3-pro-image"
random.seed(7)

for sub in ("authentic", "shallow_fake", "deepfake"):
    os.makedirs(f"{VOL}/{sub}", exist_ok=True)

rows = []   # manifest rows

def _now():
    return datetime.datetime.utcnow().isoformat()

# COMMAND ----------
# MAGIC %md ## 1. Authentic — sample real photos from flower_photos

# COMMAND ----------
classes = ["roses", "tulips", "daisy", "sunflowers", "dandelion"]
src = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.jpg")
    .load("dbfs:/databricks-datasets/flower_photos/{%s}" % ",".join(classes))
    .limit(N_AUTHENTIC)
    .collect()
)
authentic_bytes = []
for i, r in enumerate(src):
    img_id = f"auth_{i:03d}"
    out = f"{VOL}/authentic/{img_id}.jpg"
    with open(out, "wb") as f:
        f.write(r.content)
    authentic_bytes.append((img_id, r.content))
    rows.append({"image_id": img_id, "path": out, "label": "authentic",
                 "source": "flower_photos", "prompt": None, "created_at": _now()})
print(f"wrote {len(authentic_bytes)} authentic images")

# COMMAND ----------
# MAGIC %md ## 2. Shallow fakes — edit real photos programmatically
# MAGIC Three documented fraud tactics: clone a region (copy-move), paste a foreign
# MAGIC block (splice), and locally re-save at a different quality (prior-edit ghost).

# COMMAND ----------
def _decode(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)

def _jpg(img, q=92):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes()

def edit_copy_move(img):
    out = img.copy(); h, w = out.shape[:2]
    s = min(h, w) // 5
    y0, x0 = h // 4, w // 4
    patch = out[y0:y0+s, x0:x0+s].copy()
    y1, x1 = h - y0 - s, w - x0 - s
    out[y1:y1+s, x1:x1+s] = patch
    return out, "copy_move"

def edit_splice(img, donor):
    out = img.copy(); h, w = out.shape[:2]
    s = min(h, w) // 4
    dh, dw = donor.shape[:2]
    block = cv2.resize(donor[: dh // 2, : dw // 2], (s, s))
    y0, x0 = h // 3, w // 3
    out[y0:y0+s, x0:x0+s] = block
    return out, "splice"

def edit_local_recompress(img):
    # Resave a sub-region at low quality, paste back -> JPEG-ghost / ELA signal.
    out = img.copy(); h, w = out.shape[:2]
    s = min(h, w) // 3
    y0, x0 = h // 2 - s // 2, w // 2 - s // 2
    region = out[y0:y0+s, x0:x0+s]
    region = _decode(_jpg(region, q=45))
    out[y0:y0+s, x0:x0+s] = region
    return out, "local_recompress"

k = 0
for i, (aid, b) in enumerate(authentic_bytes):
    img = _decode(b)
    donor = _decode(authentic_bytes[(i + 1) % len(authentic_bytes)][1])
    for editor, args in [(edit_copy_move, (img,)), (edit_splice, (img, donor)),
                         (edit_local_recompress, (img,))]:
        if k >= 15:
            break
        edited, tactic = editor(*args)
        img_id = f"shallow_{k:03d}_{tactic}"
        out = f"{VOL}/shallow_fake/{img_id}.jpg"
        with open(out, "wb") as f:
            f.write(_jpg(edited, q=90))
        rows.append({"image_id": img_id, "path": out, "label": "shallow_fake",
                     "source": f"edited:{aid}", "prompt": tactic, "created_at": _now()})
        k += 1
print(f"wrote {k} shallow-fake images")

# COMMAND ----------
# MAGIC %md ## 3. Deepfakes — generate via the Gemini image FMAPI endpoint

# COMMAND ----------
import sys
PKG_ROOT = "/Workspace/Users/<workspace-user>/insurance_firm-image-authenticity"
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)
from image_authenticity.fmapi import fmapi_chat, first_message_content

PROMPTS = [
    "a photorealistic iPhone 15 Pro with a shattered screen on a kitchen counter",
    "a close-up photo of a MacBook Pro with liquid damage and a cracked display",
    "a realistic photo of a Samsung Galaxy phone with a deep crack across the screen",
    "a smartphone that fell in water, screen flickering, on a bathroom floor",
    "a cracked tablet screen with spiderweb fractures, natural lighting",
    "a dented laptop with a broken hinge on an office desk",
    "a phone with a scorched, burnt battery swelling the case",
    "a smartwatch with a cracked face on someone's wrist",
    "a television with a large impact crack in the corner, living room",
    "a pair of wireless earbuds crushed and broken on pavement",
]

def generate_image(prompt):
    resp = fmapi_chat(
        IMAGE_ENDPOINT,
        messages=[{"role": "user",
                   "content": f"Generate a single {prompt}. Photorealistic, as if taken "
                              f"on a phone camera for an insurance claim."}],
        max_tokens=4096,
    )
    content = first_message_content(resp)
    # content is a JSON string: [{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]
    items = json.loads(content) if isinstance(content, str) else content
    for it in items:
        if isinstance(it, dict) and it.get("type") == "image_url":
            uri = it["image_url"]["url"]
            return base64.b64decode(uri.split(",", 1)[1])
    raise ValueError("no image in response")

made = 0
for i, p in enumerate(PROMPTS[:N_DEEPFAKE]):
    try:
        png = generate_image(p)
        img_id = f"deepfake_{i:03d}"
        out = f"{VOL}/deepfake/{img_id}.png"
        with open(out, "wb") as f:
            f.write(png)
        rows.append({"image_id": img_id, "path": out, "label": "deepfake",
                     "source": IMAGE_ENDPOINT, "prompt": p, "created_at": _now()})
        made += 1
        print(f"  generated {img_id} ({len(png)} bytes)")
    except Exception as e:
        print(f"  !! failed on prompt {i}: {e}")
print(f"wrote {made} deepfake images")

# COMMAND ----------
# MAGIC %md ## 4. Write the manifest table

# COMMAND ----------
import pandas as pd
mdf = spark.createDataFrame(pd.DataFrame(rows))
(mdf.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.raw.sample_manifest"))
display(spark.table(f"{CATALOG}.raw.sample_manifest").groupBy("label").count())

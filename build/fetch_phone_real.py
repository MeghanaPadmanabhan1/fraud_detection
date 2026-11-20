"""Fetch REAL phone-camera photos (original JPEG with EXIF) from Commons 'Taken with <phone>' categories, as extra
genuine negatives. Drops photos with a detected face (PII rule) and files over 9 MB.

    SSL_CERT_FILE=<certifi> ml_env/bin/python -I build/fetch_phone_real.py <out_dir>
"""
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "insurance_firmFraudWorkbenchPOC/1.0 (Databricks field-eng demo)"}
CATS = ["Taken with iPhone 15", "Taken with iPhone 16", "Taken with iPhone 13", "Taken with iPhone 17",
        "Taken with Samsung Galaxy A52 5G", "Taken with Samsung Galaxy Note 20", "Taken with Samsung Galaxy A23 5G",
        "Taken with Samsung Galaxy Note 10", "Taken with iPhone Air", "Taken with Samsung Galaxy A03s"]
PER, TRY = 3, 15
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
random.seed(9)


def get(url, binary=False):
    for a in range(8):
        try:
            d = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read()
            return d if binary else json.loads(d)
        except Exception as e:
            if "429" in str(e) or "timed out" in str(e):
                time.sleep(20 * (a + 1)); continue
            raise
    raise RuntimeError("rate limited")


def api(**p):
    p["format"] = "json"
    time.sleep(1.5)
    return get("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(p))


kept = json.load(open(os.path.join(OUT, "index.json"))) if os.path.exists(os.path.join(OUT, "index.json")) else []
done = {k["category"] for k in kept}                 # resume: categories already fetched are skipped
for cat in CATS:
    if cat in done:
        continue
    try:
        ts = [m["title"] for m in api(action="query", list="categorymembers", cmtitle="Category:" + cat,
                                      cmlimit=200, cmtype="file")["query"]["categorymembers"]]
    except Exception as e:
        print("category failed", cat, e); continue
    random.shuffle(ts)
    got = 0
    for i in range(0, min(len(ts), TRY), 10):
        d = api(action="query", titles="|".join(ts[i:i + 10]), prop="imageinfo", iiprop="url|mime|size|extmetadata", iiurlwidth=1600)
        for p in d["query"]["pages"].values():
            ii = (p.get("imageinfo") or [{}])[0]
            if got >= PER or ii.get("mime") != "image/jpeg" or min(ii.get("width", 0), ii.get("height", 0)) < 1200 \
                    or ii.get("size", 0) > 9_000_000:
                continue
            try:
                raw = get(ii.get("thumburl") or ii["url"], True)       # 1600 px rendition: far lighter on the rate limit
            except Exception as e:
                print("  skip", p["title"], str(e)[:60], flush=True)
                continue
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            g = cv2.cvtColor(cv2.resize(img, (800, int(800 * img.shape[0] / img.shape[1]))), cv2.COLOR_BGR2GRAY)
            if len(FACE.detectMultiScale(g, 1.1, 5, minSize=(40, 40))):
                continue
            fn = f"real_{len(kept):03d}.jpg"
            open(os.path.join(OUT, fn), "wb").write(raw)
            m = ii.get("extmetadata", {})
            kept.append(dict(file=fn, title=p["title"], page=ii.get("descriptionurl"), category=cat,
                             w=ii["width"], h=ii["height"], bytes=ii["size"],
                             licence=re.sub(r"<[^>]+>", "", m.get("LicenseShortName", {}).get("value", ""))[:60],
                             artist=re.sub(r"<[^>]+>", "", m.get("Artist", {}).get("value", ""))[:80]))
            got += 1
            time.sleep(1.5)
    print(cat, "kept", got, flush=True)
    json.dump(kept, open(os.path.join(OUT, "index.json"), "w"), indent=1)      # progress survives a crash
json.dump(kept, open(os.path.join(OUT, "index.json"), "w"), indent=1)
print(len(kept), "real phone photos kept")

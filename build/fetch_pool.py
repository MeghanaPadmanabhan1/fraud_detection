"""Fetch a pool of GENUINE damaged-device photos from Wikimedia Commons.

Usage: python -I build/fetch_pool.py <out_dir>

Downloads camera originals (thumbnails strip EXIF), records licence + artist +
source URL per image, and drops any photo with a detected face (PII rule from
the dataset spec). Writes <out_dir>/pool.json.
"""
import json, os, re, sys, time, urllib.parse, urllib.request

import cv2
import numpy as np

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
UA = {"User-Agent": "insurance_firmFraudWorkbenchPOC/1.0 (Databricks field-eng demo)"}

QUERIES = {
    "phone": ["cracked smartphone screen", "broken phone screen", "shattered phone screen",
              "broken iPhone", "cracked iPhone", "broken Samsung Galaxy", "smashed mobile phone",
              "broken smartphone display", "cracked phone back glass", "damaged mobile phone"],
    "tablet": ["broken tablet screen", "cracked iPad"],
    "laptop": ["broken laptop screen", "damaged laptop", "cracked laptop display"],
    "wearable": ["broken smartwatch", "cracked watch display"],
    "console": ["broken game console", "damaged Nintendo"],
}
# Title keyword -> device label used as the claimed device.
DEVICE_HINTS = [
    (r"iphone\s*(\d+\s*(pro|max|plus|mini)?|x[sr]?|se)?", "Apple iPhone"),
    (r"ipad", "Apple iPad"), (r"macbook", "Apple MacBook"), (r"apple watch", "Apple Watch"),
    (r"galaxy\s*[a-z]?\d*", "Samsung Galaxy"), (r"samsung", "Samsung"),
    (r"huawei\s*[a-z]*\d*", "Huawei"), (r"oneplus", "OnePlus"), (r"xiaomi|redmi", "Xiaomi"),
    (r"pixel", "Google Pixel"), (r"nokia", "Nokia"), (r"motorola|moto\s*g", "Motorola"),
    (r"lg\b", "LG"), (r"sony|xperia", "Sony Xperia"), (r"thinkpad|lenovo", "Lenovo ThinkPad"),
    (r"nintendo|switch", "Nintendo Switch"), (r"playstation|ps[345]", "Sony PlayStation"),
    (r"xbox", "Microsoft Xbox"),
]
GENERIC = {"phone": "Android smartphone", "tablet": "Android tablet", "laptop": "Laptop",
           "wearable": "Smartwatch", "console": "Game console"}
DEVICE_RE = r"phone|iphone|smartphone|galaxy|mobile|handy|ipad|tablet|laptop|macbook|notebook|watch|console|nintendo|playstation|xbox|screen|display|lcd"
DAMAGE_RE = r"broken|crack|shatter|smash|damag|dead|bent|swollen|water|defect|kaputt|gebrochen|destroy|split"
FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def api(params):
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))


def device_label(title, category):
    t = title.lower()
    for pat, label in DEVICE_HINTS:
        m = re.search(pat, t)
        if m:
            tail = m.group(0).strip()
            if label.startswith(("Apple iPhone", "Samsung Galaxy", "Huawei")) and len(tail.split()) > 1:
                return label.split()[0] + " " + tail.title().replace("Iphone", "iPhone")
            return label
    return GENERIC[category]


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()[:120]


# Optional: extra queries + existing pool to dedupe against (augment mode).
if len(sys.argv) > 2:
    QUERIES = {"phone": [], "tablet": [], "laptop": [], "wearable": [], "console": []}
    for q in sys.argv[2].split(";"):
        cat, query = q.split(":", 1)
        QUERIES[cat].append(query)
EXISTING = set()
if len(sys.argv) > 3:
    EXISTING = {x["title"] for x in json.load(open(sys.argv[3]))}
seen, pool = set(EXISTING), []
for category, queries in QUERIES.items():
    for q in queries:
        try:
            d = api({"action": "query", "format": "json", "generator": "search",
                     "gsrsearch": q, "gsrnamespace": 6, "gsrlimit": 15, "prop": "imageinfo",
                     "iiprop": "url|mime|size|extmetadata", "iiurlwidth": 1280})
        except Exception as e:
            print("query failed", q, e); continue
        for p in (d.get("query") or {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [{}])[0]
            title = p.get("title", "")
            if ii.get("mime") != "image/jpeg" or title in seen:
                continue
            tl = title.lower()
            if not (re.search(DEVICE_RE, tl) and re.search(DAMAGE_RE, tl)):
                continue
            if min(ii.get("width", 0), ii.get("height", 0)) < 700 or ii.get("size", 0) > 15_000_000:
                continue
            seen.add(title)
            meta = ii.get("extmetadata", {})
            pool.append({
                "title": title, "url": ii.get("thumburl") or ii["url"], "page": ii.get("descriptionurl"),
                "category": category, "device": device_label(title, category),
                "licence": strip_html(meta.get("LicenseShortName", {}).get("value")),
                "artist": strip_html(meta.get("Artist", {}).get("value")),
            })
        time.sleep(0.3)
print(f"{len(pool)} candidate originals")

kept = []
for i, item in enumerate(pool):
    raw = None
    for attempt in range(4):
        try:
            raw = urllib.request.urlopen(urllib.request.Request(item["url"], headers=UA), timeout=60).read()
            break
        except Exception as e:
            if "429" in str(e):
                time.sleep(5 * (attempt + 1)); continue
            print("download failed", item["title"], e); break
    if raw is None:
        continue
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    small = cv2.resize(img, (800, int(img.shape[0] * 800 / img.shape[1])))
    faces = FACE.detectMultiScale(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1.1, 6, minSize=(60, 60))
    if len(faces):
        print("  skip (face detected):", item["title"]); continue
    fn = f"pool_{len(kept):03d}.jpg"
    open(os.path.join(OUT, fn), "wb").write(raw)
    kept.append({**item, "file": fn, "bytes": len(raw)})
    time.sleep(1.2)

json.dump(kept, open(os.path.join(OUT, "pool.json"), "w"), indent=1)
print(f"kept {len(kept)} photos")
from collections import Counter
print(Counter(k["category"] for k in kept), Counter(k["licence"] for k in kept).most_common(5))

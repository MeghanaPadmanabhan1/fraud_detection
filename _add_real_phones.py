"""Fetch real (camera, license-clean) phone photos from Wikimedia Commons,
apply the same programmatic shallow-fake edits, and stage them for upload.

Keeps the base genuinely real (no AI provenance) so 'authentic' stays authentic.
"""
import io, json, os, urllib.parse, urllib.request, datetime
import cv2, numpy as np

OUT = os.path.expanduser("~/asurion-image-authenticity/_phones")
os.makedirs(f"{OUT}/authentic", exist_ok=True)
os.makedirs(f"{OUT}/shallow_fake", exist_ok=True)
UA = {"User-Agent": "AsurionPOC/1.0 (field-eng demo; contact meghana)"}

def commons_urls(query, n=8):
    q = urllib.parse.quote(query)
    url = (f"https://commons.wikimedia.org/w/api.php?action=query&format=json"
           f"&generator=search&gsrsearch={q}&gsrnamespace=6&gsrlimit={n}"
           f"&prop=imageinfo&iiprop=url|mime|size&iiurlwidth=900")
    req = urllib.request.Request(url, headers=UA)
    d = json.load(urllib.request.urlopen(req, timeout=25))
    out = []
    for p in (d.get("query", {}) or {}).get("pages", {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        if ii.get("mime") in ("image/jpeg", "image/png"):
            out.append((p.get("title", ""), ii.get("thumburl") or ii.get("url")))
    return out

# Collect unique real phone photos across a few damage-relevant queries.
seen, cands = set(), []
for q in ["cracked smartphone screen", "broken phone screen", "shattered phone screen",
          "smartphone broken display", "damaged mobile phone"]:
    for title, u in commons_urls(q, 8):
        key = u.split("/")[-1].split("?")[0]
        if key not in seen:
            seen.add(key); cands.append((title, u))

def fetch(u):
    req = urllib.request.Request(u, headers=UA)
    raw = urllib.request.urlopen(req, timeout=25).read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return img

imgs = []
for title, u in cands:
    try:
        im = fetch(u)
        if im is None:
            continue
        h, w = im.shape[:2]
        if min(h, w) < 200:
            continue
        if w > 900:
            im = cv2.resize(im, (900, int(h * 900 / w)))
        imgs.append(im)
    except Exception as e:
        print("skip", u, e)
    if len(imgs) >= 12:
        break
print(f"downloaded {len(imgs)} real phone photos")

def jpg(img, q=92):
    ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q]); return b.tobytes()

def edit_copy_move(img):
    out = img.copy(); h, w = out.shape[:2]; s = min(h, w) // 5
    out[h-s-h//4:h-h//4, w-s-w//4:w-w//4] = out[h//4:h//4+s, w//4:w//4+s].copy()
    return out, "copy_move"

def edit_splice(img, donor):
    out = img.copy(); h, w = out.shape[:2]; s = min(h, w) // 4
    dh, dw = donor.shape[:2]
    out[h//3:h//3+s, w//3:w//3+s] = cv2.resize(donor[:dh//2, :dw//2], (s, s))
    return out, "splice"

def edit_local_recompress(img):
    out = img.copy(); h, w = out.shape[:2]; s = min(h, w) // 3
    y, x = h//2 - s//2, w//2 - s//2
    out[y:y+s, x:x+s] = cv2.imdecode(np.frombuffer(jpg(out[y:y+s, x:x+s], 40), np.uint8), cv2.IMREAD_COLOR)
    return out, "local_recompress"

rows, ts = [], datetime.datetime.utcnow().isoformat()
# First 4 as real authentic device negatives; rest edited into shallow-fakes.
n_auth = min(4, len(imgs) // 3)
for i in range(n_auth):
    iid = f"phone_auth_{i:02d}"; path = f"/Volumes/image_authenticity/raw/images/authentic/{iid}.jpg"
    open(f"{OUT}/authentic/{iid}.jpg", "wb").write(jpg(imgs[i]))
    rows.append((iid, path, "authentic", "wikimedia_commons", None, ts))

editors = [edit_copy_move, edit_splice, edit_local_recompress]
k = 0
for i in range(n_auth, len(imgs)):
    ed = editors[k % 3]
    out, tactic = (ed(imgs[i], imgs[(i+1) % len(imgs)]) if ed is edit_splice else ed(imgs[i]))
    iid = f"phone_shallow_{k:02d}_{tactic}"; path = f"/Volumes/image_authenticity/raw/images/shallow_fake/{iid}.jpg"
    open(f"{OUT}/shallow_fake/{iid}.jpg", "wb").write(jpg(out, 90))
    rows.append((iid, path, "shallow_fake", "wikimedia_commons_edited", tactic, ts))
    k += 1

json.dump(rows, open(f"{OUT}/manifest_rows.json", "w"))
print(f"staged {n_auth} authentic + {k} shallow-fake phone images")
print("sample rows:", rows[:2])

"""Build the workbench gold set v0 from the genuine device-photo pool.

Usage: DATABRICKS_CONFIG_PROFILE=e2 python -I build/build_dataset.py <wb_data_dir> <repo_dir>

Follows the companion dataset spec: genuine base pool -> manipulated variants
per class, synthetic claim records, receipts (correct + mismatched), pixel masks
where pixels change, and licence/provenance per image. All claimants, dates and
EXIF timelines are SYNTHETIC; photos are CC/PD from Wikimedia Commons.
"""
import base64, datetime as dt, io, json, os, random, sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WB, REPO = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
random.seed(11); np.random.seed(11)

OUT = os.path.join(WB, "dataset")
for d in ("images", "receipts", "masks"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

TODAY = dt.date(2026, 10, 27)
VOL = "/Volumes/image_authenticity/workbench/claims_media"

# ---- genuine pool selection (distinct scenes; near-duplicate angles trimmed) ----
P1 = json.load(open(os.path.join(WB, "pool_raw/pool.json")))
P2 = json.load(open(os.path.join(WB, "pool_raw2/pool.json")))
keep1 = ["001", "002", "004", "007", "009", "010", "012", "014", "016", "017", "018", "019",
         "020", "021", "022", "023", "024", "025", "026", "027", "029", "030", "032"]
keep2 = ["001", "002", "006", "007"]
pool = [dict(p, path=os.path.join(WB, "pool_raw", p["file"])) for p in P1 if p["file"][5:8] in keep1]
pool += [dict(p, path=os.path.join(WB, "pool_raw2", p["file"])) for p in P2 if p["file"][5:8] in keep2]
for p in pool:  # tidy device labels
    t = p["title"].lower()
    if "lcd" in t: p["device"], p["category"] = "Laptop (LCD panel)", "laptop"
    if "lumia" in t: p["device"] = "Microsoft Lumia"
    if "clone" in t: p["device"] = "Apple iPhone (unbranded)"
    if "watch" in t: p["category"] = "wearable"
    p["device"] = p["device"].replace("Huawei Huawei", "Huawei")
random.shuffle(pool)
print(len(pool), "genuine sources")

G, H = pool[:19], pool[19:]          # G: claimed in our history; H: never claimed ("found online")

PRICE = {"Apple iPhone": 899, "Apple Watch": 399, "Samsung": 699, "Google Pixel": 499,
         "Huawei": 449, "OnePlus": 649, "Nokia": 59, "Microsoft Lumia": 199, "Laptop": 1249,
         "Android smartphone": 349}
CAPTURE = [("Apple", "iPhone 14"), ("Apple", "iPhone 13 Pro"), ("samsung", "SM-S911B"),
           ("Google", "Pixel 7"), ("Xiaomi", "2201117TG"), ("Apple", "iPhone 15 Pro Max")]
CHANNEL = ["mobile_app", "mobile_app", "web", "adjuster_portal"]
DAMAGE = {"phone": "Cracked screen after a drop", "laptop": "Display not working after impact",
          "wearable": "Cracked watch face", "tablet": "Shattered display"}


def price_for(dev):
    for k, v in PRICE.items():
        if dev.startswith(k):
            return v
    return 349


def load(path, max_side=1600):
    img = cv2.imread(path)
    s = max(img.shape[:2])
    if s > max_side:
        f = max_side / s
        img = cv2.resize(img, (int(img.shape[1] * f), int(img.shape[0] * f)), interpolation=cv2.INTER_AREA)
    return img


def jpeg_with_exif(img_bgr, out, make=None, model=None, when=None, software=None, q=92):
    """Save JPEG with optional synthetic EXIF (Make/Model/DateTimeOriginal/Software)."""
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    kw = {"quality": q}
    if make or software or when:
        ex = Image.Exif()
        if make: ex[271] = make
        if model: ex[272] = model
        if software: ex[305] = software
        if when:
            s = when.strftime("%Y:%m:%d %H:%M:%S")
            ex[306] = s
            ex.get_ifd(0x8769)[36867] = s
            ex.get_ifd(0x8769)[36868] = s
        kw["exif"] = ex.tobytes()
    pil.save(out, "JPEG", **kw)


def rand_time(d):
    return dt.datetime.combine(d, dt.time(random.randint(8, 21), random.randint(0, 59), random.randint(0, 59)))


claims, n = [], 0
cust = iter(random.sample(range(1000, 9999), 400))


def new_claim(**kw):
    global n
    n += 1
    cid = f"CLM-{1000 + n}"
    claim_date = kw.pop("claim_date")
    incident = claim_date - dt.timedelta(days=random.randint(0, 3))
    dev = kw.pop("device")
    rec = dict(claim_id=cid, claimant_id=f"CUST-{next(cust)}", claim_date=claim_date.isoformat(),
               incident_date=incident.isoformat(), device_claimed=dev,
               damage_described=kw.pop("damage", "Cracked screen after a drop"),
               channel=random.choice(CHANNEL), claim_value=float(price_for(dev)),
               purchase_date=(claim_date - dt.timedelta(days=random.randint(150, 800))).isoformat(),
               image_path=None, receipt_path=None, mask_path=None, receipt_truth=None)
    rec.update(kw)
    claims.append(rec)
    return rec


def mask_from_rect(shape, x, y, w, h, out):
    m = np.zeros(shape[:2], np.uint8); m[y:y + h, x:x + w] = 255
    cv2.imwrite(out, m)


# ---------------- genuine claims (+ document fraud) from G ----------------
g_claims = []
for i, p in enumerate(G):
    cd = dt.date(2026, 8, 25) + dt.timedelta(days=random.randint(0, 36))
    fam, cls = ("document", "receipt_mismatch") if i < 5 else ("genuine", "genuine")
    c = new_claim(claim_date=cd, device=p["device"], damage=DAMAGE.get(p["category"], DAMAGE["phone"]),
                  family=fam, fraud_class=cls, source_title=p["title"], source_page=p["page"],
                  licence=p["licence"], artist=p["artist"], source_id=p["file"])
    img = load(p["path"])
    mk, md = random.choice(CAPTURE)
    shot = rand_time(dt.date.fromisoformat(c["incident_date"]))
    fn = f"{c['claim_id']}.jpg"
    jpeg_with_exif(img, os.path.join(OUT, "images", fn), mk, md, shot)
    c["image_path"] = f"{VOL}/images/{fn}"
    c["notes"] = "Genuine camera photo; capture EXIF re-based to the synthetic claim timeline."
    c["_img"] = img; c["_capture"] = (mk, md)
    g_claims.append(c)

# ---------------- shallowfake: reuse of G photos by other claimants ----------------
reuse_specs = ["exact"] * 4 + ["recompressed"] * 3 + ["cropped"] * 3
for spec, orig in zip(reuse_specs, random.sample([c for c in g_claims if c["family"] == "genuine"], 10)):
    cd = dt.date.fromisoformat(orig["claim_date"]) + dt.timedelta(days=random.randint(12, 45))
    cd = min(cd, TODAY - dt.timedelta(days=1))
    c = new_claim(claim_date=cd, device=orig["device_claimed"], family="shallowfake",
                  fraud_class=f"reuse_{spec}", source_title=orig["source_title"],
                  source_page=orig["source_page"], licence=orig["licence"], artist=orig["artist"],
                  source_id=orig["source_id"], reuse_of=orig["claim_id"])
    fn = f"{c['claim_id']}.jpg"; out = os.path.join(OUT, "images", fn)
    src = os.path.join(OUT, "images", f"{orig['claim_id']}.jpg")
    if spec == "exact":
        open(out, "wb").write(open(src, "rb").read())
        c["notes"] = f"Byte-identical resubmission of {orig['claim_id']}'s photo by a different claimant."
    elif spec == "recompressed":
        im = orig["_img"]; im = cv2.resize(im, (int(im.shape[1] * .8), int(im.shape[0] * .8)))
        jpeg_with_exif(im, out, q=72)
        c["notes"] = f"{orig['claim_id']}'s photo resized + recompressed (messaging-app forward), EXIF lost."
    else:
        im = orig["_img"]; h, w = im.shape[:2]
        im = im[int(h * .12):int(h * .86), int(w * .1):int(w * .9)]
        jpeg_with_exif(im, out, q=88)
        c["notes"] = f"{orig['claim_id']}'s photo cropped to hide context, EXIF stripped."
    c["image_path"] = f"{VOL}/images/{fn}"

# ---------------- H-derived classes ----------------
H_imgs = [(p, load(p["path"])) for p in H]
hcycle = iter([H_imgs[i % len(H_imgs)] for i in range(200)])


def h_claim(family, cls, notes, **kw):
    p, img = next(hcycle)
    cd = dt.date(2026, 9, 1) + dt.timedelta(days=random.randint(0, 55))
    c = new_claim(claim_date=cd, device=p["device"], family=family, fraud_class=cls,
                  source_title=p["title"], source_page=p["page"], licence=p["licence"],
                  artist=p["artist"], source_id=p["file"], **kw)
    c["notes"] = notes
    return c, p, img.copy()


def screenshot(img):
    W, Hh = 1179, 2556
    canvas = np.full((Hh, W, 3), 0, np.uint8)
    f = W / img.shape[1]; ph = int(img.shape[0] * f)
    y0 = (Hh - ph) // 2
    canvas[y0:y0 + ph] = cv2.resize(img, (W, ph))
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil)
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 46)
    d.text((70, 38), "9:41", fill="white", font=font)
    d.rounded_rectangle((W - 150, 48, W - 70, 84), 8, outline="white", width=4)
    d.rectangle((W - 144, 54, W - 100, 78), fill="white")
    d.text((W // 2 - 120, Hh - 150), "All Photos", fill="white", font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def rephotograph(img):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    j = lambda: random.uniform(0.02, 0.06)
    dst = np.float32([[w * j(), h * j()], [w * (1 - j()), h * j()], [w * (1 - j()), h * (1 - j())], [w * j(), h * (1 - j())]])
    warped = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(20, 20, 20))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a1, a2 = np.deg2rad(random.uniform(2, 6)), np.deg2rad(random.uniform(-6, -2))
    g1 = np.sin(2 * np.pi * (xx * np.cos(a1) + yy * np.sin(a1)) / 3.1)
    g2 = np.sin(2 * np.pi * (xx * np.cos(a2) + yy * np.sin(a2)) / 3.3)
    moire = (g1 * g2)[..., None]
    out = warped.astype(np.float32) * (1 - 0.10 * (moire + 1) / 2)
    glare = np.exp(-(((xx - w * .7) ** 2) / (2 * (w * .18) ** 2) + ((yy - h * .3) ** 2) / (2 * (h * .2) ** 2)))
    out = out + 70 * glare[..., None]
    out = cv2.GaussianBlur(np.clip(out, 0, 255).astype(np.uint8), (3, 3), 0)
    return out


def upscale(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    ok, b = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 55])
    small = cv2.imdecode(b, cv2.IMREAD_COLOR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def feather_paste(dst, patch, x, y):
    h, w = patch.shape[:2]
    alpha = np.zeros((h, w), np.float32); m = max(4, min(h, w) // 12)
    alpha[m:-m, m:-m] = 1; alpha = cv2.GaussianBlur(alpha, (0, 0), m / 2)[..., None]
    roi = dst[y:y + h, x:x + w].astype(np.float32)
    dst[y:y + h, x:x + w] = (alpha * patch + (1 - alpha) * roi).astype(np.uint8)


def meta_choice(c, img_out, out, i):
    """Pixel-edit metadata realism: 50% clean camera EXIF, 25% editor tag, 25% stripped."""
    mk, md = random.choice(CAPTURE)
    shot = rand_time(dt.date.fromisoformat(c["incident_date"]))
    r = i % 4
    if r in (0, 1): jpeg_with_exif(img_out, out, mk, md, shot)
    elif r == 2: jpeg_with_exif(img_out, out, mk, md, shot, software="Adobe Photoshop 25.2 (Macintosh)")
    else: jpeg_with_exif(img_out, out, q=90)


# shallowfake: screenshot, re-photograph, metadata-only
for i in range(4):
    c, p, img = h_claim("shallowfake", "screenshot", "Someone else's photo, screenshotted (status bar, PNG, no camera metadata).")
    fn = f"{c['claim_id']}.png"; cv2.imwrite(os.path.join(OUT, "images", fn), screenshot(img))
    c["image_path"] = f"{VOL}/images/{fn}"
for i in range(4):
    c, p, img = h_claim("shallowfake", "rephotograph", "Photo displayed on a monitor and re-photographed (moiré, glare, perspective). Fresh camera EXIF.")
    fn = f"{c['claim_id']}.jpg"; mk, md = random.choice(CAPTURE)
    jpeg_with_exif(rephotograph(img), os.path.join(OUT, "images", fn), mk, md, rand_time(dt.date.fromisoformat(c["incident_date"])))
    c["image_path"] = f"{VOL}/images/{fn}"
for i in range(4):
    c, p, img = h_claim("shallowfake", "metadata_old_photo" if i < 2 else "metadata_future_date",
                        "Genuine pixels; EXIF capture date contradicts the claim timeline.")
    inc = dt.date.fromisoformat(c["incident_date"])
    when = rand_time(inc - dt.timedelta(days=random.randint(150, 420))) if i < 2 else \
        rand_time(dt.date.fromisoformat(c["claim_date"]) + dt.timedelta(days=random.randint(4, 20)))
    fn = f"{c['claim_id']}.jpg"; mk, md = random.choice(CAPTURE)
    jpeg_with_exif(img, os.path.join(OUT, "images", fn), mk, md, when)
    c["image_path"] = f"{VOL}/images/{fn}"

# deepfake: splice, copy-move, upscale
for i in range(4):
    c, p, img = h_claim("deepfake", "splice", "Damage region from another device photo pasted in (feathered).")
    donor = H_imgs[(i + 3) % len(H_imgs)][1]
    h, w = img.shape[:2]; s = int(min(h, w) * .32)
    dh, dw = donor.shape[:2]
    patch = cv2.resize(donor[dh // 2 - s // 2: dh // 2 + s // 2, dw // 2 - s // 2: dw // 2 + s // 2], (s, s))
    x, y = int(w * .45 - s / 2), int(h * .5 - s / 2)
    feather_paste(img, patch, x, y)
    fn = f"{c['claim_id']}.jpg"; meta_choice(c, img, os.path.join(OUT, "images", fn), i)
    mask_from_rect(img.shape, x, y, s, s, os.path.join(OUT, "masks", f"{c['claim_id']}.png"))
    c["image_path"] = f"{VOL}/images/{fn}"; c["mask_path"] = f"{VOL}/masks/{c['claim_id']}.png"
for i in range(4):
    c, p, img = h_claim("deepfake", "copy_move", "Existing damage cloned to a second location to look worse.")
    h, w = img.shape[:2]; s = int(min(h, w) * .22)
    sx, sy = int(w * .3), int(h * .35); tx, ty = int(w * .58), int(h * .55)
    feather_paste(img, img[sy:sy + s, sx:sx + s].copy(), tx, ty)
    fn = f"{c['claim_id']}.jpg"; meta_choice(c, img, os.path.join(OUT, "images", fn), i + 1)
    mask_from_rect(img.shape, tx, ty, s, s, os.path.join(OUT, "masks", f"{c['claim_id']}.png"))
    c["image_path"] = f"{VOL}/images/{fn}"; c["mask_path"] = f"{VOL}/masks/{c['claim_id']}.png"
for i in range(4):
    c, p, img = h_claim("deepfake", "upscale", "Low-resolution, heavily compressed image upscaled 4x to pass as an original.")
    fn = f"{c['claim_id']}.jpg"; meta_choice(c, upscale(img), os.path.join(OUT, "images", fn), i)
    c["image_path"] = f"{VOL}/images/{fn}"

# ---------------- deepfake: Gemini edits + full generations ----------------
from databricks.sdk import WorkspaceClient
W = WorkspaceClient()


def gemini(content):
    r = W.api_client.do("POST", "/serving-endpoints/databricks-gemini-3-pro-image/invocations",
                        body={"messages": [{"role": "user", "content": content}], "max_tokens": 4096})
    items = r["choices"][0]["message"]["content"]
    items = json.loads(items) if isinstance(items, str) else items
    imgs = [it["image_url"]["url"] for it in items if isinstance(it, dict) and it.get("type") == "image_url"]
    return base64.b64decode(imgs[-1].split(",", 1)[1])


EDITS = ["make the screen crack much worse: add a large spiderweb fracture across the whole display",
         "add a deep crack running diagonally across the screen and chip the corner of the glass",
         "make the device look water damaged with discoloured blotches on the screen",
         "add severe shattering to the glass with missing shards near the bottom",
         "add a crack across the back glass and a dent in the frame"]
for i, prompt in enumerate(EDITS):
    c, p, img = h_claim("deepfake", "ai_edited_damage", f"Real photo edited by a generative model: '{prompt}'.")
    ok, b = cv2.imencode(".jpg", cv2.resize(img, (1024, int(img.shape[0] * 1024 / img.shape[1]))), [cv2.IMWRITE_JPEG_QUALITY, 90])
    try:
        png = gemini([{"type": "text", "text": f"Edit this photo: {prompt}. Keep everything else identical. Return the edited image."},
                      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b.tobytes()).decode()}}])
    except Exception as e:
        print("edit failed", e); claims.remove(c); continue
    edited = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    base = cv2.resize(img, (edited.shape[1], edited.shape[0]))
    diff = cv2.GaussianBlur(cv2.absdiff(edited, base).max(axis=2), (0, 0), 3)
    cv2.imwrite(os.path.join(OUT, "masks", f"{c['claim_id']}.png"), ((diff > 40) * 255).astype(np.uint8))
    c["mask_path"] = f"{VOL}/masks/{c['claim_id']}.png"
    if i % 2 == 0:   # as returned by the generator (provenance metadata intact)
        fn = f"{c['claim_id']}.png"; open(os.path.join(OUT, "images", fn), "wb").write(png)
        c["notes"] += " Generator output kept as-is (provenance metadata intact)."
    else:            # laundered: re-encoded with fake camera EXIF
        fn = f"{c['claim_id']}.jpg"; mk, md = random.choice(CAPTURE)
        jpeg_with_exif(edited, os.path.join(OUT, "images", fn), mk, md, rand_time(dt.date.fromisoformat(c["incident_date"])))
        c["notes"] += " Laundered: re-encoded as JPEG with fake camera EXIF (provenance stripped)."
    c["image_path"] = f"{VOL}/images/{fn}"
    print("  edit ok", c["claim_id"])

GENS = [("Apple iPhone 15 Pro", "an iPhone 15 Pro with a shattered screen lying on a kitchen counter"),
        ("Samsung Galaxy S24", "a Samsung Galaxy S24 with a cracked display on a car seat"),
        ("Google Pixel 8", "a Google Pixel 8 with spiderweb cracks on the screen, held over a desk"),
        ("Apple iPhone 14", "an iPhone 14 with cracked back glass on a bathroom sink"),
        ("Apple Watch Series 9", "an Apple Watch Series 9 with a cracked face on a wooden table"),
        ("Samsung Galaxy A54", "a Samsung Galaxy A54 with a broken corner and cracked screen on pavement"),
        ("Apple iPhone 13", "an iPhone 13 with a dead, cracked display on a bed sheet"),
        ("Apple MacBook Air", "a MacBook Air with a cracked screen on an office desk")]
for i, (dev, prompt) in enumerate(GENS):
    cd = dt.date(2026, 9, 1) + dt.timedelta(days=random.randint(0, 55))
    c = new_claim(claim_date=cd, device=dev, family="deepfake", fraud_class="ai_generated",
                  source_title="Generated: databricks-gemini-3-pro-image", source_page=None,
                  licence="Synthetic (generated)", artist="AI-generated", source_id=f"gen_{i}",
                  damage="Cracked screen after a drop" if "Watch" not in dev else "Cracked watch face")
    try:
        png = gemini(f"Generate a single photorealistic photo of {prompt}, taken on a phone camera for an insurance claim.")
    except Exception as e:
        print("gen failed", e); claims.remove(c); continue
    if i % 2 == 0:
        fn = f"{c['claim_id']}.png"; open(os.path.join(OUT, "images", fn), "wb").write(png)
        c["notes"] = "Fully AI-generated; generator output kept as-is (provenance metadata intact)."
    else:
        fn = f"{c['claim_id']}.jpg"; mk, md = random.choice(CAPTURE)
        jpeg_with_exif(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR), os.path.join(OUT, "images", fn),
                       mk, md, rand_time(dt.date.fromisoformat(c["incident_date"])))
        c["notes"] = "Fully AI-generated, laundered: re-encoded JPEG with fake camera EXIF (provenance stripped)."
    c["image_path"] = f"{VOL}/images/{fn}"
    print("  gen ok", c["claim_id"])

# ---------------- receipts ----------------
FONT = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 24)
FONT_B = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 30)
STORES = ["TECHMART #0412", "PHONEZONE OUTLET", "BESTCELL STORES", "CITY ELECTRONICS"]


def draw_receipt(store, date_s, item, serial, total, out):
    W_, H_ = 620, 900
    im = Image.new("RGB", (W_, H_), (250, 248, 240)); d = ImageDraw.Draw(im)
    y = 40
    d.text((W_ // 2 - 150, y), store, fill="black", font=FONT_B); y += 60
    d.text((40, y), "123 Market St, Nashville TN", fill="black", font=FONT); y += 40
    d.text((40, y), f"DATE: {date_s}", fill="black", font=FONT); y += 40
    d.text((40, y), f"RECEIPT #: {random.randint(100000, 999999)}", fill="black", font=FONT); y += 60
    d.line((40, y, W_ - 40, y), fill="black", width=2); y += 20
    d.text((40, y), item[:32], fill="black", font=FONT); y += 36
    d.text((40, y), f"SN: ****{serial}", fill="black", font=FONT); y += 50
    tax = round(total * 0.0925 / 1.0925, 2)
    for label, val in [("SUBTOTAL", total - tax), ("TAX", tax), ("TOTAL", total)]:
        d.text((40, y), label, fill="black", font=FONT_B if label == "TOTAL" else FONT)
        d.text((W_ - 220, y), f"${val:,.2f}", fill="black", font=FONT_B if label == "TOTAL" else FONT); y += 44
    y += 30; d.text((40, y), "VISA ****" + str(random.randint(1000, 9999)), fill="black", font=FONT); y += 40
    d.text((40, y), "THANK YOU FOR SHOPPING!", fill="black", font=FONT)
    arr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    M = cv2.getRotationMatrix2D((W_ / 2, H_ / 2), random.uniform(-3, 3), 1.0)
    arr = cv2.warpAffine(arr, M, (W_, H_), borderValue=(90, 90, 95))
    arr = np.clip(arr.astype(np.int16) + np.random.normal(0, 4, arr.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(out, arr, [cv2.IMWRITE_JPEG_QUALITY, 88])


def other_device(dev):
    for alt in ["Apple iPhone 12", "Samsung Galaxy A14", "Google Pixel 6a", "Motorola Moto G"]:
        if alt.split()[0] not in dev:
            return alt
    return "Motorola Moto G"


doc_kinds = ["amount", "amount", "date", "device", "device"]
for c in g_claims:
    is_doc = c["family"] == "document"
    if not is_doc and random.random() > 0.75:
        continue   # some genuine claims have no receipt
    truth = {"total": c["claim_value"], "date": c["purchase_date"], "item": c["device_claimed"]}
    if is_doc:
        kind = doc_kinds.pop(0)
        if kind == "amount":   # one digit differs from the claimed value
            s = f"{int(c['claim_value'])}"; d0 = int(s[0])
            truth["total"] = float(str(d0 - 3 if d0 > 3 else d0 + 3) + s[1:])
        elif kind == "date":
            truth["date"] = (dt.date.fromisoformat(c["claim_date"]) + dt.timedelta(days=9)).isoformat()
        else:
            truth["item"] = other_device(c["device_claimed"])
        c["notes"] = f"Genuine photo, but the receipt does not support the claim ({kind} mismatch)."
        c["receipt_mismatch"] = kind
    fn = f"{c['claim_id']}_receipt.jpg"
    d = dt.date.fromisoformat(truth["date"])
    draw_receipt(random.choice(STORES), d.strftime("%m/%d/%Y"), truth["item"], random.randint(1000, 9999),
                 truth["total"], os.path.join(OUT, "receipts", fn))
    c["receipt_path"] = f"{VOL}/receipts/{fn}"; c["receipt_truth"] = json.dumps(truth)

for c in claims:
    c.pop("_img", None); c.pop("_capture", None)
    c.setdefault("reuse_of", None); c.setdefault("receipt_mismatch", None)
json.dump(claims, open(os.path.join(OUT, "claims.json"), "w"), indent=1, default=str)
from collections import Counter
print(len(claims), "claims")
print(Counter(c["family"] for c in claims)); print(Counter(c["fraud_class"] for c in claims))

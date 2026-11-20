import csv, glob, json, os, sys
import numpy as np
ROOT = os.path.expanduser("~/insurance_firm-image-authenticity"); W = ROOT + "/wb_data"
rng = np.random.default_rng(0)

def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None, :]).sum() + .5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))
def boot(s, y, B=800):
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]; v = []
    for _ in range(B):
        idx = np.r_[rng.choice(pi, len(pi)), rng.choice(ni, len(ni))]; v.append(auc(s[idx], y[idx]))
    return np.percentile(v, [2.5, 97.5])
def logit(p): p = np.clip(p, 1e-12, 1 - 1e-12); return np.log(p / (1 - p))
def rank(x): return np.argsort(np.argsort(x)) / (len(x) - 1)

def spai_csv(path, keyfn):
    d = {}
    for r in csv.DictReader(open(path)):
        d[keyfn(r["image"])] = float(r["spai"])
    return d

def load_v1(view):
    it = json.load(open(W + "/ml_v1/items.json")); ids = [i["id"] for i in it]
    S = json.load(open(W + f"/ml_v1/scores_{view}.json"))
    sp = spai_csv(W + f"/ml_v1/spai_{view}/{view}.csv", lambda s: s.rsplit(".", 1)[0])
    y = np.array([i["label"] for i in it]); g = np.array([i["group"] for i in it])
    X = np.c_[S["clipdet_latent10k_plus"], logit(np.array([sp[i] for i in ids])), S["upscale"]]
    return X, y, g

def load_v0(view):
    claims = json.load(open(W + "/dataset/claims.json")); ids = [c["claim_id"] for c in claims]
    z = np.load(W + f"/ml_exp/ml_{view}.npz", allow_pickle=True); cid = list(z["claim_ids"])
    clip = np.array([z["clipdet_latent10k_plus"][cid.index(i)] for i in ids])
    C = json.load(open(W + f"/ml_exp/classical_{view}.json")); cc = C["claim_ids"]
    up = np.array([C["scores"]["upscale"][cc.index(i)] for i in ids])
    sp = spai_csv(glob.glob(W + f"/ml_exp/spai_{view}/*.csv")[0], lambda s: s.rsplit(".", 1)[0])
    ai = np.array([c["fraud_class"] in ("ai_generated", "ai_edited_damage") for c in claims])
    neg = np.array([c["family"] in ("genuine", "document") for c in claims])
    keep = ai | neg
    X = np.c_[clip, logit(np.array([sp[i] for i in ids])), up][keep]
    return X, ai[keep].astype(int)

def fit_lr(X, y, l2=1.0, it=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Z @ w + b))); g = p - y
        w -= lr * (Z.T @ g / len(y) + l2 * w / len(y)); b -= lr * g.mean()
    return lambda Xn: ((Xn - mu) / sd) @ w + b, w

NAMES = ["CLIP detector", "SPAI", "Upscale (OpenCV)"]
for view in ("orig", "norm"):
    X, y, g = load_v1(view)
    print(f"\n===== v1 held-out, view={view} ({int(y.sum())} AI / {int((y == 0).sum())} real) =====")
    combos = {"CLIP": [0], "SPAI": [1], "Upscale": [2], "CLIP+SPAI (rank avg)": [0, 1],
              "CLIP+Upscale (rank avg)": [0, 2], "CLIP+SPAI+Upscale (rank avg)": [0, 1, 2]}
    for n, ix in combos.items():
        s = np.mean([rank(X[:, i]) for i in ix], axis=0); a = auc(s, y); lo, hi = boot(s, y)
        gem = g == None
        gm = np.array([x.startswith("gemini") for x in g]); wb = np.array([x.startswith("web") for x in g])
        def sub(m): return auc(s[(y == 0) | m], y[(y == 0) | m])
        print(f"  {n:32s} AUC {a:.2f} [{lo:.2f}-{hi:.2f}]   Gemini-only {sub(gm):.2f}   other generators {sub(wb):.2f}")
    # train on v0 (frozen), test on v1
    X0, y0 = load_v0('orig' if view == 'orig' else 'social')      # gold v0 has no 'norm' view; social is the closest
    f, w = fit_lr(X0, y0)
    s = f(X); a = auc(s, y); lo, hi = boot(s, y)
    print(f"  {'LogReg trained on gold v0 -> v1':32s} AUC {a:.2f} [{lo:.2f}-{hi:.2f}]   weights(z) CLIP {w[0]:+.2f} SPAI {w[1]:+.2f} Upscale {w[2]:+.2f}")
    # and reverse: train on v1, test on v0
    f1, w1 = fit_lr(X, y)
    s0 = f1(X0); a0 = auc(s0, y0); lo0, hi0 = boot(s0, y0)
    print(f"  {'LogReg trained on v1 -> gold v0':32s} AUC {a0:.2f} [{lo0:.2f}-{hi0:.2f}]   (v0 singles: CLIP {auc(X0[:,0],y0):.2f}  SPAI {auc(X0[:,1],y0):.2f}  Upscale {auc(X0[:,2],y0):.2f})")

# OR-rule at the detectors' default thresholds (what the rules engine effectively does with several amber/red checks)
print("\n===== OR rule at default thresholds, v1 =====")
for view in ("orig", "norm"):
    it = json.load(open(W + "/ml_v1/items.json")); S = json.load(open(W + f"/ml_v1/scores_{view}.json"))
    ids = [i["id"] for i in it]; y = np.array([i["label"] for i in it])
    sp = spai_csv(W + f"/ml_v1/spai_{view}/{view}.csv", lambda s: s.rsplit(".", 1)[0])
    f = {"CLIP": np.array(S["clipdet_latent10k_plus"]) > 0, "SPAI": np.array([sp[i] for i in ids]) >= 0.5,
         "Upscale": np.array(S["upscale"]) > 0.5}
    for n, m in (("CLIP", f["CLIP"]), ("CLIP or SPAI", f["CLIP"] | f["SPAI"]), ("CLIP and SPAI", f["CLIP"] & f["SPAI"]),
                 ("CLIP or Upscale", f["CLIP"] | f["Upscale"]), ("any of the three", f["CLIP"] | f["SPAI"] | f["Upscale"]),
                 ("at least 2 of 3", (f["CLIP"].astype(int) + f["SPAI"] + f["Upscale"]) >= 2)):
        print(f"  {view:5s} {n:20s} recall {m[y==1].mean():4.0%}  false alarms {m[y==0].mean():4.0%}")

# reuse: SSCD + ORB
R = json.load(open(W + "/ml_v1/reuse_results.json")); rows = R["rows"]
pos = [r for r in rows if r["kind"] == "positive"]; neg = [r for r in rows if r["kind"] == "negative"]
def hit(r, m): return (r[f"{m}_score"] >= R["thresholds"][m]) if m != "phash" else (r["phash_score"] <= R["thresholds"]["phash"])
print("\n===== reuse: SSCD + ORB =====")
for n, fn in (("SSCD alone", lambda r: hit(r, "sscd")), ("ORB alone", lambda r: hit(r, "orb")),
              ("SSCD or ORB", lambda r: hit(r, "sscd") or hit(r, "orb")),
              ("SSCD and ORB (ORB confirms)", lambda r: hit(r, "sscd") and hit(r, "orb")),
              ("SSCD or pHash", lambda r: hit(r, "sscd") or hit(r, "phash"))):
    print(f"  {n:30s} found {sum(fn(r) and r['sscd_top' if 'sscd' in n.lower() else 'orb_top'] == r['source'] for r in pos):2d}/80   false hits {sum(fn(r) for r in neg)}/40")
print("  ORB false-hit queries:", [(r['id'], r['orb_score']) for r in neg if hit(r, 'orb')])

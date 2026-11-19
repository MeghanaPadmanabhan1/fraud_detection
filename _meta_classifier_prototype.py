"""Prototype: replace the hand-weighted composite with a LEARNED stacking
meta-classifier over the per-signal scores.

Honest framing: 54 samples is tiny, so we use leave-one-out CV (train on 53,
predict the 1 held out) to get an unbiased accuracy estimate, and compare to the
current rule-based composite on the same images.
"""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

det = json.load(open("/tmp/det.json"))
scores = json.load(open("/tmp/scores.json"))

SLOTS = ["spectral", "ela_global", "learned", "ela_local",
         "copy_move", "noise", "jpeg_ghost", "watermark"]

# Pivot detections -> per-image feature vector (impute missing/errored with 0.5).
feat, label = {}, {}
for r in det:
    iid = r["image_id"]; label[iid] = r["true_label"]
    feat.setdefault(iid, {})[r["slot"]] = (
        float(r["score"]) if r["score"] not in (None, "") else np.nan)

ids = sorted(feat)
X = np.array([[feat[i].get(s, np.nan) for s in SLOTS] for i in ids])
X = np.where(np.isnan(X), 0.5, X)          # impute
y3 = np.array([label[i] for i in ids])      # authentic / deepfake / shallow_fake
y2 = np.where(y3 == "authentic", "authentic", "fake")  # binary

print(f"{len(ids)} images | classes: {dict(zip(*np.unique(y3, return_counts=True)))}")

def loo(X, y):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    pred = cross_val_predict(clf, X, y, cv=LeaveOneOut())
    return pred

# ---- Binary: fake vs authentic ----
p2 = loo(X, y2)
acc2 = accuracy_score(y2, p2)
rec_fake = recall_score(y2, p2, pos_label="fake")
prec_fake = precision_score(y2, p2, pos_label="fake")
print("\n=== META-CLASSIFIER (binary: fake vs authentic), leave-one-out CV ===")
print(f"accuracy={acc2:.2f}  fake-recall={rec_fake:.2f}  fake-precision={prec_fake:.2f}")

# ---- 3-class: authentic / deepfake / shallow_fake ----
p3 = loo(X, y3)
acc3 = accuracy_score(y3, p3)
print("\n=== META-CLASSIFIER (3-class), leave-one-out CV ===")
print(f"accuracy={acc3:.2f}")
classes = ["authentic", "deepfake", "shallow_fake"]
cm = confusion_matrix(y3, p3, labels=classes)
print("confusion (rows=true, cols=pred):", classes)
for c, row in zip(classes, cm):
    print(f"  {c:13s} {row}")

# ---- Learned feature importance (fit on all data, standardized coefs) ----
full = make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=2000, class_weight="balanced"))
full.fit(X, y2)
coefs = full.named_steps["logisticregression"].coef_[0]
print("\n=== what the model relies on (binary 'fake' standardized weights) ===")
for s, c in sorted(zip(SLOTS, coefs), key=lambda t: -abs(t[1])):
    print(f"  {s:12s} {c:+.2f}")

# ---- Baseline: current rule-based composite on the same images ----
verd = {s["image_id"]: s["verdict"] for s in scores}
tb = np.array([("authentic" if label[i] == "authentic" else "fake") for i in ids])
rb = np.array([("authentic" if verd.get(i) == "likely_authentic" else "fake") for i in ids])
print("\n=== RULE-BASED composite (verdict != likely_authentic => fake) ===")
print(f"accuracy={accuracy_score(tb, rb):.2f}  "
      f"fake-recall={recall_score(tb, rb, pos_label='fake'):.2f}  "
      f"fake-precision={precision_score(tb, rb, pos_label='fake'):.2f}")

"""POST-HOC sensitivity check (not part of the pre-specified protocol): does SSCD's weak recall on screenshots come
from the square 320x320 resize I used? The SSCD README allows either Resize([320,320]) or Resize(288) on the short edge.

    python -I build/ml_exp_sscd_sensitivity.py
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ml_exp_analyze as A
from ml_exp_common import EXP, WB, variant_path

M = os.path.join(WB, "ml_models", "sscd")
NORM = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
PRE = {"square 320x320 (used in the main run)": T.Compose([T.Resize([320, 320]), NORM]),
       "short edge 288 (aspect kept)": T.Compose([T.Resize(288), NORM])}

order = sorted(range(A.N), key=lambda i: (A.claims[i]["claim_date"], A.claims[i]["claim_id"]))
rank = {i: r for r, i in enumerate(order)}
src = [c["source_id"] for c in A.claims]
earlier = [[j for j in range(A.N) if rank[j] < rank[i]] for i in range(A.N)]
pos = np.array([any(src[j] == src[i] for j in earlier[i]) for i in range(A.N)])

rows = []
for variant in ("orig", "social"):
    for model_name in ("sscd_disc_mixup", "sscd_disc_large"):
        net = torch.jit.load(os.path.join(M, f"{model_name}.torchscript.pt"), map_location="cpu").eval()
        for pname, tf in PRE.items():
            with torch.no_grad():
                E = np.stack([torch.nn.functional.normalize(net(tf(Image.open(variant_path(c, variant)).convert("RGB")).unsqueeze(0))[0], dim=0).numpy()
                              for c in A.claims])
            Sim = E @ E.T
            sc = np.array([max([Sim[i, j] for j in earlier[i]], default=Sim.min()) for i in range(A.N)])
            thr = sc[~pos].max()
            r = {"variant": variant, "model": model_name, "preprocessing": pname,
                 "auc": A.auc(sc[pos], sc[~pos]), "recall_at_0fp": float((sc[pos] > thr).mean())}
            for k in ("screenshot", "rephotograph", "reuse_cropped", "splice"):
                m = pos & (A.cls == k)
                r[k] = float((sc[m] > thr).mean())
            rows.append(r)
            print(f"{variant:6s} {model_name:16s} {pname:40s} AUC {r['auc']:.2f}  R@0FP {r['recall_at_0fp']:.0%}  "
                  f"screenshot {r['screenshot']:.0%}  re-photo {r['rephotograph']:.0%}  cropped {r['reuse_cropped']:.0%}  splice {r['splice']:.0%}", flush=True)
json.dump(rows, open(os.path.join(EXP, "sscd_sensitivity.json"), "w"), indent=1)

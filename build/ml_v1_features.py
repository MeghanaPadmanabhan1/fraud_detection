"""Score every v1 image, per variant: Cozzolino CLIP detector (LLR>0 = synthetic) + the app's OpenCV/NumPy checks
+ confound probes (aspect ratio, bytes per pixel). SPAI runs separately (own environment).

    ml_env/bin/python -I build/ml_v1_features.py <orig|social|norm>
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from image_authenticity import detectors  # noqa: F401,E402
from image_authenticity.core import build   # noqa: E402
from image_authenticity.detectors import _util   # noqa: E402
from image_authenticity.workbench import checks   # noqa: E402

variant = sys.argv[1]
OUT = os.path.join(ROOT, "wb_data", "ml_v1")
M = os.path.join(ROOT, "wb_data", "ml_models")
items = json.load(open(os.path.join(OUT, "items.json")))
paths = [it[variant] for it in items]
torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))
res = {"ids": [it["id"] for it in items]}

# ---- CLIP detector (authors' code; backbone from local safetensors)
import huggingface_hub   # noqa: E402
huggingface_hub.hf_hub_download = lambda *a, **k: os.path.join(M, "hf_clip", "open_clip_pytorch_model.safetensors")
sys.path.insert(0, os.path.join(M, "clipdet_repo"))
from networks import create_architecture, load_weights   # noqa: E402
from utils.processing import make_normalize   # noqa: E402
model = create_architecture("opencliplinearnext_clipL14commonpool").to("cpu").eval()
tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop((224, 224)), make_normalize("clip")])
with torch.no_grad():
    feats = torch.cat([model.forward_features(tf(Image.open(p).convert("RGB")).unsqueeze(0)) for p in paths])
    for head in ("clipdet_latent10k_plus", "clipdet_latent10k"):
        load_weights(model, os.path.join(M, "clipdet_weights", head, "weights.pth"))
        res[head] = model.forward_head(feats).reshape(len(paths), -1).mean(1).numpy().tolist()
        print(head, "flagged synthetic:", int((np.array(res[head]) > 0).sum()), "/", len(paths), flush=True)

# ---- the app's own pixel-level checks + probes
DET = {"ela": ("ela_local", "ela_heatmap_v1"), "jpeg_ghost": ("jpeg_ghost", "jpeg_ghost_v1"), "noise": ("noise", "noise_residual_v1"),
       "copy_move_orb": ("copy_move", "orb_v1"), "copy_move_sift": ("copy_move", "sift_v1"),
       "spectral_fft": ("spectral", "fft_radial_v1"), "spectral_dct": ("spectral", "dct_stats_v1"),
       "ela_uniformity": ("ela_global", "ela_uniformity_v1")}
dets = {k: build(*v) for k, v in DET.items()}
for k in list(DET) + ["upscale", "recapture", "aspect_probe", "bytes_per_pixel_probe"]:
    res[k] = []
for p in paths:
    raw = open(p, "rb").read()
    img = _util.decode_bgr(raw)
    for k, d in dets.items():
        s = d.analyze(img, raw, {}).score
        res[k].append(float(s) if s is not None else 0.0)
    meta = checks.read_metadata(raw)
    res["upscale"].append(checks.upscale_check(img)["score"])
    res["recapture"].append(checks.recapture_check(img, meta)["score"])
    h, w = img.shape[:2]
    res["aspect_probe"].append(abs(float(np.log(w / h))))
    res["bytes_per_pixel_probe"].append(len(raw) / (w * h))
json.dump(res, open(os.path.join(OUT, f"scores_{variant}.json"), "w"))
print("saved", variant)

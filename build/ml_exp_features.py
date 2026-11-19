"""Learned-model scores for one image variant: SSCD (copy-detection descriptors) and the Cozzolino et al.
CLIP-based synthetic-image detector. Runs the AUTHORS' code unmodified; the only change is that the CLIP
backbone is read from a local safetensors copy instead of downloading a pickle .bin from the Hub.

    python -I build/ml_exp_features.py <orig|social>
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_exp_common import EXP, WB, load_claims, variant_path

variant = sys.argv[1]
M = os.path.join(WB, "ml_models")
torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))
torch.manual_seed(0)
claims = load_claims()
paths = [variant_path(c, variant) for c in claims]
os.makedirs(EXP, exist_ok=True)
out, timing = {"claim_ids": np.array([c["claim_id"] for c in claims])}, {}

# ----------------------------------------------------------------------------- SSCD
sscd_tf = T.Compose([T.Resize([320, 320]), T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
for name in ("sscd_disc_mixup", "sscd_disc_large"):
    net = torch.jit.load(os.path.join(M, "sscd", f"{name}.torchscript.pt"), map_location="cpu").eval()
    embs, secs = [], []
    with torch.no_grad():
        for p in paths:
            x = sscd_tf(Image.open(p).convert("RGB")).unsqueeze(0)
            t0 = time.perf_counter(); e = net(x)[0]; secs.append(time.perf_counter() - t0)
            embs.append(torch.nn.functional.normalize(e, dim=0).numpy())     # already L2-normalised; be explicit
    out[name] = np.stack(embs).astype("float32")
    timing[name] = float(np.median(secs))
    print(f"{name}: {out[name].shape}, median {timing[name] * 1000:.0f} ms/img (CPU)", flush=True)

# ----------------------------------------------------------------------------- Cozzolino CLIP detector
import huggingface_hub
LOCAL_BACKBONE = os.path.join(M, "hf_clip", "open_clip_pytorch_model.safetensors")
huggingface_hub.hf_hub_download = lambda *a, **k: LOCAL_BACKBONE          # authors call this inside OpenClipLinear
sys.path.insert(0, os.path.join(M, "clipdet_repo"))
from networks import create_architecture, load_weights                    # noqa: E402  (authors' code)
from utils.processing import make_normalize                               # noqa: E402

model = create_architecture("opencliplinearnext_clipL14commonpool").to("cpu").eval()
clip_tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop((224, 224)),
                     make_normalize("clip")])
feats, secs = [], []
with torch.no_grad():
    for p in paths:
        x = clip_tf(Image.open(p).convert("RGB")).unsqueeze(0)
        t0 = time.perf_counter(); f = model.forward_features(x); secs.append(time.perf_counter() - t0)
        feats.append(f)
timing["clip_backbone"] = float(np.median(secs))
feats = torch.cat(feats)
print(f"CLIP ViT-L/14 features: {tuple(feats.shape)}, median {timing['clip_backbone'] * 1000:.0f} ms/img (CPU)", flush=True)
for head in ("clipdet_latent10k_plus", "clipdet_latent10k"):
    load_weights(model, os.path.join(M, "clipdet_weights", head, "weights.pth"))
    with torch.no_grad():
        llr = model.forward_head(feats).reshape(len(paths), -1).mean(1).numpy()
    out[head] = llr.astype("float64")
    print(f"{head}: LLR range [{llr.min():.2f}, {llr.max():.2f}], flagged synthetic (LLR>0): {(llr > 0).sum()}/{len(llr)}",
          flush=True)

np.savez(os.path.join(EXP, f"ml_{variant}.npz"), **out)
json.dump(timing, open(os.path.join(EXP, f"ml_{variant}_timing.json"), "w"))
print("saved", os.path.join(EXP, f"ml_{variant}.npz"))

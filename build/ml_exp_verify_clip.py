"""Validity checks for the Cozzolino CLIP-detector pipeline (a negative result is only meaningful if the
pipeline is verifiably correct):
  1. every backbone tensor in memory equals the tensor in the downloaded safetensors file (weights really loaded);
  2. the linear head in memory equals the verified head file;
  3. the visual tower is not random: zero-shot CLIP picks the right caption for real photos from our set.

    python -I build/ml_exp_verify_clip.py
"""
import os
import sys

import huggingface_hub
import numpy as np
import open_clip
import torch
from PIL import Image
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_exp_common import WB, load_claims, orig_path

M = os.path.join(WB, "ml_models")
ST = os.path.join(M, "hf_clip", "open_clip_pytorch_model.safetensors")
huggingface_hub.hf_hub_download = lambda *a, **k: ST
sys.path.insert(0, os.path.join(M, "clipdet_repo"))
from networks import create_architecture, load_weights   # noqa: E402

model = create_architecture("opencliplinearnext_clipL14commonpool").eval()
load_weights(model, os.path.join(M, "clipdet_weights", "clipdet_latent10k_plus", "weights.pth"))

# 1. backbone tensors vs file
file_sd = load_file(ST)
vis = {k[len("visual."):]: v for k, v in file_sd.items() if k.startswith("visual.")}
mem = model.bb[0].visual.state_dict()
checked = bad = 0
for k, v in mem.items():
    if k in vis:
        checked += 1
        bad += int(not torch.equal(v.float(), vis[k].float()))
missing = [k for k in mem if k not in vis]
print(f"1. backbone: {checked} visual tensors compared with the safetensors file, {bad} differ; "
      f"{len(missing)} in-memory tensors without a file counterpart {missing[:3]}")

# 2. head vs file
head = torch.load(os.path.join(M, "clipdet_weights", "clipdet_latent10k_plus", "weights.pth"), map_location="cpu",
                  weights_only=True)["model"]
hw = {k.replace("module.", ""): v for k, v in head.items()}
print("2. head: keys", list(hw), "| in-memory fc.weight equals file:",
      torch.equal(model.fc.weight, next(v for k, v in hw.items() if k.endswith("weight"))),
      "| weight norm", float(model.fc.weight.norm()))

# 3. zero-shot sanity with the full CLIP (text tower) on real photos of our set
full, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained=ST)
full.eval(); tok = open_clip.get_tokenizer("ViT-L-14")
claims = [c for c in load_claims() if c["family"] == "genuine"][:6]
caps = ["a photo of a flower", "a photo of a broken smartphone", "a photo of a cat", "a photo of a laptop"]
with torch.no_grad():
    t = full.encode_text(tok(caps), normalize=True)
    for c in claims:
        i = full.encode_image(pre(Image.open(orig_path(c)).convert("RGB")).unsqueeze(0), normalize=True)
        p = (100 * i @ t.T).softmax(-1)[0]
        print(f"3. zero-shot {c['claim_id']}: top caption = '{caps[int(p.argmax())]}' (p={float(p.max()):.2f})")

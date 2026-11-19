"""Inspect a torch checkpoint WITHOUT trusting it: restricted load first; on failure list the pickle's imports
statically (pickletools, nothing executed). Usage: python -I build/ml_inspect_ckpt.py <checkpoint.pth>"""
import pickletools, sys, zipfile
import torch

import importlib
p = sys.argv[1]
allow = [a for a in sys.argv[2:] if "." in a]          # e.g. yacs.config.CfgNode (reviewed from the static scan)
for a in allow:
    mod, cls = a.rsplit(".", 1)
    torch.serialization.add_safe_globals([getattr(importlib.import_module(mod), cls)])
try:
    ck = torch.load(p, map_location="cpu", weights_only=True)
    print("restricted load (weights_only=True): OK -> contains only tensors / plain containers")
except Exception as e:
    print("restricted load FAILED:", type(e).__name__, str(e)[:700])
    with zipfile.ZipFile(p) as z:
        data = z.read([n for n in z.namelist() if n.endswith("data.pkl")][0])
    strings, globs = [], set()
    for op, arg, _ in pickletools.genops(data):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            strings.append(arg)
        elif op.name == "GLOBAL":
            globs.add(str(arg).replace(" ", "."))
        elif op.name == "STACK_GLOBAL" and len(strings) >= 2:
            globs.add(f"{strings[-2]}.{strings[-1]}")
    print("pickle imports (static scan):")
    for g in sorted(globs):
        print("  ", g)
    sys.exit(0)

def describe(o, depth=0, key="root"):
    pad = "  " * depth
    if isinstance(o, dict):
        print(f"{pad}{key}: dict[{len(o)}]")
        if depth < 1:
            for k, v in list(o.items())[:12]:
                describe(v, depth + 1, k)
    elif torch.is_tensor(o):
        print(f"{pad}{key}: tensor {tuple(o.shape)} {o.dtype}")
    else:
        print(f"{pad}{key}: {type(o).__name__} = {str(o)[:100]}")
describe(ck)
sd = ck.get("model", ck) if isinstance(ck, dict) else ck
if isinstance(sd, dict):
    n = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    print(f"state-dict tensors: {sum(torch.is_tensor(v) for v in sd.values())}, parameters: {n/1e6:.1f}M")
    print("first keys:", list(sd)[:6]); print("last keys:", list(sd)[-4:])

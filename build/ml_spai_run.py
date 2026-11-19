"""Run SPAI (Karageorgiou et al., CVPR 2025; github.com/mever-team/spai) on CPU.

The authors' code runs unmodified. Process-level shims only (all documented in the experiment report):
  1. Checkpoint loading goes through a RESTRICTED unpickler that allows exactly the six globals found by a
     static scan of the checkpoint (tensor rebuild, OrderedDict, set, and the yacs config container, which is
     replaced by a plain dict). Anything else raises -- the file cannot execute code.
  2. `.cuda()` is mapped to a no-op so the CLI's hard-coded CUDA calls run on CPU.
  3. `clip`, `neptune`, `tsnecuda` are stubbed: they are imported at module level but never used on the
     `infer` path (verified by reading the source).

    python -I build/ml_spai_run.py check                 # load the checkpoint and print its structure
    python -I build/ml_spai_run.py infer <orig|social>   # score every claim image of a variant
"""
import collections
import glob
import os
import pickle
import sys
import time
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_exp_common import EXP, SOCIAL_DIR, WB, load_claims, variant_path   # noqa: E402

M = os.path.join(WB, "ml_models")
SPAI_SRC, CKPT = os.path.join(M, "spai_src"), os.path.join(M, "spai", "spai.pth")

# ---- 1. restricted unpickler -------------------------------------------------------------------------
class _CfgStandIn(dict):
    """Stand-in for yacs.config.CfgNode (training config stored inside the checkpoint; unused at inference)."""

_ALLOWED = {("collections", "OrderedDict"): collections.OrderedDict, ("__builtin__", "set"): set,
            ("builtins", "set"): set, ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
            ("yacs.config", "CfgNode"): _CfgStandIn}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return _ALLOWED[(module, name)]
        raise pickle.UnpicklingError(f"blocked global in checkpoint: {module}.{name}")


_restricted = types.ModuleType("restricted_pickle")          # torch.load reads pickle_module.__name__
_restricted.Unpickler, _restricted.UnpicklingError = _RestrictedUnpickler, pickle.UnpicklingError
_restricted.load = lambda f, **kw: _RestrictedUnpickler(f, **kw).load()
_restricted.Pickler, _restricted.dump, _restricted.HIGHEST_PROTOCOL = pickle.Pickler, pickle.dump, pickle.HIGHEST_PROTOCOL
_real_torch_load = torch.load


def restricted_load(f, map_location=None, pickle_module=None, *, weights_only=None, **kw):
    return _real_torch_load(f, map_location=map_location, pickle_module=_restricted, weights_only=False, **kw)


torch.load = restricted_load

mode = sys.argv[1]
if mode == "check":
    ck = torch.load(CKPT, map_location="cpu")
    print("loaded through the restricted unpickler; top-level keys:", list(ck))
    sd = ck["model"]
    n = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
    print(f"model tensors: {sum(torch.is_tensor(v) for v in sd.values())}, parameters: {n / 1e6:.1f}M")
    for k in ck:
        if k != "model":
            print(f"  {k}: {type(ck[k]).__name__} = {str(ck[k])[:120]}")
    sys.exit(0)

# ---- 2/3. shims ----------------------------------------------------------------------------------------
for stub in ("clip", "neptune", "tsnecuda"):
    sys.modules[stub] = types.ModuleType(stub)
torch.Tensor.cuda = lambda self, *a, **k: self
torch.nn.Module.cuda = lambda self, *a, **k: self
torch.cuda.max_memory_allocated = lambda *a, **k: 0
torch.cuda.empty_cache = lambda *a, **k: None

if mode == "dir":                                    # python ml_spai_run.py dir <image_dir> <output_dir>
    variant, limit, img_dir, out_dir = "dir", None, sys.argv[2], sys.argv[3]
else:
    variant = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
    img_dir = os.path.join(WB, "dataset", "images") if variant == "orig" else SOCIAL_DIR
    out_dir = os.path.join(EXP, f"spai_{variant}" + (f"_first{limit}" if limit else ""))
if limit:                                           # timing / smoke test on a handful of images
    import shutil
    img_dir_small = os.path.join(EXP, f"_spai_smoke_{variant}")
    shutil.rmtree(img_dir_small, ignore_errors=True); os.makedirs(img_dir_small)
    for c in load_claims()[:limit]:
        shutil.copy(variant_path(c, variant), img_dir_small)
    img_dir = img_dir_small

sys.path.insert(0, SPAI_SRC)
torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))
from spai.__main__ import cli                       # noqa: E402  (authors' code)

t0 = time.time()
cli(["infer", "--cfg", os.path.join(SPAI_SRC, "configs", "spai.yaml"), "--batch-size", "1",
     "--input", img_dir, "--model", CKPT, "--output", out_dir, "--tag", "spai",
     "--opt", "DATA.NUM_WORKERS", "0", "--opt", "DATA.TEST_PREFETCH_FACTOR", "None",
     "--opt", "DATA.PREFETCH_FACTOR", "None"],   # no worker processes (macOS spawn re-imports this script)
    standalone_mode=False)
n = len(glob.glob(os.path.join(img_dir, "*")))
if variant == "orig" and not limit:
    import json
    json.dump({"s_per_img": (time.time() - t0) / n}, open(os.path.join(EXP, "spai_timing.json"), "w"))
print(f"SPAI finished {n} images in {time.time() - t0:.0f}s ({(time.time() - t0) / n:.1f} s/img on CPU); outputs in {out_dir}")

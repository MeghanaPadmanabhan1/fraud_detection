"""Generate fully AI-made damaged-device photos with three Gemini image models (same 10 prompts each).

    DATABRICKS_CONFIG_PROFILE=e2 python -I build/gen_ai_v1.py <out_dir>
Writes <out_dir>/<model>_<i>.png (generator output untouched) and <out_dir>/index.json.
"""
import base64
import concurrent.futures as cf
import json
import os
import sys

from databricks.sdk import WorkspaceClient

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
W = WorkspaceClient()
MODELS = ["databricks-gemini-3-pro-image", "databricks-gemini-3-1-flash-image", "databricks-gemini-3-1-flash-lite-image"]
SCENES = [
    ("Apple iPhone 12", "an iPhone 12 with a cracked screen held in a hand outdoors"),
    ("Samsung Galaxy S22", "a Samsung Galaxy S22 with a shattered display on a wooden desk"),
    ("Google Pixel 6", "a Google Pixel 6 with a cracked back glass on a bedside table"),
    ("Apple iPad", "an iPad with a spiderweb crack across the screen on a sofa"),
    ("OnePlus 9", "a OnePlus 9 with a broken screen and a bent frame on concrete"),
    ("Apple iPhone SE", "an older iPhone SE with a chipped corner and cracked glass next to a coffee mug"),
    ("Samsung Galaxy Tab", "a Samsung Galaxy Tab with a half-dead cracked display on a kitchen table"),
    ("Motorola Moto G", "a Motorola Moto G with a cracked screen on a car dashboard"),
    ("Apple iPhone 11", "an iPhone 11 with a cracked screen and water droplets on it, close up"),
    ("Dell laptop", "a Dell laptop with a cracked LCD panel on an office desk"),
]


def gen(job):
    model, i = job
    dev, prompt = SCENES[i]
    try:
        r = W.api_client.do("POST", f"/serving-endpoints/{model}/invocations", body={
            "messages": [{"role": "user", "content":
                          f"Generate a single photorealistic photo of {prompt}, taken on a phone camera for an insurance claim."}],
            "max_tokens": 4096})
        items = r["choices"][0]["message"]["content"]
        items = json.loads(items) if isinstance(items, str) else items
        urls = [it["image_url"]["url"] for it in items if isinstance(it, dict) and it.get("type") == "image_url"]
        png = base64.b64decode(urls[-1].split(",", 1)[1])
    except Exception as e:
        return dict(model=model, i=i, device=dev, prompt=prompt, error=str(e)[:200])
    fn = f"{model.replace('databricks-', '')}_{i}.png"
    open(os.path.join(OUT, fn), "wb").write(png)
    return dict(model=model, i=i, device=dev, prompt=prompt, file=fn, bytes=len(png))


with cf.ThreadPoolExecutor(4) as ex:
    res = list(ex.map(gen, [(m, i) for m in MODELS for i in range(len(SCENES))]))
json.dump(res, open(os.path.join(OUT, "index.json"), "w"), indent=1)
ok = [r for r in res if "file" in r]
print(len(ok), "generated;", len(res) - len(ok), "failed")
for r in res:
    if "error" in r:
        print(" fail", r["model"], r["i"], r["error"])

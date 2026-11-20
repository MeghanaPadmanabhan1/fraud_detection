# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Select a model per functionality, run, and evaluate
# MAGIC
# MAGIC **Functionality** = one detection capability (e.g. *Spectral analysis*,
# MAGIC *Copy-move*, *Learned / vision-LLM*). **Model** = the concrete implementation
# MAGIC plugged into that functionality. Each functionality below is a **widget**
# MAGIC (top of the notebook) whose dropdown lists the models available for it —
# MAGIC plus `(off)` to ablate it.
# MAGIC
# MAGIC Pick a model per functionality → **Run all** → read the per-functionality
# MAGIC evaluation at the bottom. Switch one model, re-run, and watch that
# MAGIC functionality's separation on its own test images change.
# MAGIC
# MAGIC > **⚠️ Serverless environment:** set **Environment version = 2** in the
# MAGIC > Environment side panel before running. The floating/default version has
# MAGIC > intermittently crashed the kernel on native imports; version 2 is the
# MAGIC > one the scheduled job runs on. This notebook is also kept **pandas-free**
# MAGIC > to minimize that native-import surface.

# COMMAND ----------
# MAGIC %pip install -q --no-deps opencv-python-headless pillow
# MAGIC # --no-deps: do NOT let pip upgrade numpy (breaks the runtime ABI)
# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import sys, json
from collections import defaultdict

PKG_ROOT = "/Workspace/Users/<workspace-user>/insurance_firm-image-authenticity"
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from image_authenticity import get_config
from image_authenticity.core import available
from image_authenticity.pipeline import Pipeline

CATALOG = "image_authenticity"
HEATMAP_DIR = f"/Volumes/{CATALOG}/results/heatmaps"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Model catalog — the menu behind each functionality widget
# MAGIC Maps every functionality to its arm, composite weight, the ground-truth
# MAGIC class it is tested against (`target`), and the selectable models. Add a
# MAGIC model by registering a `Detector` for the slot and adding a line here.

# COMMAND ----------
FUNCTIONALITIES = {
    "spectral": {
        "label": "Deepfake · Spectral analysis", "arm": "deepfake", "weight": 0.15,
        "target": "deepfake",
        "models": {
            "FFT radial power spectrum": ("fft_radial_v1", {}),
            "DCT high-frequency energy": ("dct_stats_v1", {}),
        }},
    "ela_global": {
        "label": "Deepfake · ELA uniformity", "arm": "deepfake", "weight": 0.15,
        "target": "deepfake",
        "models": {"ELA uniformity (q90)": ("ela_uniformity_v1", {"quality": 90})}},
    "learned": {
        "label": "Deepfake · Learned (vision LLM)", "arm": "deepfake", "weight": 0.70,
        "target": "deepfake",
        "models": {  # first entry is the widget default; Llama won the A/B
            "Llama-4 Maverick (VLM)":  ("vlm_zeroshot_v1", {"endpoint": "databricks-llama-4-maverick"}),
            "Claude Sonnet 5 (VLM)":   ("vlm_zeroshot_v1", {"endpoint": "databricks-claude-sonnet-5"}),
            "GPT-5 (VLM)":             ("vlm_zeroshot_v1", {"endpoint": "databricks-gpt-5"}),
            "Gemini 3.1 Pro (VLM)":    ("vlm_zeroshot_v1", {"endpoint": "databricks-gemini-3-1-pro"}),
        }},
    "ela_local": {
        "label": "Shallowfake · ELA heatmap", "arm": "shallow", "weight": 0.30,
        "target": "shallow_fake",
        "models": {"ELA localized hotspots (q90)": ("ela_heatmap_v1", {"quality": 90})}},
    "copy_move": {
        "label": "Shallowfake · Copy-move", "arm": "shallow", "weight": 0.25,
        "target": "shallow_fake",
        "models": {
            "ORB keypoints": ("orb_v1", {}),
            "SIFT keypoints": ("sift_v1", {}),
        }},
    "noise": {
        "label": "Shallowfake · Noise / splice", "arm": "shallow", "weight": 0.25,
        "target": "shallow_fake",
        "models": {"Noise-residual inconsistency": ("noise_residual_v1", {})}},
    "jpeg_ghost": {
        "label": "Shallowfake · JPEG ghost", "arm": "shallow", "weight": 0.20,
        "target": "shallow_fake",
        "models": {"JPEG ghost (quality sweep)": ("jpeg_ghost_v1", {})}},
    "watermark": {
        "label": "Provenance · Watermark / C2PA", "arm": "watermark", "weight": 1.00,
        "target": "deepfake",
        "models": {"C2PA + IPTC + SynthID stub": ("c2pa_synthid_v1", {})}},
}
OFF = "(off — skip this functionality)"

print("Models registered in the package, by functionality (slot):")
for slot, impls in available().items():
    print(f"  {slot:12s} -> {impls}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Functionality widgets (also driven by job parameters)
# MAGIC Creates one dropdown per functionality. These same names are the job's
# MAGIC **base parameters** — a headless job run can override any of them without
# MAGIC touching the UI. We do NOT call `removeAll()` here, because that would wipe
# MAGIC parameters passed by a job.

# COMMAND ----------
for slot, meta in FUNCTIONALITIES.items():
    choices = list(meta["models"].keys()) + [OFF]
    dbutils.widgets.dropdown(slot, list(meta["models"].keys())[0], choices, meta["label"])
print("Widgets ready. Interactive: use the bar above. Headless: pass job parameters.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Build the pipeline from the current selections
# MAGIC A selection may arrive from a widget OR a job parameter, and may be given
# MAGIC as the model's display label, its impl name, or (for the VLM) its endpoint
# MAGIC — `resolve_model` accepts any of those, or `off`/`(off)` to skip.

# COMMAND ----------
def resolve_model(slot, choice, meta):
    c = (choice or "").strip()
    if c.lower() in ("off", "(off)", "none", "", OFF.lower()):
        return None
    if c in meta["models"]:                       # exact display label
        return meta["models"][c]
    for impl, params in meta["models"].values():  # impl name, or VLM endpoint
        if c == impl or c == params.get("endpoint"):
            return (impl, params)
    raise ValueError(
        f"Unknown model {choice!r} for functionality {slot!r}. "
        f"Options: {list(meta['models']) + ['off']}")


cfg = {"slots": {}, "composite": get_config()["composite"]}
selection = []
for slot, meta in FUNCTIONALITIES.items():
    choice = dbutils.widgets.get(slot)
    resolved = resolve_model(slot, choice, meta)
    if resolved is None:
        selection.append((meta["label"], "— off —"))
        continue
    impl, params = resolved
    cfg["slots"][slot] = {"impl": impl, "arm": meta["arm"],
                          "weight": meta["weight"], "params": params}
    selection.append((meta["label"], f"{choice}  [{impl}]"))

print("Active selection (functionality -> model):")
for f, m in selection:
    print(f"  {f:34s} : {m}")
if not cfg["slots"]:
    raise ValueError("All functionalities are off — enable at least one.")
pipe = Pipeline(cfg)

# COMMAND ----------
# MAGIC %md ## Run the selected pipeline over the test images

# COMMAND ----------
manifest = spark.table(f"{CATALOG}.raw.sample_manifest").collect()
detection_rows, score_rows = [], []
for m in manifest:
    with open(m.path, "rb") as f:
        img_bytes = f.read()
    out = pipe.run(img_bytes, image_id=m.image_id,
                   context={"claim_context": m.prompt}, heatmap_dir=HEATMAP_DIR)
    comp = out["composite"]
    for r in out["results"]:
        row = r.as_row(); row.update({"image_id": m.image_id, "true_label": m.label, "path": m.path})
        detection_rows.append(row)
    score_rows.append({
        "image_id": m.image_id, "true_label": m.label, "path": m.path,
        "fraud_score": comp["fraud_score"], "deepfake_confidence": comp["deepfake_confidence"],
        "shallowfake_confidence": comp["shallowfake_confidence"], "watermark_signal": comp["watermark_signal"],
        "verdict": comp["verdict"], "narrative": comp["narrative"],
        "fired_signals_json": json.dumps(comp["fired_signals"]),
    })
print(f"scored {len(score_rows)} images with the selected models")

# COMMAND ----------
# MAGIC %md ## Persist to Delta (pandas-free — explicit Spark schemas)

# COMMAND ----------
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType)

DET_SCHEMA = StructType([
    StructField("detector", StringType()), StructField("slot", StringType()),
    StructField("score", DoubleType()), StructField("label", StringType()),
    StructField("evidence_json", StringType()), StructField("heatmap_path", StringType()),
    StructField("error", StringType()), StructField("image_id", StringType()),
    StructField("true_label", StringType()), StructField("path", StringType()),
])
SCORE_SCHEMA = StructType([
    StructField("image_id", StringType()), StructField("true_label", StringType()),
    StructField("path", StringType()), StructField("fraud_score", DoubleType()),
    StructField("deepfake_confidence", DoubleType()), StructField("shallowfake_confidence", DoubleType()),
    StructField("watermark_signal", DoubleType()), StructField("verdict", StringType()),
    StructField("narrative", StringType()), StructField("fired_signals_json", StringType()),
])

def _tuples(rows, schema):
    fields = [f.name for f in schema.fields]
    return [tuple(r.get(f) for f in fields) for r in rows]

(spark.createDataFrame(_tuples(detection_rows, DET_SCHEMA), DET_SCHEMA)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.results.detections"))
(spark.createDataFrame(_tuples(score_rows, SCORE_SCHEMA), SCORE_SCHEMA)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.results.scores"))
print("wrote image_authenticity.results.detections and .scores")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Per-functionality evaluation
# MAGIC For each active functionality, its **test set** is the authentic images
# MAGIC (negatives) plus the fake class it targets (positives). `separation` =
# MAGIC mean score on the target class minus mean score on authentic — **higher is a
# MAGIC better discriminator**. Switch a model in the widget, re-run, and compare.

# COMMAND ----------
# Aggregate mean score per (slot, true_label) in plain Python (no pandas).
bucket = defaultdict(lambda: defaultdict(list))
errs = defaultdict(int)
for r in detection_rows:
    if r.get("error"):
        errs[r["slot"]] += 1
    if r.get("score") is not None:
        bucket[r["slot"]][r["true_label"]].append(r["score"])

def _mean(xs):
    return round(sum(xs) / len(xs), 3) if xs else None

eval_rows = []
for slot, meta in FUNCTIONALITIES.items():
    if slot not in cfg["slots"]:
        continue
    by_lbl = bucket[slot]
    neg, tgt = _mean(by_lbl.get("authentic", [])), _mean(by_lbl.get(meta["target"], []))
    sep = round(tgt - neg, 3) if (neg is not None and tgt is not None) else None
    eval_rows.append({
        "functionality": meta["label"], "model": dbutils.widgets.get(slot),
        "target_class": meta["target"], "mean_authentic": neg,
        "mean_deepfake": _mean(by_lbl.get("deepfake", [])),
        "mean_shallow_fake": _mean(by_lbl.get("shallow_fake", [])),
        "separation_vs_authentic": sep, "errors": errs[slot],
    })
eval_rows.sort(key=lambda r: (r["separation_vs_authentic"] is None,
                              -(r["separation_vs_authentic"] or 0)))

print("Per-functionality discrimination with the currently selected models:")
hdr = ["functionality", "model", "target_class", "mean_authentic",
       "mean_deepfake", "mean_shallow_fake", "separation_vs_authentic", "errors"]
print(" | ".join(hdr))
for r in eval_rows:
    print(" | ".join(str(r[h]) for h in hdr))

# Tag every eval row with run metadata so runs accumulate for comparison.
import uuid, datetime
RUN_TS = datetime.datetime.utcnow().isoformat()
try:
    _ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    _rid = _ctx.currentRunId()
    RUN_ID = _rid.get().toString() if _rid.isDefined() else str(uuid.uuid4())
except Exception:
    RUN_ID = str(uuid.uuid4())
# Signature of the whole configuration (functionality=model; ...) for grouping.
SELECTION_SIG = "; ".join(f"{s}={dbutils.widgets.get(s)}" for s in cfg["slots"])

MODEL_EVAL_SCHEMA = StructType([
    StructField("run_ts", StringType()), StructField("run_id", StringType()),
    StructField("functionality", StringType()), StructField("model", StringType()),
    StructField("target_class", StringType()), StructField("mean_authentic", DoubleType()),
    StructField("mean_deepfake", DoubleType()), StructField("mean_shallow_fake", DoubleType()),
    StructField("separation_vs_authentic", DoubleType()), StructField("errors", IntegerType()),
    StructField("selection_signature", StringType()),
])
me_fields = [f.name for f in MODEL_EVAL_SCHEMA.fields]
me_tuples = []
for r in eval_rows:
    rr = dict(r); rr.update({"run_ts": RUN_TS, "run_id": RUN_ID, "selection_signature": SELECTION_SIG})
    me_tuples.append(tuple(rr.get(f) for f in me_fields))

# Append so model/version comparisons across runs are queryable side by side.
(spark.createDataFrame(me_tuples, MODEL_EVAL_SCHEMA)
    .write.mode("append").option("mergeSchema", "true")
    .saveAsTable(f"{CATALOG}.results.model_eval"))
print(f"appended {len(me_tuples)} rows to {CATALOG}.results.model_eval (run_id={RUN_ID})")

print("\nThis run:")
display(spark.table(f"{CATALOG}.results.model_eval").where(f"run_id = '{RUN_ID}'"))

# COMMAND ----------
# MAGIC %md ## Composite outcome by true label (overall pipeline)

# COMMAND ----------
comp_bucket = defaultdict(lambda: defaultdict(list))
verdicts = defaultdict(lambda: defaultdict(int))
for r in score_rows:
    verdicts[r["true_label"]][r["verdict"]] += 1
    for k in ("fraud_score", "deepfake_confidence", "shallowfake_confidence"):
        if r.get(k) is not None:
            comp_bucket[r["true_label"]][k].append(r[k])

print("Mean composite scores by true label:")
for lbl in sorted(comp_bucket):
    vals = {k: _mean(v) for k, v in comp_bucket[lbl].items()}
    print(f"  {lbl:13s} {vals}")
print("\nVerdicts by true label:")
for lbl in sorted(verdicts):
    print(f"  {lbl:13s} {dict(verdicts[lbl])}")

display(spark.table(f"{CATALOG}.results.scores")
        .select("image_id", "true_label", "verdict", "fraud_score", "narrative"))

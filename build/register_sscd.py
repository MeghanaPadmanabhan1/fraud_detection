"""Register SSCD (Meta, CVPR 2022, MIT) as a Unity Catalog model so it can sit behind a Model Serving endpoint.

Request: {"dataframe_records": [{"image_b64": "<base64 PNG/JPEG>"}]}  ->  {"predictions": [[512 floats]]}
The model does the resize (short edge 288, aspect kept, INTER_AREA) and ImageNet normalisation itself, so any
client sending an encoded image gets the same vector as image_authenticity.workbench.checks.sscd_embedding.

    DATABRICKS_CONFIG_PROFILE=e2 ml_env/bin/python -I build/register_sscd.py
"""
import base64
import os
import sys

import cv2
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "wb_data", "ml_models", "sscd", "sscd_disc_mixup.torchscript.pt")
NAME = "image_authenticity.workbench.sscd"


class SSCD(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        torch.set_num_threads(2)
        self.torch = torch
        self.net = torch.jit.load(context.artifacts["weights"], map_location="cpu").eval()

    def predict(self, context, model_input, params=None):
        out = []
        for b64 in model_input["image_b64"]:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            k = 288 / min(h, w)
            rgb = cv2.cvtColor(cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))),
                                          interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = (rgb.astype(np.float32) / 255 - np.float32([0.485, 0.456, 0.406])) / np.float32([0.229, 0.224, 0.225])
            with self.torch.no_grad():
                v = self.net(self.torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float())[0]
                out.append(self.torch.nn.functional.normalize(v, dim=0).numpy().astype(np.float32))
        return np.stack(out)


def _example():
    ok, buf = cv2.imencode(".png", np.random.default_rng(0).integers(0, 255, (288, 384, 3), dtype=np.uint8))
    return pd.DataFrame({"image_b64": [base64.b64encode(buf.tobytes()).decode()]})


if __name__ == "__main__":
    ex = _example()
    m = SSCD()
    ctx = type("C", (), {"artifacts": {"weights": WEIGHTS}})()
    m.load_context(ctx)
    sig = infer_signature(ex, m.predict(ctx, ex))
    if len(sys.argv) > 1 and sys.argv[1] == "--dry-run":
        print("signature:", sig)
        sys.exit(0)
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/meghana.padmanabhan@databricks.com/sscd-copy-detection")
    with mlflow.start_run(run_name="register-sscd-disc-mixup"):
        info = mlflow.pyfunc.log_model(
            name="sscd", python_model=SSCD(), artifacts={"weights": WEIGHTS}, signature=sig, input_example=ex,
            registered_model_name=NAME,
            pip_requirements=["torch==2.14.0", "opencv-python-headless<5", "numpy<2.3", "pandas", "mlflow==3.16.0"])
    print("registered", NAME, info.model_uri)

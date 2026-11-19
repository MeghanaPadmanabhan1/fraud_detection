"""Deepfake arm (learned) -- vision-LLM zero-shot authenticity + context check.

Uses a Databricks Foundation Model API vision endpoint (Claude / Gemini / GPT /
Llama-4) as a zero-shot detector. This is the "foundation model serving" story
for Databricks, and it simultaneously covers Mandela's "contextual analysis"
and "image recognition" asks: the same call can read the claimed device model,
flag physically implausible damage, and compare against claim context.

No GPU / torch required -- it's an API call. Swap in a CLIP/DIRE/UnivFD linear
probe for this slot when a trained model is available.
"""
from __future__ import annotations

import base64
import json

import cv2

from ..core import (
    LABEL_AI,
    LABEL_AUTHENTIC,
    LABEL_INDETERMINATE,
    LABEL_MANIPULATED,
    Detector,
    register,
)

_PROMPT = """You are a forensic image analyst for an insurance claims fraud team.
Analyze the attached claim photo for signs that it is NOT an authentic,
unedited camera photo of a real device.

Assess, independently:
1. ai_generated: is the whole image synthetic (diffusion/GAN/Gemini/Midjourney)?
   Look for too-perfect textures, impossible reflections, warped text/logos,
   nonsensical fine detail, over-smooth noise.
2. manipulated: is a real photo edited (spliced/cloned/painted damage,
   removed serial)? Look for inconsistent lighting/shadows, blending seams,
   resolution/noise mismatches between regions.
3. image recognition / context: what device is shown (brand + model guess)?
   Any physically implausible or internally inconsistent details?

Respond with ONLY a compact JSON object, no prose:
{"ai_generated_prob": 0.0-1.0, "manipulated_prob": 0.0-1.0,
 "device_guess": "...", "anomalies": ["..."], "rationale": "one sentence"}"""


@register
class VLMZeroShotDetector(Detector):
    slot = "learned"
    name = "vlm_zeroshot_v1"
    version = "1.0"

    def analyze(self, image_bgr, image_bytes=None, context=None):
        try:
            from ..fmapi import fmapi_chat, first_message_content

            endpoint = self.config.get("endpoint", "databricks-claude-sonnet-5")
            # Re-encode to a modest JPEG to keep the payload small.
            ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf.tobytes()).decode()
            data_uri = f"data:image/jpeg;base64,{b64}"

            claim_ctx = ""
            if context and context.get("claim_context"):
                claim_ctx = f"\n\nClaim context: {context['claim_context']}"

            resp = fmapi_chat(
                endpoint,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _PROMPT + claim_ctx},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                max_tokens=400,
                # NB: some endpoints (e.g. claude-sonnet-5) reject `temperature`;
                # omit it and let the detector stay portable across models.
                temperature=self.config.get("temperature"),
            )
            raw = first_message_content(resp)
            parsed = _extract_json(raw)
            ai_p = float(parsed.get("ai_generated_prob", 0.0))
            man_p = float(parsed.get("manipulated_prob", 0.0))
            score = max(ai_p, man_p)
            if score < 0.5:
                label = LABEL_AUTHENTIC
            elif ai_p >= man_p:
                label = LABEL_AI
            else:
                label = LABEL_MANIPULATED

            return self.result(
                float(score),
                label,
                evidence={
                    "endpoint": endpoint,
                    "ai_generated_prob": round(ai_p, 3),
                    "manipulated_prob": round(man_p, 3),
                    "device_guess": parsed.get("device_guess"),
                    "anomalies": parsed.get("anomalies", []),
                    "rationale": parsed.get("rationale"),
                },
            )
        except Exception as e:  # pragma: no cover
            return self.result(None, LABEL_INDETERMINATE, error=repr(e))


def _extract_json(text: str) -> dict:
    if not text:
        return {}
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):]
    try:
        return json.loads(s)
    except Exception:
        a, b = s.find("{"), s.rfind("}")
        if a >= 0 and b > a:
            try:
                return json.loads(s[a : b + 1])
            except Exception:
                return {"rationale": text[:200]}
        return {"rationale": text[:200]}

"""Slow-path model calls (vision detector, receipt OCR, explainer) via the
Foundation Model API. Shared by the batch cache builder and the live app."""
from __future__ import annotations

import base64
import json
import concurrent.futures as cf
import time

import cv2

from ..detectors.vlm import _PROMPT, _extract_json
from ..fmapi import first_message_content, fmapi_chat
from .checks import parse_receipt_text

OCR_PROMPT = ("Read this purchase receipt. Respond with ONLY compact JSON: "
              '{"store": "...", "date": "YYYY-MM-DD", "item": "the product line exactly as printed", '
              '"total": 0.00}. Use null for anything you cannot read.')


def _data_uri(img_bgr, side=1024, q=85):
    h, w = img_bgr.shape[:2]
    if max(h, w) > side:
        f = side / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _tokens(resp):
    u = resp.get("usage") or {}
    return int(u.get("total_tokens") or (u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)) or 0)


# ----------------------------------------------------------------- SSCD behind Model Serving
_SSCD_POOL = cf.ThreadPoolExecutor(2)
_SSCD_CLIENT = []
_SSCD_DOWN_UNTIL = 0.0       # circuit breaker: after a timeout/error, skip the endpoint briefly instead of stalling claims


def sscd_remote(endpoint, img_bgr, timeout=8.0, cooldown=60.0):
    """SSCD descriptor from a Model Serving endpoint, or None if it is cold/unreachable (caller falls back).
    The client does the resize (short edge 288, INTER_AREA) and sends a lossless PNG, which is ~140 KB and
    makes the endpoint's own resize a no-op, so the vector equals the in-process one (cosine 1.0000)."""
    global _SSCD_DOWN_UNTIL
    import numpy as np
    if time.time() < _SSCD_DOWN_UNTIL:
        return None
    h, w = img_bgr.shape[:2]
    k = 288 / min(h, w)
    small = cv2.resize(img_bgr, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    body = {"dataframe_records": [{"image_b64": base64.b64encode(cv2.imencode(".png", small)[1].tobytes()).decode()}]}

    def call():
        if not _SSCD_CLIENT:                         # building the SDK client costs ~1.2 s; do it once
            from databricks.sdk import WorkspaceClient
            _SSCD_CLIENT.append(WorkspaceClient())
        return _SSCD_CLIENT[0].api_client.do("POST", f"/serving-endpoints/{endpoint}/invocations", body=body)

    try:
        v = np.array(_SSCD_POOL.submit(call).result(timeout=timeout)["predictions"][0], np.float32)
        return v / (np.linalg.norm(v) + 1e-9)
    except Exception:
        _SSCD_DOWN_UNTIL = time.time() + cooldown
        return None


def vlm_detect(endpoint, img_bgr, claim=None):
    ctx = ""
    if claim:
        ctx = (f"\n\nClaim context: device claimed = {claim['device_claimed']}; "
               f"damage described = {claim['damage_described']}.")
    t0 = time.perf_counter()
    try:
        resp = fmapi_chat(endpoint, [{"role": "user", "content": [
            {"type": "text", "text": _PROMPT + ctx},
            {"type": "image_url", "image_url": {"url": _data_uri(img_bgr)}}]}], max_tokens=500)
        parsed = _extract_json(first_message_content(resp))
        return {"ai_generated_prob": float(parsed.get("ai_generated_prob") or 0),
                "manipulated_prob": float(parsed.get("manipulated_prob") or 0),
                "device_guess": parsed.get("device_guess"), "anomalies": parsed.get("anomalies") or [],
                "rationale": parsed.get("rationale"), "latency_ms": (time.perf_counter() - t0) * 1000,
                "tokens": _tokens(resp), "error": None}
    except Exception as e:
        return {"error": repr(e)[:300], "latency_ms": (time.perf_counter() - t0) * 1000, "tokens": 0}


def vlm_ocr(endpoint, receipt_bytes):
    img = cv2.imdecode(__import__("numpy").frombuffer(receipt_bytes, "uint8"), cv2.IMREAD_COLOR)
    t0 = time.perf_counter()
    try:
        resp = fmapi_chat(endpoint, [{"role": "user", "content": [
            {"type": "text", "text": OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": _data_uri(img, side=1200, q=90)}}]}], max_tokens=300)
        p = _extract_json(first_message_content(resp))
        total = p.get("total")
        if isinstance(total, str):
            total = float(total.replace("$", "").replace(",", "")) if total.strip() else None
        return {"fields": {"total": total, "date": p.get("date"), "item": p.get("item")},
                "latency_ms": (time.perf_counter() - t0) * 1000, "tokens": _tokens(resp), "error": None}
    except Exception as e:
        return {"error": repr(e)[:300], "latency_ms": (time.perf_counter() - t0) * 1000, "tokens": 0}


def parsed_document_fields(parsed_variant_json: str):
    """Fields from a Databricks ai_parse_document result (JSON string of the VARIANT)."""
    try:
        d = json.loads(parsed_variant_json) if isinstance(parsed_variant_json, str) else parsed_variant_json
        els = (d.get("document") or {}).get("elements") or []
        text = "\n".join(e.get("content", "") for e in els)
    except Exception:
        text = str(parsed_variant_json)
    return parse_receipt_text(text), text


def explain(endpoint, prompt):
    t0 = time.perf_counter()
    try:
        resp = fmapi_chat(endpoint, [{"role": "user", "content": prompt}], max_tokens=250)
        return {"text": first_message_content(resp), "latency_ms": (time.perf_counter() - t0) * 1000,
                "tokens": _tokens(resp)}
    except Exception as e:
        return {"text": None, "error": repr(e)[:300], "latency_ms": (time.perf_counter() - t0) * 1000, "tokens": 0}

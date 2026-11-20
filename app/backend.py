"""Data access for the Fraud Defence Workbench app.

SQL via the Statement Execution API (SQL warehouse resource), media via the Files
API (UC volume), slow-path model calls via Foundation Model API. The app's service
principal holds all credentials; nothing reaches the browser.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient

import lakebase
from image_authenticity.workbench import checks, models
from image_authenticity.workbench.engine import EngineContext

SCHEMA = os.getenv("WB_SCHEMA", "image_authenticity.workbench")
LAKEFLOW = os.getenv("WB_LAKEFLOW_SCHEMA", "image_authenticity.lakeflow")   # gold tables built by the Lakeflow pipeline
WAREHOUSE = os.getenv("DATABRICKS_WAREHOUSE_ID")


@st.cache_resource
def client() -> WorkspaceClient:
    return WorkspaceClient()


def sql(statement: str, params: dict | None = None) -> pd.DataFrame:
    from databricks.sdk.service.sql import StatementParameterListItem
    w = client()
    p = [StatementParameterListItem(name=k, value=None if v is None else str(v)) for k, v in (params or {}).items()]
    r = w.statement_execution.execute_statement(statement=statement, warehouse_id=WAREHOUSE,
                                                wait_timeout="50s", parameters=p or None)
    while r.status.state.value in ("PENDING", "RUNNING"):
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"SQL failed: {r.status.error.message if r.status.error else r.status.state}")
    if not r.manifest or not r.manifest.schema or not r.manifest.schema.columns:
        return pd.DataFrame()
    cols = [c.name for c in r.manifest.schema.columns]
    rows = list(r.result.data_array or []) if r.result else []
    chunk = r.result.next_chunk_index if r.result else None
    while chunk is not None:
        nxt = w.statement_execution.get_statement_result_chunk_n(r.statement_id, chunk)
        rows += nxt.data_array or []
        chunk = nxt.next_chunk_index
    df = pd.DataFrame(rows, columns=cols)
    for c, meta in zip(cols, r.manifest.schema.columns):
        t = (meta.type_name.value if meta.type_name else "").upper()
        if t in ("DOUBLE", "FLOAT", "DECIMAL", "INT", "LONG", "BIGINT", "SHORT"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif t == "BOOLEAN":
            df[c] = df[c].map({"true": True, "false": False})
    return df


def T(name):
    return f"{SCHEMA}.{name}"


# ------------------------------------------------------------------ cached reads
@st.cache_data(ttl=600, show_spinner=False)
def claims() -> pd.DataFrame:
    df = sql(f"SELECT * FROM {T('claims')} ORDER BY claim_id")
    df["claim_value"] = pd.to_numeric(df["claim_value"])
    return df


@st.cache_data(ttl=300, show_spinner=False)
def runs() -> pd.DataFrame:
    return sql(f"SELECT run_id, config_name, claim_id, verdict, confidence, families, route, explanation, "
               f"sync_latency_ms, async_latency_ms, tokens, run_ts FROM {T('runs')}")


@st.cache_data(ttl=300, show_spinner=False)
def stage_results() -> pd.DataFrame:
    return sql(f"SELECT config_name, claim_id, sid, stage_num, stage_name, techs, status, score, severity, "
               f"latency_ms, tokens FROM {T('stage_results')}")


@st.cache_data(ttl=300, show_spinner=False)
def tech_scores() -> pd.DataFrame:
    return sql(f"SELECT * FROM {T('tech_scores')}")


@st.cache_data(ttl=60, show_spinner=False)
def labels() -> pd.DataFrame:
    return sql(f"SELECT * FROM {T('labels')} ORDER BY labeled_at DESC")


@st.cache_data(ttl=60, show_spinner=False)
def audit() -> pd.DataFrame:
    return sql(f"SELECT * FROM {T('audit_log')} ORDER BY ts DESC")


@st.cache_data(ttl=300, show_spinner=False)
def governance() -> pd.DataFrame:
    return sql(f"SELECT * FROM {T('governance_items')} ORDER BY due_date")


@st.cache_data(ttl=60, show_spinner=False)
def saved_configs() -> pd.DataFrame:
    return sql(f"SELECT * FROM {T('saved_configs')} ORDER BY created_at DESC")


def _err(v):
    """Cached error column: None/NaN/'' all mean 'no error'."""
    return v if isinstance(v, str) and v.strip() else None


@st.cache_data(ttl=3600, show_spinner=False)
def _vlm_cache() -> dict:
    df = sql(f"SELECT claim_id, endpoint, ai_generated_prob, manipulated_prob, device_guess, rationale, "
             f"latency_ms, tokens, error FROM {T('vlm_cache')}")
    out = {}
    for r in df.itertuples(index=False):
        d = r._asdict()
        d["error"] = _err(d.get("error"))
        for k in ("latency_ms", "tokens", "ai_generated_prob", "manipulated_prob"):
            d[k] = 0.0 if d.get(k) != d.get(k) or d.get(k) is None else float(d[k])
        d["tokens"] = int(d["tokens"])
        out[(r.claim_id, r.endpoint)] = {**d, "cached": True}
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _ocr_cache() -> dict:
    df = sql(f"SELECT claim_id, engine, fields, latency_ms, tokens, error FROM {T('ocr_cache')}")
    out = {}
    for r in df.itertuples(index=False):
        out[(r.claim_id, r.engine)] = {"fields": json.loads(r.fields) if r.fields else None,
                                       "latency_ms": float(r.latency_ms or 0), "tokens": int(r.tokens or 0),
                                       "error": _err(r.error), "cached": True}
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def _explain_cache() -> dict:
    df = sql(f"SELECT claim_id, endpoint, prompt, text, latency_ms, tokens FROM {T('explain_cache')}")
    return {(r.claim_id, r.endpoint, r.prompt): {"text": r.text, "latency_ms": float(r.latency_ms or 0),
                                                 "tokens": int(r.tokens or 0), "cached": True}
            for r in df.itertuples(index=False)}


# ------------------------------------------------------------------ media
@st.cache_resource(show_spinner=False)
def _media_store() -> dict:
    return {}


def media(path: str) -> bytes:
    store = _media_store()
    if path not in store:
        store[path] = client().files.download(path).contents.read()
    return store[path]


def image(path: str):
    return cv2.imdecode(np.frombuffer(media(path), np.uint8), cv2.IMREAD_COLOR)


def preload_media(paths):
    todo = [p for p in paths if isinstance(p, str) and p and p not in _media_store()]
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(media, todo))


# ------------------------------------------------------------------ engine context
@st.cache_resource(show_spinner=False)
def reuse_index():
    df = sql(f"SELECT claim_id, claimant_id, claim_date, image_path, phash, dhash, embedding, indexed_at "
             f"FROM {T('hash_index')}")
    try:                                   # SSCD vectors live in their own table; the index works without them
        sv = {r.claim_id: np.array(json.loads(r.sscd), np.float32)
              for r in sql(f"SELECT claim_id, sscd FROM {T('sscd_index')}").itertuples(index=False)}
    except Exception:
        sv = {}
    idx = checks.ReuseIndex(built_at=dt.datetime.now())
    for r in df.itertuples(index=False):
        idx.entries.append(checks.IndexEntry(r.claim_id, r.claimant_id, str(r.claim_date), r.image_path,
                                             int(r.phash), int(r.dhash), np.array(json.loads(r.embedding), np.float32),
                                             sscd=sv.get(r.claim_id)))
    return idx


def rebuild_index():
    """Recompute hashes from the claims table + volume (index freshness = table freshness)."""
    c = claims()
    preload_media(c["image_path"].tolist())
    idx = reuse_index()
    old = {e.claim_id: e.sscd for e in idx.entries}      # keep stored SSCD vectors when torch is not installed
    idx.entries = [checks.make_entry(r._asdict(), image(r.image_path), with_sscd=True) for r in c.itertuples(index=False)]
    for e, r in zip(idx.entries, c.itertuples(index=False)):   # vectors: in-process, else endpoint, else keep stored
        e.sscd = e.sscd if e.sscd is not None else (sscd_embed(image(r.image_path)) if SSCD_ENDPOINT else None)
        e.sscd = e.sscd if e.sscd is not None else old.get(e.claim_id)
    idx.built_at = dt.datetime.now()
    return idx


@st.cache_resource(show_spinner=False)
def _live_store() -> dict:
    return {"vlm": {}, "ocr": {}, "explain": {}}


SSCD_ENDPOINT = os.getenv("SSCD_ENDPOINT")


def sscd_embed(img):
    """SSCD vector from the Model Serving endpoint (None if not configured / cold / unreachable)."""
    return models.sscd_remote(SSCD_ENDPOINT, img) if SSCD_ENDPOINT else None


def context(live_models: bool) -> EngineContext:
    vc, oc, ec, live = _vlm_cache(), _ocr_cache(), _explain_cache(), _live_store()

    def vlm(cid, ep, img):
        k = (cid, ep)
        if k in vc and not vc[k].get("error"):
            return vc[k]
        if k in live["vlm"]:
            return live["vlm"][k]
        if not live_models:
            return None
        claim = claims().set_index("claim_id").loc[cid].to_dict()
        claim["claim_id"] = cid
        live["vlm"][k] = r = dict(models.vlm_detect(ep, img, claim), cached=False)
        return r

    def ocr(cid, engine, receipt):
        k = (cid, engine)
        if k in oc and not oc[k].get("error"):
            return oc[k]
        if k in live["ocr"]:
            return live["ocr"][k]
        if not live_models or not engine.startswith("vlm_ocr:"):
            return None
        live["ocr"][k] = r = dict(models.vlm_ocr(engine.split(":", 1)[1], receipt), cached=False)
        return r

    def explain(ep, prompt, cid):
        k = (cid, ep, prompt)
        if k in ec:
            return ec[k]
        if k in live["explain"]:
            return live["explain"][k]
        if not live_models:
            return None
        live["explain"][k] = r = dict(models.explain(ep, prompt), cached=False)
        return r

    return EngineContext(index=reuse_index(), index_loader=lambda e: image(e.image_path),
                         vlm=vlm, ocr=ocr, explain=explain, sscd=sscd_embed)


def upload_context(claim: dict, live_models: bool = True) -> EngineContext:
    """Engine context for a photo someone just uploaded. Model answers about the upload are kept in THIS session
    only (never in the shared cache above), keyed by the upload's content hash. Nothing is written anywhere."""
    ss = st.session_state.setdefault("_upload_live", {"vlm": {}, "ocr": {}})

    def vlm(cid, ep, img):
        k = (cid, ep)
        if k not in ss["vlm"]:
            if not live_models:
                return None
            r = dict(models.vlm_detect(ep, img, claim), cached=False)
            if r.get("error"):
                return r                                    # do not remember failures; the next check retries
            ss["vlm"][k] = r
        return ss["vlm"][k]

    def ocr(cid, engine, receipt):
        k = (cid, engine)
        if k not in ss["ocr"]:
            if not live_models or not engine.startswith("vlm_ocr:"):
                return None
            r = dict(models.vlm_ocr(engine.split(":", 1)[1], receipt), cached=False)
            if r.get("error"):
                return r
            ss["ocr"][k] = r
        return ss["ocr"][k]

    return EngineContext(index=reuse_index(), index_loader=lambda e: image(e.image_path), vlm=vlm, ocr=ocr,
                         explain=None, sscd=sscd_embed)


# ------------------------------------------------------------------ writes
def current_user() -> str:
    try:
        h = st.context.headers
        return h.get("X-Forwarded-Email") or h.get("X-Forwarded-Preferred-Username") or "local-demo-user"
    except Exception:
        return "local-demo-user"


def log(action: str, target: str, details: str):
    sql(f"INSERT INTO {T('audit_log')} VALUES (current_timestamp(), :actor, :action, :target, :details)",
        {"actor": current_user(), "action": action, "target": target, "details": details[:2000]})
    audit.clear()
    lakebase.add_audit(current_user(), action, target, details)       # operational copy, best effort


def add_label(claim_id, label, reason, note, model_verdict, config_name):
    sql(f"INSERT INTO {T('labels')} VALUES (:c, :l, :r, :n, :u, :v, :cfg, current_timestamp())",
        {"c": claim_id, "l": label, "r": reason, "n": note, "u": current_user(), "v": model_verdict,
         "cfg": config_name})
    lakebase.add_decision(claim_id, label, reason, note, current_user(), model_verdict, config_name)  # Lakebase, best effort
    log("label", claim_id, f"{label} / {reason}" + (f" — {note}" if note else ""))
    labels.clear()


def save_config(name, cfg, notes):
    sql(f"INSERT INTO {T('saved_configs')} VALUES (:n, :j, :notes, :u, current_timestamp())",
        {"n": name, "j": json.dumps(cfg), "notes": notes, "u": current_user()})
    log("save_config", name, notes or "")
    saved_configs.clear()

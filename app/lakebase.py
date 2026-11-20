"""Lakebase (managed Postgres) for the operational side of the workbench.

Delta + the SQL warehouse hold the analytics tables. Small, frequent, transactional writes (a reviewer's decision,
an audit entry) go to Lakebase, which answers in milliseconds. Every call is best-effort: if Lakebase is not
reachable the caller carries on with the Delta write, and a 60 s circuit breaker stops repeated slow failures.
The database credential is fetched in memory per connection and never stored.
"""
from __future__ import annotations

import os
import time
import uuid

INSTANCE = os.getenv("LAKEBASE_INSTANCE", "")
DB = os.getenv("LAKEBASE_DB", "fraud_ops")
HOST = os.getenv("LAKEBASE_HOST", "")
_DOWN_UNTIL = 0.0
LAST_ERROR = ""


def enabled() -> bool:
    return bool(INSTANCE)


def _connect():
    import psycopg
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    host = HOST or w.database.get_database_instance(name=INSTANCE).read_write_dns
    cred = w.database.generate_database_credential(request_id=str(uuid.uuid4()), instance_names=[INSTANCE])
    user = w.config.client_id or w.current_user.me().user_name
    return psycopg.connect(host=host, dbname=DB, user=user, password=cred.token, sslmode="require",
                           connect_timeout=5, autocommit=True)


def run(statement: str, params: tuple = (), fetch: bool = False):
    """Returns (ok, rows). Never raises."""
    global _DOWN_UNTIL, LAST_ERROR
    if not enabled() or time.time() < _DOWN_UNTIL:
        return False, []
    try:
        with _connect() as c, c.cursor() as cur:
            cur.execute(statement, params)
            return True, (cur.fetchall() if fetch else [])
    except Exception as e:                      # noqa: BLE001 - best effort by design
        LAST_ERROR = repr(e)[:200]
        _DOWN_UNTIL = time.time() + 60
        return False, []


def add_decision(claim_id, decision, reason, note, reviewer, model_verdict, config_name) -> bool:
    return run("INSERT INTO ops.review_decisions(claim_id,decision,reason_code,note,reviewer,model_verdict,config_name) "
               "VALUES (%s,%s,%s,%s,%s,%s,%s)",
               (claim_id, decision, reason, note, reviewer, model_verdict, config_name))[0]


def add_audit(actor, action, target, details) -> bool:
    return run("INSERT INTO ops.audit_log(actor,action,target,details) VALUES (%s,%s,%s,%s)",
               (actor, action, target, (details or "")[:2000]))[0]


def recent_decisions(limit: int = 15):
    ok, rows = run("SELECT decided_at, claim_id, decision, reason_code, reviewer, model_verdict "
                   "FROM ops.review_decisions ORDER BY decided_at DESC LIMIT %s", (limit,), fetch=True)
    return ok, rows


def status() -> dict:
    t0 = time.perf_counter()
    ok, rows = run("SELECT (SELECT count(*) FROM ops.review_decisions), (SELECT count(*) FROM ops.audit_log)", fetch=True)
    return {"ok": ok, "ms": round((time.perf_counter() - t0) * 1000), "decisions": rows[0][0] if rows else None,
            "audit": rows[0][1] if rows else None, "error": "" if ok else LAST_ERROR or "not configured"}

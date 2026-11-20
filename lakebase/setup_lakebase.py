# Databricks notebook source
# MAGIC %md
# MAGIC # Lakebase setup and smoke test
# MAGIC Creates the operational tables (review decisions, audit log) in the Lakebase Postgres database `fraud_ops`,
# MAGIC then proves the synced review queue (from the Lakeflow gold table) and the operational tables work.
# MAGIC Runs inside Databricks, because the Postgres endpoint is reachable from the workspace network.

# COMMAND ----------
import json, uuid, psycopg
from databricks.sdk import WorkspaceClient

INSTANCE, DB = "fraud-workbench-db", "fraud_ops"
w = WorkspaceClient()
inst = w.database.get_database_instance(name=INSTANCE)
cred = w.database.generate_database_credential(request_id=str(uuid.uuid4()), instance_names=[INSTANCE])
me = w.current_user.me().user_name
conn = psycopg.connect(host=inst.read_write_dns, dbname=DB, user=me, password=cred.token, sslmode="require")
conn.autocommit = True
cur = conn.cursor()
out = {"connected_as": me, "postgres": None}

# COMMAND ----------
cur.execute("SELECT version()"); out["postgres"] = cur.fetchone()[0][:40]
cur.execute("CREATE SCHEMA IF NOT EXISTS ops")
cur.execute("""CREATE TABLE IF NOT EXISTS ops.review_decisions(
  id BIGSERIAL PRIMARY KEY, claim_id TEXT NOT NULL,
  decision TEXT NOT NULL CHECK (decision IN ('confirm_fraud','confirm_genuine','overturn_needs_more_info')),
  reason_code TEXT, note TEXT, reviewer TEXT NOT NULL, model_verdict TEXT, config_name TEXT,
  decided_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
cur.execute("CREATE INDEX IF NOT EXISTS review_decisions_claim ON ops.review_decisions(claim_id, decided_at DESC)")
cur.execute("""CREATE TABLE IF NOT EXISTS ops.audit_log(
  id BIGSERIAL PRIMARY KEY, ts TIMESTAMPTZ NOT NULL DEFAULT now(), actor TEXT NOT NULL,
  action TEXT NOT NULL, target TEXT, details TEXT)""")

# COMMAND ----------
# Smoke test 1: the Lakeflow gold review queue, synced into Postgres (public.review_queue)
cur.execute("SELECT count(*), min(queue_rank), max(queue_rank) FROM public.review_queue")
out["synced_review_queue_rows_min_max"] = [int(x) for x in cur.fetchone()]
cur.execute("SELECT queue_rank, claim_id, verdict, claim_value FROM public.review_queue ORDER BY queue_rank LIMIT 3")
out["queue_top3"] = [[str(v) for v in r] for r in cur.fetchall()]

# Smoke test 2: a transactional write and read-back, with latency
import time
t0 = time.perf_counter()
cur.execute("INSERT INTO ops.audit_log(actor,action,target,details) VALUES (%s,'lakebase_setup','ops','smoke test') RETURNING id", (me,))
rid = cur.fetchone()[0]
cur.execute("SELECT actor, action FROM ops.audit_log WHERE id = %s", (rid,))
out["write_then_read"] = list(cur.fetchone())
out["write_read_ms"] = round((time.perf_counter() - t0) * 1000, 1)
cur.execute("""SELECT table_schema||'.'||table_name FROM information_schema.tables
               WHERE table_schema IN ('ops','public') ORDER BY 1""")
out["tables"] = [r[0] for r in cur.fetchall()]
print(json.dumps(out, indent=1))
dbutils.notebook.exit(json.dumps(out))

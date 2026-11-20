# Databricks notebook source
# MAGIC %md
# MAGIC # Read back what the app wrote to Lakebase
# MAGIC Proof that reviewer decisions and audit entries made in the app are stored in Postgres (`ops` schema).

# COMMAND ----------
import json, uuid, psycopg
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
inst = w.database.get_database_instance(name="fraud-workbench-db")
cred = w.database.generate_database_credential(request_id=str(uuid.uuid4()), instance_names=["fraud-workbench-db"])
c = psycopg.connect(host=inst.read_write_dns, dbname="fraud_ops", user=w.current_user.me().user_name, password=cred.token, sslmode="require")
cur = c.cursor()
out = {}
cur.execute("SELECT count(*) FROM ops.review_decisions"); out["review_decisions_rows"] = cur.fetchone()[0]
cur.execute("SELECT count(*) FROM ops.audit_log"); out["audit_log_rows"] = cur.fetchone()[0]
cur.execute("SELECT decided_at::text, claim_id, decision, reason_code, model_verdict, config_name FROM ops.review_decisions ORDER BY decided_at DESC LIMIT 5")
out["latest_decisions"] = [list(r) for r in cur.fetchall()]
cur.execute("SELECT ts::text, action, target FROM ops.audit_log ORDER BY ts DESC LIMIT 5")
out["latest_audit"] = [list(r) for r in cur.fetchall()]
print(json.dumps(out, indent=1))
dbutils.notebook.exit(json.dumps(out))

# Databricks notebook source
# MAGIC %md
# MAGIC # Check the app's database access, running AS the app's service principal
# MAGIC Proves the role works and is limited: it can insert and read decisions, and cannot delete or create.

# COMMAND ----------
import json, uuid, psycopg
from databricks.sdk import WorkspaceClient

INSTANCE, DB = "fraud-workbench-db", "fraud_ops"
w = WorkspaceClient()
inst = w.database.get_database_instance(name=INSTANCE)
cred = w.database.generate_database_credential(request_id=str(uuid.uuid4()), instance_names=[INSTANCE])
who = w.config.client_id or w.current_user.me().user_name
c = psycopg.connect(host=inst.read_write_dns, dbname=DB, user=who, password=cred.token, sslmode="require")
c.autocommit = True
cur = c.cursor()
out = {"connected_as": "<app-role>"}
cur.execute("INSERT INTO ops.review_decisions(claim_id,decision,reason_code,note,reviewer,model_verdict,config_name) "
            "VALUES ('SMOKE-TEST','confirm_genuine','other','role check','app-role-check','clear','Production') RETURNING id")
rid = cur.fetchone()[0]
cur.execute("SELECT claim_id, decision FROM ops.review_decisions WHERE id=%s", (rid,))
out["insert_then_read"] = list(cur.fetchone())
def attempt(label, sql):
    try:
        cur.execute(sql); out[label] = "ALLOWED (unexpected)"
    except Exception as e:
        out[label] = "denied: " + type(e).__name__
attempt("delete_decisions", "DELETE FROM ops.review_decisions WHERE claim_id='SMOKE-TEST'")
attempt("create_table", "CREATE TABLE ops.should_not_exist(x int)")
attempt("read_other_schema", "SELECT count(*) FROM public.review_queue")
print(json.dumps(out, indent=1))
dbutils.notebook.exit(json.dumps(out))

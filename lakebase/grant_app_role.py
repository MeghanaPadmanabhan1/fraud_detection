# Databricks notebook source
# MAGIC %md
# MAGIC # Give the app's database role the least access it needs
# MAGIC The app writes reviewer decisions and audit entries to two tables and reads recent decisions back.
# MAGIC It gets exactly that: use of the `ops` schema, insert and select on the two tables, and their id sequences.
# MAGIC No create, update, delete or access to anything else. Run as the data owner.

# COMMAND ----------
import json, uuid, psycopg
from databricks.sdk import WorkspaceClient

dbutils.widgets.text("app_role", "")
APP_ROLE = dbutils.widgets.get("app_role")
assert APP_ROLE, "pass the app service principal client id as app_role"
INSTANCE, DB = "fraud-workbench-db", "fraud_ops"
w = WorkspaceClient()
inst = w.database.get_database_instance(name=INSTANCE)
cred = w.database.generate_database_credential(request_id=str(uuid.uuid4()), instance_names=[INSTANCE])
me = w.current_user.me().user_name
c = psycopg.connect(host=inst.read_write_dns, dbname=DB, user=me, password=cred.token, sslmode="require")
c.autocommit = True
cur = c.cursor()
q = f'"{APP_ROLE}"'
for s in [f"GRANT CONNECT ON DATABASE {DB} TO {q}",
          f"GRANT USAGE ON SCHEMA ops TO {q}",
          f"GRANT SELECT, INSERT ON ops.review_decisions, ops.audit_log TO {q}",
          f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA ops TO {q}"]:
    cur.execute(s)

# COMMAND ----------
cur.execute("""SELECT table_name, string_agg(privilege_type, ',' ORDER BY privilege_type)
               FROM information_schema.role_table_grants WHERE grantee = %s AND table_schema = 'ops' GROUP BY 1 ORDER BY 1""", (APP_ROLE,))
res = {"role": "<app-role>", "table_privileges": cur.fetchall()}
print(json.dumps(res, indent=1))
dbutils.notebook.exit(json.dumps(res))

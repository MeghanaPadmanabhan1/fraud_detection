"""Write the raw claim feed that the Lakeflow pipeline ingests: three JSON-lines batch files.

The 66 synthetic claim records live in the `claims` table (built by build/build_dataset.py from public Wikimedia
Commons photos). This script exports them as three files, as if a claims system dropped them in batches, so the
pipeline's Auto Loader ingest has something real to pick up. Copy the output to <volume>/landing/claims/.

Usage (workspace credentials in the environment, DATABRICKS_WAREHOUSE_ID set):
    python pipeline/make_landing_feed.py <out_dir> [catalog.schema]
"""
import json
import os
import sys

from databricks.sdk import WorkspaceClient

out_dir = sys.argv[1]
schema = sys.argv[2] if len(sys.argv) > 2 else "image_authenticity.workbench"
os.makedirs(out_dir, exist_ok=True)

w = WorkspaceClient()
r = w.statement_execution.execute_statement(
    statement=f"SELECT * EXCEPT(_rescued_data) FROM {schema}.claims ORDER BY claim_id",
    warehouse_id=os.environ["DATABRICKS_WAREHOUSE_ID"], wait_timeout="50s")
cols = [c.name for c in r.manifest.schema.columns]
rows = [dict(zip(cols, v)) for v in (r.result.data_array or [])]
for d in rows:                                    # numbers back to numbers; everything else stays text
    d["claim_value"] = float(d["claim_value"]) if d["claim_value"] is not None else None

for i, (a, b) in enumerate([(0, 30), (30, 52), (52, len(rows))], 1):    # three batches: 30, 22, 14 claims
    with open(os.path.join(out_dir, f"claims_batch_{i}.jsonl"), "w") as f:
        for d in rows[a:b]:
            f.write(json.dumps(d) + "\n")
print(f"{len(rows)} claims written to {out_dir}")

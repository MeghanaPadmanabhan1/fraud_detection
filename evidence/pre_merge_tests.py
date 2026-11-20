# Pre-merge test runner. Set FORBIDDEN_PATTERN (regex) and fill the <placeholders> (warehouse, pipeline, workspace user, app URL) and run from app/ with workspace credentials in the environment.
import os, sys, json, re, subprocess, time, glob, io, contextlib
REPO="/tmp/tb"; os.chdir(REPO)
R=[]  # (group, name, ok, detail)
def rec(g,n,ok,d=""):
    R.append((g,n,bool(ok),d)); print(("PASS " if ok else "FAIL ")+f"[{g}] {n}"+(f"  -- {d}" if d else ""),flush=True)
def sh(c): return subprocess.run(c,shell=True,capture_output=True,text=True,cwd=REPO)
files=sh("git ls-files").stdout.split("\n")[:-1]

# ---- A. static ----
bad=[]
for f in files:
    if f.endswith(".py"):
        try: compile(open(f,encoding="utf-8").read(),f,"exec")
        except Exception as e: bad.append(f"{f}: {e}")
rec("static",f"all {sum(f.endswith('.py') for f in files)} Python files compile",not bad,"; ".join(bad[:3]))
bad=[]
for f in files:
    if f.endswith(".json"):
        try: json.load(open(f))
        except Exception as e: bad.append(f)
rec("static",f"all {sum(f.endswith('.json') for f in files)} JSON files parse",not bad,", ".join(bad[:3]))
import yaml
y=yaml.safe_load(open("app/app.yaml")); env={e["name"]:e.get("value",e.get("valueFrom")) for e in y["env"]}
rec("static","app.yaml parses and has the expected settings",set(env)>={"DATABRICKS_WAREHOUSE_ID","WB_SCHEMA","SSCD_ENDPOINT","WB_LAKEFLOW_SCHEMA","GENIE_SPACE_ID","LAKEBASE_INSTANCE","LAKEBASE_DB"},f"{sorted(env)}")
rec("static","pipeline SQL defines the 6 datasets (2 bronze, 1 silver, 3 gold)",len(re.findall(r"CREATE OR REFRESH (STREAMING TABLE|MATERIALIZED VIEW)",open("pipeline/claims_pipeline.sql").read()))==6)
rec("static","requirements pin an SDK with the Lakebase and Genie APIs and psycopg",
    "databricks-sdk>=0.169.0" in open("app/requirements.txt").read() and "psycopg" in open("app/requirements.txt").read())

# ---- B. hygiene ----
PAT=re.compile(os.environ["FORBIDDEN_PATTERN"],re.I)   # regex of strings that must not appear: customer name, personal ids, workspace ids, token prefixes
hits=[]
for f in files:
    try: t=open(f,encoding="utf-8").read()
    except Exception: continue
    m=PAT.search(t)
    if m: hits.append(f"{f}:{m.group(0)}")
rec("hygiene","no customer name, personal identifiers, workspace IDs or secrets in any tracked text file",not hits,", ".join(hits[:4]))
num=sh("git diff --numstat origin/main HEAD").stdout.split("\n")
binary=[l for l in num if l.startswith("-\t-")]
rec("hygiene","no binary files (images, weights) added versus main",not binary,str(binary[:2]))
big=[(int(a)+int(b),f) for l in num if l and not l.startswith("-") for a,b,f in [l.split("\t")]]
sizes=[(os.path.getsize(f),f) for f in sh("git diff --name-only --diff-filter=AM origin/main HEAD").stdout.split("\n") if f and os.path.exists(f)]
rec("hygiene","largest added or changed file is under 1 MB",max(sizes)[0]<1_000_000,f"{max(sizes)[1]} {max(sizes)[0]//1024} KB")
mt=sh("git merge-tree --write-tree origin/main HEAD")
rec("merge","merges into main with no conflicts",mt.returncode==0,mt.stdout.strip()[:80] if mt.returncode else "clean")
ig=sh("git diff --name-only origin/main HEAD").stdout.split("\n")
rec("merge","change set limited to expected areas",all(f.split("/")[0] in {"app","pipeline","lakebase","genie","uc","evidence","deck","README.md",".gitignore","POC_SUMMARY.md","ML_EXPERIMENT.md","wb_data","notebooks","build","_add_real_phones.py","dashboard","image_authenticity"} or f=="" for f in ig),f"{len(ig)-1} files")

# ---- C. parity with live e2 app ----
os.makedirs("/tmp/e2live",exist_ok=True)
subprocess.run("rm -rf /tmp/e2live/* && databricks workspace export-dir /Workspace/Users/<workspace-user>/fraud-defence-workbench /tmp/e2live >/dev/null 2>&1; find /tmp/e2live -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null; rm -rf /tmp/e2live/.databricks",shell=True)
d=sh("diff -rq -x __pycache__ /tmp/e2live app").stdout.strip().split("\n")
d=[x for x in d if x]
rd=sh("diff /tmp/e2live/README.md app/README.md").stdout.strip().split("\n"); rd_ok=all(l.startswith(("<",">","---")) and ("<workspace" in l or "databricksapps" in l or l.startswith("---")) or re.match(r"^\d+c\d+$",l) for l in rd if l)
rec("parity","live e2 app code equals the branch except app.yaml (placeholders vs real values) and identifier placeholders in README",all(("app.yaml" in x or "README.md" in x) for x in d) and rd_ok,"; ".join(d)[:200] or "identical")

# ---- D. live data vs evidence ----
sys.path.insert(0,"/tmp/e2b")
from q import sql
rows,_=sql("SELECT 'silver',count(*) FROM image_authenticity.lakeflow.silver_claims UNION ALL SELECT 'gold_decisions',count(*) FROM image_authenticity.lakeflow.gold_claim_decisions UNION ALL SELECT 'queue',count(*) FROM image_authenticity.lakeflow.gold_review_queue UNION ALL SELECT 'lakebase_synced',count(*) FROM fraud_lakebase.public.review_queue")
c={a:int(b) for a,b in rows}
rec("data","pipeline row counts match the committed evidence (66 / 66 / 57)",c["silver"]==66 and c["gold_decisions"]==66 and c["queue"]==57,str(c))
rec("data","Lakebase synced queue equals the Lakeflow gold queue",c["lakebase_synced"]==c["queue"])
rows,_=sql("SELECT sum(CASE WHEN is_fraud AND flagged THEN 1 ELSE 0 END), sum(CASE WHEN is_fraud THEN 1 ELSE 0 END), sum(CASE WHEN is_fraud AND firm_call THEN 1 ELSE 0 END), sum(CASE WHEN NOT is_fraud AND firm_call THEN 1 ELSE 0 END), sum(CASE WHEN NOT is_fraud AND flagged THEN 1 ELSE 0 END), round(sum(CASE WHEN is_fraud AND flagged THEN claim_value ELSE 0 END)), round(sum(CASE WHEN is_fraud THEN claim_value ELSE 0 END)) FROM image_authenticity.lakeflow.gold_claim_decisions")
a=[int(float(x)) for x in rows[0]]
rec("data","headline numbers in the deck match the data (51/52 flagged, 43 firm, 0 genuine firm, 6 genuine flagged, $33,919 of $34,618)",a==[51,52,43,0,6,33919,34618],str(a))
rows,_=sql("SELECT count(*) FROM image_authenticity.lakeflow.silver_claims WHERE claim_id IS NULL OR claim_value<=0 OR image_bytes IS NULL")
rec("data","silver has no rows breaking its quality rules",int(rows[0][0])==0)
ev=json.load(open("evidence/04_lakebase_smoke_test.json"))
rec("data","Lakebase evidence records 57 synced rows and a sub-100 ms write",ev["synced_review_queue_rows_min_max"][0]==57 and ev["write_read_ms"]<100,f"{ev['write_read_ms']} ms")
dk=open("deck/DECK.md").read()
rec("data","deck quotes the same numbers",all(s in dk for s in ["51 of 52","43 of 52","$33,919","$34,618","80 of 80"]))

# ---- E. unit tests of new code ----
sys.path.insert(0,REPO+"/app"); os.chdir(REPO+"/app")
os.environ.pop("LAKEBASE_INSTANCE",None)
import importlib, lakebase
importlib.reload(lakebase)
rec("unit","lakebase does nothing and never raises when not configured",lakebase.run("SELECT 1")==(False,[]) and lakebase.add_audit("a","b","c","d") is False)
os.environ["LAKEBASE_INSTANCE"]="no-such-instance"; os.environ["LAKEBASE_HOST"]="127.0.0.1"
importlib.reload(lakebase)
t0=time.time(); r1=lakebase.run("SELECT 1"); t1=time.time()-t0
t0=time.time(); r2=lakebase.run("SELECT 1"); t2=time.time()-t0
rec("unit","an unreachable Lakebase fails soft, then the circuit breaker answers instantly",r1==(False,[]) and r2==(False,[]) and t2<0.05,f"first {t1:.1f}s, second {t2*1000:.1f} ms")
os.chdir(REPO+"/app")
import types
os.environ["DATABRICKS_WAREHOUSE_ID"]="<warehouse-id>"
import backend as B
calls=[]
_orig=(B.sql,B.audit,B.labels)
B.sql=lambda s,p=None: calls.append(s[:40])
B.audit=types.SimpleNamespace(clear=lambda:None); B.labels=types.SimpleNamespace(clear=lambda:None)
try:
    B.add_label("CLM-TEST","confirm_fraud","other","n","likely_fraud","Production"); ok=True
except Exception as e: ok=False; print(e)
B.sql,B.audit,B.labels=_orig
rec("unit","saving a review decision still writes to Delta and does not raise when Lakebase is down",ok and any("INSERT INTO" in c for c in calls),f"{len(calls)} Delta writes")
import genie_client
class FakeW:
    class genie:
        @staticmethod
        def start_conversation_and_wait(s,q):
            A=types.SimpleNamespace
            return A(conversation_id="c",message_id="m",status=A(value="COMPLETED"),attachments=[A(text=A(content="hi"),query=None,attachment_id="a")])
r=genie_client.ask(FakeW,"sp","q")
rec("unit","genie client parses a text-only answer",r["text"]=="hi" and r["rows"]==[])

# ---- F. live end to end ----
os.environ.update(GENIE_SPACE_ID=open("/tmp/e2b/space.txt").read().strip(),LAKEBASE_INSTANCE="fraud-workbench-db",WB_LAKEFLOW_SCHEMA="image_authenticity.lakeflow",DATABRICKS_WAREHOUSE_ID="<warehouse-id>")
os.environ.pop("LAKEBASE_HOST",None)
from databricks.sdk import WorkspaceClient
w=WorkspaceClient()
g=genie_client.ask(w,os.environ["GENIE_SPACE_ID"],"How many claims did we flag as likely fraud?")
nums=re.findall(r"\d+",json.dumps(g["rows"])+g["text"])
rec("live","Genie answers the headline question with the right number (43)","43" in nums and g["sql"],f"rows={g['rows'][:1]}")
import urllib.request
rq=urllib.request.Request("https://fraud-defence-workbench-<workspace-id>.aws.databricksapps.com/_stcore/health",headers={"Authorization":"Bearer "+os.environ["DATABRICKS_TOKEN"]})
rec("live","deployed app is healthy",urllib.request.urlopen(rq,timeout=30).status==200)
ep=w.serving_endpoints.get("image-authenticity-sscd")
rec("live","SSCD endpoint is READY with scale-to-zero off",str(ep.state.ready.value)=="READY" and not ep.config.served_entities[0].scale_to_zero_enabled)
pl=w.pipelines.get("<pipeline-id>")
rec("live","Lakeflow pipeline exists and last update completed",pl.latest_updates[0].state.value=="COMPLETED",pl.latest_updates[0].state.value)

# ---- G. pages (AppTest on the pushed code) ----
from streamlit.testing.v1 import AppTest
def page(p,after=None):
    at=AppTest.from_file(f"{REPO}/app/views/{p}",default_timeout=180).run()
    if after: after(at)
    return at,[str(e.value)[:120] for e in at.exception]
at,ex=page("business.py"); m={x.label:x.value for x in at.metric}
rec("pages","Business view renders with the right headline",not ex and m.get("Fraud claims flagged")=="51 of 52" and m.get("Firm “likely fraud” calls")=="43",str(m)[:140])
at,ex=page("ask.py"); rec("pages","Ask the data renders with sample questions",not ex and len(at.button)>=5,ex[0] if ex else f"{len(at.button)} buttons")
at.button[0].click().run(timeout=180); ex=[str(e.value)[:120] for e in at.exception]
rec("pages","Ask the data returns an answer to a sample question",not ex and len(at.chat_message)>=2,ex[0] if ex else f"{len(at.chat_message)} messages")
at,ex=page("review.py"); rec("pages","Review queue renders and shows the Lakebase panel or its fallback",not ex,ex[0] if ex else "")

p=sum(1 for r in R if r[2]); print(f"\nSUMMARY: {p} passed, {len(R)-p} failed, {len(R)} total")
open("/tmp/e2b/premerge_results.json","w").write(json.dumps(R))

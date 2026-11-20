import sys, os
sys.path.insert(0, "<repo>/app"); os.chdir("<repo>/app")
from streamlit.testing.v1 import AppTest
res=[]
def chk(name,at):
    exc=[e.value for e in at.exception]; res.append((name,not exc,exc[:1]))
    print(("PASS " if not exc else "FAIL ")+name, exc[:1] if exc else "")
at=AppTest.from_file("<repo>/app/views/business.py",default_timeout=120).run(); chk("business page renders",at)
print("  metrics:",[ (m.label,m.value) for m in at.metric])
print("  dataframes:",len(at.dataframe))
at=AppTest.from_file("<repo>/app/views/ask.py",default_timeout=120).run(); chk("ask page renders",at)
print("  sample buttons:",len(at.button))
at.button[1].click().run(timeout=180); chk("ask page answers a sample question",at)
print("  chat messages:",[ (m.name, (m.markdown[0].value[:90] if m.markdown else '')) for m in at.chat_message][:4])
at=AppTest.from_file("<repo>/app/views/review.py",default_timeout=180).run(); chk("review page renders (with Lakebase panel)",at)
print("  info/alerts:",[i.value[:110] for i in at.info][:2])

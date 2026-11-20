# Evidence that the build ran

Every file here is real output captured from the e2 workspace, as text. Nothing is a screenshot.

| File | What it shows | How it was produced |
|---|---|---|
| `01_lakeflow_pipeline_output.md` | Row counts for bronze, silver and gold, the three landed batch files, decisions vs ground truth, top of the review queue | SQL queries on `image_authenticity.lakeflow.*` after pipeline update `25d39f9e` COMPLETED |
| `02_lakeflow_event_log.txt` | The pipeline's own event log: each flow planned, started, completed | `pipelines list-pipeline-events` |
| `03_genie_answers.md` | Five business questions asked of the Genie space: Genie's wording, generated SQL, returned rows | Genie Conversation API |
| `04_lakebase_smoke_test.json` | PostgreSQL 16 connection, the synced review queue (57 rows), a write and read-back in 14.7 ms | Serverless job run of `lakebase/setup_lakebase.py` |
| `05_app_page_tests.txt` | The new app pages rendered headlessly against real data (business metrics, a Genie answer, review page) | Streamlit AppTest, script in `app_page_tests.py` |
| `06_unity_catalog_governance.md` | Comments, tags (including personal-data tags on `claimant_id`) and grants as applied and read back | `uc/governance.sql` |
| `07_pre_merge_tests.txt` | 29 checks on a fresh clone of this branch: code and data hygiene, clean merge, parity with the live app, data and deck numbers, unit tests of the new code, live Genie, endpoint, pipeline and page tests | `pre_merge_tests.py` |
| `../ML_EXPERIMENT.md`, `../wb_data/ml_exp/*.md`, `../wb_data/ml_v1/results.md` | The detection experiments: SSCD vs pHash vs ORB, CLIP and SPAI detectors, combiner tests | scripts in `build/` |

Data is synthetic claim records over public Wikimedia Commons photos. No customer data is used.

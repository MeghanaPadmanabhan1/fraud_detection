# Architecture

![Architecture diagram](architecture.svg)

The same picture as text, so it can be read without rendering the image:

```mermaid
flowchart LR
  subgraph IN[1 Inputs]
    FEED[Claim feed: JSON batches]
    PHOTO[Claim photos]
    RCPT[Receipts]
    VOL[(UC volume claims_media)]
    FEED --> VOL
    PHOTO --> VOL
    RCPT --> VOL
  end
  subgraph LF[2 Lakeflow pipeline: serverless, incremental]
    BC[bronze_claims: Auto Loader]
    BI[bronze_claim_images]
    SC[silver_claims: typed, quality rules]
    GD[gold_claim_decisions]
    GS[gold_fraud_summary]
    GQ[gold_review_queue]
    BC --> SC
    BI --> SC
    SC --> GD
    GD --> GS
    GD --> GQ
  end
  subgraph AI[3 Intelligence]
    ENG[Detection engine: 5 checks, rules decide]
    SSCD[SSCD endpoint: Model Serving]
    VLM[Vision models: scene and receipt]
    DELTA[(Engine results in Delta: runs, findings)]
    ENG --> SSCD
    ENG --> VLM
    ENG --> DELTA
  end
  subgraph SV[4 Serving]
    LB[(Lakebase Postgres: synced queue, decisions, audit)]
    GENIE[Genie space over gold tables]
    WH[SQL warehouse]
  end
  subgraph APP[5 Databricks app]
    UI[Claims, Business view, Ask the data, Compare, Admin]
  end
  VOL --> BC
  VOL --> BI
  VOL -->|photos| ENG
  DELTA -->|verdicts joined in gold| GD
  GQ --> LB
  GD --> GENIE
  GD --> WH
  LB --> UI
  GENIE --> UI
  WH --> UI
  UI -.->|live calls| SSCD
  UI -.->|live calls| VLM
  UI -->|decisions, audit| LB
  UC{{Unity Catalog governs all of it}} -.-> LF
  UC -.-> SV
  UC -.-> AI
```

## Components and where they live

| Component | What it does | In this repo |
|---|---|---|
| UC volume `claims_media` | Landing zone for the claim feed, photos, receipts and masks | `pipeline/claims_pipeline.sql` (reads it) |
| Lakeflow pipeline | Auto Loader ingest, quality rules, gold tables for the business | `pipeline/claims_pipeline.sql`, `evidence/01_*`, `evidence/02_*` |
| Detection engine | Five checks and a rules engine that decides, with a written reason | `image_authenticity/workbench/` |
| SSCD endpoint | Learned copy detection, always on, called live for new photos | `build/register_sscd.py`, `app/` (client in `models.py`) |
| Vision models | Scene reading and receipt OCR through Foundation Model endpoints | `image_authenticity/workbench/models.py` |
| Lakebase | Review queue synced from gold; decisions and audit log written in well under a second | `lakebase/`, `app/lakebase.py`, `evidence/04_*` |
| Genie space | Plain-English questions over the gold tables | `genie/`, `app/genie_client.py`, `evidence/03_*` |
| Databricks app | What investigators and managers use | `app/` |
| Unity Catalog | Catalog, schemas, volume, model, tags, comments, least-privilege grants | `uc/governance.sql`, `evidence/06_*` |

Order note: "3 Intelligence" sits below the pipeline in the picture because it both reads the photos from the volume and feeds its verdicts back into the gold tables.

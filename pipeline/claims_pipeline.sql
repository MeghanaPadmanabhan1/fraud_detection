-- Lakeflow Spark Declarative Pipeline: raw claim feed -> governed, decision-ready tables.
-- Target: image_authenticity.lakeflow.  Source: UC volume landing zone + the detection engine's run tables.
-- Bronze = raw as landed. Silver = typed and quality-checked. Gold = what the business and Genie read.

-- ---------- BRONZE: Auto Loader (incremental) ----------
CREATE OR REFRESH STREAMING TABLE bronze_claims
COMMENT 'Raw claim records exactly as landed (JSON lines), with source file and ingest time'
AS SELECT *, _metadata.file_path AS source_file, current_timestamp() AS ingested_at
FROM STREAM read_files('/Volumes/image_authenticity/workbench/claims_media/landing/claims/', format => 'json');

CREATE OR REFRESH STREAMING TABLE bronze_claim_images
COMMENT 'Photo files in the claims volume: path and size only (the image bytes are not copied into a table)'
AS SELECT _metadata.file_path AS image_file, regexp_extract(_metadata.file_path, '[^/]+$', 0) AS image_name, length AS image_bytes, _metadata.file_modification_time AS image_modified_at
FROM STREAM read_files('/Volumes/image_authenticity/workbench/claims_media/images/', format => 'binaryFile');

-- ---------- SILVER: typed, deduplicated, quality-checked ----------
CREATE OR REFRESH MATERIALIZED VIEW silver_claims (
  CONSTRAINT claim_has_id        EXPECT (claim_id IS NOT NULL)                ON VIOLATION DROP ROW,
  CONSTRAINT value_is_positive   EXPECT (claim_value > 0)                     ON VIOLATION DROP ROW,
  CONSTRAINT incident_not_future EXPECT (incident_date <= claim_date),
  CONSTRAINT photo_is_in_volume  EXPECT (image_bytes IS NOT NULL)
)
COMMENT 'One row per claim, typed, with the photo file size. Rows with no id or a non-positive value are dropped; other checks are counted.'
AS SELECT
  c.claim_id, c.claimant_id,
  to_date(c.claim_date) AS claim_date, to_date(c.incident_date) AS incident_date, to_date(c.purchase_date) AS purchase_date,
  c.device_claimed, c.damage_described, c.channel,
  CAST(c.claim_value AS DOUBLE) AS claim_value,
  c.image_path, c.receipt_path,
  i.image_bytes,
  c.family AS true_family, c.fraud_class AS true_fraud_class,
  c.family <> 'genuine' AS is_fraud,
  c.source_file, c.ingested_at
FROM (SELECT *, row_number() OVER (PARTITION BY claim_id ORDER BY ingested_at DESC) AS rn FROM bronze_claims) c
LEFT JOIN bronze_claim_images i ON i.image_name = regexp_extract(c.image_path, '[^/]+$', 0)
WHERE c.rn = 1;

-- ---------- GOLD: decisions, summary and the review queue ----------
CREATE OR REFRESH MATERIALIZED VIEW gold_claim_decisions
COMMENT 'One row per claim: the claim, the detection engine verdict (Production setup), its top reason, and whether that matches the known label'
AS WITH latest AS (
  SELECT *, row_number() OVER (PARTITION BY claim_id ORDER BY run_ts DESC) AS rn
  FROM image_authenticity.workbench.runs WHERE config_name = 'Production'),
top_reason AS (
  SELECT claim_id, check_name, text,
         row_number() OVER (PARTITION BY claim_id ORDER BY CASE severity WHEN 'red' THEN 0 ELSE 1 END, score DESC) AS rn
  FROM image_authenticity.workbench.findings WHERE config_name = 'Production' AND severity IN ('red','amber'))
SELECT s.claim_id, s.claimant_id, s.claim_date, s.device_claimed, s.damage_described, s.channel, s.claim_value,
       s.is_fraud, s.true_family, s.true_fraud_class,
       r.verdict, round(r.confidence, 3) AS confidence, r.route, r.explanation,
       t.check_name AS top_check, t.text AS top_reason,
       r.verdict IN ('likely_fraud','needs_review') AS flagged,
       r.verdict = 'likely_fraud' AS firm_call,
       (s.is_fraud AND r.verdict IN ('likely_fraud','needs_review')) AS fraud_caught,
       (NOT s.is_fraud AND r.verdict IN ('likely_fraud','needs_review')) AS genuine_flagged
FROM silver_claims s
LEFT JOIN latest r ON r.claim_id = s.claim_id AND r.rn = 1
LEFT JOIN top_reason t ON t.claim_id = s.claim_id AND t.rn = 1;

CREATE OR REFRESH MATERIALIZED VIEW gold_fraud_summary
COMMENT 'Per fraud type: how many claims, how much value, and how many the engine flagged or called likely fraud'
AS SELECT true_family AS family, true_fraud_class AS fraud_type,
       count(*) AS claims, round(sum(claim_value), 2) AS claim_value_total,
       sum(CASE WHEN flagged THEN 1 ELSE 0 END) AS flagged_claims,
       sum(CASE WHEN firm_call THEN 1 ELSE 0 END) AS firm_calls,
       round(sum(CASE WHEN flagged THEN claim_value ELSE 0 END), 2) AS flagged_value,
       round(avg(CASE WHEN flagged THEN 1.0 ELSE 0.0 END), 3) AS flag_rate
FROM gold_claim_decisions GROUP BY 1, 2;

CREATE OR REFRESH MATERIALIZED VIEW gold_review_queue
COMMENT 'Claims a person should look at, strongest first: firm fraud calls, then needs-review, with value at stake'
AS SELECT claim_id, claimant_id, device_claimed, claim_value, verdict, confidence, route, top_check, top_reason,
       row_number() OVER (ORDER BY CASE verdict WHEN 'likely_fraud' THEN 0 ELSE 1 END, confidence DESC, claim_value DESC) AS queue_rank
FROM gold_claim_decisions WHERE flagged;

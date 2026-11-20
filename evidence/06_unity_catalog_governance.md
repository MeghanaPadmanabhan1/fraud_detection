# Unity Catalog governance: applied and verified (workspace e2)

Statements from `uc/governance.sql`, run as the data owner. Output below is queried back from `information_schema`.

Note: the first run of the tag statements was rejected because this workspace restricts the shared tag keys `domain` and `pii` to a fixed vocabulary. The script now uses project-specific keys (`fd_*`).

Applied 10 of 10 statements.

### Table tags
```sql
SELECT table_name, tag_name, tag_value FROM image_authenticity.information_schema.table_tags WHERE schema_name='lakeflow' ORDER BY 1,2
```
table_name | tag_name | tag_value
gold_claim_decisions | fd_data_origin | synthetic
gold_claim_decisions | fd_domain | claims_fraud
gold_claim_decisions | fd_layer | gold
gold_review_queue | fd_data_origin | synthetic
gold_review_queue | fd_domain | claims_fraud
gold_review_queue | fd_layer | gold
silver_claims | fd_data_origin | synthetic
silver_claims | fd_domain | claims_fraud
silver_claims | fd_layer | silver

### Column tags (personal data)
```sql
SELECT table_name, column_name, tag_name, tag_value FROM image_authenticity.information_schema.column_tags WHERE schema_name='lakeflow' ORDER BY 1,2,3
```
table_name | column_name | tag_name | tag_value
gold_claim_decisions | claimant_id | fd_contains_pii | true
gold_claim_decisions | claimant_id | fd_data_class | customer_identifier
silver_claims | claimant_id | fd_contains_pii | true
silver_claims | claimant_id | fd_data_class | customer_identifier

### Table comments
```sql
SELECT table_name, comment FROM image_authenticity.information_schema.tables WHERE table_schema='lakeflow' AND comment IS NOT NULL ORDER BY 1
```
table_name | comment
__materialization_mat_dfb5bddb_24e7_40ed_8cc2_a2371e337066_bronze_claim_images_1 | Photo files in the claims volume: path and size only (the image bytes are not copied into a table)
__materialization_mat_dfb5bddb_24e7_40ed_8cc2_a2371e337066_bronze_claims_1 | Raw claim records exactly as landed (JSON lines), with source file and ingest time
__materialization_mat_dfb5bddb_24e7_40ed_8cc2_a2371e337066_gold_claim_decisions_1 | One row per claim: the claim, the detection engine verdict (Production setup), its top reason, and whether that matches the known label
__materialization_mat_dfb5bddb_24e7_40ed_8cc2_a2371e337066_gold_fraud_summary_1 | Per fraud type: how many claims, how much value, and how many the engine flagged or called likely fraud
__materialization_mat_dfb5bddb_24e7_40ed_8cc2_a2371e337066_gold_review_queue_1 | Claims a person should look at, strongest first: firm fraud calls, then needs-review, with value at stake
__materialization_mat_dfb5bddb_24e7_40ed_8cc2_a2371e337066_silver_claims_1 | One row per claim, typed, with the photo file size. Rows with no id or a non-positive value are dropped; other checks are counted.
bronze_claim_images | Photo files in the claims volume: path and size only (the image bytes are not copied into a table)
bronze_claims | Raw claim records exactly as landed (JSON lines), with source file and ingest time
gold_claim_decisions | One row per claim: detection verdict, top reason, and whether it matches the known label.
gold_fraud_summary | Per fraud type: claims, value, flagged claims, firm calls.
gold_review_queue | Claims a person should review, strongest first.
silver_claims | One row per claim, typed, with the photo file size. Rows with no id or a non-positive value are dropped; other checks are counted.

### Who can read the lakeflow schema
```sql
SELECT grantee, privilege_type FROM image_authenticity.information_schema.schema_privileges WHERE schema_name='lakeflow' ORDER BY 1,2
```
grantee | privilege_type
<app-service-principal-client-id> | SELECT
<app-service-principal-client-id> | USE_SCHEMA
57e7dc57-63b9-4114-a2ad-0a6226794a22 | ALL_PRIVILEGES
57e7dc57-63b9-4114-a2ad-0a6226794a22 | MANAGE
f52e6a8a-14a5-413f-8b44-1b9037bda546 | SELECT
f52e6a8a-14a5-413f-8b44-1b9037bda546 | USE_SCHEMA

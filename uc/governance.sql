-- Unity Catalog governance for the claims-fraud data product (run as the data owner).
-- Everything here is metadata: no data moves and the app keeps working unchanged.

-- 1. Documentation that shows up in Catalog Explorer and is read by Genie
COMMENT ON SCHEMA image_authenticity.lakeflow IS 'Lakeflow pipeline output: bronze (raw), silver (clean), gold (decisions for the business). Synthetic claims, public photos.';
COMMENT ON TABLE image_authenticity.lakeflow.gold_claim_decisions IS 'One row per claim: detection verdict, top reason, and whether it matches the known label.';
COMMENT ON TABLE image_authenticity.lakeflow.gold_review_queue IS 'Claims a person should review, strongest first.';
COMMENT ON TABLE image_authenticity.lakeflow.gold_fraud_summary IS 'Per fraud type: claims, value, flagged claims, firm calls.';

-- 2. Classification tags: what each table is, and which column identifies a person.
--    Project-specific keys (fd_*) because this workspace restricts the shared keys (domain, pii) to its own vocabulary.
ALTER TABLE image_authenticity.lakeflow.silver_claims SET TAGS ('fd_domain' = 'claims_fraud', 'fd_layer' = 'silver', 'fd_data_origin' = 'synthetic');
ALTER TABLE image_authenticity.lakeflow.gold_claim_decisions SET TAGS ('fd_domain' = 'claims_fraud', 'fd_layer' = 'gold', 'fd_data_origin' = 'synthetic');
ALTER TABLE image_authenticity.lakeflow.gold_review_queue SET TAGS ('fd_domain' = 'claims_fraud', 'fd_layer' = 'gold', 'fd_data_origin' = 'synthetic');
ALTER TABLE image_authenticity.lakeflow.silver_claims ALTER COLUMN claimant_id SET TAGS ('fd_contains_pii' = 'true', 'fd_data_class' = 'customer_identifier');
ALTER TABLE image_authenticity.lakeflow.gold_claim_decisions ALTER COLUMN claimant_id SET TAGS ('fd_contains_pii' = 'true', 'fd_data_class' = 'customer_identifier');

-- 3. Least-privilege access: the app's service principal reads the lakeflow schema and nothing else there
GRANT USE SCHEMA, SELECT ON SCHEMA image_authenticity.lakeflow TO `<app-service-principal-client-id>`;

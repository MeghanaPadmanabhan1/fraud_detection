# Lakeflow pipeline: real output (workspace e2, pipeline fraud-claims-lakeflow)

Captured by querying the pipeline's tables after update 25d39f9e-4478-4294-b140-ac30e3beacb3 (full refresh) COMPLETED.

### Row counts per pipeline table
```sql
SELECT 'bronze_claims' t,count(*) n FROM image_authenticity.lakeflow.bronze_claims UNION ALL SELECT 'bronze_claim_images',count(*) FROM image_authenticity.lakeflow.bronze_claim_images UNION ALL SELECT 'silver_claims',count(*) FROM image_authenticity.lakeflow.silver_claims UNION ALL SELECT 'gold_claim_decisions',count(*) FROM image_authenticity.lakeflow.gold_claim_decisions UNION ALL SELECT 'gold_fraud_summary',count(*) FROM image_authenticity.lakeflow.gold_fraud_summary UNION ALL SELECT 'gold_review_queue',count(*) FROM image_authenticity.lakeflow.gold_review_queue
```
t | n
silver_claims | 66
bronze_claims | 66
bronze_claim_images | 66
gold_claim_decisions | 66
gold_fraud_summary | 14
gold_review_queue | 57

### Auto Loader landed three batch files
```sql
SELECT regexp_extract(source_file,'[^/]+$',0) file, count(*) claims FROM image_authenticity.lakeflow.bronze_claims GROUP BY 1 ORDER BY 1
```
file | claims
claims_batch_1.jsonl | 30
claims_batch_2.jsonl | 22
claims_batch_3.jsonl | 14

### Silver: photos matched to claims
```sql
SELECT count(*) claims, sum(CASE WHEN image_bytes IS NOT NULL THEN 1 ELSE 0 END) with_photo, round(avg(image_bytes)/1024) avg_photo_kb FROM image_authenticity.lakeflow.silver_claims
```
claims | with_photo | avg_photo_kb
66 | 66 | 420.0

### Gold: decisions vs ground truth
```sql
SELECT is_fraud, verdict, count(*) claims, round(sum(claim_value),2) claim_value FROM image_authenticity.lakeflow.gold_claim_decisions GROUP BY 1,2 ORDER BY 1 DESC,2
```
is_fraud | verdict | claims | claim_value
true | clear | 1 | 699.0
true | likely_fraud | 43 | 28617.0
true | needs_review | 8 | 5302.0
false | clear | 8 | 3072.0
false | needs_review | 6 | 3594.0

### Gold: fraud summary by type
```sql
SELECT * FROM image_authenticity.lakeflow.gold_fraud_summary ORDER BY family, fraud_type
```
family | fraud_type | claims | claim_value_total | flagged_claims | firm_calls | flagged_value | flag_rate
deepfake | ai_edited_damage | 5 | 3095.0 | 5 | 5 | 3095.0 | 1.000
deepfake | ai_generated | 8 | 5342.0 | 7 | 6 | 4643.0 | 0.875
deepfake | copy_move | 4 | 2896.0 | 4 | 2 | 2896.0 | 1.000
deepfake | splice | 4 | 3046.0 | 4 | 2 | 3046.0 | 1.000
deepfake | upscale | 4 | 3046.0 | 4 | 4 | 3046.0 | 1.000
document | receipt_mismatch | 5 | 3995.0 | 5 | 5 | 3995.0 | 1.000
genuine | genuine | 14 | 6666.0 | 6 | 0 | 3594.0 | 0.429
shallowfake | metadata_future_date | 2 | 1548.0 | 2 | 2 | 1548.0 | 1.000
shallowfake | metadata_old_photo | 2 | 1348.0 | 2 | 2 | 1348.0 | 1.000
shallowfake | rephotograph | 4 | 3046.0 | 4 | 4 | 3046.0 | 1.000
shallowfake | reuse_cropped | 3 | 1057.0 | 3 | 1 | 1057.0 | 1.000
shallowfake | reuse_exact | 4 | 1706.0 | 4 | 4 | 1706.0 | 1.000
shallowfake | reuse_recompressed | 3 | 1597.0 | 3 | 3 | 1597.0 | 1.000
shallowfake | screenshot | 4 | 2896.0 | 4 | 3 | 2896.0 | 1.000

### Gold: top of the review queue
```sql
SELECT queue_rank, claim_id, device_claimed, claim_value, verdict, confidence, top_check FROM image_authenticity.lakeflow.gold_review_queue ORDER BY queue_rank LIMIT 8
```
queue_rank | claim_id | device_claimed | claim_value | verdict | confidence | top_check
1 | CLM-1058 | Microsoft Lumia | 199.0 | likely_fraud | 0.998 | IPTC / C2PA provenance
2 | CLM-1056 | Apple iPhone Xs | 899.0 | likely_fraud | 0.997 | IPTC / C2PA provenance
3 | CLM-1054 | OnePlus | 649.0 | likely_fraud | 0.994 | IPTC / C2PA provenance
4 | CLM-1040 | Apple iPhone Xs | 899.0 | likely_fraud | 0.992 | Perceptual hash (pHash) near-match
5 | CLM-1041 | OnePlus | 649.0 | likely_fraud | 0.992 | Perceptual hash (pHash) near-match
6 | CLM-1022 | OnePlus | 649.0 | likely_fraud | 0.985 | Perceptual hash (pHash) near-match
7 | CLM-1046 | OnePlus | 649.0 | likely_fraud | 0.983 | JPEG ghost / double compression
8 | CLM-1050 | Microsoft Lumia | 199.0 | likely_fraud | 0.983 | Resampling / upscale detector

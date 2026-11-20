# Genie space answers (real output)

Space: Fraud claims: ask the data, over image_authenticity.lakeflow gold tables. Each question was asked through the Genie Conversation API.

## How many claims did we flag as likely fraud, and how much money is that?
status: COMPLETED
SQL generated:
```sql
SELECT COUNT(*) AS `likely_fraud_claims`, ROUND(SUM(`claim_value`), 2) AS `likely_fraud_value` FROM `image_authenticity`.`lakeflow`.`gold_claim_decisions` WHERE `verdict` = 'likely_fraud'
```
Result:
likely_fraud_claims | likely_fraud_value
43 | 28617.0
Genie said: We flagged **43** claims as **likely fraud**, totaling **$28,617.00** in claim value. In this result, the average claim value per **likely fraud** claim is about **$665.51**.

## Which fraud types do we catch least often?
status: COMPLETED
SQL generated:
```sql
SELECT `fraud_type`, `claims`, `flagged_claims`, `flag_rate`
FROM `image_authenticity`.`lakeflow`.`gold_fraud_summary`
WHERE `family` <> 'genuine'
  AND `flag_rate` IS NOT NULL
  AND `claims` IS NOT NULL
ORDER BY `flag_rate` ASC, `claims` DESC
```
Result:
fraud_type | claims | flagged_claims | flag_rate
ai_generated | 8 | 7 | 0.875
ai_edited_damage | 5 | 5 | 1.000
receipt_mismatch | 5 | 5 | 1.000
splice | 4 | 4 | 1.000
copy_move | 4 | 4 | 1.000
screenshot | 4 | 4 | 1.000
upscale | 4 | 4 | 1.000
rephotograph | 4 | 4 | 1.000
Genie said: The fraud type caught least often is **ai_generated**, with **7 of 8** claims flagged for a **87.5%** catch rate. All other fraud types in the results were caught on **100%** of claims, so **ai_generated** is the only category showing any misses.

## Which genuine claims were flagged by mistake?
status: COMPLETED
SQL generated:
```sql
SELECT `claim_id`, `claimant_id`, `claim_date`, `device_claimed`, `channel`, `claim_value`, `verdict`, `confidence`, `route`, `top_check`, `top_reason`, `explanation` FROM `image_authenticity`.`lakeflow`.`gold_claim_decisions` WHERE `is_fraud` = false AND `flagged` = true
```
Result:
claim_id | claimant_id | claim_date | device_claimed | channel | claim_value | verdict | confidence | route | top_check | top_reason | explanation
CLM-1007 | CUST-1492 | 2026-09-02 | OnePlus | web | 649.0 | needs_review | 0.525 | Analyst review queue | Llama-4 Maverick (vision model) | Llama-4 Maverick rates this photo 70% likely digitally altered: “The image is likely real but the damage may be edited or enhanced.” | Needs review (52% confidence). The evidence points to a deepfake — pixels were generated or altered. Key evidence: (1) Llama-4 Maverick rates this photo 70% likely digitally altered: “The image is likely real but the damage may be edited or enhanced.” Send to an analyst for a second look.
CLM-1015 | CUST-9188 | 2026-09-06 | Huawei P7 | adjuster_portal | 449.0 | needs_review | 0.528 | Analyst review queue | Error Level Analysis (ELA) | ELA: a localized region re-compresses differently from the rest of the image (peak z 10.663) — typical of pasted or painted pixels. | Needs review (53% confidence). The evidence points to a deepfake — pixels were generated or altered. Key evidence: (1) ELA: a localized region re-compresses differently from the rest of the image (peak z 10.663) — typical of pasted or painted pixels. Send to an analyst for a second look.
CLM-1019 | CUST-5557 | 2026-09-21 | OnePlus | adjuster_portal | 649.0 | needs_review | 0.45 | Analyst review queue | Llama-4 Maverick (context) | The device in the photo looks like “iPhone”, but the claim is for OnePlus. | Needs review (45% confidence). The evidence points to a shallowfake — the pixels may be real, but the photo's story is false. Key evidence: (1) The device in the photo looks like “iPhone”, but the claim is for OnePlus. Send to an analyst for a second look.
CLM-1013 | CUST-4828 | 2026-08-29 | OnePlus | web | 649.0 | needs_review | 0.45 | Analyst review queue | Llama-4 Maverick (vision model) | Llama-4 Maverick rates this photo 60% likely digitally altered: “Cracked glass appears digitally enhanced or manipulated.” | Needs review (45% confidence). The evidence points to a deepfake — pixels were generated or altered. Key evidence: (1) Llama-4 Maverick rates this photo 60% likely digitally altered: “Cracked glass appears digitally enhanced or manipulated.” Send to an analyst for a second look.
CLM-1009 | CUST-6345 | 2026-08-31 | Samsung Galaxy S2 | web | 699.0 | needs_review | 0.45 | Analyst review queue | Llama-4 Maverick (vision model) | Llama-4 Maverick rates this photo 60% likely digitally altered: “The image appears to be a real photo, but the device model and damage are inconsistent with the claim.” | Needs review (45% confidence). The evidence points to a deepfake — pixels were generated or altered. Key evidence: (1) Llama-4 Maverick rates this photo 60% likely digitally altered: “The image appears to be a real photo, but the device model and damage are inconsistent with the claim.” Send to an analyst for a second look.
CLM-1006 | CUST-4963 | 2026-09-14 | Google Pixel | mobile_app | 499.0 | needs_review | 0.595 | Analyst review queue | JPEG ghost / double compression | JPEG ghost: a cluster of blocks carries a different compression history (51% of the image) — evidence of a prior local edit. | Needs review (59% confidence). The evidence points to a deepfake — pixels were generated or altered. Key evidence: (1) JPEG ghost: a cluster of blocks carries a different compression history (51% of the image) — evidence of a prior local edit. Send to an analyst for a second look.
Genie said: There are **6 genuine claims** that were flagged by mistake, and **all 6** were sent to the **Analyst review queue** with a **needs_review** verdict rather than a firm fraud call. Examples include:
- **CLM-1006** — **Google Pixel**, **$499**, **mobile_app**, confidence **59.5%**
- **CLM-1007** — **OnePlus**, **$649**, **web**, confidence **52.5%**
- **CLM-1015** — **Huawei P7**, **$449**, **adjuster_portal**, confidence **52.8%**
- **CLM-1019** — **OnePlus**, **$649**, **adjuster_portal**, confidence **45.0%**
- **CLM-1013** — **OnePlus**, **$649**, **web**, confidence **45.0%**

These mistaken flags were all borderline review cases, with confidence ranging from **45.0% to 59.5%**.

## What are the top 5 claims in the review queue?
status: COMPLETED
SQL generated:
```sql
SELECT `claim_id`, `claimant_id`, `device_claimed`, `claim_value`, `verdict`, `confidence`, `route`, `top_check`, `top_reason`, `queue_rank`
FROM `image_authenticity`.`lakeflow`.`gold_review_queue`
WHERE `queue_rank` IS NOT NULL
ORDER BY `queue_rank` ASC
LIMIT 5
```
Result:
claim_id | claimant_id | device_claimed | claim_value | verdict | confidence | route | top_check | top_reason | queue_rank
CLM-1058 | CUST-1981 | Microsoft Lumia | 199.0 | likely_fraud | 0.998 | Priority investigator queue (99%-confidence route) | IPTC / C2PA provenance | The file declares itself AI-generated: IPTC DigitalSourceType = trainedAlgorithmicMedia, embedded by the image generator. | 1
CLM-1056 | CUST-2655 | Apple iPhone Xs | 899.0 | likely_fraud | 0.997 | Priority investigator queue (99%-confidence route) | IPTC / C2PA provenance | The file declares itself AI-generated: IPTC DigitalSourceType = trainedAlgorithmicMedia, embedded by the image generator. | 2
CLM-1054 | CUST-3205 | OnePlus | 649.0 | likely_fraud | 0.994 | Priority investigator queue (99%-confidence route) | IPTC / C2PA provenance | The file declares itself AI-generated: IPTC DigitalSourceType = trainedAlgorithmicMedia, embedded by the image generator. | 3
CLM-1040 | CUST-8700 | Apple iPhone Xs | 899.0 | likely_fraud | 0.992 | Priority investigator queue (99%-confidence route) | Perceptual hash (pHash) near-match | This photo was already submitted in CLM-1048 by a different claimant (CUST-2427, filed 2026-09-08, 23 days earlier) — pHash distance 2/64. | 4
CLM-1041 | CUST-7151 | OnePlus | 649.0 | likely_fraud | 0.992 | Priority investigator queue (99%-confidence route) | Perceptual hash (pHash) near-match | This photo was already submitted in CLM-1049 by a different claimant (CUST-6098, filed 2026-09-29, 25 days earlier) — pHash distance 0/64. | 5
Genie said: The top 5 claims currently in the review queue are all **likely fraud** cases in the **priority investigator queue**.

- **Rank 1:** **CLM-1058** — Microsoft Lumia, **$199**, confidence **0.998**
- **Rank 2:** **CLM-1056** — Apple iPhone Xs, **$899**, confidence **0.997**
- **Rank 3:** **CLM-1054** — OnePlus, **$649**, confidence **0.994**
- **Rank 4:** **CLM-1040** — Apple iPhone Xs, **$899**, confidence **0.992**
- **Rank 5:** **CLM-1041** — OnePlus, **$649**, confidence **0.992**

Across these 5 claims, the queue is dominated by **firm-call fraud cases** with very high confidence, ranging from **0.992 to 0.998**.

## How much claim value is in claims we cleared?
status: COMPLETED
SQL generated:
```sql
SELECT SUM(`claim_value`) AS `cleared_claim_value`
FROM `image_authenticity`.`lakeflow`.`gold_claim_decisions`
WHERE `verdict` IS NOT NULL
  AND `verdict` ILIKE '%clear%'
```
Result:
cleared_claim_value
3771.0
Genie said: The total claim value for claims with a verdict matching **clear** is **3771.0**. Based on the visible result, the cleared-claims total is **3771.0**.

# Option C: one learned score from all signals vs the rules engine (gold set v0)

66 claims (52 fraud, 14 genuine). Cross-validated, claims from the same source photo never split across train/test. Rules engine = its own confidence score (thresholds were calibrated on these genuine photos, so it is favoured).

## Result

| Method | AUC | recall @ 0 false alarms | @ 1 (7%) | @ 2 (14%) | @ 6 (43%) |
|---|---|---|---|---|---|
| Rules engine (hand-written, Production) | 0.96 | 88% | 88% | 88% | 98% |
| Learned, logistic regression, all signals | 0.98 ± 0.01 | 76% ± 7 | 97% ± 1 | 97% ± 1 | 100% ± 1 |
| Learned, random forest, all signals | 0.96 ± 0.02 | 86% ± 5 | 90% ± 4 | 92% ± 3 | 97% ± 2 |
| Learned, logistic regression, without metadata checks (s1) | 0.93 ± 0.01 | 61% ± 5 | 89% ± 3 | 91% ± 3 | 95% ± 1 |
| Learned, random forest, without metadata checks (s1) | 0.90 ± 0.02 | 74% ± 6 | 80% ± 5 | 83% ± 6 | 92% ± 3 |
| Learned, logistic regression, without vision models | 0.96 ± 0.01 | 91% ± 3 | 93% ± 3 | 94% ± 2 | 96% ± 1 |
| Learned, random forest, without vision models | 0.93 ± 0.01 | 83% ± 3 | 86% ± 3 | 88% ± 3 | 94% ± 2 |

± is the spread over 30 repeats of the CV split (not sampling uncertainty: with 14 genuine claims, one claim moves a rate by 7 points).

## By fraud type: recall at 1/14 false alarms, when the model HAS seen that type vs has NEVER seen it

| Fraud type | n | Rules engine | Logistic, seen | Logistic, unseen | Forest, seen | Forest, unseen |
|---|---|---|---|---|---|---|
| receipt_mismatch | 5 | 100% | 100% | 0% | 47% | 10% |
| reuse_exact | 4 | 100% | 100% | 100% | 100% | 100% |
| reuse_recompressed | 3 | 100% | 100% | 100% | 100% | 100% |
| reuse_cropped | 3 | 33% | 100% | 100% | 100% | 100% |
| screenshot | 4 | 100% | 100% | 100% | 100% | 100% |
| rephotograph | 4 | 100% | 100% | 100% | 100% | 100% |
| metadata_old_photo | 2 | 100% | 100% | 100% | 95% | 88% |
| metadata_future_date | 2 | 100% | 100% | 100% | 100% | 100% |
| splice | 4 | 75% | 100% | 97% | 97% | 97% |
| copy_move | 4 | 75% | 100% | 100% | 100% | 100% |
| upscale | 4 | 100% | 100% | 100% | 100% | 100% |
| ai_edited_damage | 5 | 100% | 100% | 100% | 100% | 100% |
| ai_generated | 8 | 75% | 81% | 77% | 74% | 62% |
| **all fraud (claim-weighted)** | 52 | 88% | 97% | 87% | 90% | 85% |

## What they lean on (fit on all 66 claims; for reading only)

Logistic, largest standardised weights: orb +0.80, exif_rules +0.69, exif_full +0.67, vlm:databricks-llama-4-maverick +0.57, upscale +0.41, sscd_maxcos_prior +0.41, dhash +0.37, ai_parse_document +0.36

Forest, largest importances: sscd_maxcos_prior 0.15, exif_full 0.12, exif_rules 0.11, orb 0.11, vlm:databricks-llama-4-maverick 0.06, ela 0.05, upscale 0.04, vlm:databricks-claude-sonnet-5 0.03

printf '\n## Fairer rules baselines (added after the first run: Production uses only pHash for reuse, the learned models also see ORB and SSCD)\n\nRules engine presets scored by their own confidence, same 66 claims. recall @k = fraud caught when at most k of 14 genuine claims score higher.\n\n```\n'

cat /tmp/stack_extra.txt

printf '```\n'

printf '\n## Fairer rules baselines (added after the first run: Production uses only pHash for reuse, the learned models also see ORB and SSCD)\n\nRules engine presets scored by their own confidence, same 66 claims. recall @k = fraud caught when at most k of 14 genuine claims score higher.\n\n```\n'

cat /tmp/stack_extra.txt

printf '```\n'

## Fairer rules baselines (added after the first run: Production uses only pHash for reuse, the learned models also see ORB and SSCD)

Rules engine presets scored by their own confidence, same 66 claims. recall @k = fraud caught when at most k of 14 genuine claims score higher.

```
Rules-engine presets (own confidence score):
preset             AUC    @0    @1    @2    @6  flagged-fraud / flagged-genuine   reuse_cropped@1  splice@1  copy_move@1  ai_generated@1
Production        0.96   88%   88%   88%   98%  51/52 / 6/14      33%     75%     75%     75%
Challenger A      0.97   94%   94%   94%   96%  50/52 / 4/14     100%     75%     75%     88%
Full forensic     0.97   90%   96%   96%   96%  52/52 / 9/14     100%     75%    100%     88%
Fast path only    0.87   69%   71%   79%   79%  41/52 / 2/14      33%     50%     50%     62%
```

# Stop paying fake device claims: catch them from the photo and the receipt

*Fraud Defence Workbench · insurance_firm device-protection claims · all figures from a synthetic test set of 66 claims*

---

## 1. The outcome

**Of 52 fraudulent test claims, 51 were flagged for a person, and 43 were firm "likely fraud" calls with no genuine claim among them.**
Every call comes with a plain-English reason and the evidence behind it, and a person makes the final decision.

| What the buyer cares about | Measured on the test set |
|---|---|
| Fraud caught | **51 of 52** flagged (98%); **43 of 52** firm calls (83%) |
| Money at stake flagged | **$33,919 of $34,618** of fraudulent claim value (98%) |
| Wrongly accused genuine customers | **0 of 14** received a firm "likely fraud" call |
| Genuine claims sent to a person | 6 of 14 (43%), all as "needs a human look", never as fraud |
| Explainability | every decision lists the checks that fired, in plain words |

*The 43% is the cost to manage, and it is the number we would work down first (slide 6).*

---

## 2. The problem

Customers file device-protection claims with a photo of the damage and a receipt. Fraud now arrives in three forms:

- **Reused photo** (shallowfake): a real picture, but taken from another claim, cropped, screenshotted or forwarded.
- **Generated or edited photo** (deepfake): an AI-made image, or damage painted onto a real phone.
- **Receipt that does not support the claim**: wrong amount, date or item.

A person looking at a photo cannot reliably tell these apart, and they cannot compare against thousands of earlier claims.

---

## 3. What each stakeholder gets

**Executive sponsor**
- Fewer fraudulent payouts, with the savings measurable per fraud type.
- A control that is explainable: every decision lists the checks that fired. Nothing is auto-denied.
- A clear dial between catching more fraud and bothering more genuine customers.

**Domain owner (claims investigations)**
- A ranked review queue: strongest cases first, with value at stake.
- Plain reasons ("This photo appears in an earlier claim") instead of scores.
- Their decisions are saved and feed the next round of tuning.
- Ask questions in plain English: "Which fraud types do we catch least often?"

---

## 4. The end-to-end data journey (all on Databricks)

| Stage | What runs | Where to look |
|---|---|---|
| **Lakeflow** ingest | Auto Loader reads raw claim files and photos into bronze, then silver (with quality rules), then gold | `pipeline/claims_pipeline.sql`, `evidence/01_*`, `evidence/02_*` |
| **Unity Catalog** governance | One catalog for tables, a volume and the SSCD model; tags mark personal data; least-privilege grants | `uc/governance.sql`, `evidence/06_*` |
| **Lakebase** serving | The review queue is synced into Postgres; reviewer decisions and the audit log are written there in milliseconds | `lakebase/setup_lakebase.py`, `evidence/04_*` |
| **ML and Gen AI** | SSCD copy detection on Model Serving; vision models read the scene and the receipt; a rules engine decides; an explainer words it | `image_authenticity/workbench/`, `ML_EXPERIMENT.md` |
| **Genie** | A space over the gold tables answers business questions with the SQL shown | `genie/`, `evidence/03_*` |
| **Databricks app** | Claim inspector, photo upload, business view, ask-the-data, compare methods, admin | `app/`, `evidence/05_*` |

---

## 5. How we know it works

- **Held-out tests with thresholds frozen first.** Nothing was tuned on the second test set.
- **Reuse detection (80 altered copies, 40 photos never seen before):**

| Method | Found | False hits |
|---|---|---|
| Quick fingerprint (pHash) | 34% | 0 of 40 |
| Feature matching (ORB) | 79 of 80 | 3 of 40 |
| **SSCD (learned)** | **80 of 80** | **0 of 40** |

- **AI-image detectors were tested and not shipped.** Two published detectors (a CLIP-based one and SPAI) scored 0.55 to 0.79 AUC on 52 AI images against 32 real photos, and one flagged up to 31% of real photos. We use a vision model's read as a hint only. Image shape alone separated one generator at 0.97, which is why we distrust any single score.
- **A learned combiner did not beat the rules.** On a fraud type held out of training its recall fell from 97% to 87%, so the explainable rules stay.

---

## 6. Honest limits, and what we would do next

- **Small and synthetic.** 66 claims built from public photos and made-up records. Treat these as direction, not guarantees.
- **False alarms are the main cost.** The weak editing checks fire on forwarded photos. Work to do: tune them on real investigator labels, and route only strong evidence to investigators.
- **AI-made photos are the weak area** (7 of 8 AI-generated claims flagged), and new generators keep arriving.
- **Not checked yet:** photos copied from the internet. Today reuse is checked against earlier claims only.

---

## 7. Illustrative economics (assumptions, not measurements)

Per 1,000 claims, assuming **3% are fraudulent** (30 claims), average fraudulent claim **$666** (the test-set average), and **$20** of analyst time per manual look. Detection rates are the measured ones above.

| | Firm calls only | Firm calls plus "needs a human look" |
|---|---|---|
| Fraud claims caught | ~25 (83%) | ~29 (98%) |
| Fraud value stopped | ~$16,500 | ~$19,600 |
| Genuine claims sent to a person | ~0 (0 of 14 in test) | ~420 (43% in test) |
| Review cost | ~$500 | ~$8,900 |

Reading it: the firm-call route pays for itself on its own. The wider route catches about five more fraud claims per 1,000 but costs far more to review, so it only makes sense once the weak checks are tuned. **Replace these assumptions with real figures** (fraud rate, average payout, review cost) **before presenting externally.**

---

## 8. The ask

1. Approve a pilot on a slice of real claims, with investigators labelling outcomes in the app.
2. Name the fraud rate and review-cost figures so the economics can be replaced with real numbers.
3. Decide the risk appetite: firm calls only, or firm calls plus the human-look queue.

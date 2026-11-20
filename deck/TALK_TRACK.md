# Role-play talk track (25 minutes)

All numbers come from the 66-claim synthetic test set. Say "in our test" every time.

**Before you start (off the clock)**
- Open the app, the deck, the Genie room and the architecture diagram.
- Pick three claims: a reused photo (CLM-1008, a crop of an earlier claim), a genuine claim that reads "Looks fine", and one receipt mismatch.
- The review decision button works: a decision appears under "Operational store (Lakebase)" at the bottom of the Review page. It also adds a row to the reviewer-agreement table, so use a claim you don't mind labelling.

---

## 1. Set the scene (0:00 to 3:00)

**Say:** "Your customers file a device-protection claim with a photo and a receipt. Today a person looks at them. They can't compare a photo against every earlier claim, and they can't tell a real photo from an AI-made one. That's where fraud gets through, and where honest customers get slowed down.

The question I'm answering is: **can we stop paying fake claims earlier, without accusing honest customers?**

I'll show you three things in about fifteen minutes. The result in your terms. One fraudulent claim and one honest claim, so you can see how a decision is made and explained. And how your team asks questions of the data in plain English. Then I want your questions, and I'll tell you where this doesn't work yet."

## 2. Lead with the outcome (3:00 to 5:00)

**Show:** the Business view.

**Say:** "Here's the headline. We tested on 66 claims: 52 fraudulent, 14 genuine.
- **51 of the 52 fraud claims were flagged.**
- **43 were firm 'likely fraud' calls, and none of the 14 genuine customers got one.**
- The flagged fraud adds up to about **$33,900 of $34,600** at stake.

Two caveats up front. This is synthetic data, and **6 of the 14 genuine claims were sent to a person for a look**, a 43% rate. That's the number I'd work down first."

**Land it:** "It's a dial: firm calls only, or firm calls plus a human-look queue."

## 3. How it fits together (5:00 to 6:30)

**Show:** the architecture diagram.

**Say:** "Left to right: raw claim files and photos land in one governed volume. A **Lakeflow pipeline** turns them into clean tables and a ranked review queue. A **detection engine** with a learned copy-detection model and vision models adds a verdict and a written reason. The queue is served through **Lakebase** and a **Genie** space, and one **Databricks app** puts it in front of your team. **Unity Catalog** governs all of it, with one set of permissions and no data copied out."

## 4. Demo (6:30 to 17:00)

**A. A fraudulent claim (6:30 to 10:30)**
- **Show:** open the reused-photo claim. Read the banner: "Likely fraud", with a reason in plain words. Switch the photo view to **Same as an earlier claim**, and open **See the evidence**.
- **Say:** "An investigator gets a sentence and the proof. A learned model fingerprints every photo, so it finds copies even when cropped, screenshotted or forwarded. On a held-out test it **found all 80 altered copies with no false hits on 40 photos it had never seen**. The older quick fingerprint found a third."

**B. An honest claim (10:30 to 12:30)**
- **Show:** open the genuine claim: "Looks fine · nothing flagged". Switch to the heat map and read the caption: "A guide, not proof."
- **Say:** "That map always lights up the busiest parts of a photo, even a genuine one. We relabelled it so nobody reads it as an accusation. The decision comes from the checks, not the picture."

**C. A new photo (12:30 to 14:00)**
- **Show:** the **Check a new photo** page, with a sample photo and receipt.
- **Say:** "Your team can drop in a photo and receipt that aren't in the system yet. Only a fingerprint of the file and the verdict are logged. The photo isn't stored."

**D. Ask the data (14:00 to 17:00)**
- **Show:** **Ask the data**. Click "Which fraud types do we catch least often?", then ask "And how many of those are firm calls?"
- **Say:** "Genie writes the query and shows it, so an analyst can verify it. The weakest area is AI-generated photos, 7 of 8 caught. I'd rather you hear that from me."
- If Genie asks a clarifying question, answer it. It is not fully predictable, which is why you rehearse.

## 5. What it means (17:00 to 19:00)

**Say:** "In your terms: **fraud leakage** drops because most fraudulent value is flagged before payout. **Customer experience** is protected: no honest customer got a fraud call, and the ones sent for a look are routed as 'needs a look'. **Investigator time** goes further because the queue is ranked and every case has a reason.

Using assumptions I'll make explicit, per 1,000 claims at a 3% fraud rate, firm calls alone stop about **$16,500** of fraud for about **$500** of review time. Adding the human-look queue stops about $3,000 more but costs about **$8,900** of review time. So I'd start with firm calls and tune the rest on your real labels. **Those are illustrative. Give me your fraud rate, average payout and review cost and I'll replace mine.**"

## 6. Questions (19:00 to 24:00)

Answer in this shape: one sentence for the room, detail for the asker, one sentence for the other person.

### From the business stakeholder

- **Cost and payback?** "Two parts: platform usage and review time. On my assumptions the firm-call route pays for itself. The heavy step runs on one small always-on endpoint, under a second per photo. I'd size real cost in the pilot, not guess today."
- **What if it accuses an honest customer?** "It never denies a claim. Firm calls go to an investigator with the evidence, and a person decides. None of 14 genuine claims got one, but that's only 14, so I won't promise zero. The pilot measures it weekly."
- **Time to value?** "The pipeline, app and models already exist. A slice of real claims and a couple of investigators labelling gets you real measurements in weeks. The long pole is data access and agreeing the review workflow."
- **Why trust fake data?** "Don't treat it as a forecast. It shows the method works and where it fails. The settings were frozen before the second test. The pilot replaces my numbers with yours."
- **Why not buy an AI-image detector?** "We tested two. They scored 0.55 to 0.79 where 0.5 is a coin flip, and one flagged nearly a third of real photos. I kept them out. Reuse detection, your bigger risk, works well."

### From the technical stakeholder

- **Architecture?** "Volume, then Lakeflow (bronze, silver, gold), then the detection engine on Model Serving and foundation models, then Lakebase and Genie, then the app. Unity Catalog across all of it."
- **Why rules, not one learned model?** "We tested a combiner. On a fraud type it hadn't seen, recall fell from 97% to 87%, and it missed every receipt mismatch. With 66 labels, rules are explainable to an auditor. I'd revisit with thousands of real labels."
- **Data quality?** "Silver has declared rules: a claim needs an id and a positive value, and the photo must exist. Rows breaking those are dropped, other checks are counted. Bronze keeps the raw record and its source file."
- **Security and PII?** "Everything stays in the workspace. Claimant IDs are tagged as personal data, the app has read-only access to what it needs, uploads store only a hash and verdict, and every action is audit-logged. Before real data I'd add column masking on the claimant ID."
- **Integration?** "Today it's a review tool with its own queue. The connector is a claim event in and a verdict out. I haven't built it, and I'd scope it in the pilot."
- **How does it fail?** "New AI generators will beat the vision hint. The reuse check only knows earlier claims, not the internet. The weak editing checks cause the 43%. If the copy-detection endpoint is down, the screen says 'didn't run', not a green tick."
- **Is Lakebase wired in?** "Yes. A reviewer's decision lands in Delta for analytics and in Lakebase for the live queue. From the app the round trip measured under half a second, since each write opens a fresh secure connection. Inside the database the write itself takes about 15 milliseconds."

**If you are pushed on a number:** "Good challenge. That's from 66 claims, so the uncertainty is wide. I'd rather show you how we measure it on real claims."
**If you don't know:** "I don't know yet. Here's how I'd find out, and I'll get back to you by [day]." Never invent a figure.

## 7. Close (24:00 to 25:00)

**Say:** "In our test we flagged 51 of 52 fraudulent claims, with 43 firm calls and none on an honest customer. Every call has a reason a person can check, and your team can ask questions in plain English. The honest limits are a small synthetic set, a 43% human-look rate to tune down, and AI-made photos as the weak spot.

My ask: approve a pilot on a slice of real claims, give me your fraud rate, payout and review cost, and choose whether we start with firm calls only. Who should I work with on the claims side?"

---

**Do not say:** anything about production results, or any guarantee. Use "your" and "insurance_firm", never a real customer name.

**Rehearse first:** run the five Genie sample questions. Answers vary, so know what you will say if one comes back differently.

## Quick reference: numbers you can say

| Fact | Number |
|---|---|
| Test set | 66 claims: 52 fraud, 14 genuine; synthetic records over public photos |
| Fraud flagged | 51 of 52; 43 firm "likely fraud" calls; $33,919 of $34,618 flagged |
| Genuine claims given a firm call | 0 of 14 |
| Genuine claims sent for a look | 6 of 14 (43%) |
| Reuse test (held-out) | SSCD 80/80 with 0 of 40 false hits; pHash 34%; ORB 79/80 with 3 false hits |
| AI-image detectors | 0.55 to 0.79 AUC; one flagged 28 to 31% of real photos; not shipped |
| AI-generated claims flagged | 7 of 8 |
| Learned combiner | 97% to 87% recall on an unseen fraud type; receipt mismatches missed |
| Review queue | 57 claims synced to Lakebase |
| Lakebase write and read-back | 14.7 ms |
| Economics | Illustrative: 3% fraud, $666 average claim, $20 per review |

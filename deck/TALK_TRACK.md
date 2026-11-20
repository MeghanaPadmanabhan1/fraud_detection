# Role-play talk track (25 minutes)

You are presenting to two people in the same room:

- **Business stakeholder** (the executive who funds it): cares about fraud losses, customer experience, cost, risk and time to value.
- **Technical stakeholder** (architect or data lead who lives with it): cares about architecture, data quality, security, integration and how it fails.

Rules for yourself: lead with the outcome, say it in their KPIs, then show it, then say what it means. When a question comes from the technical stakeholder, give real detail, then close with one sentence the executive can use.

All numbers below come from the 66-claim synthetic test set and the held-out set. Say "in our test" every time. Never say "in production".

---

## Before you start (2 minutes, off the clock)

Open these tabs: the app, the deck, the Genie space, the pipeline run. Pick claims in advance:
- A **reused photo** (CLM-1008, a crop of an earlier claim).
- A **genuine claim** that reads "Looks fine".
- One claim with a **receipt mismatch**.

Run through the app once. If the review page shows a Lakebase "not reachable" notice, say so honestly and do not click the decision button live.

---

## Part 1: Set the scene (0:00 to 3:00)

**Say:**

"Thanks for the time. Let me frame what I'm going to show you and why.

Your customers file a device-protection claim with two things: a photo of the damage and a receipt. Today a person looks at them. They cannot compare a photo against every earlier claim, and they cannot tell a real photo from an AI-made one. That is where fraud gets through, and it is also where honest customers get slowed down because everything looks equally suspicious.

The business question I'm answering is: **can we stop paying fake claims earlier, without accusing honest customers?**

In the next fifteen minutes I'll do three things. First, show you the result in your terms. Second, take one fraudulent claim and one honest claim through the system so you can see how a decision is made and explained. Third, show how your claims team asks questions of the data in plain English. Then I'd like your questions, and I'll tell you where this does not work yet."

**Why this works:** it names the customer, the problem, and the three things you will show (the setup criterion).

---

## Part 2: Tell the outcome first (3:00 to 5:00)

**Tell** (open the Business view page):

"Here is the headline. We tested on 66 claims: 52 fraudulent, 14 genuine. **The system flagged 51 of the 52 fraud claims, and 43 of those were firm 'likely fraud' calls.** None of the 14 genuine customers received a firm fraud call. The fraudulent claims we flagged add up to about **$33,900 of $34,600** at stake.

Two caveats up front, because you will trust the rest more. This is synthetic data, built from public photos and made-up claim records. And **6 of the 14 genuine claims were sent to a person for a look**, which is a 43% rate. That is the number I would work down first, and I'll show you how."

**Land it:** "So the value is a dial. Firm calls only gives you high-confidence fraud with no false accusations in our test. Adding the human-look queue catches a few more, at a real cost in review time."

---

## Part 3: Show (5:00 to 15:00)

### 3a. A fraudulent claim (5:00 to 9:00)

**Tell:** "I'll open a claim where someone reused a photo from an earlier claim, cropped."

**Show:** Claims page, select the reused-photo claim.
- Point at the banner: "Likely fraud", the one-line reason in plain words.
- Choose **Same as an earlier claim** in the photo selector: the two photos side by side.
- Open **See the evidence**: the similarity score and the earlier claim ID and filing date.

**Say:** "An investigator doesn't get a score, they get a sentence and the proof. This is the check I'm most proud of. A learned model called SSCD fingerprints every photo, so it finds copies even when they are cropped, screenshotted or forwarded. In a held-out test, **it found all 80 altered copies, with no false hits on 40 photos it had never seen**. The older quick-fingerprint method found about a third of them."

**Land it (business):** "The most common fraud route, the reused photo, is caught without anyone remembering seeing it before."

### 3b. An honest claim (9:00 to 11:00)

**Show:** open the genuine claim.
- Banner reads "Looks fine · nothing flagged".
- Switch the photo view to the heat map and say the caption out loud: "A guide, not proof."

**Say:** "I want to show you the heat map because someone on your team will ask. It always lights up the busiest parts of a photo, even on a genuine one. We learned that and relabelled it, so nobody reads a warm patch as an accusation. The decision comes from the checks, not the picture."

**Land it:** "This is what protects the honest customer: the system says 'nothing flagged' and means it."

### 3c. A receipt, and upload (11:00 to 12:30)

**Show:** the **Check a new photo** page. Upload a sample photo and a receipt, run it.

**Say:** "Your team can also drop in a photo and receipt that are not in the system yet, and get the same answer. Only a fingerprint of the file and the verdict are logged. The photo is not stored."

### 3d. Ask the data in plain English (12:30 to 15:00)

**Tell:** "Now the manager's view. They don't want to open claims one at a time."

**Show:** the **Ask the data** page. Click "Which fraud types do we catch least often?" Then type a follow-up: "And how many of those are firm calls?"

**Say:** "That answer comes from a Genie space over the checked results. It shows the query it wrote, so an analyst can verify it. The weakest area in our test is AI-generated photos, 7 of 8 caught, and I'd rather you hear that from me."

**Land it:** "A claims manager answers their own questions in seconds, and every answer can be audited."

---

## Part 4: Tell what it means (15:00 to 17:00)

**Say:**

"Let me bring it back to your numbers.

- **Fraud leakage:** in our test the system stopped the large majority of fraudulent value before payout.
- **Customer experience:** no honest customer received a fraud call. The ones sent for a look are routed as 'needs a look', not as fraud.
- **Investigator time:** the queue is ranked, every case has a written reason, and decisions are saved to feed the next round of tuning.

Using assumptions I'll make explicit, per 1,000 claims at a 3% fraud rate, firm calls alone stop roughly **$16,500** of fraud and cost about **$500** in review time. Adding the human-look queue stops about $3,000 more but costs about **$8,900** in review time. That is why I recommend starting with firm calls and tuning the rest on your real labels. **These are illustrative. I'd like your actual fraud rate, average payout and review cost to replace mine.**"

---

## Part 5: Questions (17:00 to 25:00)

Answer in this shape: **one sentence for the room, detail for the asker, one sentence for the other person.**

### From the business stakeholder

**"What does this cost, and what is the payback?"**
"Two parts: the Databricks platform, which is usage-based, and review time. In our model the firm-call route pays for itself on its own: about $16,500 of fraud stopped per 1,000 claims for about $500 of review time, at a 3% fraud rate and my assumptions. The platform cost for 1,000 claims is small because the heavy step, the photo fingerprint, runs on a small always-on endpoint at under a second per photo. I'd size the real cost in the pilot. I'd rather give you a measured figure than a guess today." *(For the technical person:* "The endpoint is one small CPU instance, so cost is flat until volume is high.")

**"What if it wrongly accuses a customer?"**
"It never denies a claim. A firm call goes to an investigator with the evidence, and a person decides. In our test, none of the 14 genuine claims got a firm call, but that's 14 claims, so I wouldn't promise zero. The pilot measures it on real claims, and it's the number I'd report to you every week."

**"How long until we see value?"**
"The pilot is the fast part, because the pipeline, the app and the models already exist. A slice of real claims and a couple of investigators labelling outcomes gets you real measurements in weeks, not quarters. The long pole is access to real claim data and agreeing the review workflow."

**"Why should I trust a number from fake data?"**
"You shouldn't take it as a forecast, and I've said so. What it does show is that the method works and where it fails. We froze the settings before the second test, so that second result wasn't tuned. The pilot replaces my numbers with yours."

**"Why not just buy an AI-image detector?"**
"We tested two published ones. They scored between 0.55 and 0.79 on a 0 to 1 scale, where 0.5 is a coin flip, and one flagged nearly a third of real photos. We kept them out. Reuse detection, which is the bigger share of your fraud, works well. For AI-made images we use a vision model as a hint and say plainly it's the weak spot."

### From the technical stakeholder

**"Walk me through the architecture."**
"Raw claim files and photos land in a Unity Catalog volume. A **Lakeflow pipeline** uses Auto Loader to ingest them into bronze, applies quality rules into silver, and builds gold tables. The detection engine, a deterministic rules engine over five checks, writes its verdicts to Delta. SSCD runs on **Model Serving**. Gold tables feed a **Genie space**, and the review queue is synced to **Lakebase** so the app's writes are fast and transactional. The **Databricks App** reads all of it." *(For the executive:* "One platform, one set of permissions, no data copied out.")

**"Why a rules engine instead of one learned model?"**
"We tested a learned combiner. It matched the rules on the data it saw, but on a fraud type it hadn't seen its recall dropped from 97% to 87%, and it missed every receipt mismatch. With 66 labelled claims, rules are also explainable to an investigator and an auditor. Once you have thousands of real labels, I'd revisit it, and the app already captures those labels."

**"How do you handle data quality?"**
"The silver table has declared rules: a claim must have an id and a positive value, an incident date can't be after the claim date, and the photo must exist in the volume. Rows that break the first two are dropped, and the others are counted. Bronze keeps the raw record and its source file, so nothing is lost." *(Point at the event log if asked.)*

**"What about security and PII?"**
"Everything stays inside the workspace. Unity Catalog governs the tables, columns that identify a person are tagged, and the app's identity has only read access to the schemas it needs. Uploads are checked for type and size, nothing is stored except a hash and the verdict, and every decision is audit-logged. One thing I'd add before real data: column masking on the claimant identifier."

**"How does it integrate with our claims system?"**
"Today it is a review tool with its own queue. The integration is a claim-created event in, and a verdict plus reasons out, via an API or a table the claims system reads. I haven't built that connector, and I'd scope it in the pilot."

**"How does it fail? What about drift?"**
"Three ways. New AI generators will beat the vision hint, so we'd re-test on each new generator and watch the flag rate. The reuse index only knows claims it has seen, so it won't catch a photo taken from the internet; adding a web search is a planned option and needs a privacy review because the photo would leave the platform. And weak signals like re-save patterns produce false alarms, which is the 43%. If the SSCD endpoint is down, the system falls back to the older checks and says 'didn't run' on the screen, rather than showing a green tick."

**"Latency and scale?"**
"The photo fingerprint takes under a second per call on a small endpoint. The pipeline is incremental, so only new files are processed. I haven't load-tested this, so I wouldn't quote a throughput figure."

### Handling pushback

- **If challenged on a number:** "Good challenge. That figure is from the 66-claim set, so the uncertainty is wide. I'd rather show you how we'll measure it on real claims."
- **If you don't know:** "I don't know yet. Here's how I'd find out, and I'll get it to you by [day]." Never invent a figure.
- **If they say it's too complicated:** return to the one sentence: "Firm calls only. High-confidence fraud, a written reason, a person decides."

---

## Closing (last 90 seconds)

"To sum up. In our test we flagged 51 of 52 fraudulent claims, with 43 firm calls and none on an honest customer. Every call comes with a reason a person can check, and your team can ask questions in plain English. The honest limits are a small synthetic test set, a 43% human-look rate to tune down, and AI-made photos as the weak spot.

What I'd ask of you: approve a pilot on a slice of real claims, give me your real fraud rate, payout and review cost so the economics are yours, and choose whether we start with firm calls only. Who should I work with on the claims side to start?"

---

## Quick reference: numbers you can say

| Fact | Number |
|---|---|
| Test set | 66 claims: 52 fraud, 14 genuine; synthetic records over public photos |
| Fraud flagged | 51 of 52; 43 firm "likely fraud" calls; $33,919 of $34,618 flagged |
| Genuine claims given a firm call | 0 of 14 |
| Genuine claims sent for a look | 6 of 14 (43%) |
| Reuse test (held-out) | SSCD 80/80, 0 of 40 false hits; pHash 34%; ORB 79/80 with 3 false hits |
| AI-image detectors | 0.55 to 0.79 AUC; one flagged 28 to 31% of real photos; not shipped |
| AI-generated claims flagged | 7 of 8 |
| Learned combiner | 97% to 87% recall on an unseen fraud type; receipt mismatches missed |
| Reviewer queue | 57 claims synced to Lakebase |
| Lakebase write and read-back | 14.7 ms |
| Economics | Illustrative: 3% fraud, $666 average claim, $20 per review |

## Do not say

- Anything about production results, or "we guarantee".
- That Lakebase writes work from the app until the app's database role is granted and you have seen it work.
- Any real customer name. Use "your" and "insurance_firm".

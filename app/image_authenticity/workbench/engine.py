"""Run a claim through the configured eight stages and decide.

Every check returns plain-language Findings tied to the stage + technology that
produced them. The rules engine decides deterministically; an explainer (template
or vision-language model) only words the decision. Nothing is auto-denied.
"""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from ..core import build
from ..detectors import copy_move as _cm
from ..detectors import _util
from . import checks
from .catalog import STAGE_BY_ID, STAGES, VLMS, option_label

SEV_RANK = {"red": 3, "amber": 2, "green": 1, "info": 0}
STAGE_WEIGHT = {"s1": 0.9, "s2": 0.95, "s3": 0.6, "s4": 0.75, "s5": 0.85}


@dataclass
class Finding:
    stage: str
    check: str
    severity: str           # red | amber | green | info
    text: str
    family: Optional[str] = None   # deepfake | shallowfake | document | None
    score: Optional[float] = None
    evidence: dict = field(default_factory=dict)


@dataclass
class StageResult:
    sid: str
    num: int
    name: str
    techs: list
    status: str             # ran | skipped | pending | failed
    path: str
    score: Optional[float] = None
    severity: str = "info"
    latency_ms: float = 0.0
    tokens: int = 0
    cached: bool = False
    findings: list = field(default_factory=list)


@dataclass
class ClaimRun:
    claim_id: str
    verdict: str
    confidence: float
    families: list
    route: str
    reasons: list
    explanation: str
    explainer: str
    stages: list
    sync_latency_ms: float
    async_latency_ms: float
    tokens: int
    overlays: dict = field(default_factory=dict)

    def stage(self, sid):
        return next(s for s in self.stages if s.sid == sid)

    def to_row(self):
        d = asdict(self); d.pop("overlays", None)
        return d


@dataclass
class EngineContext:
    index: checks.ReuseIndex
    index_loader: Optional[Callable] = None        # entry -> bgr image (for ORB)
    vlm: Optional[Callable] = None                 # (claim_id, endpoint, img_bgr) -> dict|None
    ocr: Optional[Callable] = None                 # (claim_id, engine, receipt_bytes) -> dict|None
    explain: Optional[Callable] = None             # (endpoint, prompt) -> dict|None
    sscd: Optional[Callable] = None                # img_bgr -> 512-d vector | None (Model Serving endpoint)


_DET = {}


def _det(slot, name):
    if (slot, name) not in _DET:
        _DET[(slot, name)] = build(slot, name)
    return _DET[(slot, name)]


def _sev(score, amber, red):
    if score is None:
        return "info"
    return "red" if score >= red else "amber" if score >= amber else "green"


def _max_sev(findings):
    return max((f.severity for f in findings), key=lambda s: SEV_RANK[s], default="info")


def _days(a, b):
    return (a - b).days


# ======================================================================= stages
def _s0(claim, ctx, cfg, img, raw, meta, ov):
    tech = cfg["tech"]
    return [Finding("s0", option_label("s0", tech), "info",
                    f"Submitted via {claim.get('channel', 'upload').replace('_', ' ')} (native upload) — no capture-time "
                    "signature, so provenance relies on the metadata checks in stage 1.")], None


def _s1(claim, ctx, cfg, img, raw, meta, ov):
    p, tech = cfg["params"], cfg["tech"]
    label = option_label("s1", tech)
    fs = []
    inc = dt.date.fromisoformat(claim["incident_date"])
    filed = dt.date.fromisoformat(claim["claim_date"])
    if tech == "exif_full":
        if meta["ai_provenance"]:
            fs.append(Finding("s1", "IPTC / C2PA provenance", "red",
                              "The file declares itself AI-generated: IPTC DigitalSourceType = "
                              "trainedAlgorithmicMedia, embedded by the image generator.", "deepfake", 0.97))
        elif meta["c2pa"]:
            fs.append(Finding("s1", "IPTC / C2PA provenance", "info",
                              "A C2PA manifest is present (relying on it in decisions is pending legal approval)."))
    camera = meta.get("make") or meta.get("model")
    if not camera and not meta.get("taken"):
        sev = p.get("missing_metadata", "amber")
        fs.append(Finding("s1", label, sev if sev in SEV_RANK else "amber",
                          "No camera metadata — EXIF has been stripped. That is consistent with a screenshot, a "
                          "messaging-app forward or deliberate stripping; there is no settled rule, so it is flagged.",
                          "shallowfake", 0.5))
    sw = (meta.get("software") or "").strip("\x00")
    if sw and any(e in sw.lower() for e in checks.EDITORS):
        fs.append(Finding("s1", label, "amber", f"Editing software is recorded in the EXIF: “{sw}”.",
                          "deepfake", 0.6, {"software": sw}))
    taken = meta.get("taken")
    if taken:
        td = taken.date()
        if td < inc - dt.timedelta(days=p.get("max_photo_age_days", 30)):
            fs.append(Finding("s1", label, "red",
                              f"The photo was taken on {td:%d %b %Y} — {_days(inc, td)} days before the reported "
                              f"incident ({inc:%d %b %Y}). An older photo cannot show damage from this incident.",
                              "shallowfake", 0.92, {"taken": td.isoformat()}))
        elif td > filed + dt.timedelta(days=1):
            fs.append(Finding("s1", label, "red",
                              f"The photo is dated {td:%d %b %Y}, {_days(td, filed)} days after the claim was filed — "
                              "the capture date contradicts the claim.", "shallowfake", 0.9, {"taken": td.isoformat()}))
        elif td < inc - dt.timedelta(days=1):
            fs.append(Finding("s1", label, "amber",
                              f"The photo was taken {_days(inc, td)} days before the reported incident.", "shallowfake", 0.5))
        else:
            fs.append(Finding("s1", label, "green",
                              f"Photo taken {td:%d %b %Y} — consistent with the incident on {inc:%d %b %Y}.", None, 0.03))
    if camera:
        fs.append(Finding("s1", label, "info",
                          f"Captured on {meta.get('make') or ''} {meta.get('model') or ''}".strip()
                          + " (the capturing phone — recorded separately from the damaged device)."))
    return fs, max((f.score for f in fs if f.score is not None), default=0.03)


def _s2(claim, ctx, cfg, img, raw, meta, ov):
    p, tech = cfg["params"], cfg["tech"]
    label = option_label("s2", tech)
    prior = ctx.index.prior(claim)
    best, best_sim, detail = None, 0.0, ""
    if tech in ("phash", "dhash"):
        q = checks.phash(img) if tech == "phash" else checks.dhash(img)
        lim = p["phash_max"] if tech == "phash" else p["dhash_max"]
        for e in prior:
            d = checks.hamming(q, e.phash if tech == "phash" else e.dhash)
            sim = 1 - d / 64
            if sim > best_sim:
                best, best_sim, detail = e, sim, f"{'pHash' if tech == 'phash' else 'dHash'} distance {d}/64"
        matched = best is not None and (1 - best_sim) * 64 <= lim
    elif tech == "embed":
        q = checks.embedding(img)
        for e in prior:
            sim = float(np.dot(q, e.emb))
            if sim > best_sim:
                best, best_sim, detail = e, sim, f"embedding cosine {sim:.3f}"
        matched = best is not None and best_sim >= p["embed_min_cos"]
    elif tech == "sscd":
        entry = next((e for e in ctx.index.entries if e.claim_id == claim["claim_id"]), None)
        q = entry.sscd if entry is not None else None
        if q is None and ctx.sscd is not None:
            q = ctx.sscd(img)                          # Model Serving endpoint; None when cold/unreachable
        if q is None and checks.sscd_live_available():
            q = checks.sscd_embedding(img)
        if q is None:
            return [Finding("s2", label, "info", "SSCD vector unavailable for this photo: it is not in the index, "
                            "and the SSCD endpoint did not answer in time. Pick pHash or ORB, or retry.", None, 0.03,
                            {"unavailable": True})], 0.03
        skipped = 0
        for e in prior:
            v = ctx.index.sscd_for(e, ctx.index_loader)
            if v is None:
                skipped += 1
                continue
            sim = float(np.dot(q, v))
            if sim > best_sim:
                best, best_sim, detail = e, sim, f"SSCD cosine {sim:.3f}"
        matched = best is not None and best_sim >= p["sscd_min_cos"]
    else:  # orb
        qf = checks.orb_features(img)
        top = 0
        for e in prior:
            n = checks.orb_inliers(qf, ctx.index.orb_for(e, ctx.index_loader))
            if n > top:
                top, best = n, e
        best_sim = min(1.0, top / 200)
        detail = f"{top} geometrically consistent keypoint matches"
        matched = best is not None and top >= p["orb_min_inliers"]
    age = ""
    if ctx.index.built_at:
        mins = max(0, int((dt.datetime.now() - ctx.index.built_at).total_seconds() // 60))
        age = f", index refreshed {mins} min ago" if mins < 120 else f", index refreshed {mins // 60} h ago"
    if matched:
        days = _days(dt.date.fromisoformat(claim["claim_date"]), dt.date.fromisoformat(best.claim_date))
        same = best.claimant_id == claim["claimant_id"]
        ev = {"match_claim_id": best.claim_id, "match_image_path": best.image_path,
              "similarity": round(best_sim, 3), "detail": detail}
        if same:
            return [Finding("s2", label, "amber",
                            f"The same claimant used this photo before, in {best.claim_id} ({days} days earlier) — {detail}.",
                            "shallowfake", 0.6, ev)], 0.6
        return [Finding("s2", label, "red",
                        f"This photo was already submitted in {best.claim_id} by a different claimant "
                        f"({best.claimant_id}, filed {best.claim_date}, {days} days earlier) — {detail}.",
                        "shallowfake", 0.96, ev)], 0.96
    near = f" Closest: {best.claim_id} ({detail})." if best is not None and tech != "orb" else ""
    if tech == "sscd" and skipped:
        near += f" {skipped} prior claims have no SSCD vector yet."
    return [Finding("s2", label, "green",
                    f"No earlier claim uses this photo — searched {len(prior)} prior claims{age}.{near}", None, 0.03)], 0.03


PIXEL_TEXT = {
    "ela": ("ELA: a localized region re-compresses differently from the rest of the image (peak z {ela_peak_z}) — "
            "typical of pasted or painted pixels.", "ELA: compression error is consistent across the image."),
    "jpeg_ghost": ("JPEG ghost: a cluster of blocks carries a different compression history ({ghost_block_frac:.0%} of "
                   "the image) — evidence of a prior local edit.", "JPEG ghost: one consistent compression history."),
    "noise": ("Noise residual: a contiguous region has a different sensor-noise pattern ({noise_outlier_frac:.1%} of "
              "blocks) — consistent with content spliced from another source.",
              "Noise residual: sensor noise is consistent across the frame."),
    "copy_move_orb": ("Copy-move (ORB): {clone_pairs} matching feature pairs between distant regions — part of the "
                      "image appears to be cloned.", "Copy-move (ORB): no cloned regions found."),
    "copy_move_sift": ("Copy-move (SIFT): {clone_pairs} matching feature pairs between distant regions — part of the "
                       "image appears to be cloned.", "Copy-move (SIFT): no cloned regions found."),
    "upscale": ("Resampling: almost no fine detail near full resolution (HF ratio {hf_log_ratio}) — the image looks "
                "upscaled from a much smaller original.", "Resampling: native-resolution detail is present."),
}
PIXEL_DET = {"ela": ("ela_local", "ela_heatmap_v1"), "jpeg_ghost": ("jpeg_ghost", "jpeg_ghost_v1"),
             "noise": ("noise", "noise_residual_v1"), "copy_move_orb": ("copy_move", "orb_v1"),
             "copy_move_sift": ("copy_move", "sift_v1")}


def _s3(claim, ctx, cfg, img, raw, meta, ov):
    p = cfg["params"]
    techs = cfg["tech"] if isinstance(cfg["tech"], list) else [cfg["tech"]]
    fs = []
    for t in techs:
        label = option_label("s3", t)
        amber, red = p["thresholds"].get(t, [0.6, 0.85])
        if t == "recapture":
            r = checks.recapture_check(img, meta)
            sev = _sev(r["score"], amber, red)
            if sev in ("red", "amber"):
                kind = r["kind"] or "screenshot or screen photo"
                text = f"Recapture: this looks like a {kind}, not an original camera photo — " + "; ".join(r["cues"]) + "."
            else:
                text = "Recapture: no screenshot or screen-photo signatures."
            fs.append(Finding("s3", label, sev, text, "shallowfake" if sev != "green" else None, r["score"], r))
            continue
        if t == "upscale":
            r = checks.upscale_check(img)
        else:
            res = _det(*PIXEL_DET[t]).analyze(img, raw, {})
            r = dict(res.evidence, score=res.score)
        sev = _sev(r["score"], amber, red)
        hi, lo = PIXEL_TEXT[t]
        try:
            text = (hi if sev in ("red", "amber") else lo).format(**r)
        except (KeyError, ValueError):
            text = hi if sev in ("red", "amber") else lo
        fs.append(Finding("s3", label, sev, text, "deepfake" if sev != "green" else None, r["score"],
                          {k: v for k, v in r.items() if k != "note"}))
    if ov is not None:
        ov["ELA heatmap"] = checks.ela_overlay(img)
        ov["Noise map"] = checks.noise_overlay(img)
        ov["JPEG ghost map"] = checks.ghost_overlay(img)
        if any(t.startswith("copy_move") for t in techs):
            g = _util.to_gray(img)
            orb = cv2.ORB_create(nfeatures=2000)
            kps, desc = orb.detectAndCompute(g, None)
            if desc is not None and len(kps) > 10:
                pts = np.array([k.pt for k in kps])
                ov["Copy-move pairs"] = _cm._clone_pairs(kps, desc, cv2.NORM_HAMMING, pts, 40, 0.5)
    return fs, max((f.score for f in fs if f.score is not None), default=None)


SPECTRAL = {"spectral_fft": ("spectral", "fft_radial_v1", "Spectral (FFT)"),
            "spectral_dct": ("spectral", "dct_stats_v1", "Spectral (DCT)"),
            "ela_uniformity": ("ela_global", "ela_uniformity_v1", "ELA uniformity")}


def _s4(claim, ctx, cfg, img, raw, meta, ov, sr):
    p = cfg["params"]
    techs = cfg["tech"] if isinstance(cfg["tech"], list) else [cfg["tech"]]
    fs = []
    for t in techs:
        if t.startswith("vlm:"):
            ep = t[4:]; name = VLMS.get(ep, ep)
            r = ctx.vlm(claim["claim_id"], ep, img) if ctx.vlm else None
            if not r or r.get("error"):
                sr.status = "pending"
                fs.append(Finding("s4", f"{name} (vision model)", "info",
                                  f"{name}: deep check pending (async path) — result not yet available."))
                continue
            sr.latency_ms += r.get("latency_ms", 0); sr.tokens += r.get("tokens", 0); sr.cached |= r.get("cached", False)
            ai, man = float(r.get("ai_generated_prob") or 0), float(r.get("manipulated_prob") or 0)
            why = (r.get("rationale") or "").strip().rstrip(".")
            if max(ai, man) >= p["amber"]:
                what = "AI-generated" if ai >= man else "digitally altered"
                fs.append(Finding("s4", f"{name} (vision model)", _sev(max(ai, man), p["amber"], p["red"]),
                                  f"{name} rates this photo {max(ai, man):.0%} likely {what}: “{why}.”",
                                  "deepfake", max(ai, man), {"ai_generated_prob": ai, "manipulated_prob": man}))
            else:
                fs.append(Finding("s4", f"{name} (vision model)", "green",
                                  f"{name} sees no sign of AI generation or editing ({max(ai, man):.0%}).", None,
                                  max(ai, man), {"ai_generated_prob": ai, "manipulated_prob": man}))
            guess = r.get("device_guess")
            if guess:
                gb, cb = checks.brand_of(guess), checks.brand_of(claim["device_claimed"])
                if gb and cb and gb != cb:
                    fs.append(Finding("s4", f"{name} (context)", "amber",
                                      f"The device in the photo looks like “{guess}”, but the claim is for "
                                      f"{claim['device_claimed']}.", "shallowfake", 0.6, {"device_guess": guess}))
                else:
                    fs.append(Finding("s4", f"{name} (context)", "info",
                                      f"Device read by {name}: “{guess}” (claim: {claim['device_claimed']})."))
        else:
            slot, impl, name = SPECTRAL[t]
            res = _det(slot, impl).analyze(img, raw, {})
            sev = _sev(res.score, *p.get("thresholds", {}).get(t, [p["amber"], p["red"]]))
            text = (f"{name}: frequency statistics look synthetic (score {res.score:.2f})." if sev != "green"
                    else f"{name}: frequency statistics look like a camera photo.")
            fs.append(Finding("s4", name, sev, text, "deepfake" if sev != "green" else None, res.score, res.evidence))
    return fs, max((f.score for f in fs if f.score is not None), default=None)


def _s5(claim, ctx, cfg, receipt, sr):
    tech = cfg["tech"]; label = option_label("s5", tech)
    if not receipt:
        return [Finding("s5", label, "info", "No receipt was submitted with this claim.")], None
    r = ctx.ocr(claim["claim_id"], tech, receipt) if ctx.ocr else None
    if not r or r.get("error"):
        sr.status = "pending"
        return [Finding("s5", label, "info", "Receipt OCR pending (async path).")], None
    sr.latency_ms += r.get("latency_ms", 0); sr.tokens += r.get("tokens", 0); sr.cached |= r.get("cached", False)
    f = r.get("fields") or {}
    fs = []
    total, rdate, item = f.get("total"), f.get("date"), f.get("item")
    if total is None:
        fs.append(Finding("s5", label, "amber", "OCR could not read the receipt total.", "document", 0.5))
    elif abs(float(total) - float(claim["claim_value"])) > 0.01:
        fs.append(Finding("s5", label, "red", f"Receipt total ${float(total):,.2f} does not match the claimed value "
                          f"${float(claim['claim_value']):,.2f}.", "document", 0.95, {"total": total}))
    if rdate is None:
        fs.append(Finding("s5", label, "amber", "OCR could not read the receipt date.", "document", 0.5))
    elif rdate > claim["claim_date"]:
        fs.append(Finding("s5", label, "red", f"The receipt is dated {rdate} — after the claim was filed "
                          f"({claim['claim_date']}).", "document", 0.95, {"date": rdate}))
    elif rdate != claim["purchase_date"]:
        fs.append(Finding("s5", label, "amber", f"Receipt date {rdate} differs from the purchase date on the claim "
                          f"({claim['purchase_date']}).", "document", 0.55, {"date": rdate}))
    m = checks.item_matches(item, claim["device_claimed"])
    if m is None:
        fs.append(Finding("s5", label, "amber", "OCR could not read the item on the receipt.", "document", 0.5))
    elif not m:
        fs.append(Finding("s5", label, "red", f"The receipt is for “{item}”, but the claim is for "
                          f"{claim['device_claimed']}.", "document", 0.95, {"item": item}))
    if not fs:
        fs.append(Finding("s5", label, "green", "Receipt matches the claim: total, purchase date and device agree.",
                          None, 0.03, f))
    return fs, max(x.score for x in fs)


# ======================================================================= decide
VERDICT_TEXT = {"likely_fraud": "Likely fraud", "needs_review": "Needs review", "clear": "Clear"}
FAMILY_TEXT = {"shallowfake": "a shallowfake — the pixels may be real, but the photo's story is false",
               "deepfake": "a deepfake — pixels were generated or altered",
               "document": "a document problem — the receipt does not support the claim"}


def decide(stages, params):
    reasons = sorted([f for s in stages for f in s.findings if f.severity in ("red", "amber")],
                     key=lambda f: (-SEV_RANK[f.severity], -(f.score or 0)))
    # Noisy-OR over FIRED findings only: every point of confidence traces back to a named
    # check that crossed its calibrated threshold (sub-threshold scores never add up).
    conf = 1.0
    for f in reasons:
        if f.score is not None and f.stage in STAGE_WEIGHT:
            conf *= 1 - STAGE_WEIGHT[f.stage] * float(f.score)
    conf = 1 - conf
    reds = [f for f in reasons if f.severity == "red"]
    if reds or conf >= params["fraud"]:
        verdict = "likely_fraud"
    elif reasons or conf >= params["review"]:
        verdict = "needs_review"
    else:
        verdict = "clear"
    if verdict == "likely_fraud":
        conf = max(conf, 0.8)
        multi = len({f.stage for f in reds}) >= 2
        route = ("Priority investigator queue (99%-confidence route)" if conf >= 0.99 or multi
                 else "Investigator queue")
    elif verdict == "needs_review":
        route = "Analyst review queue"
    else:
        route = "Straight-through (no human review needed)"
    fams = []
    for f in reasons:
        if f.family and f.family not in fams:
            fams.append(f.family)
    return verdict, round(conf, 4), fams, route, reasons


def template_explanation(verdict, conf, fams, route, reasons, stages):
    head = f"{VERDICT_TEXT[verdict]} ({conf:.0%} confidence)."
    if verdict == "clear":
        ran = [s.name for s in stages if s.status == "ran" and s.findings]
        return f"{head} No check raised a concern across {len(ran)} stages ({', '.join(ran)}). {route}."
    fam = ""
    if fams:
        fam = " The evidence points to " + "; and ".join(FAMILY_TEXT[f] for f in fams[:2]) + "."
    top = " ".join(f"({i + 1}) {r.text}" for i, r in enumerate(reasons[:3]))
    action = ("Route to an investigator — do not auto-deny on the score alone." if verdict == "likely_fraud"
              else "Send to an analyst for a second look.")
    return f"{head}{fam} Key evidence: {top} {action}"


def explanation_prompt(claim, verdict, conf, fams, reasons):
    ev = "\n".join(f"- [{r.severity}] stage {r.stage} · {r.check}: {r.text}" for r in reasons[:8]) or "- no concerns"
    return (f"You are explaining a fraud-screening decision to an insurance claims analyst.\n"
            f"Claim {claim['claim_id']}: {claim['device_claimed']}, '{claim['damage_described']}', "
            f"value ${claim['claim_value']:,.0f}, filed {claim['claim_date']}.\n"
            f"Decision (made by a deterministic rules engine): {VERDICT_TEXT[verdict]}, confidence {conf:.0%}.\n"
            f"Evidence:\n{ev}\n\nWrite 2-3 plain sentences explaining why, citing only the evidence above. "
            f"Do not invent facts. Do not recommend denying the claim; say what the analyst should check.")


# ======================================================================= run
RUNNERS = {"s0": _s0, "s1": _s1, "s2": _s2, "s3": _s3}


def run_claim(claim, image_bytes, receipt_bytes, config, ctx: EngineContext, want_overlays=False) -> ClaimRun:
    img = _util.decode_bgr(image_bytes)
    meta = checks.read_metadata(image_bytes)
    ov = {} if want_overlays else None
    stages = []
    for st in STAGES:
        sid = st["id"]; cfg = config[sid]
        techs = cfg["tech"] if isinstance(cfg["tech"], list) else [cfg["tech"]]
        sr = StageResult(sid, st["num"], st["name"], [option_label(sid, t) for t in techs], "ran", st["path"])
        if not cfg.get("enabled", True) or (st["mode"] == "multi" and not techs):
            sr.status = "skipped"; stages.append(sr); continue
        if sid in ("s6", "s7"):
            stages.append(sr); continue
        t0 = time.perf_counter()
        try:
            if sid in RUNNERS:
                fs, score = RUNNERS[sid](claim, ctx, cfg, img, image_bytes, meta, ov)
            elif sid == "s4":
                fs, score = _s4(claim, ctx, cfg, img, image_bytes, meta, ov, sr)
            else:
                fs, score = _s5(claim, ctx, cfg, receipt_bytes, sr)
        except Exception as e:  # one failing check must not sink the claim
            fs, score = [Finding(sid, ", ".join(sr.techs), "info", f"Check failed: {e!r}")], None
            sr.status = "failed"
        # Cached slow-path results already carry their recorded API latency; add local compute time.
        sr.latency_ms += (time.perf_counter() - t0) * 1000
        sr.findings, sr.score, sr.severity = fs, score, _max_sev(fs)
        stages.append(sr)

    p6 = config["s6"]["params"]
    verdict, conf, fams, route, reasons = decide(stages, p6)
    s6 = next(s for s in stages if s.sid == "s6")
    explainer = p6.get("explainer", "template")
    expl = template_explanation(verdict, conf, fams, route, reasons, stages)
    t0 = time.perf_counter()
    if explainer.startswith("vlm:") and ctx.explain:
        r = ctx.explain(explainer[4:], explanation_prompt(claim, verdict, conf, fams, reasons), claim["claim_id"])
        if r and r.get("text"):
            expl = r["text"].strip(); s6.tokens += r.get("tokens", 0); s6.cached = r.get("cached", False)
            s6.latency_ms += r.get("latency_ms", 0)
    s6.latency_ms += (time.perf_counter() - t0) * 1000
    s6.techs = [option_label("s6", config["s6"]["tech"]), option_label("s6", explainer)]
    s6.findings = [Finding("s6", "Rules engine", {"likely_fraud": "red", "needs_review": "amber", "clear": "green"}[verdict],
                           f"Decision: {VERDICT_TEXT[verdict]} ({conf:.0%}). {len([r for r in reasons if r.severity == 'red'])} "
                           f"red and {len([r for r in reasons if r.severity == 'amber'])} amber findings.")]
    s6.severity = s6.findings[0].severity; s6.score = conf
    s7 = next(s for s in stages if s.sid == "s7")
    if s7.status != "skipped":
        s7.findings = [Finding("s7", "Routing", "info", f"Routed to: {route}.")]
    sync = sum(s.latency_ms for s in stages if s.sid in ("s0", "s1", "s2", "s3"))
    asyn = sum(s.latency_ms for s in stages if s.sid in ("s4", "s5", "s6"))
    return ClaimRun(claim["claim_id"], verdict, conf, fams, route, reasons, expl, explainer, stages,
                    round(sync, 1), round(asyn, 1), sum(s.tokens for s in stages), ov or {})

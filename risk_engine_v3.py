"""
risk_engine_v3.py — SoundVision V3
=====================================
Implements the Vector-Intersection risk model:

    R_raw = (mass_weight × vel_effective × path_intersection_prob) / distance_m

Extended with:
  - Stationary suppression: decaying risk for static objects
  - TTC-gated urgency multiplier
  - EMA score smoothing per track (prevents jitter in alerts)
  - Score trend analysis (linear slope of smoothed score history)
  - Multi-threat output sorted by risk
  - Smart directional description: encodes both lateral position AND
    movement vector (crossing, approaching, receding, converging)

Risk Score Interpretation
--------------------------
  ≥ 300 → CRITICAL (stop immediately)
  ≥ 120 → HIGH     (urgent caution)
  ≥  45 → MEDIUM   (be aware)
  ≥  15 → LOW      (informational)
  <  15 → ignore

All scores are in arbitrary units (AU); thresholds are tuned for
the chest-mount pedestrian use case.

Directional Model
------------------
Direction is described in two parts that are combined into a natural
spoken phrase:

  1. LATERAL ZONE  — where the object is relative to user centreline:
       centre   : |lat| < 0.35 m
       near-L/R : 0.35 ≤ |lat| < 1.0 m
       far-L/R  : 1.0 ≤ |lat| < 2.0 m
       edge-L/R : |lat| ≥ 2.0 m

  2. MOTION CLASS  — how the object is moving relative to the user:
       approaching     : closing along Z (vz < -threshold)
       receding        : moving away along Z (vz > +threshold)
       crossing-left   : lateral motion left (vx < -threshold)
       crossing-right  : lateral motion right (vx > +threshold)
       converging      : closing AND crossing toward centre
       stationary/slow : speed < threshold

  These combine into phrases a blind user can act on immediately, e.g.:
    "car crossing from your right"
    "person ahead approaching fast"
    "bicycle on your left, moving away"
    "truck far right, crossing toward you"
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import numpy as np

from spatial_v3 import Object3D
from config import Config, CFG

log = logging.getLogger("SV3.RiskEngine")


# ─────────────────────────────────────────────────────────────────────────────
# Severity constants
# ─────────────────────────────────────────────────────────────────────────────

class Severity:
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"
    CLEAR    = "CLEAR"


# ─────────────────────────────────────────────────────────────────────────────
# Directional descriptor — the smart upgrade
# ─────────────────────────────────────────────────────────────────────────────

# Minimum speed (m/frame at 10 AI fps ≈ 0.05 m/s) to consider an object moving.
# Below this we call it stationary for directional purposes.
_VEL_THRESHOLD_MF = 0.005   # metres per AI-frame  (≈ 0.05 m/s at 10 fps)

# Lateral zone thresholds (metres from centreline, using *closest-point* X)
_ZONE_CENTRE   = 0.35   # |lat| < 0.35 m  → "directly ahead / behind"
_ZONE_NEAR     = 1.00   # 0.35 ≤ |lat| < 1.00 → "on your left/right"
_ZONE_FAR      = 2.00   # 1.00 ≤ |lat| < 2.00 → "far left/right"
# |lat| ≥ 2.00 → "on the edge, far left/right"

# Crossing dominance: if lateral velocity > Z velocity AND above threshold,
# call it a crossing motion rather than an approach.
_CROSS_DOMINANCE = 0.7   # vx must be this fraction of vel_mag to be "crossing"


def _lateral_zone(lat_m: float) -> Tuple[str, str]:
    """
    Returns (position_phrase, side_word) for the object's lateral position.

    position_phrase : e.g. "on your left", "far to your right", "ahead"
    side_word       : "left", "right", or "" for centre
    """
    abs_lat = abs(lat_m)
    side = "left" if lat_m < 0 else "right"

    if abs_lat < _ZONE_CENTRE:
        return "ahead", ""
    elif abs_lat < _ZONE_NEAR:
        return f"on your {side}", side
    elif abs_lat < _ZONE_FAR:
        return f"far to your {side}", side
    else:
        return f"on the far edge to your {side}", side


def _motion_phrase(
    vx: float,
    vz: float,
    lat_m: float,
    is_stationary: bool,
    ttc_s: float,
) -> str:
    """
    Classify the object's motion vector into a spoken phrase fragment.

    vx : lateral velocity (m/frame)  — positive = moving right
    vz : forward velocity (m/frame)  — negative = approaching user
    lat_m : object lateral position  — positive = object is on right side
    is_stationary : spatial analyser flag
    ttc_s : time-to-collision in seconds

    Returns a phrase like:
      "approaching"
      "approaching fast"
      "crossing toward you"
      "crossing left to right"
      "moving away"
      "stationary"
      "" (no useful motion info)
    """
    if is_stationary:
        return "stationary"

    vel_mag = math.sqrt(vx ** 2 + vz ** 2)
    if vel_mag < _VEL_THRESHOLD_MF:
        return "stationary"

    closing  = -vz   # positive = object getting closer
    opening  =  vz   # positive = object moving away
    crossing = abs(vx)

    # "Fast" qualifier: TTC < 5 s or closing speed > 0.15 m/frame (≈1.5 m/s)
    fast = ttc_s < 5.0 or closing > 0.15

    # ── Crossing dominant ──────────────────────────────────────────────────
    # Object is moving laterally more than it is approaching/receding
    if crossing > vel_mag * _CROSS_DOMINANCE and crossing > _VEL_THRESHOLD_MF:
        moving_right = vx > 0

        # Is the object crossing TOWARD or AWAY from the user's path centre?
        # If object is on the right and moving left → converging on user
        toward_centre = (lat_m > 0 and not moving_right) or \
                        (lat_m < 0 and moving_right) or \
                        (abs(lat_m) < _ZONE_CENTRE)

        if toward_centre:
            if closing > _VEL_THRESHOLD_MF:
                return "crossing toward you" if not fast else "crossing toward you fast"
            return "crossing toward you"
        else:
            direction_word = "left to right" if moving_right else "right to left"
            return f"crossing {direction_word}"

    # ── Pure approach / recession ──────────────────────────────────────────
    if closing > _VEL_THRESHOLD_MF and closing >= opening:
        return "approaching fast" if fast else "approaching"

    if opening > _VEL_THRESHOLD_MF and opening > closing:
        return "moving away"

    # ── Mixed: slight lateral + slight approach ────────────────────────────
    if closing > _VEL_THRESHOLD_MF * 0.5:
        return "approaching" if not fast else "approaching fast"

    return ""


def describe_direction(obj: Object3D) -> str:
    """
    Build a full, natural spoken direction phrase for a detected object.

    Uses BOTH the object's 3D position AND its velocity vector so that
    the blind user receives immediately actionable spatial information.

    Examples:
      "directly ahead, approaching fast"
      "on your left, crossing toward you"
      "far to your right, crossing left to right"
      "on your right, stationary"
      "ahead, moving away"
      "on your left, approaching"

    Implementation notes:
      - lateral_m uses the CENTROID X (spatial average of the whole object)
        for the position phrase, which represents the object's bulk.
      - Velocity vector (vx, vz) comes from the track's regression fit,
        which is robust to single-frame noise.
      - is_stationary flag from SpatialAnalyzerV3 overrides motion analysis
        when the object has been confirmed static for ≥5 consecutive frames.
    """
    vx, _vy, vz = obj.velocity
    lat_m = float(obj.lateral_m)   # centroid X, metres from centre

    pos_phrase, _side = _lateral_zone(lat_m)
    motion = _motion_phrase(vx, vz, lat_m, obj.is_stationary, obj.ttc_s)

    if motion:
        return f"{pos_phrase}, {motion}"
    return pos_phrase


# ─────────────────────────────────────────────────────────────────────────────
# Threat record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThreatRecord:
    obj:           Object3D
    score:         float       # smoothed risk score
    raw_score:     float       # unsmoothed, for diagnostics
    severity:      str
    ttc_s:         float
    trend:         float       # score slope (positive = worsening)
    direction:     str         # full spoken direction phrase
    distance_m:    float


# ─────────────────────────────────────────────────────────────────────────────
# Risk Engine V3
# ─────────────────────────────────────────────────────────────────────────────

class RiskEngineV3:
    """
    Vector-Intersection risk engine.

    Core formula
    -------------
    velocity_magnitude = ||(vx, vz)||   [m/frame, forward + lateral only]

    path_intersection = P(object will enter user corridor)
                      = pre-computed [0,1] from SpatialAnalyzerV3

    mass_weight = per-class danger weight (config.py)

    vel_effective = vel_mag * 0.40 + closing_vz * 1.60
      (closing component weighted 4× over pure lateral speed)

    base_score = mass_weight × vel_effective × path_intersection
                 / max(distance_m, 0.5)

    Then multiplied by:
      × TTC urgency factor   (non-linear, peaks at TTC < 2 s → ×10)
      × proximity boost      (exponential for dist < 6 m)
      × size factor          (mask area fraction, capped at 2.5×)
      × stationary decay     (if object hasn't moved for N frames)

    Finally EMA-smoothed per track to eliminate jitter.

    Key accuracy fixes vs. previous version
    -----------------------------------------
    FIX-A : vel_effective weights — closing weight raised from 1.5 to 1.6,
            lateral weight lowered from 0.5 to 0.4 (crossing is still
            dangerous but head-on closure is the primary kill vector).
    FIX-B : vel_effective now also adds a convergence bonus: if the object
            is crossing AND its lateral motion is reducing |lat_m| (i.e.
            converging on user's path), an additional 0.5× closing_vz
            term is added.
    FIX-C : `_direction` replaced by module-level `describe_direction`,
            which encodes both position and motion vector.
    FIX-D : `_score_trend` docstring corrected — it returns a linear slope
            (1st derivative), not a 2nd derivative.  The slope is the
            operationally correct signal: rising slope = worsening.
    FIX-E : Stale-track decay now uses per-track last-seen timestamp
            instead of a blind 0.85-per-call decay that would over-penalise
            objects whose inference frame rate varies.
    FIX-F : `_severity` TTC-HIGH branch now only upgrades to HIGH if the
            score already qualifies for MEDIUM (≥ TIER_MEDIUM), preventing
            false HIGH alerts for distant slow objects with middling TTC.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rc = cfg.risk

        # Per-track smoothed scores
        self._smooth_scores:    Dict[int, float]  = {}
        # Per-track score history for trend analysis
        self._score_histories:  Dict[int, deque]  = defaultdict(
            lambda: deque(maxlen=rc.score_history_len)
        )
        # Stationary decay accumulator
        self._stationary_decay: Dict[int, float]  = defaultdict(lambda: 1.0)
        # Per-track last-seen AI-frame index (for stale-track decay)
        self._last_seen:        Dict[int, int]    = {}
        self._frame_idx:        int               = 0

    # ── Public ────────────────────────────────────────────────────────────

    def evaluate(self, objects: List[Object3D]) -> Optional[ThreatRecord]:
        """Return the single highest-risk ThreatRecord, or None."""
        threats = self.evaluate_all(objects)
        return threats[0] if threats else None

    def evaluate_all(self, objects: List[Object3D]) -> List[ThreatRecord]:
        """Return all threats above TIER_LOW, sorted highest-risk first."""
        rc     = self.cfg.risk
        threats: List[ThreatRecord] = []

        active_ids = {o.track_id for o in objects}
        self._frame_idx += 1

        # FIX-E: stale-track decay — decay proportional to frames-since-seen
        for tid in list(self._smooth_scores.keys()):
            if tid not in active_ids:
                frames_absent = self._frame_idx - self._last_seen.get(tid, self._frame_idx)
                decay = 0.85 ** max(frames_absent, 1)
                self._smooth_scores[tid] *= decay
                if self._smooth_scores[tid] < 1.0:
                    del self._smooth_scores[tid]
                    self._last_seen.pop(tid, None)

        for obj in objects:
            self._last_seen[obj.track_id] = self._frame_idx
            raw    = self._compute_raw(obj)
            smooth = self._apply_smoothing(obj.track_id, raw, obj.is_stationary)

            if smooth < rc.TIER_LOW:
                continue

            sev   = self._severity(smooth, obj.ttc_s)
            trend = self._score_trend(obj.track_id)
            dir_  = describe_direction(obj)   # FIX-C

            obj.risk_score = smooth
            threats.append(ThreatRecord(
                obj        = obj,
                score      = round(smooth, 1),
                raw_score  = round(raw, 1),
                severity   = sev,
                ttc_s      = obj.ttc_s,
                trend      = round(trend, 2),
                direction  = dir_,
                distance_m = round(obj.distance_m, 2),
            ))

        threats.sort(key=lambda t: t.score, reverse=True)
        return threats

    # ── Core score computation ─────────────────────────────────────────────

    def _compute_raw(self, obj: Object3D) -> float:
        rc  = self.cfg.risk
        vx, vy, vz = obj.velocity
        mass   = rc.HAZARD_WEIGHTS.get(obj.label, 3.0)
        dist   = max(obj.distance_m, 0.5)
        path_p = obj.path_intersection

        # ── 1. Velocity vector magnitude (lateral + forward) ───────────────
        vel_mag = math.sqrt(vx ** 2 + vz ** 2)     # metres/frame
        vel_mag = max(vel_mag, 0.02)                # minimum ambient motion

        # ── 2. Closing component (object approaching user) ─────────────────
        # vz negative = object approaching (Z decreasing toward user)
        closing_vz = max(-vz, 0.0)   # only count approach component

        # ── 3. Convergence bonus (FIX-B) ───────────────────────────────────
        # If object is crossing AND moving toward the user's centreline,
        # add a fraction of closing_vz to signal the converging threat.
        lat_m = float(obj.lateral_m)
        crossing_toward = (lat_m > 0 and vx < -_VEL_THRESHOLD_MF) or \
                          (lat_m < 0 and vx >  _VEL_THRESHOLD_MF)
        convergence_bonus = closing_vz * 0.5 if crossing_toward else 0.0

        # ── 4. Effective velocity (FIX-A: adjusted weights) ───────────────
        vel_effective = (
            vel_mag       * 0.40 +   # ambient motion contribution
            closing_vz    * 1.60 +   # head-on closure (primary danger signal)
            convergence_bonus         # crossing-toward bonus
        )

        # ── 5. Base score: (mass × velocity × path_p) / distance ──────────
        base = (mass * vel_effective * max(path_p, 0.05)) / dist

        # ── 6. TTC urgency multiplier ──────────────────────────────────────
        ttc_mult = self._ttc_multiplier(obj.ttc_s)

        # ── 7. Proximity exponential boost (very close = exponential danger) ─
        prox_mult = math.exp(max(0.0, (6.0 - dist)) * 0.35)

        # ── 8. Mask size factor (normalised by frame area) ─────────────────
        mask_pixels = obj.inst.mask.sum()
        total_pixels = obj.inst.mask.size + 1
        mask_area_frac = mask_pixels / total_pixels
        size_mult = 1.0 + min(mask_area_frac * 8.0, 2.5)

        raw = base * ttc_mult * prox_mult * size_mult
        return float(np.clip(raw, 0.0, 2000.0))

    # ── Smoothing ─────────────────────────────────────────────────────────

    def _apply_smoothing(self, tid: int, raw: float, is_stationary: bool) -> float:
        rc    = self.cfg.risk
        alpha = rc.risk_ema_alpha

        # Stationary decay: suppress risk of objects confirmed not moving
        if is_stationary:
            self._stationary_decay[tid] = max(
                self._stationary_decay[tid] * rc.stationary_decay,
                rc.stationary_min,
            )
        else:
            self._stationary_decay[tid] = min(
                self._stationary_decay[tid] * 1.15, 1.0
            )

        raw_decayed = raw * self._stationary_decay[tid]

        # Asymmetric EMA: react faster to rising risk than falling risk.
        # This ensures sudden new threats are not dampened by a low prior.
        if tid not in self._smooth_scores:
            self._smooth_scores[tid] = raw_decayed
        else:
            prev = self._smooth_scores[tid]
            # Rising: use full alpha.  Falling: use 55% alpha (slower decay).
            a = alpha if raw_decayed >= prev else alpha * 0.55
            self._smooth_scores[tid] = a * raw_decayed + (1.0 - a) * prev

        smooth = self._smooth_scores[tid]
        self._score_histories[tid].append(smooth)
        return smooth

    # ── Trend ─────────────────────────────────────────────────────────────

    def _score_trend(self, tid: int) -> float:
        """
        Linear slope of the smoothed score history (1st derivative).   FIX-D
        Positive = risk worsening; negative = risk improving.
        Uses the most recent 8 samples (or all available if fewer).
        """
        hist = list(self._score_histories[tid])
        if len(hist) < 4:
            return 0.0
        arr = np.array(hist[-8:], dtype=np.float64)
        t   = np.arange(len(arr), dtype=np.float64)
        slope = float(np.polyfit(t, arr, 1)[0])
        return slope

    # ── Helpers ───────────────────────────────────────────────────────────

    def _ttc_multiplier(self, ttc_s: float) -> float:
        """
        Non-linear TTC urgency multiplier.

        TTC ≥ 100 s  → 0.85  (essentially no threat)
        TTC =  10 s  → 1.00
        TTC =   5 s  → 2.00
        TTC =   2 s  → 5.00
        TTC =   0 s  → 10.0

        Breakpoints are intentionally conservative: even 5 s TTC should
        produce a clear warning for a blind pedestrian.
        """
        if ttc_s >= 100.0:
            return 0.85
        if ttc_s >= 10.0:
            return 1.0 + (10.0 - ttc_s) * 0.02     # 1.00 → 1.00 (flat zone)
        if ttc_s >=  5.0:
            return 1.0 + (10.0 - ttc_s) * 0.20     # 1.00 → 2.00
        if ttc_s >=  2.0:
            return 2.0 + ( 5.0 - ttc_s) * 1.00     # 2.00 → 5.00
        return      5.0 + ( 2.0 - ttc_s) * 2.50    # 5.00 → 10.0

    def _severity(self, score: float, ttc_s: float) -> str:
        """
        Map (score, TTC) to a severity tier.

        TTC overrides:
          TTC < TTC_CRITICAL_S               → always CRITICAL
          TTC < TTC_HIGH_S AND score ≥ MEDIUM → HIGH    (FIX-F)

        Score thresholds:
          ≥ TIER_CRITICAL → CRITICAL
          ≥ TIER_HIGH     → HIGH
          ≥ TIER_MEDIUM   → MEDIUM
          else            → LOW
        """
        rc = self.cfg.risk

        # Immediate collision imminent
        if ttc_s < rc.TTC_CRITICAL_S:
            return Severity.CRITICAL

        # FIX-F: only upgrade to HIGH if score already warrants MEDIUM or better
        if ttc_s < rc.TTC_HIGH_S and score >= rc.TIER_MEDIUM:
            return Severity.HIGH

        if score >= rc.TIER_CRITICAL:
            return Severity.CRITICAL
        if score >= rc.TIER_HIGH:
            return Severity.HIGH
        if score >= rc.TIER_MEDIUM:
            return Severity.MEDIUM
        return Severity.LOW

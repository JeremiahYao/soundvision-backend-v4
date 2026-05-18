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
  ≥   8 → LOW      (informational)  [LOWERED from 15 — live calibration]
  <   8 → ignore

All scores are in arbitrary units (AU); thresholds are tuned for
the chest-mount pedestrian use case.

Directional Model
------------------
Direction is described in two parts combined into a natural spoken phrase:

  1. LATERAL ZONE  — where the object is relative to user centreline:
       centre   : |lat| < 0.35 m  → "ahead"
       near-L/R : 0.35–1.0 m      → "on your left/right"
       far-L/R  : 1.0–2.0 m       → "far to your left/right"
       edge-L/R : ≥ 2.0 m         → "on the far edge to your left/right"

  2. MOTION CLASS  — how the object is moving relative to the user:
       approaching / approaching fast
       crossing toward you / crossing toward you fast
       crossing left to right / crossing right to left
       moving away
       stationary
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
# Directional descriptor
# ─────────────────────────────────────────────────────────────────────────────

_VEL_THRESHOLD_MF = 0.005   # m/frame ≈ 0.05 m/s at 10 fps
_ZONE_CENTRE      = 0.35    # |lat| < 0.35 m
_ZONE_NEAR        = 1.00    # 0.35–1.00 m
_ZONE_FAR         = 2.00    # 1.00–2.00 m
_CROSS_DOMINANCE  = 0.70    # vx must be this fraction of vel_mag to be "crossing"


def _lateral_zone(lat_m: float) -> Tuple[str, str]:
    """Returns (position_phrase, side_word)."""
    abs_lat = abs(lat_m)
    side    = "left" if lat_m < 0 else "right"
    if abs_lat < _ZONE_CENTRE:
        return "ahead", ""
    elif abs_lat < _ZONE_NEAR:
        return f"on your {side}", side
    elif abs_lat < _ZONE_FAR:
        return f"far to your {side}", side
    else:
        return f"on the far edge to your {side}", side


def _motion_phrase(
    vx: float, vz: float, lat_m: float,
    is_stationary: bool, ttc_s: float,
) -> str:
    """Classify object motion into a spoken phrase fragment."""
    if is_stationary:
        return "stationary"

    vel_mag = math.sqrt(vx ** 2 + vz ** 2)
    if vel_mag < _VEL_THRESHOLD_MF:
        return "stationary"

    closing  = -vz
    opening  =  vz
    crossing = abs(vx)
    fast     = ttc_s < 5.0 or closing > 0.15

    if crossing > vel_mag * _CROSS_DOMINANCE and crossing > _VEL_THRESHOLD_MF:
        moving_right   = vx > 0
        toward_centre  = (
            (lat_m > 0 and not moving_right) or
            (lat_m < 0 and moving_right) or
            (abs(lat_m) < _ZONE_CENTRE)
        )
        if toward_centre:
            return "crossing toward you fast" if fast else "crossing toward you"
        direction_word = "left to right" if moving_right else "right to left"
        return f"crossing {direction_word}"

    if closing > _VEL_THRESHOLD_MF and closing >= opening:
        return "approaching fast" if fast else "approaching"

    if opening > _VEL_THRESHOLD_MF and opening > closing:
        return "moving away"

    if closing > _VEL_THRESHOLD_MF * 0.5:
        return "approaching fast" if fast else "approaching"

    return ""


def describe_direction(obj: Object3D) -> str:
    """
    Full spoken direction phrase: position + motion.

    Examples:
      "directly ahead, approaching fast"
      "on your left, crossing toward you"
      "far to your right, crossing left to right"
      "on your right, stationary"
    """
    vx, _vy, vz = obj.velocity
    lat_m       = float(obj.lateral_m)
    pos_phrase, _ = _lateral_zone(lat_m)
    motion        = _motion_phrase(vx, vz, lat_m, obj.is_stationary, obj.ttc_s)
    return f"{pos_phrase}, {motion}" if motion else pos_phrase


# ─────────────────────────────────────────────────────────────────────────────
# Threat record
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThreatRecord:
    obj:           Object3D
    score:         float
    raw_score:     float
    severity:      str
    ttc_s:         float
    trend:         float
    direction:     str
    distance_m:    float


# ─────────────────────────────────────────────────────────────────────────────
# Risk Engine V3
# ─────────────────────────────────────────────────────────────────────────────

class RiskEngineV3:
    """
    Vector-Intersection risk engine.

    Key accuracy fixes vs. previous version
    -----------------------------------------
    FIX-A : vel_effective closing weight raised 1.5 → 1.6, lateral 0.5 → 0.4
    FIX-B : convergence bonus when object crosses toward user's centreline
    FIX-C : direction replaced by describe_direction() — encodes motion
    FIX-D : _score_trend docstring corrected (slope, not 2nd derivative)
    FIX-E : stale-track decay proportional to frames-absent (not flat 0.85)
    FIX-F : TTC-HIGH severity upgrade requires score ≥ TIER_MEDIUM
    FIX-G : stationary_vel_floor applied by InferenceThread before evaluate_all()
            so stationary in-path objects always produce a meaningful score
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rc = cfg.risk
        self._smooth_scores:    Dict[int, float] = {}
        self._score_histories:  Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=rc.score_history_len)
        )
        self._stationary_decay: Dict[int, float] = defaultdict(lambda: 1.0)
        self._last_seen:        Dict[int, int]   = {}
        self._frame_idx:        int              = 0

    # ── Public ────────────────────────────────────────────────────────────

    def evaluate(self, objects: List[Object3D]) -> Optional[ThreatRecord]:
        threats = self.evaluate_all(objects)
        return threats[0] if threats else None

    def evaluate_all(self, objects: List[Object3D]) -> List[ThreatRecord]:
        rc      = self.cfg.risk
        threats: List[ThreatRecord] = []
        active  = {o.track_id for o in objects}
        self._frame_idx += 1

        # Stale-track decay — proportional to absence duration (FIX-E)
        for tid in list(self._smooth_scores.keys()):
            if tid not in active:
                absent = self._frame_idx - self._last_seen.get(tid, self._frame_idx)
                self._smooth_scores[tid] *= 0.85 ** max(absent, 1)
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
            dir_  = describe_direction(obj)

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

        # Velocity magnitude (lateral + forward)
        vel_mag = math.sqrt(vx ** 2 + vz ** 2)
        vel_mag = max(vel_mag, 0.02)

        # Closing component (approaching user)
        closing_vz = max(-vz, 0.0)

        # Convergence bonus: crossing AND moving toward centreline (FIX-B)
        lat_m = float(obj.lateral_m)
        crossing_toward = (
            (lat_m > 0 and vx < -_VEL_THRESHOLD_MF) or
            (lat_m < 0 and vx >  _VEL_THRESHOLD_MF)
        )
        convergence_bonus = closing_vz * 0.5 if crossing_toward else 0.0

        # Effective velocity (FIX-A)
        vel_effective = (
            vel_mag    * 0.40 +
            closing_vz * 1.60 +
            convergence_bonus
        )

        # Base score
        base = (mass * vel_effective * max(path_p, 0.05)) / dist

        # Multipliers
        ttc_mult  = self._ttc_multiplier(obj.ttc_s)
        prox_mult = math.exp(max(0.0, (6.0 - dist)) * 0.35)

        mask_area_frac = obj.inst.mask.sum() / (obj.inst.mask.size + 1)
        size_mult      = 1.0 + min(mask_area_frac * 8.0, 2.5)

        raw = base * ttc_mult * prox_mult * size_mult
        return float(np.clip(raw, 0.0, 2000.0))

    # ── Smoothing ─────────────────────────────────────────────────────────

    def _apply_smoothing(self, tid: int, raw: float, is_stationary: bool) -> float:
        rc    = self.cfg.risk
        alpha = rc.risk_ema_alpha

        if is_stationary:
            self._stationary_decay[tid] = max(
                self._stationary_decay[tid] * rc.stationary_decay,
                rc.stationary_min,   # floor raised to 0.50 in config
            )
        else:
            self._stationary_decay[tid] = min(
                self._stationary_decay[tid] * 1.15, 1.0
            )

        raw_decayed = raw * self._stationary_decay[tid]

        if tid not in self._smooth_scores:
            self._smooth_scores[tid] = raw_decayed
        else:
            prev = self._smooth_scores[tid]
            a    = alpha if raw_decayed >= prev else alpha * 0.55
            self._smooth_scores[tid] = a * raw_decayed + (1.0 - a) * prev

        smooth = self._smooth_scores[tid]
        self._score_histories[tid].append(smooth)
        return smooth

    # ── Trend ─────────────────────────────────────────────────────────────

    def _score_trend(self, tid: int) -> float:
        """Linear slope of score history. Positive = worsening. (FIX-D)"""
        hist = list(self._score_histories[tid])
        if len(hist) < 4:
            return 0.0
        arr   = np.array(hist[-8:], dtype=np.float64)
        t     = np.arange(len(arr), dtype=np.float64)
        return float(np.polyfit(t, arr, 1)[0])

    # ── Helpers ───────────────────────────────────────────────────────────

    def _ttc_multiplier(self, ttc_s: float) -> float:
        if ttc_s >= 100.0:  return 0.85
        if ttc_s >= 10.0:   return 1.0 + (10.0 - ttc_s) * 0.02
        if ttc_s >=  5.0:   return 1.0 + (10.0 - ttc_s) * 0.20
        if ttc_s >=  2.0:   return 2.0 + ( 5.0 - ttc_s) * 1.00
        return               5.0 + ( 2.0 - ttc_s) * 2.50

    def _severity(self, score: float, ttc_s: float) -> str:
        rc = self.cfg.risk
        if ttc_s < rc.TTC_CRITICAL_S:
            return Severity.CRITICAL
        # FIX-F: only upgrade via TTC if score already warrants MEDIUM
        if ttc_s < rc.TTC_HIGH_S and score >= rc.TIER_MEDIUM:
            return Severity.HIGH
        if score >= rc.TIER_CRITICAL:   return Severity.CRITICAL
        if score >= rc.TIER_HIGH:       return Severity.HIGH
        if score >= rc.TIER_MEDIUM:     return Severity.MEDIUM
        return Severity.LOW

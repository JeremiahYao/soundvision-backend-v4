"""
risk_engine_v3.py — SoundVision V3
=====================================
Vector-Intersection risk engine.

    R_raw = (mass_weight × velocity_eff × path_intersection) / distance_m
          × ttc_multiplier × proximity_boost × size_factor × pose_risk_factor

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Changes in this version (v4) — Seated person handling
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIX-POSE-3  pose_risk_factor integrated into _compute_raw().
            A seated person has pose_risk_factor = 0.20 (set by
            PostureClassifier in perception.py, propagated through
            Object3D in spatial_v3.py).

            The factor multiplies the raw score BEFORE EMA smoothing,
            so seated people consistently score ~5× lower than an
            equivalent standing person at the same distance.

FIX-POSE-4  Severity cap for seated/low-risk persons.
            Even if a seated person somehow accumulates a high smoothed
            score (e.g., right after standing up), _severity() caps
            them at MEDIUM when pose_risk_factor < 0.5. This prevents
            CRITICAL or HIGH alerts for bench-sitters and wheelchair
            users who are not on the walking path.

FIX-POSE-5  Direction label for seated persons uses "person_seated"
            in the ThreatRecord so the HUD and TTS guidance system
            can display and speak a more informative label.
            e.g. "person seated on your left, 2.3 metres"
            instead of "person left, 2.3 metres".

Risk Score Tiers (unchanged):
  ≥ 300 → CRITICAL
  ≥ 120 → HIGH
  ≥  45 → MEDIUM
  ≥  15 → LOW
  <  15 → ignored
"""

from __future__ import annotations

import math
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Optional, Dict

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

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rc = cfg.risk
        self._smooth_scores:    Dict[int, float]  = {}
        self._score_histories:  Dict[int, deque]  = defaultdict(
            lambda: deque(maxlen=rc.score_history_len)
        )
        self._stationary_decay: Dict[int, float]  = defaultdict(lambda: 1.0)

    # ── Public ────────────────────────────────────────────────────────────

    def evaluate(self, objects: List[Object3D]) -> Optional[ThreatRecord]:
        threats = self.evaluate_all(objects)
        return threats[0] if threats else None

    def evaluate_all(self, objects: List[Object3D]) -> List[ThreatRecord]:
        rc = self.cfg.risk
        threats: List[ThreatRecord] = []

        active_ids = {o.track_id for o in objects}
        for tid in list(self._smooth_scores.keys()):
            if tid not in active_ids:
                self._smooth_scores[tid] *= 0.85
                if self._smooth_scores[tid] < 1.0:
                    del self._smooth_scores[tid]

        for obj in objects:
            raw    = self._compute_raw(obj)
            smooth = self._apply_smoothing(obj.track_id, raw, obj.is_stationary)

            if smooth < rc.TIER_LOW:
                continue

            # FIX-POSE-4: severity cap for seated persons
            sev   = self._severity(smooth, obj.ttc_s, obj.pose_risk_factor)
            trend = self._score_trend(obj.track_id)
            dir_  = self._direction(obj)

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

        # velocity
        vel_mag       = max(math.sqrt(vx**2 + vz**2), 0.02)
        closing_vz    = max(-vz, 0.0)
        vel_effective = vel_mag * 0.5 + closing_vz * 1.5

        # base
        base = (mass * vel_effective * max(path_p, 0.05)) / dist

        # multipliers
        ttc_mult  = self._ttc_multiplier(obj.ttc_s)
        prox_mult = math.exp(max(0, (6.0 - dist)) * 0.35)
        mask_area_frac = obj.inst.mask.sum() / (obj.inst.mask.size + 1)
        size_mult = 1.0 + min(mask_area_frac * 8.0, 2.5)

        # FIX-POSE-3: pose risk factor applied to raw score
        pose_mult = obj.pose_risk_factor   # 1.0 standing, 0.20 seated

        raw = base * ttc_mult * prox_mult * size_mult * pose_mult
        return float(np.clip(raw, 0.0, 2000.0))

    # ── Smoothing ─────────────────────────────────────────────────────────

    def _apply_smoothing(self, tid, raw, is_stationary) -> float:
        rc    = self.cfg.risk
        alpha = rc.risk_ema_alpha

        if is_stationary:
            self._stationary_decay[tid] *= rc.stationary_decay
            self._stationary_decay[tid]  = max(self._stationary_decay[tid], 0.20)
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
            self._smooth_scores[tid] = a * raw_decayed + (1 - a) * prev

        smooth = self._smooth_scores[tid]
        self._score_histories[tid].append(smooth)
        return smooth

    # ── Trend ─────────────────────────────────────────────────────────────

    def _score_trend(self, tid) -> float:
        hist = list(self._score_histories[tid])
        if len(hist) < 4:
            return 0.0
        arr = np.array(hist[-8:], dtype=float)
        t   = np.arange(len(arr), dtype=float)
        return float(np.polyfit(t, arr, 1)[0])

    # ── Helpers ───────────────────────────────────────────────────────────

    def _ttc_multiplier(self, ttc_s) -> float:
        if ttc_s >= 100:  return 0.85
        if ttc_s >= 10:   return 1.0 + (10 - ttc_s) * 0.02
        if ttc_s >= 5:    return 1.0 + (10 - ttc_s) * 0.20
        if ttc_s >= 2:    return 2.0 + (5  - ttc_s) * 1.00
        return 5.0 + (2 - ttc_s) * 2.50

    def _severity(self, score: float, ttc_s: float,
                  pose_risk_factor: float = 1.0) -> str:
        rc = self.cfg.risk

        # FIX-POSE-4: seated persons capped at MEDIUM
        if pose_risk_factor < 0.5:
            if score >= rc.TIER_MEDIUM:
                return Severity.MEDIUM
            if score >= rc.TIER_LOW:
                return Severity.LOW
            return Severity.CLEAR

        # Normal severity logic for standing persons and vehicles
        if ttc_s < rc.TTC_CRITICAL_S:
            return Severity.CRITICAL
        if ttc_s < rc.TTC_HIGH_S and score >= rc.TIER_MEDIUM:
            return Severity.HIGH
        if score >= rc.TIER_CRITICAL:  return Severity.CRITICAL
        if score >= rc.TIER_HIGH:      return Severity.HIGH
        if score >= rc.TIER_MEDIUM:    return Severity.MEDIUM
        return Severity.LOW

    @staticmethod
    def _direction(obj: Object3D) -> str:
        lat = obj.lateral_m
        if abs(lat) < 0.40:    return "directly ahead"
        if lat < -1.0:         return "far left"
        if lat < -0.40:        return "left"
        if lat >  1.0:         return "far right"
        return "right"

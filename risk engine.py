"""
risk_engine_v3.py — SoundVision V3
=====================================
Implements the Vector-Intersection risk model:

    R_raw = (mass_weight × |velocity| × path_intersection_prob) / distance_m

Extended with:
  - Stationary suppression: decaying risk for static objects
  - TTC-gated urgency multiplier
  - EMA score smoothing per track (prevents jitter in alerts)
  - Score trend analysis (accelerating = higher danger)
  - Multi-threat output sorted by risk

Risk Score Interpretation
--------------------------
  ≥ 300 → CRITICAL (stop immediately)
  ≥ 120 → HIGH     (urgent caution)
  ≥  45 → MEDIUM   (be aware)
  ≥  15 → LOW      (informational)
  <  15 → ignore

All scores are in arbitrary units (AU); thresholds are tuned for
the chest-mount pedestrian use case.
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
    score:         float       # smoothed risk score
    raw_score:     float       # unsmoothed, for diagnostics
    severity:      str
    ttc_s:         float
    trend:         float       # score acceleration (positive = worsening)
    direction:     str         # spoken direction: "ahead", "left", "right", etc.
    distance_m:    float


# ─────────────────────────────────────────────────────────────────────────────
# Risk Engine V3
# ─────────────────────────────────────────────────────────────────────────────

class RiskEngineV3:
    """
    Vector-Intersection risk engine.

    Core formula
    -------------
    velocity_magnitude = ||(vx, vz)||   [m/frame, forward+lateral only]

    path_intersection = P(object will enter user corridor)
                      = pre-computed [0,1] from SpatialAnalyzerV3

    mass_weight = per-class danger weight (config.py)

    base_score = mass_weight × velocity_magnitude × path_intersection
                 / max(distance_m, 0.5)

    Then multiplied by:
      × TTC urgency factor   (higher when TTC < 5 s)
      × stationary decay     (if object hasn't moved for N frames)
      × size factor          (large masks = large objects = more danger)

    Finally EMA-smoothed per track to eliminate jitter.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rc = cfg.risk

        # Per-track smoothed scores
        self._smooth_scores: Dict[int, float] = {}
        # Per-track score history for trend analysis
        self._score_histories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=rc.score_history_len)
        )
        # Stationary decay accumulator
        self._stationary_decay: Dict[int, float] = defaultdict(lambda: 1.0)

    # ── Public ────────────────────────────────────────────────────────────

    def evaluate(self, objects: List[Object3D]) -> Optional[ThreatRecord]:
        """Return the single highest-risk ThreatRecord, or None."""
        threats = self.evaluate_all(objects)
        return threats[0] if threats else None

    def evaluate_all(self, objects: List[Object3D]) -> List[ThreatRecord]:
        """Return all threats above TIER_LOW, sorted highest-risk first."""
        rc = self.cfg.risk
        threats: List[ThreatRecord] = []

        active_ids = {o.track_id for o in objects}

        # Decay scores for objects no longer tracked
        for tid in list(self._smooth_scores.keys()):
            if tid not in active_ids:
                self._smooth_scores[tid] *= 0.85
                if self._smooth_scores[tid] < 1.0:
                    del self._smooth_scores[tid]

        for obj in objects:
            raw = self._compute_raw(obj)
            smooth = self._apply_smoothing(obj.track_id, raw, obj.is_stationary)

            if smooth < rc.TIER_LOW:
                continue

            sev   = self._severity(smooth, obj.ttc_s)
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

        # ── 1. Velocity vector magnitude (lateral + forward only) ──────────
        vel_mag = math.sqrt(vx**2 + vz**2)     # metres/frame
        vel_mag = max(vel_mag, 0.02)            # minimum ambient motion

        # ── 2. Closing component: forward velocity toward user ─────────────
        # vz negative = object approaching (Z decreasing)
        closing_vz = max(-vz, 0.0)             # only count approach
        vel_effective = vel_mag * 0.5 + closing_vz * 1.5   # weight closing

        # ── 3. Base score: (mass × velocity × path_p) / distance ──────────
        base = (mass * vel_effective * max(path_p, 0.05)) / dist

        # ── 4. TTC urgency multiplier ──────────────────────────────────────
        ttc_mult = self._ttc_multiplier(obj.ttc_s)

        # ── 5. Proximity exponential boost (very close = exponential danger) ─
        prox_mult = math.exp(max(0, (6.0 - dist)) * 0.35)

        # ── 6. Mask size factor (normalised by frame area) ─────────────────
        mask_area_frac = obj.inst.mask.sum() / (obj.inst.mask.size + 1)
        size_mult = 1.0 + min(mask_area_frac * 8.0, 2.5)

        raw = base * ttc_mult * prox_mult * size_mult
        return float(np.clip(raw, 0.0, 2000.0))

    # ── Smoothing ─────────────────────────────────────────────────────────

    def _apply_smoothing(self, tid: int, raw: float, is_stationary: bool) -> float:
        rc    = self.cfg.risk
        alpha = rc.risk_ema_alpha

        # Stationary decay
        if is_stationary:
            self._stationary_decay[tid] *= rc.stationary_decay
            self._stationary_decay[tid]  = max(self._stationary_decay[tid], 0.20)
        else:
            self._stationary_decay[tid] = min(
                self._stationary_decay[tid] * 1.15, 1.0
            )

        raw_decayed = raw * self._stationary_decay[tid]

        # EMA smoothing — asymmetric: react faster to increases
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

    def _score_trend(self, tid: int) -> float:
        """
        Score acceleration (2nd derivative of score history).
        Positive = worsening situation; negative = improving.
        """
        hist = list(self._score_histories[tid])
        if len(hist) < 4:
            return 0.0
        arr = np.array(hist[-8:], dtype=float)
        t   = np.arange(len(arr), dtype=float)
        slope = float(np.polyfit(t, arr, 1)[0])
        return slope

    # ── Helpers ───────────────────────────────────────────────────────────

    def _ttc_multiplier(self, ttc_s: float) -> float:
        """
        Non-linear TTC urgency multiplier.
        TTC  → 999 s : mult = 0.8  (no threat)
        TTC  →  10 s : mult = 1.0
        TTC  →   5 s : mult = 2.0
        TTC  →   2 s : mult = 5.0
        TTC  →   0 s : mult = 10.0
        """
        if ttc_s >= 100:
            return 0.85
        if ttc_s >= 10:
            return 1.0 + (10 - ttc_s) * 0.02         # 1.0 → 1.0
        if ttc_s >= 5:
            return 1.0 + (10 - ttc_s) * 0.20         # 1.0 → 2.0
        if ttc_s >= 2:
            return 2.0 + (5 - ttc_s)  * 1.00         # 2.0 → 5.0
        return 5.0 + (2 - ttc_s) * 2.50              # 5.0 → 10.0

    def _severity(self, score: float, ttc_s: float) -> str:
        rc = self.cfg.risk
        # TTC override: even moderate score becomes CRITICAL if TTC < 2.5 s
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
        """Convert lateral_m to a spoken direction word."""
        lat = obj.lateral_m
        if abs(lat) < 0.40:
            return "directly ahead"
        if lat < -1.0:
            return "far left"
        if lat < -0.40:
            return "left"
        if lat >  1.0:
            return "far right"
        return "right"

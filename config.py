"""
config.py — SoundVision V3
===========================
Centralized parameter store for the entire pipeline.
All tunable values live here — no magic numbers elsewhere.

Architecture: chest-mounted camera, pedestrian navigation.

CRITICAL CALIBRATION NOTES (updated after live-frame analysis)
--------------------------------------------------------------
Corridor geometry
  Previous near-width (0.80 m) was dangerously narrow — people standing
  one shoulder-width off-centre registered path_intersection ≈ 0 and
  scored below TIER_LOW, producing false "path clear" outputs.

  Real pedestrian clearance requirements:
    Single adult body width   : ~0.45 m
    Comfortable passing gap   : ~0.30 m each side
    Minimum safe corridor     : 0.45 + 0.30×2 = 1.05 m → rounded to 1.10 m
    Far end (6 m ahead)       : 2.00 m  (corridor widens with walking speed)

  corridor_width_near_m raised 0.80 → 1.10
  corridor_width_far_m  raised 1.60 → 2.20

Stationary suppression
  Previous floor (0.20) caused standing people to fall below TIER_LOW.
  A standing person at 1.5 m is just as dangerous as a slow-moving one.
  stationary_min raised 0.20 → 0.50   (never suppress more than 50%)
  stationary_decay softened 0.92 → 0.96 (slower decay per-frame)

Segmentation confidence
  conf_thresh lowered 0.40 → 0.30 to reduce missed detections of partially
  occluded pedestrians. Slightly more false positives are acceptable;
  false negatives are life-threatening.

TIER_LOW threshold
  Lowered 15.0 → 8.0 so that nearby stationary objects are announced.
  The audio cooldown system prevents annoying repetition; the risk engine
  is the wrong place to silently discard real obstacles.

TTS speech rate
  Lowered 160 → 145 wpm — intelligibility under stress favours slower
  speech for short urgent phrases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Camera & Optics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CameraConfig:
    """
    Physical camera parameters for the chest-mount setup.

    chest_height_m : metres above ground (average adult chest ≈ 1.2 m)
    hfov_deg       : horizontal FOV in degrees
    vfov_deg       : vertical FOV in degrees
    mount_tilt_deg : nominal downward tilt of lens from horizontal
                     (positive = looking down; chest mount ≈ 10–15°)
    """
    chest_height_m:          float = 1.20
    hfov_deg:                float = 69.0
    vfov_deg:                float = 43.0
    mount_tilt_deg:          float = 12.0

    # Auto-calibration horizon search band (fraction of frame height)
    horizon_search_top:      float = 0.25
    horizon_search_bot:      float = 0.75

    # Maximum correctable roll (degrees) — clamps auto-calibration
    max_roll_correction_deg: float = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Depth Estimation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DepthConfig:
    """
    MiDaS depth model configuration and metric scaling parameters.

    MiDaS outputs inverse-relative depth d̃ (higher = closer).
    Metric conversion:  Z_metric = scale / (d̃ + shift)

    scale and shift are estimated per-frame from the ground plane.
    """
    # Model: "MiDaS_small" (fast CPU) | "DPT_Hybrid" (accurate GPU)
    model_type:                  str   = "MiDaS_small"

    # Metric conversion priors (ground-calibrated each frame)
    default_scale:               float = 4.0
    default_shift:               float = 0.10

    # EMA for scale/shift calibration stability
    calib_ema_alpha:             float = 0.12

    # Savitzky-Golay temporal smoothing (odd window, degree ≤ window-1)
    sg_window_len:               int   = 7
    sg_poly_order:               int   = 2

    # EMA fallback alpha (used until SG buffer fills)
    ema_alpha:                   float = 0.20

    # Valid metric depth range (metres)
    min_depth_m:                 float = 0.30
    max_depth_m:                 float = 25.0

    # Ego-motion compensation threshold (metres/AI-frame)
    ego_motion_depth_threshold:  float = 0.08


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SegmentationConfig:
    """
    YOLO-seg inference parameters and COCO class groupings.

    conf_thresh lowered to 0.30 (was 0.40).
    For a safety-critical assistive device, missing a real person
    (false negative) is far more dangerous than announcing a phantom
    detection (false positive). The cooldown system handles spurious alerts.
    """
    model_path:   str   = "yolo11n-seg.pt"   # falls back to yolov8n-seg.pt
    conf_thresh:  float = 0.30               # LOWERED from 0.40 — safety priority
    iou_thresh:   float = 0.45
    input_imgsz:  int   = 640

    # Obstacle classes (COCO-80 IDs → label strings)
    OBSTACLE_CLASSES: Dict[int, str] = field(default_factory=lambda: {
        0:  "person",
        1:  "bicycle",
        2:  "car",
        3:  "motorcycle",
        5:  "bus",
        7:  "truck",
        9:  "traffic light",
        11: "stop sign",
        24: "backpack",
    })

    # Per-class physical heights (metres) — used as metric-depth anchors
    REAL_HEIGHTS_M: Dict[str, float] = field(default_factory=lambda: {
        "person":        1.75,
        "bicycle":       1.10,
        "car":           1.50,
        "motorcycle":    1.20,
        "bus":           3.00,
        "truck":         2.80,
        "traffic light": 2.50,
        "stop sign":     1.80,
        "backpack":      0.55,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Risk Engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskConfig:
    """
    Parameters for the vector-intersection risk engine.

    Core formula:
        R = (mass_weight * velocity_eff * path_intersection) / distance_m
          * ttc_multiplier * proximity_boost * size_factor
          * stationary_decay_factor

    CRITICAL CHANGES (live-frame calibration):
      corridor_width_near_m : 0.80 → 1.10  (was missing people off-centre)
      corridor_width_far_m  : 1.60 → 2.20  (proportional increase)
      stationary_decay      : 0.92 → 0.96  (slower; standing ≠ safe)
      stationary_min        : 0.20 → 0.50  (standing person still scores 50%)
      TIER_LOW              : 15.0 → 8.0   (nearby objects must be announced)
      vel_effective_floor   : see risk_engine — raised minimum ambient vel
    """
    HAZARD_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "person":        6.0,
        "bicycle":       9.0,
        "car":          20.0,
        "motorcycle":   14.0,
        "bus":          28.0,
        "truck":        28.0,
        "traffic light": 3.0,
        "stop sign":     2.0,
        "backpack":      4.0,
        "unknown":       3.0,
    })

    # Severity tier thresholds (risk score units)
    TIER_CRITICAL:         float = 300.0
    TIER_HIGH:             float = 120.0
    TIER_MEDIUM:           float =  45.0
    TIER_LOW:              float =   8.0   # LOWERED from 15.0

    # TTC thresholds (seconds)
    TTC_CRITICAL_S:        float =  2.5
    TTC_HIGH_S:            float =  5.0
    TTC_MEDIUM_S:          float = 10.0

    # Walking corridor full-width at near / far ends (metres)
    # WIDENED from (0.80, 1.60) — previous values missed people off-centre.
    # 1.10 m near = one adult body (0.45 m) + 0.325 m clearance each side.
    corridor_width_near_m: float = 1.10
    corridor_width_far_m:  float = 2.20

    # Stationary suppression
    # SOFTENED: stationary_min raised so standing people are never silent.
    # A parked car or standing person is still an obstacle.
    stationary_decay:      float = 0.96   # per-frame decay when static (was 0.92)
    stationary_min:        float = 0.50   # floor multiplier (was 0.20)

    # Risk score EMA smoothing
    risk_ema_alpha:        float = 0.35

    # Minimum corridor intersection fraction to register as threat
    min_intersection_frac: float = 0.05   # LOWERED from 0.08

    # Velocity estimation history window (frames)
    velocity_window:       int   = 8

    # Score history length for trend analysis
    score_history_len:     int   = 10

    # Minimum ambient velocity injected for stationary objects.
    # This ensures stationary objects in the corridor produce a non-trivial
    # base score purely from proximity + mass, not from motion.
    # Value: 0.08 m/frame ≈ 0.8 m/s — represents the user's own walking
    # speed closing on a stationary obstacle (i.e. the user IS the moving
    # party, so relative approach still exists).
    stationary_vel_floor:  float = 0.08


# ─────────────────────────────────────────────────────────────────────────────
# Guidance / Audio
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GuidanceConfig:
    """
    Alert cooldown and TTS parameters.

    Audio design principles for blind navigation:
      1. Urgency first  — CRITICAL starts with "Stop"
      2. Count matters  — "2 people ahead" is more useful than repeating the alert
      3. Relatable distance — "arm's reach" beats "0.7 metres"
      4. Clear action   — tell user what to DO, not just what's there
      5. Rate           — slower speech (145 wpm) is more intelligible under
                          the cognitive load of navigation
    """
    COOLDOWN_S: Dict[str, float] = field(default_factory=lambda: {
        "CRITICAL": 1.5,
        "HIGH":     3.0,
        "MEDIUM":   5.0,
        "LOW":     10.0,
    })
    tts_rate:      int   = 145          # LOWERED from 160 — better intelligibility
    clear_msg:     str   = "Path ahead is clear."
    clear_delay_s: float = 4.0

    # Distance bands for human-relatable distance phrases
    # These replace raw metre values in spoken output.
    # Key = upper bound (metres), value = spoken phrase
    # Evaluated in ascending order; first match wins.
    DISTANCE_PHRASES: Dict[float, str] = field(default_factory=lambda: {
         0.7: "right in front of you",   # arm's reach
         1.2: "very close",              # 1–2 steps
         2.0: "close",                   # 2–3 steps
         3.5: "nearby",                  # 3–5 steps
         6.0: "ahead",                   # 5–8 steps
        12.0: "in the distance",         # 8–15 steps
    })
    # Beyond 12 m: "far ahead" — rarely needs action immediately

    # Number of obstacles to announce in a multi-threat scene
    max_announced_threats: int = 2


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline / Performance
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    """Threading, frame-skip, and I/O configuration."""
    inference_queue_size: int   = 2
    target_ai_fps:        float = 10.0
    depth_frame_scale:    float = 0.50

    # Output video
    output_fourcc:        str   = "mp4v"
    output_ext:           str   = "mp4"
    output_dir:           str   = "/content"

    # HUD display toggles
    hud_font_scale:       float = 0.75
    hud_thickness:        int   = 2


# ─────────────────────────────────────────────────────────────────────────────
# Master Config Bundle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """
    Master configuration bundle.

    Call CFG.compute_intrinsics(width, height) once at pipeline startup
    to derive pixel-space focal lengths from the physical FOV settings.
    """
    camera:   CameraConfig       = field(default_factory=CameraConfig)
    depth:    DepthConfig        = field(default_factory=DepthConfig)
    seg:      SegmentationConfig = field(default_factory=SegmentationConfig)
    risk:     RiskConfig         = field(default_factory=RiskConfig)
    guidance: GuidanceConfig     = field(default_factory=GuidanceConfig)
    pipeline: PipelineConfig     = field(default_factory=PipelineConfig)

    # Derived pixel-space camera intrinsics (set by compute_intrinsics)
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    def compute_intrinsics(self, width: int, height: int) -> None:
        """
        Populate pixel-space focal lengths from FOV + resolution.

        fx = (W/2) / tan(HFOV/2)
        fy = (H/2) / tan(VFOV/2)
        """
        self.fx = (width  / 2.0) / math.tan(math.radians(self.camera.hfov_deg / 2.0))
        self.fy = (height / 2.0) / math.tan(math.radians(self.camera.vfov_deg / 2.0))
        self.cx = width  / 2.0
        self.cy = height / 2.0


# Pipeline-wide singleton — import this from all modules
CFG = Config()

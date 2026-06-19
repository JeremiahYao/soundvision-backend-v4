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

  corridor_width_near_m raised 0.80 → 1.10
  corridor_width_far_m  raised 1.60 → 2.20

Stationary suppression
  stationary_min raised 0.20 → 0.50
  stationary_decay softened 0.92 → 0.96

Segmentation confidence
  conf_thresh lowered 0.40 → 0.30

TIER_LOW threshold
  Lowered 15.0 → 8.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Camera & Optics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CameraConfig:
    chest_height_m:          float = 1.20
    hfov_deg:                float = 69.0
    vfov_deg:                float = 43.0
    mount_tilt_deg:          float = 12.0
    horizon_search_top:      float = 0.25
    horizon_search_bot:      float = 0.75
    max_roll_correction_deg: float = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Depth Estimation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DepthConfig:
    model_type:                  str   = "MiDaS_small"
    default_scale:               float = 4.0
    default_shift:               float = 0.10
    calib_ema_alpha:             float = 0.12
    sg_window_len:               int   = 7
    sg_poly_order:               int   = 2
    ema_alpha:                   float = 0.20
    min_depth_m:                 float = 0.30
    max_depth_m:                 float = 25.0
    ego_motion_depth_threshold:  float = 0.08


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SegmentationConfig:
    model_path:   str   = "yolo11n-seg.pt"
    conf_thresh:  float = 0.30
    iou_thresh:   float = 0.45
    input_imgsz:  int   = 640

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

    TIER_CRITICAL:         float = 300.0
    TIER_HIGH:             float = 120.0
    TIER_MEDIUM:           float =  45.0
    TIER_LOW:              float =   8.0

    TTC_CRITICAL_S:        float =  2.5
    TTC_HIGH_S:            float =  5.0
    TTC_MEDIUM_S:          float = 10.0

    corridor_width_near_m: float = 1.10
    corridor_width_far_m:  float = 2.20

    stationary_decay:      float = 0.96
    stationary_min:        float = 0.50

    risk_ema_alpha:        float = 0.35
    min_intersection_frac: float = 0.05
    velocity_window:       int   = 8
    score_history_len:     int   = 10
    stationary_vel_floor:  float = 0.08


# ─────────────────────────────────────────────────────────────────────────────
# Guidance / Audio
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GuidanceConfig:
    COOLDOWN_S: Dict[str, float] = field(default_factory=lambda: {
        "CRITICAL": 1.5,
        "HIGH":     3.0,
        "MEDIUM":   5.0,
        "LOW":     10.0,
    })
    tts_rate:      int   = 145
    clear_msg:     str   = "Path ahead is clear."
    clear_delay_s: float = 4.0

    DISTANCE_PHRASES: Dict[float, str] = field(default_factory=lambda: {
         0.7: "right in front of you",
         1.2: "very close",
         2.0: "close",
         3.5: "nearby",
         6.0: "ahead",
        12.0: "in the distance",
    })

    max_announced_threats: int = 2


# ─────────────────────────────────────────────────────────────────────────────
# Text Reader  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TextReaderConfig:
    """
    EasyOCR configuration for the text-reading feature.

    languages:
        List of language codes EasyOCR loads.
        'en'      — English (all Latin-script Singapore signage)
        'ch_sim'  — Simplified Chinese (MRT maps, hawker signs, product labels)
        Adding languages increases model size and load time.

    min_confidence:
        OCR result confidence threshold (0–1).
        Results below this are discarded as likely noise.
        0.4 is a safe starting point; raise to 0.5 if you get too many
        phantom words on textured backgrounds.

    min_text_length:
        Minimum character length of a text block to be spoken.
        Filters out single stray characters (punctuation, specks).
    """
    languages:        List[str] = field(default_factory=lambda: ["en", "ch_sim"])
    min_confidence:   float     = 0.40
    min_text_length:  int       = 2


# ─────────────────────────────────────────────────────────────────────────────
# Voice Control  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VoiceConfig:
    """
    Whisper speech recognition and hotkey configuration.

    whisper_model:
        Size of the local Whisper model to load.
        'tiny'   — fastest, ~150 MB download, good enough for short commands
        'base'   — more accurate, ~290 MB, still fast on Jetson GPU
        For laptop CPU testing, stick with 'tiny'.

    record_seconds:
        How long to record after the trigger key is pressed.
        4 seconds is enough for any command. Keep it short so feedback is fast.

    sample_rate:
        Audio sample rate in Hz. 16000 is Whisper's native rate — do not change.

    trigger_key:
        Keyboard key that activates voice input.
        'f1'   — recommended for laptop testing (corner key, easy to find by touch)
        Change to 'space' if you prefer, but space conflicts with some apps.
        On the Jetson hardware, this is replaced by a physical GPIO button.
    """
    whisper_model:  str   = "tiny"
    record_seconds: float = 4.0
    sample_rate:    int   = 16000
    trigger_key:    str   = "f1"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline / Performance
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    inference_queue_size: int   = 2
    target_ai_fps:        float = 5.0   # 5 FPS for laptop CPU; raise to 10 on Jetson
    depth_frame_scale:    float = 0.50
    output_fourcc:        str   = "mp4v"
    output_ext:           str   = "mp4"
    output_dir:           str   = "/content"
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
    camera:      CameraConfig      = field(default_factory=CameraConfig)
    depth:       DepthConfig       = field(default_factory=DepthConfig)
    seg:         SegmentationConfig= field(default_factory=SegmentationConfig)
    risk:        RiskConfig        = field(default_factory=RiskConfig)
    guidance:    GuidanceConfig    = field(default_factory=GuidanceConfig)
    pipeline:    PipelineConfig    = field(default_factory=PipelineConfig)
    text_reader: TextReaderConfig  = field(default_factory=TextReaderConfig)
    voice:       VoiceConfig       = field(default_factory=VoiceConfig)

    # Derived pixel-space camera intrinsics (set by compute_intrinsics)
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    def compute_intrinsics(self, width: int, height: int) -> None:
        self.fx = (width  / 2.0) / math.tan(math.radians(self.camera.hfov_deg / 2.0))
        self.fy = (height / 2.0) / math.tan(math.radians(self.camera.vfov_deg / 2.0))
        self.cx = width  / 2.0
        self.cy = height / 2.0


# Pipeline-wide singleton — import this from all modules
CFG = Config()

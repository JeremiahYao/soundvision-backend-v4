"""
config.py — SoundVision V3
===========================
Centralized parameter store for the entire pipeline.
All tunable values live here — no magic numbers elsewhere.

Architecture: chest-mounted camera, pedestrian navigation.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Camera & Optics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CameraConfig:
    """
    Physical camera parameters for the chest-mount setup.

    chest_height_m : metres above ground — average adult chest ≈ 1.2 m
    hfov_deg       : horizontal field of view (degrees)
    vfov_deg       : vertical field of view (degrees)
    mount_tilt_deg : nominal downward tilt of lens from horizontal
                     (positive = looking down; chest mount ≈ 10–15°)
    """
    chest_height_m:    float = 1.20
    hfov_deg:          float = 69.0    # common wide-angle mobile lens
    vfov_deg:          float = 43.0
    mount_tilt_deg:    float = 12.0    # lens tilted slightly downward

    # Auto-calibration: ground-plane horizon search band [top%, bot%] of frame
    horizon_search_top:  float = 0.25
    horizon_search_bot:  float = 0.70

    # Roll correction: max correctable roll angle in degrees
    max_roll_correction_deg: float = 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Depth Estimation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DepthConfig:
    """
    MiDaS depth model configuration and metric scaling parameters.

    MiDaS outputs inverse-relative depth (disparity-like).
    We convert to metric using:  metric_m = scale / (raw_depth + shift)
    Scale & shift are estimated per-frame from the ground-plane mask.
    """
    model_type:       str   = "MiDaS_small"   # "MiDaS_small" | "DPT_Hybrid"
    input_size:       int   = 256              # resize short side to this

    # Metric conversion (ground-plane bootstrapped)
    default_scale:    float = 3.5
    default_shift:    float = 0.15

    # Savitzky-Golay filter for depth smoothing (per-pixel temporal)
    sg_window_len:    int   = 7     # must be odd
    sg_poly_order:    int   = 2

    # EMA fallback alpha (lower = smoother, higher = more responsive)
    ema_alpha:        float = 0.18

    # Valid depth range in metres
    min_depth_m:      float = 0.3
    max_depth_m:      float = 30.0

    # Ego-motion compensation: depth change threshold below which
    # the object is considered "stationary relative to observer" (metres/frame)
    ego_motion_depth_threshold: float = 0.08


# ─────────────────────────────────────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SegmentationConfig:
    """
    YOLOv11-seg (or YOLOv8-seg) configuration.

    COCO class IDs for each semantic category used by the pipeline.
    """
    model_path:   str   = "yolo11n-seg.pt"    # falls back to yolov8n-seg.pt
    conf_thresh:  float = 0.40
    iou_thresh:   float = 0.45
    input_imgsz:  int   = 640

    # ── COCO class groupings ──────────────────────────────────────────────
    # Ground-plane classes (define the walkable corridor)
    GROUND_CLASSES: Tuple[int, ...] = (
        13,   # bench (proxy for sidewalk furniture)
    )

    # The actual walkable surface detection relies on low-position heuristics
    # when explicit segmentation classes are absent; see perception.py.

    # Obstacle classes — objects that can collide with the user
    OBSTACLE_CLASSES: Dict[int, str] = field(default_factory=lambda: {
        0:  "person",
        1:  "bicycle",
        2:  "car",
        3:  "motorcycle",
        5:  "bus",
        7:  "truck",
        9:  "traffic light",
        11: "stop sign",
        24: "backpack",   # pedestrian proxy when person mask fails
    })

    # ── Per-class physical dimensions (metres) for metric anchor ─────────
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

    Risk formula (per obstacle):
        R = (mass_weight * velocity_magnitude * path_intersection_prob) / distance_m
    Scaled and clamped to [0, 1000].
    """

    # Object mass/danger weights
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

    # Severity tier thresholds (raw risk score)
    TIER_CRITICAL:  float = 300.0
    TIER_HIGH:      float = 120.0
    TIER_MEDIUM:    float = 45.0
    TIER_LOW:       float = 15.0

    # TTC thresholds (seconds)
    TTC_CRITICAL_S:  float = 2.5
    TTC_HIGH_S:      float = 5.0
    TTC_MEDIUM_S:    float = 10.0

    # Path corridor width at user position (metres, full width)
    corridor_width_near_m: float = 0.80   # at 0 m ahead
    corridor_width_far_m:  float = 1.60   # at max_depth_m ahead

    # Stationary suppression: decay factor per frame when object is static
    stationary_decay: float = 0.92

    # EMA for smoothing risk scores (prevents jitter in audio alerts)
    risk_ema_alpha:   float = 0.35

    # Minimum intersection area (fraction of obstacle mask) to trigger risk
    min_intersection_frac: float = 0.08

    # Velocity estimation window (frames)
    velocity_window: int = 8

    # Score history for trend analysis
    score_history_len: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# Guidance / Audio
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GuidanceConfig:
    """Alert cooldown and TTS parameters."""

    # Minimum seconds between spoken alerts per tracked object
    COOLDOWN_S: Dict[str, float] = field(default_factory=lambda: {
        "CRITICAL": 1.5,
        "HIGH":     3.0,
        "MEDIUM":   5.0,
        "LOW":      9.0,
    })

    tts_rate:      int   = 160     # words per minute (pyttsx3)
    clear_msg:     str   = "Path is clear."
    clear_delay_s: float = 4.0     # wait this long before speaking "clear"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline / Performance
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    """Threading, frame-skip, and I/O configuration."""

    # Async depth inference: runs in parallel thread
    depth_thread_enabled:   bool  = True

    # Async segmentation: also parallel (requires 2× GPU memory)
    seg_thread_enabled:     bool  = True

    # Maximum queued frames in inference queues (drop-oldest policy)
    inference_queue_size:   int   = 2

    # Target AI inference rate (Hz); adaptive skipper adjusts to meet this
    target_ai_fps:          float = 10.0

    # Performance: resize frame before depth inference
    depth_frame_scale:      float = 0.50    # 0.5 = half-res for depth model

    # Output
    output_fourcc:          str   = "XVID"
    output_ext:             str   = "avi"
    output_dir:             str   = "/content"

    # Display
    hud_font_scale:         float = 0.75
    hud_thickness:          int   = 2


# ─────────────────────────────────────────────────────────────────────────────
# Master config bundle
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    camera:      CameraConfig      = field(default_factory=CameraConfig)
    depth:       DepthConfig       = field(default_factory=DepthConfig)
    seg:         SegmentationConfig = field(default_factory=SegmentationConfig)
    risk:        RiskConfig        = field(default_factory=RiskConfig)
    guidance:    GuidanceConfig    = field(default_factory=GuidanceConfig)
    pipeline:    PipelineConfig    = field(default_factory=PipelineConfig)

    # Derived: focal lengths (pixels) computed from FOV + frame size
    # Call cfg.compute_intrinsics(width, height) before pipeline starts.
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    def compute_intrinsics(self, width: int, height: int):
        """Populate pixel-space camera intrinsics from FOV + resolution."""
        import math
        self.fx = (width  / 2.0) / math.tan(math.radians(self.camera.hfov_deg / 2.0))
        self.fy = (height / 2.0) / math.tan(math.radians(self.camera.vfov_deg / 2.0))
        self.cx = width  / 2.0
        self.cy = height / 2.0


# Singleton used throughout the pipeline
CFG = Config()

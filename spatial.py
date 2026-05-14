"""
spatial_v3.py — SoundVision V3
================================
Converts 2D segmentation masks + metric depth into a full 3D coordinate
system anchored to the ground plane.

Mathematical Foundation
------------------------
We use a standard pinhole camera model with known intrinsics (fx, fy, cx, cy).

For any pixel (u, v) with metric depth d (metres from camera), the 3D point
in camera coordinates is:

    X_cam = (u - cx) * d / fx       [metres, lateral]
    Y_cam = (v - cy) * d / fy       [metres, vertical — positive DOWN]
    Z_cam = d                        [metres, forward]

Ground Plane Frame
------------------
We rotate from camera frame to a "world" frame where Y=0 is the ground,
X is lateral (right = positive), Z is forward.

The camera is mounted at height H above the ground with a downward tilt θ.
The rotation from camera → world around the X-axis by angle θ gives:

    X_w =  X_cam
    Y_w =  Y_cam * cos(θ) - Z_cam * sin(θ)
    Z_w =  Y_cam * sin(θ) + Z_cam * cos(θ)

After applying this rotation, ground points satisfy Y_w ≈ -H (camera height).
The user stands at (X_w=0, Z_w=0). Forward distance is Z_w.

Walking Corridor (Dynamic Trapezoid)
-------------------------------------
The danger zone is a trapezoid in the (X_w, Z_w) ground plane:
  - Near edge (Z_w = 0): width = corridor_near_m (e.g. 0.8 m)
  - Far edge  (Z_w = Z_max): width = corridor_far_m  (e.g. 1.6 m)

A point (X_w, Z_w) is inside the corridor if:
    |X_w| ≤ (corridor_near_m/2) + (Z_w / Z_max) * ((corridor_far_m - corridor_near_m) / 2)

Pixel-Space Corridor
---------------------
We project the corridor trapezoid back to 2D:
    u = fx * X_w / Z_w + cx
    v = fy * (Y_world_ref) / Z_w + cy
where Y_world_ref is the camera height (i.e. ground level).

This gives 4 corners of the corridor trapezoid overlaid on the frame.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

from perception import PerceptionOutput, InstanceMask
from config import CFG, Config

log = logging.getLogger("SV3.Spatial")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Object3D:
    """
    A detected obstacle fully described in 3D world coordinates.
    """
    inst:               InstanceMask       # original 2D detection
    label:              str
    track_id:           int

    # 3D position of the closest point of the mask to the user (metres)
    closest_point_m:    Tuple[float, float, float]   # (X, Y, Z) world frame
    distance_m:         float                          # Euclidean distance
    forward_distance_m: float                          # Z_w only

    # Centroid in 3D
    centroid_m:         Tuple[float, float, float]

    # Lateral offset in world frame (metres, + = right)
    lateral_m:          float

    # Velocity vector (metres/frame) — populated by SpatialAnalyzer
    velocity:           Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # TTC from depth delta (seconds) — populated by SpatialAnalyzer
    ttc_s:              float = 999.0

    # Intersection probability with walking corridor [0,1]
    path_intersection:  float = 0.0

    # Is the object stationary relative to ego-motion?
    is_stationary:      bool  = False

    # Raw risk score (set by RiskEngine)
    risk_score:         float = 0.0


@dataclass
class CorridorTrapezoid:
    """
    Walking corridor in both 3D world and 2D pixel coordinates.
    """
    # World-frame corners: (X_w, Z_w) pairs — near-left, near-right, far-right, far-left
    world_corners:  List[Tuple[float, float]]

    # Pixel-space corners: (u, v) pairs — same order
    pixel_corners:  List[Tuple[int, int]]

    # Pixel mask of the corridor on the frame (bool H×W)
    corridor_mask:  np.ndarray

    # Corridor half-width at any given Z
    near_half_w:    float
    far_half_w:     float
    z_max:          float


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate transforms
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_3d(
    u: float, v: float, depth_m: float,
    fx: float, fy: float, cx: float, cy: float,
    tilt_rad: float
) -> Tuple[float, float, float]:
    """
    Convert a single pixel (u,v) + metric depth to world 3D coordinates.
    Applies camera-tilt rotation around X-axis.

    Returns (X_w, Y_w, Z_w) in metres.
    """
    # Camera-frame coordinates
    X_cam = (u - cx) * depth_m / fx
    Y_cam = (v - cy) * depth_m / fy
    Z_cam = depth_m

    # Rotate by tilt angle around X-axis (camera tilted downward → positive tilt)
    cos_t = math.cos(tilt_rad)
    sin_t = math.sin(tilt_rad)

    X_w =  X_cam
    Y_w =  Y_cam * cos_t - Z_cam * sin_t
    Z_w =  Y_cam * sin_t + Z_cam * cos_t

    return X_w, Y_w, Z_w


def world_to_pixel(
    X_w: float, Y_w: float, Z_w: float,
    fx: float, fy: float, cx: float, cy: float,
    tilt_rad: float,
    camera_height_m: float
) -> Optional[Tuple[int, int]]:
    """
    Project a 3D world point back to pixel (u, v).
    Returns None if point is behind camera (Z_cam ≤ 0).
    """
    if Z_w <= 0.01:
        return None

    # Inverse tilt rotation (world → camera)
    cos_t = math.cos(tilt_rad)
    sin_t = math.sin(tilt_rad)

    X_cam =  X_w
    Y_cam =  Y_w * cos_t + Z_w * sin_t
    Z_cam = -Y_w * sin_t + Z_w * cos_t

    if Z_cam <= 0.01:
        return None

    u = int(fx * X_cam / Z_cam + cx)
    v = int(fy * Y_cam / Z_cam + cy)
    return u, v


def mask_to_3d_points(
    mask: np.ndarray,
    depth_smooth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    tilt_rad: float,
    subsample: int = 8,
) -> Optional[np.ndarray]:
    """
    Convert all True pixels in `mask` to 3D world coordinates.

    Parameters
    ----------
    subsample : Only process every Nth pixel (speed optimisation).

    Returns
    -------
    np.ndarray of shape (N, 3) in metres, or None if mask is empty.
    """
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None

    # Subsample
    idx   = np.arange(0, len(rows), subsample)
    rows  = rows[idx]
    cols  = cols[idx]

    depths = depth_smooth[rows, cols].astype(np.float64)

    # Camera frame
    X_cam = (cols - cx) * depths / fx
    Y_cam = (rows - cy) * depths / fy
    Z_cam = depths

    # Tilt rotation (vectorised)
    cos_t = math.cos(tilt_rad)
    sin_t = math.sin(tilt_rad)

    X_w =  X_cam
    Y_w =  Y_cam * cos_t - Z_cam * sin_t
    Z_w =  Y_cam * sin_t + Z_cam * cos_t

    pts = np.stack([X_w, Y_w, Z_w], axis=1)   # (N, 3)

    # Filter out points behind camera and above reasonable height
    valid = (Z_w > 0.1) & (Z_w < 35.0)
    return pts[valid] if valid.any() else None


# ─────────────────────────────────────────────────────────────────────────────
# Corridor Builder
# ─────────────────────────────────────────────────────────────────────────────

class CorridorBuilder:
    """
    Constructs the 3D walking corridor trapezoid for each frame.
    Adapts dynamically to:
      - Horizon line (auto-calibrated from ground mask)
      - Camera roll (corrects trapezoid lean)
      - Estimated tilt
    """

    def __init__(self, cfg: Config, frame_h: int, frame_w: int):
        self.cfg     = cfg
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.near_hw = cfg.risk.corridor_width_near_m / 2.0
        self.far_hw  = cfg.risk.corridor_width_far_m  / 2.0
        self.z_max   = cfg.depth.max_depth_m

    def build(
        self,
        horizon_y: int,
        roll_deg: float,
        tilt_rad: float,
    ) -> CorridorTrapezoid:
        """
        Build the corridor for this frame.
        Returns a CorridorTrapezoid with world corners, pixel corners, and mask.
        """
        # World-frame corners of trapezoid in (X_w, Z_w) ground plane
        # (Y_w ≈ -camera_height for ground-level points)
        world_corners = [
            (-self.near_hw, 0.5),                # near-left
            ( self.near_hw, 0.5),                # near-right
            ( self.far_hw,  self.z_max),         # far-right
            (-self.far_hw,  self.z_max),         # far-left
        ]

        # Project to pixels
        pix_corners = []
        camera_height = self.cfg.camera.chest_height_m
        for (Xw, Zw) in world_corners:
            Yw = -camera_height   # ground level
            pt = world_to_pixel(
                Xw, Yw, Zw,
                CFG.fx, CFG.fy, CFG.cx, CFG.cy,
                tilt_rad, camera_height
            )
            if pt is None:
                # Fallback: project to horizon
                pt = (int(self.frame_w / 2 + Xw * CFG.fx / max(Zw, 0.1)), horizon_y)
            pix_corners.append(pt)

        # Roll correction: rotate pixel corners around frame centre
        if abs(roll_deg) > 0.5:
            pix_corners = self._apply_roll(pix_corners, roll_deg)

        # Rasterise corridor polygon to mask
        pts_arr = np.array(pix_corners, dtype=np.int32).reshape((-1, 1, 2))
        mask    = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_arr], 1)
        corridor_mask = mask.astype(bool)

        return CorridorTrapezoid(
            world_corners = world_corners,
            pixel_corners = pix_corners,
            corridor_mask = corridor_mask,
            near_half_w   = self.near_hw,
            far_half_w    = self.far_hw,
            z_max         = self.z_max,
        )

    def _apply_roll(self, corners, roll_deg):
        """Rotate 2D points around image centre by roll angle."""
        cx, cy = self.frame_w / 2, self.frame_h / 2
        rad    = math.radians(-roll_deg)   # counter-rotate to correct
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        corrected = []
        for (u, v) in corners:
            dx, dy = u - cx, v - cy
            u2 = int(cx + dx * cos_r - dy * sin_r)
            v2 = int(cy + dx * sin_r + dy * cos_r)
            corrected.append((u2, v2))
        return corrected

    @staticmethod
    def point_in_corridor(
        X_w: float, Z_w: float,
        near_half: float, far_half: float, z_max: float
    ) -> float:
        """
        Returns intersection probability [0,1] of a 3D point with corridor.
        Uses a soft sigmoid instead of a hard boundary.
        """
        if Z_w <= 0 or Z_w > z_max:
            return 0.0
        half_w = near_half + (far_half - near_half) * (Z_w / z_max)
        dist_from_centre = abs(X_w)
        # Soft boundary: sigmoid decay outside corridor
        margin = half_w * 0.2   # 20% of half-width as transition zone
        excess = dist_from_centre - half_w
        if excess <= 0:
            return 1.0
        elif excess > margin * 3:
            return 0.0
        else:
            return float(1.0 / (1.0 + math.exp(excess / margin * 5)))


# ─────────────────────────────────────────────────────────────────────────────
# SpatialAnalyzerV3
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAnalyzerV3:
    """
    Converts PerceptionOutput → List[Object3D].

    For each obstacle mask:
    1. Project all mask pixels to 3D world coordinates.
    2. Find closest mask point to user (minimum Z_w).
    3. Estimate velocity vector from per-track history (depth delta).
    4. Compute TTC from depth derivative (not bounding-box size).
    5. Determine corridor intersection probability.
    6. Flag stationary objects via ego-motion-compensated depth delta.
    """

    # History depth: frames to keep per track
    HISTORY_LEN = 12

    def __init__(self, cfg: Config, frame_h: int, frame_w: int):
        self.cfg      = cfg
        self.corridor = CorridorBuilder(cfg, frame_h, frame_w)

        # Per-track history: track_id → list of (X_w, Y_w, Z_w) centroid
        self._track_history:   Dict[int, List[Tuple[float,float,float]]] = {}
        # Per-track depth history for TTC
        self._depth_history:   Dict[int, List[float]] = {}
        # Per-track smoothed risk for stationary suppression
        self._stationary_cnt:  Dict[int, int] = {}

        self._tilt_rad = math.radians(cfg.camera.mount_tilt_deg)

    # ── Public ────────────────────────────────────────────────────────────

    def analyze(self, perc: PerceptionOutput) -> Tuple[List[Object3D], CorridorTrapezoid]:
        """
        Main entry point.

        Returns
        -------
        objects   : List[Object3D] with all spatial fields populated
        corridor  : CorridorTrapezoid for this frame (for HUD overlay)
        """
        # Rebuild corridor with latest horizon + roll
        corridor = self.corridor.build(
            perc.horizon_y, perc.roll_deg, self._tilt_rad
        )

        objects: List[Object3D] = []

        for inst in perc.obstacle_masks:
            obj3d = self._process_instance(inst, perc, corridor)
            if obj3d is not None:
                objects.append(obj3d)

        # Prune stale tracks
        active_ids = {o.track_id for o in objects}
        for tid in list(self._track_history.keys()):
            if tid not in active_ids:
                del self._track_history[tid]
                self._depth_history.pop(tid, None)
                self._stationary_cnt.pop(tid, None)

        return objects, corridor

    # ── Internal ──────────────────────────────────────────────────────────

    def _process_instance(
        self,
        inst: InstanceMask,
        perc: PerceptionOutput,
        corridor: CorridorTrapezoid,
    ) -> Optional[Object3D]:
        """Process one obstacle instance → Object3D."""

        pts3d = mask_to_3d_points(
            inst.mask,
            perc.depth_smooth,
            CFG.fx, CFG.fy, CFG.cx, CFG.cy,
            self._tilt_rad,
            subsample=6,
        )

        if pts3d is None or len(pts3d) < 5:
            return None

        # ── Closest point (minimum forward distance) ──────────────────────
        z_vals      = pts3d[:, 2]
        closest_idx = int(np.argmin(z_vals))
        closest_pt  = tuple(pts3d[closest_idx])
        closest_z   = float(closest_pt[2])

        if closest_z < self.cfg.depth.min_depth_m:
            closest_z = self.cfg.depth.min_depth_m

        # ── Centroid ──────────────────────────────────────────────────────
        centroid = tuple(pts3d.mean(axis=0).tolist())

        # ── Euclidean distance ─────────────────────────────────────────────
        dist_m = float(np.sqrt(centroid[0]**2 + centroid[2]**2))

        # ── Track history & velocity ───────────────────────────────────────
        tid = inst.track_id
        if tid not in self._track_history:
            self._track_history[tid]  = []
            self._depth_history[tid]  = []
            self._stationary_cnt[tid] = 0

        hist_xyz = self._track_history[tid]
        hist_xyz.append(centroid)
        if len(hist_xyz) > self.HISTORY_LEN:
            hist_xyz.pop(0)

        velocity = self._estimate_velocity(hist_xyz)

        # ── TTC from depth derivative ─────────────────────────────────────
        hist_d = self._depth_history[tid]
        hist_d.append(closest_z)
        if len(hist_d) > self.HISTORY_LEN:
            hist_d.pop(0)

        ttc_s = self._compute_ttc(hist_d)

        # ── Stationary suppression ────────────────────────────────────────
        is_stationary = self._check_stationary(tid, hist_d, velocity)

        # ── Corridor intersection ─────────────────────────────────────────
        # Use a sample of 3D points to estimate intersection fraction
        path_probs = [
            CorridorBuilder.point_in_corridor(
                p[0], p[2],
                corridor.near_half_w, corridor.far_half_w, corridor.z_max
            )
            for p in pts3d[::3]   # every 3rd point for speed
        ]
        path_intersection = float(np.mean(path_probs)) if path_probs else 0.0

        return Object3D(
            inst               = inst,
            label              = inst.label,
            track_id           = tid,
            closest_point_m    = closest_pt,
            distance_m         = dist_m,
            forward_distance_m = closest_z,
            centroid_m         = centroid,
            lateral_m          = float(centroid[0]),
            velocity           = velocity,
            ttc_s              = ttc_s,
            path_intersection  = path_intersection,
            is_stationary      = is_stationary,
        )

    def _estimate_velocity(
        self, hist: List[Tuple[float,float,float]]
    ) -> Tuple[float, float, float]:
        """
        Velocity in metres/frame via linear regression on position history.
        More robust than frame-to-frame delta.
        """
        n = len(hist)
        if n < 2:
            return (0.0, 0.0, 0.0)

        pts = np.array(hist)
        t   = np.arange(n, dtype=float)

        # polyfit slope = velocity (m/frame)
        vx = float(np.polyfit(t, pts[:, 0], 1)[0])
        vy = float(np.polyfit(t, pts[:, 1], 1)[0])
        vz = float(np.polyfit(t, pts[:, 2], 1)[0])

        return (vx, vy, vz)

    def _compute_ttc(self, depth_hist: List[float]) -> float:
        """
        Time-to-collision from depth history (pixel-depth delta method).

        TTC = current_depth / (-rate_of_change)
        where rate_of_change is derived from linear regression on depth history.

        Positive rate = object approaching (depth decreasing).
        Returns TTC in seconds (assuming ~15 AI frames/sec).
        """
        if len(depth_hist) < 3:
            return 999.0

        d   = np.array(depth_hist, dtype=np.float64)
        t   = np.arange(len(d), dtype=np.float64)

        slope = float(np.polyfit(t, d, 1)[0])   # metres per AI-frame

        # Negative slope = approaching
        if slope >= -0.005:
            return 999.0   # not closing

        ai_fps   = self.cfg.pipeline.target_ai_fps
        rate_m_s = slope * ai_fps     # m/s (negative)
        current  = float(d[-1])

        ttc = -current / rate_m_s    # seconds
        return float(np.clip(ttc, 0.1, 999.0))

    def _check_stationary(
        self,
        tid: int,
        depth_hist: List[float],
        velocity: Tuple[float, float, float],
    ) -> bool:
        """
        Object is stationary if its depth is not changing beyond the
        ego-motion threshold (user walking naturally causes depth changes).
        """
        if len(depth_hist) < 4:
            return False

        # Standard deviation of depth over window
        d_std = float(np.std(depth_hist[-6:]))
        ego_thresh = self.cfg.depth.ego_motion_depth_threshold

        speed_3d = math.sqrt(sum(v**2 for v in velocity))

        if d_std < ego_thresh and speed_3d < 0.05:
            self._stationary_cnt[tid] = self._stationary_cnt.get(tid, 0) + 1
        else:
            self._stationary_cnt[tid] = max(0, self._stationary_cnt.get(tid, 0) - 1)

        # Confirmed stationary after 5 consecutive frames of low motion
        return self._stationary_cnt[tid] >= 5

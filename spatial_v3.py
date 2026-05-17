"""
spatial_v3.py — SoundVision V3
================================
Converts 2D segmentation masks + metric depth into a full 3D coordinate
system anchored to the ground plane.

Mathematical Foundation
────────────────────────
Standard pinhole camera model with intrinsics (fx, fy, cx, cy).

For any pixel (u, v) with metric depth d (metres from camera):

    X_cam = (u - cx) * d / fx       [lateral,  metres]
    Y_cam = (v - cy) * d / fy       [vertical, metres — positive DOWN]
    Z_cam = d                        [forward,  metres]

Camera tilt θ (downward positive) → world frame via X-axis rotation:

    X_w =  X_cam
    Y_w =  Y_cam * cos(θ) - Z_cam * sin(θ)
    Z_w =  Y_cam * sin(θ) + Z_cam * cos(θ)

Ground points satisfy Y_w ≈ −H (camera height).
The user is at (X_w=0, Z_w=0). Forward distance = Z_w.

Walking Corridor
─────────────────
Trapezoid in the (X_w, Z_w) ground plane:
  near half-width (Z=0):   corridor_width_near_m / 2
  far  half-width (Z=max): corridor_width_far_m  / 2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Changes in this version (v4) — Corridor alignment fix
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIX-CORR-1  Hard lateral exclusion zone added to point_in_corridor().
            Any object whose 3D centroid is more than MAX_LATERAL_M
            metres to the side is excluded from corridor risk entirely,
            regardless of sigmoid value. This prevents cars in the
            adjacent road lane from scoring as path threats even when
            the corridor trapezoid pixel mask clips them at depth.

            MAX_LATERAL_M = 1.8 m  (typical pavement half-width).
            Vehicles rarely threaten a pedestrian from > 1.8m lateral
            without also being dead ahead; if they are that far lateral
            they are beside the path, not in it.

FIX-CORR-2  Sigmoid transition band tightened.
            margin was half_w * 0.20 (20% soft edge).
            Now half_w * 0.10 (10%) — sharper boundary between
            in-corridor and out-of-corridor. This stops objects that
            are clearly beside the path from accumulating sigmoid
            probability tail scores.

FIX-CORR-3  Corridor z_max capped at 15 m for risk calculation.
            The visual corridor still shows to depth.max_depth_m (25m)
            but path_intersection is only calculated up to 15m.
            Beyond 15m, objects are too far away to be an immediate
            pedestrian threat; vehicles at 20m+ on an adjacent road
            were contributing non-zero scores.

FIX-POSE-2  Object3D gains is_seated and pose_risk_factor fields.
            These are read from InstanceMask (set by PostureClassifier
            in perception.py) and passed to risk_engine_v3.py.
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
# Corridor tuning constants
# ─────────────────────────────────────────────────────────────────────────────

# FIX-CORR-1: maximum lateral offset (metres) for any object to be in-corridor
MAX_LATERAL_M: float = 1.8

# FIX-CORR-2: sigmoid margin as fraction of half-width (was 0.20)
CORRIDOR_SIGMOID_MARGIN_FRAC: float = 0.10

# FIX-CORR-3: maximum depth for path intersection scoring
CORRIDOR_RISK_Z_MAX_M: float = 15.0


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Object3D:
    """A detected obstacle fully described in 3D world coordinates."""
    inst:               InstanceMask
    label:              str
    track_id:           int

    closest_point_m:    Tuple[float, float, float]
    distance_m:         float
    forward_distance_m: float
    centroid_m:         Tuple[float, float, float]
    lateral_m:          float

    velocity:           Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ttc_s:              float = 999.0
    path_intersection:  float = 0.0
    is_stationary:      bool  = False
    risk_score:         float = 0.0

    # FIX-POSE-2: pose fields propagated from InstanceMask
    is_seated:          bool  = False
    pose_risk_factor:   float = 1.0


@dataclass
class CorridorTrapezoid:
    """Walking corridor in 3D world and 2D pixel coordinates."""
    world_corners:  List[Tuple[float, float]]
    pixel_corners:  List[Tuple[int, int]]
    corridor_mask:  np.ndarray
    near_half_w:    float
    far_half_w:     float
    z_max:          float


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate transforms
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_3d(u, v, depth_m, fx, fy, cx, cy, tilt_rad):
    X_cam = (u - cx) * depth_m / fx
    Y_cam = (v - cy) * depth_m / fy
    Z_cam = depth_m
    cos_t, sin_t = math.cos(tilt_rad), math.sin(tilt_rad)
    return X_cam, Y_cam * cos_t - Z_cam * sin_t, Y_cam * sin_t + Z_cam * cos_t


def world_to_pixel(X_w, Y_w, Z_w, fx, fy, cx, cy, tilt_rad, camera_height_m):
    if Z_w <= 0.01:
        return None
    cos_t, sin_t = math.cos(tilt_rad), math.sin(tilt_rad)
    X_cam =  X_w
    Y_cam =  Y_w * cos_t + Z_w * sin_t
    Z_cam = -Y_w * sin_t + Z_w * cos_t
    if Z_cam <= 0.01:
        return None
    return int(fx * X_cam / Z_cam + cx), int(fy * Y_cam / Z_cam + cy)


def mask_to_3d_points(mask, depth_smooth, fx, fy, cx, cy, tilt_rad, subsample=8):
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    idx    = np.arange(0, len(rows), subsample)
    rows, cols = rows[idx], cols[idx]
    depths = depth_smooth[rows, cols].astype(np.float64)
    X_cam  = (cols - cx) * depths / fx
    Y_cam  = (rows - cy) * depths / fy
    Z_cam  = depths
    cos_t, sin_t = math.cos(tilt_rad), math.sin(tilt_rad)
    X_w =  X_cam
    Y_w =  Y_cam * cos_t - Z_cam * sin_t
    Z_w =  Y_cam * sin_t + Z_cam * cos_t
    pts   = np.stack([X_w, Y_w, Z_w], axis=1)
    valid = (Z_w > 0.1) & (Z_w < 35.0)
    return pts[valid] if valid.any() else None


# ─────────────────────────────────────────────────────────────────────────────
# Corridor Builder
# ─────────────────────────────────────────────────────────────────────────────

class CorridorBuilder:
    def __init__(self, cfg: Config, frame_h: int, frame_w: int):
        self.cfg     = cfg
        self.frame_h = frame_h
        self.frame_w = frame_w
        self.near_hw = cfg.risk.corridor_width_near_m / 2.0
        self.far_hw  = cfg.risk.corridor_width_far_m  / 2.0
        self.z_max   = cfg.depth.max_depth_m   # visual only

    def build(self, horizon_y, roll_deg, tilt_rad) -> CorridorTrapezoid:
        world_corners = [
            (-self.near_hw, 0.5),
            ( self.near_hw, 0.5),
            ( self.far_hw,  self.z_max),
            (-self.far_hw,  self.z_max),
        ]
        pix_corners = []
        cam_h = self.cfg.camera.chest_height_m
        for (Xw, Zw) in world_corners:
            pt = world_to_pixel(
                Xw, -cam_h, Zw,
                CFG.fx, CFG.fy, CFG.cx, CFG.cy,
                tilt_rad, cam_h,
            )
            if pt is None:
                pt = (int(self.frame_w/2 + Xw*CFG.fx/max(Zw,0.1)), horizon_y)
            pix_corners.append(pt)

        if abs(roll_deg) > 0.5:
            pix_corners = self._apply_roll(pix_corners, roll_deg)

        pts_arr = np.array(pix_corners, dtype=np.int32).reshape((-1, 1, 2))
        mask    = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts_arr], 1)

        return CorridorTrapezoid(
            world_corners = world_corners,
            pixel_corners = pix_corners,
            corridor_mask = mask.astype(bool),
            near_half_w   = self.near_hw,
            far_half_w    = self.far_hw,
            z_max         = self.z_max,
        )

    def _apply_roll(self, corners, roll_deg):
        cx, cy = self.frame_w / 2, self.frame_h / 2
        rad    = math.radians(-roll_deg)
        cos_r, sin_r = math.cos(rad), math.sin(rad)
        out = []
        for (u, v) in corners:
            dx, dy = u - cx, v - cy
            out.append((int(cx + dx*cos_r - dy*sin_r),
                        int(cy + dx*sin_r + dy*cos_r)))
        return out

    @staticmethod
    def point_in_corridor(X_w, Z_w, near_half, far_half, z_max) -> float:
        """
        Returns path-intersection probability [0, 1].

        FIX-CORR-1: Hard exclusion beyond MAX_LATERAL_M.
        FIX-CORR-2: Tighter sigmoid margin (10% instead of 20%).
        FIX-CORR-3: Risk scoring capped at CORRIDOR_RISK_Z_MAX_M.
        """
        # FIX-CORR-1: absolute lateral gate
        if abs(X_w) > MAX_LATERAL_M:
            return 0.0

        # FIX-CORR-3: beyond risk horizon → no score
        if Z_w <= 0 or Z_w > CORRIDOR_RISK_Z_MAX_M:
            return 0.0

        half_w = near_half + (far_half - near_half) * (Z_w / z_max)
        excess = abs(X_w) - half_w

        if excess <= 0:
            return 1.0

        # FIX-CORR-2: tighter sigmoid
        margin = half_w * CORRIDOR_SIGMOID_MARGIN_FRAC
        if margin < 0.02:
            margin = 0.02  # prevent division by zero for very narrow near-end

        if excess > margin * 3:
            return 0.0

        return float(1.0 / (1.0 + math.exp(excess / margin * 5)))


# ─────────────────────────────────────────────────────────────────────────────
# SpatialAnalyzerV3
# ─────────────────────────────────────────────────────────────────────────────

class SpatialAnalyzerV3:
    """
    Converts PerceptionOutput → List[Object3D].

    Per-object pipeline:
      1. Project mask pixels to 3D world coordinates.
      2. Find closest mask point (minimum Z_w).
      3. Estimate velocity from track history (linear regression).
      4. Compute TTC from depth derivative.
      5. Corridor intersection probability (with FIX-CORR fixes).
      6. Stationary check via ego-motion-compensated depth delta.
      7. Propagate pose fields from InstanceMask (FIX-POSE-2).
    """

    HISTORY_LEN = 12

    def __init__(self, cfg: Config, frame_h: int, frame_w: int):
        self.cfg      = cfg
        self.corridor = CorridorBuilder(cfg, frame_h, frame_w)
        self._track_history:  Dict[int, List[Tuple[float,float,float]]] = {}
        self._depth_history:  Dict[int, List[float]] = {}
        self._stationary_cnt: Dict[int, int] = {}
        self._tilt_rad = math.radians(cfg.camera.mount_tilt_deg)

    def analyze(self, perc: PerceptionOutput) -> Tuple[List[Object3D], CorridorTrapezoid]:
        corridor = self.corridor.build(perc.horizon_y, perc.roll_deg, self._tilt_rad)
        objects: List[Object3D] = []

        for inst in perc.obstacle_masks:
            obj3d = self._process_instance(inst, perc, corridor)
            if obj3d is not None:
                objects.append(obj3d)

        active_ids = {o.track_id for o in objects}
        for tid in list(self._track_history.keys()):
            if tid not in active_ids:
                del self._track_history[tid]
                self._depth_history.pop(tid, None)
                self._stationary_cnt.pop(tid, None)

        return objects, corridor

    def _process_instance(self, inst, perc, corridor) -> Optional[Object3D]:
        pts3d = mask_to_3d_points(
            inst.mask, perc.depth_smooth,
            CFG.fx, CFG.fy, CFG.cx, CFG.cy,
            self._tilt_rad, subsample=6,
        )
        if pts3d is None or len(pts3d) < 5:
            return None

        z_vals      = pts3d[:, 2]
        closest_idx = int(np.argmin(z_vals))
        closest_pt  = tuple(pts3d[closest_idx])
        closest_z   = max(float(closest_pt[2]), self.cfg.depth.min_depth_m)

        centroid = tuple(pts3d.mean(axis=0).tolist())
        dist_m   = float(np.sqrt(centroid[0]**2 + centroid[2]**2))

        tid = inst.track_id
        if tid not in self._track_history:
            self._track_history[tid]  = []
            self._depth_history[tid]  = []
            self._stationary_cnt[tid] = 0

        self._track_history[tid].append(centroid)
        if len(self._track_history[tid]) > self.HISTORY_LEN:
            self._track_history[tid].pop(0)

        velocity = self._estimate_velocity(self._track_history[tid])

        self._depth_history[tid].append(closest_z)
        if len(self._depth_history[tid]) > self.HISTORY_LEN:
            self._depth_history[tid].pop(0)

        ttc_s         = self._compute_ttc(self._depth_history[tid])
        is_stationary = self._check_stationary(tid, self._depth_history[tid], velocity)

        path_probs = [
            CorridorBuilder.point_in_corridor(
                p[0], p[2],
                corridor.near_half_w, corridor.far_half_w, corridor.z_max
            )
            for p in pts3d[::3]
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
            # FIX-POSE-2: propagate from InstanceMask
            is_seated          = inst.is_seated,
            pose_risk_factor   = inst.pose_risk_factor,
        )

    def _estimate_velocity(self, hist):
        n = len(hist)
        if n < 2:
            return (0.0, 0.0, 0.0)
        pts = np.array(hist)
        t   = np.arange(n, dtype=float)
        vx  = float(np.polyfit(t, pts[:, 0], 1)[0])
        vy  = float(np.polyfit(t, pts[:, 1], 1)[0])
        vz  = float(np.polyfit(t, pts[:, 2], 1)[0])
        return (vx, vy, vz)

    def _compute_ttc(self, depth_hist):
        if len(depth_hist) < 3:
            return 999.0
        d     = np.array(depth_hist, dtype=np.float64)
        t     = np.arange(len(d), dtype=np.float64)
        slope = float(np.polyfit(t, d, 1)[0])
        if slope >= -0.005:
            return 999.0
        ai_fps   = self.cfg.pipeline.target_ai_fps
        rate_m_s = slope * ai_fps
        return float(np.clip(-float(d[-1]) / rate_m_s, 0.1, 999.0))

    def _check_stationary(self, tid, depth_hist, velocity):
        if len(depth_hist) < 4:
            return False
        d_std      = float(np.std(depth_hist[-6:]))
        ego_thresh = self.cfg.depth.ego_motion_depth_threshold
        speed_3d   = math.sqrt(sum(v**2 for v in velocity))
        if d_std < ego_thresh and speed_3d < 0.05:
            self._stationary_cnt[tid] = self._stationary_cnt.get(tid, 0) + 1
        else:
            self._stationary_cnt[tid] = max(0, self._stationary_cnt.get(tid, 0) - 1)
        return self._stationary_cnt[tid] >= 5

"""
perception.py — SoundVision V3
================================
Unified perception class that runs:
  1. YOLOv11-seg  → semantic instance masks
  2. MiDaS        → monocular depth map (metric-calibrated)
  3. Risk Heatmap → depth map masked by obstacle pixels only

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Changes in this version (v4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FIX-MISS-1  Person confidence threshold lowered from 0.55 → 0.42.
            The 0.55 value introduced in the ghost-fix was too aggressive
            and caused missed detections for partially occluded or
            motion-blurred people. 0.42 keeps ghost suppression while
            restoring sensitivity.

FIX-MISS-2  TRACK_IOU_THRESH split into per-class values.
            Fast-moving people can shift their bbox significantly between
            AI frames, causing IoU to drop below 0.25 → tracker treating
            each frame as a new object → never reaching MIN_CONFIRM_FRAMES.
            Person IOU gate lowered to 0.15; vehicles remain at 0.25.

FIX-MISS-3  MIN_CONFIRM_FRAMES lowered from 3 → 2.
            3 frames at 10Hz AI = 300ms minimum latency before any alert.
            2 frames = 200ms — still filters single-frame ghosts but reacts
            faster to real pedestrians.

FIX-MISS-4  Immediate-confirm bypass for close, on-path objects.
            If a new detection is estimated to be within 4m AND its bbox
            centre is in the middle third of the frame (directly ahead),
            it is confirmed in 1 frame. This catches the "person steps out
            directly in front" scenario that MIN_CONFIRM_FRAMES delays.

FIX-POSE-1  Seated person classifier (new: PostureClassifier).
            Uses three complementary signals:
              a) Bounding-box aspect ratio  (bbox_w / bbox_h > 0.80 → likely seated)
              b) Estimated real-world height from depth + bbox_h via pinhole model
                 (est_height_m < 1.10m at distance → seated/crouching)
              c) Mask centre-of-mass position relative to horizon
                 (seated people's mask CoM is very low in the frame)
            Any detection classified as seated gets:
              - label changed to "person_seated"
              - pose_risk_factor = 0.20 (stored in InstanceMask)
              - Severity capped at LOW by risk_engine_v3.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

log = logging.getLogger("SV3.Perception")


# ─────────────────────────────────────────────────────────────────────────────
# Tracker tuning  (per-class IoU gates)
# ─────────────────────────────────────────────────────────────────────────────

# FIX-MISS-2: per-class IoU gate (lower = more lenient matching)
TRACK_IOU_THRESH_BY_CLASS: Dict[str, float] = {
    "person":        0.15,   # people move fast between AI frames
    "bicycle":       0.18,
    "car":           0.28,
    "motorcycle":    0.20,
    "bus":           0.30,
    "truck":         0.30,
    "traffic light": 0.30,
    "stop sign":     0.35,
    "backpack":      0.20,
}
TRACK_IOU_THRESH_DEFAULT: float = 0.22

# FIX-MISS-3: confirmation frames
MIN_CONFIRM_FRAMES: int = 2

# FIX-MISS-4: immediate-confirm thresholds
IMMEDIATE_CONFIRM_DIST_M:   float = 4.0    # metres — objects this close bypass gate
IMMEDIATE_CONFIRM_CENTRE_F: float = 0.33   # middle third of frame width

# FIX-GHOST: per-class minimum confidence
# Lowered person to 0.42 (was 0.55 — too aggressive, caused missed detections)
PER_CLASS_CONF: Dict[str, float] = {
    "person":        0.42,   # FIX-MISS-1: was 0.55
    "bicycle":       0.45,
    "car":           0.48,
    "motorcycle":    0.45,
    "bus":           0.48,
    "truck":         0.48,
    "traffic light": 0.50,
    "stop sign":     0.50,
    "backpack":      0.55,   # kept high — still very ghost-prone
}

# Minimum mask area fraction (keeps ghost suppression)
MIN_MASK_AREA_FRAC: float = 0.0004


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InstanceMask:
    """One detected obstacle with its segmentation mask and metadata."""
    label:            str
    cls_id:           int
    conf:             float
    bbox:             Tuple[int, int, int, int]   # x1, y1, x2, y2
    mask:             np.ndarray                  # bool H×W
    track_id:         int   = -1
    # FIX-POSE-1: pose risk multiplier (1.0 = standing, 0.20 = seated)
    pose_risk_factor: float = 1.0
    is_seated:        bool  = False


@dataclass
class PerceptionOutput:
    """Complete output of one Perception.process() call."""
    frame:          np.ndarray
    depth_raw:      np.ndarray
    depth_metric:   np.ndarray
    depth_smooth:   np.ndarray
    ground_mask:    np.ndarray
    obstacle_masks: List[InstanceMask]
    risk_heatmap:   np.ndarray
    horizon_y:      int
    roll_deg:       float
    depth_scale:    float
    depth_shift:    float
    frame_id:       int   = 0
    inference_ms:   float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Posture Classifier  (FIX-POSE-1)
# ─────────────────────────────────────────────────────────────────────────────

class PostureClassifier:
    """
    Classifies detected persons as standing or seated using three signals:

    Signal A — Bounding-box aspect ratio
        bbox_w / bbox_h:
          > 0.80 → strong seated indicator (person is wide relative to height)
          < 0.50 → strong standing indicator

    Signal B — Estimated real-world height
        Using the pinhole model:  h_world = (bbox_h_px / fy) * depth_m
        If h_world < SEATED_HEIGHT_THRESH_M → likely seated

    Signal C — Mask centre-of-mass below horizon
        If the mask CoM is very close to the bottom of the bounding box
        (< 25% from the bottom), the person occupies only the lower portion
        of their bbox, consistent with being seated behind an obstacle (bench,
        car window) or crouching.

    A person is classified as seated when at least 2 of the 3 signals agree.
    """

    SEATED_ASPECT_RATIO:    float = 0.78   # bbox_w/bbox_h above this → seated signal
    SEATED_HEIGHT_THRESH_M: float = 1.10   # estimated real height below this → seated
    SEATED_COM_FRAC:        float = 0.30   # CoM in bottom fraction of bbox → seated

    def classify(
        self,
        inst:        InstanceMask,
        depth_smooth: np.ndarray,
        fy:          float,
        horizon_y:   int,
        frame_h:     int,
        frame_w:     int,
    ) -> Tuple[bool, float]:
        """
        Returns (is_seated: bool, pose_risk_factor: float).
        pose_risk_factor = 1.0 for standing, 0.20 for seated.
        Only applied to 'person' class; all other classes return (False, 1.0).
        """
        if inst.label != "person":
            return False, 1.0

        x1, y1, x2, y2 = inst.bbox
        bbox_w = max(x2 - x1, 1)
        bbox_h = max(y2 - y1, 1)

        signals_seated = 0

        # ── Signal A: aspect ratio ────────────────────────────────────────
        aspect = bbox_w / bbox_h
        if aspect > self.SEATED_ASPECT_RATIO:
            signals_seated += 1

        # ── Signal B: estimated real-world height ─────────────────────────
        if fy > 0 and inst.mask.any():
            median_depth = float(np.median(depth_smooth[inst.mask]))
            if median_depth > 0.1:
                est_height_m = (bbox_h / fy) * median_depth
                if est_height_m < self.SEATED_HEIGHT_THRESH_M:
                    signals_seated += 1

        # ── Signal C: mask CoM position in bbox ───────────────────────────
        rows, cols = np.where(inst.mask)
        if len(rows) > 0:
            com_y  = float(rows.mean())
            # How far down in the bbox is the CoM?  0=top, 1=bottom
            com_frac_in_bbox = (com_y - y1) / bbox_h
            if com_frac_in_bbox > (1.0 - self.SEATED_COM_FRAC):
                signals_seated += 1

        is_seated = signals_seated >= 2
        pose_risk_factor = 0.20 if is_seated else 1.0
        return is_seated, pose_risk_factor


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────

def _box_iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2-ax1)*(ay2-ay1))
    area_b = max(1, (bx2-bx1)*(by2-by1))
    return inter / (area_a + area_b - inter)


# ─────────────────────────────────────────────────────────────────────────────
# Stable IoU Tracker  (ghost suppression + FIX-MISS fixes)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _TrackEntry:
    track_id:  int
    label:     str
    bbox:      Tuple[int, int, int, int]
    hits:      int  = 1
    missed:    int  = 0
    confirmed: bool = False


class StableTracker:
    """
    IoU-based frame-to-frame tracker with:
      - Per-class IoU thresholds (FIX-MISS-2)
      - Reduced confirmation window (FIX-MISS-3)
      - Immediate-confirm bypass for close/central threats (FIX-MISS-4)

    Ghost suppression is preserved: single-frame blips still need 2 hits
    unless they are within IMMEDIATE_CONFIRM_DIST_M directly ahead.
    """

    MAX_MISSED = 6

    def __init__(self):
        self._tracks:  Dict[int, _TrackEntry] = {}
        self._next_id: int = 0

    def update(
        self,
        raw_detections: List[InstanceMask],
        depth_smooth:   Optional[np.ndarray] = None,
        frame_w:        int = 640,
    ) -> List[InstanceMask]:
        """
        Returns confirmed detections with stable track_ids.
        depth_smooth is used for the immediate-confirm distance estimate.
        """
        # Age all tracks
        for t in self._tracks.values():
            t.missed += 1

        matched_tids: set = set()
        matched_dis:  set = set()
        track_ids = list(self._tracks.keys())

        # Greedy IoU matching — per-class threshold
        for _ in range(max(len(track_ids), len(raw_detections))):
            best_iou = -1.0
            best_ti  = -1
            best_di  = -1

            for ti, tid in enumerate(track_ids):
                if tid in matched_tids:
                    continue
                t = self._tracks[tid]
                thresh = TRACK_IOU_THRESH_BY_CLASS.get(
                    t.label, TRACK_IOU_THRESH_DEFAULT
                )
                for di, det in enumerate(raw_detections):
                    if di in matched_dis:
                        continue
                    if det.label != t.label:
                        continue
                    iou = _box_iou(det.bbox, t.bbox)
                    if iou >= thresh and iou > best_iou:
                        best_iou = iou
                        best_ti  = ti
                        best_di  = di

            if best_di == -1:
                break

            tid = track_ids[best_ti]
            t   = self._tracks[tid]
            t.bbox   = raw_detections[best_di].bbox
            t.hits  += 1
            t.missed  = 0
            t.confirmed = t.confirmed or (t.hits >= MIN_CONFIRM_FRAMES)
            matched_tids.add(tid)
            matched_dis.add(best_di)

        # Unmatched → new tracks
        for di, det in enumerate(raw_detections):
            if di not in matched_dis:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _TrackEntry(
                    track_id=tid, label=det.label, bbox=det.bbox, hits=1
                )

        # Evict stale tracks
        for tid in [t for t, e in self._tracks.items() if e.missed > self.MAX_MISSED]:
            del self._tracks[tid]

        # FIX-MISS-4: immediate-confirm bypass
        if depth_smooth is not None:
            for tid, t in self._tracks.items():
                if t.confirmed or t.missed > 0:
                    continue
                x1, y1, x2, y2 = t.bbox
                cx_px   = (x1 + x2) / 2.0
                centre_frac = abs(cx_px / frame_w - 0.5)  # 0 = dead centre
                if centre_frac < IMMEDIATE_CONFIRM_CENTRE_F / 2.0:
                    # Estimate depth at bbox centre
                    cy_px  = int((y1 + y2) / 2)
                    cx_int = int(cx_px)
                    h_d, w_d = depth_smooth.shape
                    cy_px  = max(0, min(cy_px, h_d - 1))
                    cx_int = max(0, min(cx_int, w_d - 1))
                    est_depth = float(depth_smooth[cy_px, cx_int])
                    if est_depth < IMMEDIATE_CONFIRM_DIST_M:
                        t.confirmed = True
                        log.debug(
                            f"[Tracker] Immediate-confirm tid={tid} "
                            f"label={t.label} depth={est_depth:.1f}m"
                        )

        # Build output: confirmed + not-missed
        out: List[InstanceMask] = []
        for tid, t in self._tracks.items():
            if not t.confirmed or t.missed > 0:
                continue
            for det in raw_detections:
                thresh = TRACK_IOU_THRESH_BY_CLASS.get(
                    t.label, TRACK_IOU_THRESH_DEFAULT
                )
                if det.label == t.label and _box_iou(det.bbox, t.bbox) >= thresh:
                    det.track_id = tid
                    out.append(det)
                    break

        return out


# ─────────────────────────────────────────────────────────────────────────────
# MiDaS Depth Estimator
# ─────────────────────────────────────────────────────────────────────────────

class DepthEstimator:
    def __init__(self, model_type: str, device: torch.device):
        self.device = device
        log.info(f"[DepthEstimator] Loading MiDaS [{model_type}] on {device}…")
        self.model = torch.hub.load(
            "intel-isl/MiDaS", model_type, trust_repo=True
        )
        self.model.to(device).eval()
        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        self.transform = (
            transforms.dpt_transform
            if model_type == "DPT_Hybrid"
            else transforms.small_transform
        )
        self._infer(np.zeros((256, 256, 3), dtype=np.uint8))
        log.info("[DepthEstimator] Warmup complete.")

    @torch.inference_mode()
    def _infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        inp  = self.transform(frame_rgb).to(self.device)
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return pred.cpu().numpy().astype(np.float32)

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self._infer(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


# ─────────────────────────────────────────────────────────────────────────────
# Metric Depth Calibrator
# ─────────────────────────────────────────────────────────────────────────────

class MetricCalibrator:
    def __init__(self, default_scale, default_shift, calib_ema_alpha,
                 camera_height_m, fy):
        self._scale     = default_scale
        self._shift     = default_shift
        self._ema_alpha = calib_ema_alpha
        self._cam_h     = camera_height_m
        self._fy        = fy

    def set_fy(self, fy: float) -> None:
        self._fy = fy

    def calibrate(self, depth_raw, ground_mask, horizon_y, cy):
        rows, _ = np.where(ground_mask)
        if len(rows) >= 30 and self._fy > 0:
            delta_v   = np.maximum((rows - horizon_y).astype(np.float32), 1.0)
            tan_angle = np.maximum(np.tan(np.arctan(delta_v / self._fy)), 1e-4)
            z_anchor  = np.clip(self._cam_h / tan_angle, 0.5, 20.0)
            raw_vals  = np.maximum(depth_raw[ground_mask].astype(np.float32), 1e-6)
            scale_est = float(np.clip(
                np.median(z_anchor * (raw_vals + self._shift)), 0.5, 50.0
            ))
            self._scale = (
                self._ema_alpha * scale_est +
                (1.0 - self._ema_alpha) * self._scale
            )
        depth_metric = np.clip(
            self._scale / (depth_raw + self._shift + 1e-6), 0.3, 25.0
        ).astype(np.float32)
        return self._scale, self._shift, depth_metric


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Depth Smoother
# ─────────────────────────────────────────────────────────────────────────────

class DepthSmoother:
    def __init__(self, window=7, poly=2, ema_alpha=0.20):
        self._win   = window if window % 2 == 1 else window + 1
        self._poly  = poly
        self._alpha = ema_alpha
        self._buf:  List[np.ndarray] = []
        self._ema:  Optional[np.ndarray] = None
        try:
            from scipy.signal import savgol_coeffs
            self._sg_coeffs = savgol_coeffs(self._win, self._poly, pos=self._win - 1)
            self._use_sg    = True
        except ImportError:
            log.warning("[DepthSmoother] scipy not available; using EMA only.")
            self._use_sg = False

    def update(self, depth_metric: np.ndarray) -> np.ndarray:
        if self._ema is None:
            self._ema = depth_metric.copy()
        else:
            self._ema = self._alpha * depth_metric + (1.0 - self._alpha) * self._ema
        if not self._use_sg:
            return self._ema.astype(np.float32)
        self._buf.append(depth_metric.copy())
        if len(self._buf) > self._win:
            self._buf.pop(0)
        if len(self._buf) == self._win:
            stack    = np.stack(self._buf, axis=0)
            smoothed = np.einsum("t,thw->hw", self._sg_coeffs, stack)
            return np.clip(smoothed, 0.3, 25.0).astype(np.float32)
        return self._ema.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Ground Plane Detector
# ─────────────────────────────────────────────────────────────────────────────

class GroundPlaneDetector:
    def __init__(self, cfg):
        self._top = cfg.camera.horizon_search_top
        self._bot = cfg.camera.horizon_search_bot

    def detect(self, frame_bgr, depth_raw):
        h, w = depth_raw.shape
        y0, y1 = int(h * self._top), int(h * self._bot)
        region_d   = depth_raw[y0:y1, :]
        region_bgr = frame_bgr[y0:y1, :]

        grad_y  = np.abs(cv2.Sobel(region_d, cv2.CV_32F, 0, 1, ksize=5))
        smooth  = grad_y < np.percentile(grad_y, 45)
        hsv     = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
        low_sat = hsv[:, :, 1] < 70
        med_d   = float(np.median(region_d))
        near    = region_d < (med_d * 1.5)

        candidate = (smooth & low_sat & near).astype(np.uint8)
        k         = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
        cleaned   = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k).astype(bool)

        ground_mask = np.zeros((h, w), dtype=bool)
        ground_mask[y0:y1, :] = cleaned
        horizon_y, roll_deg = self._fit_horizon(ground_mask, h, w)
        return ground_mask, horizon_y, roll_deg

    @staticmethod
    def _fit_horizon(ground_mask, h, w):
        top_y, top_x = [], []
        for col in range(0, w, 10):
            rows = np.where(ground_mask[:, col])[0]
            if len(rows):
                top_y.append(int(rows[0]))
                top_x.append(col)
        if len(top_x) < 8:
            return int(h * 0.45), 0.0
        px = np.array(top_x, dtype=np.float32)
        py = np.array(top_y, dtype=np.float32)
        m, b = np.polyfit(px, py, 1)
        hy   = int(np.clip(b + m * (w / 2.0), 0, h - 1))
        return hy, float(np.degrees(np.arctan(m)))


# ─────────────────────────────────────────────────────────────────────────────
# YOLO Segmentation Detector
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationDetector:
    def __init__(self, cfg, device: torch.device):
        from ultralytics import YOLO
        self._obstacle_classes: Dict[int, str] = cfg.seg.OBSTACLE_CLASSES
        self._conf   = cfg.seg.conf_thresh
        self._iou    = cfg.seg.iou_thresh
        self._device = str(device)

        log.info(f"[SegDet] Loading [{cfg.seg.model_path}]…")
        try:
            self.model = YOLO(cfg.seg.model_path)
        except Exception:
            log.warning("[SegDet] Fallback to yolov8n-seg.pt")
            self.model = YOLO("yolov8n-seg.pt")

        self._run(np.zeros((640, 640, 3), dtype=np.uint8))
        log.info("[SegDet] Warmup complete.")

    def detect(self, frame: np.ndarray) -> List[InstanceMask]:
        return self._run(frame)

    def _run(self, frame: np.ndarray) -> List[InstanceMask]:
        h, w = frame.shape[:2]
        frame_pixels = h * w
        results = self.model(
            frame,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
            device=self._device,
        )[0]

        out: List[InstanceMask] = []
        if results.masks is None:
            return out

        masks_data = results.masks.data.cpu().numpy()
        boxes      = results.boxes

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            label  = self._obstacle_classes.get(cls_id)
            if label is None:
                continue

            conf_val = float(box.conf[0])
            min_conf = PER_CLASS_CONF.get(label, self._conf)
            if conf_val < min_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_mask     = masks_data[i]
            mask_resized = cv2.resize(
                raw_mask, (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            if mask_resized.sum() < frame_pixels * MIN_MASK_AREA_FRAC:
                continue

            out.append(InstanceMask(
                label    = label,
                cls_id   = cls_id,
                conf     = conf_val,
                bbox     = (x1, y1, x2, y2),
                mask     = mask_resized,
                track_id = -1,
            ))

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Risk Heatmap Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_risk_heatmap(depth_metric, obstacle_masks, ground_mask, hazard_weights):
    h, w    = depth_metric.shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    max_w   = max(hazard_weights.values(), default=28.0)
    for inst in obstacle_masks:
        if not inst.mask.any():
            continue
        base_w = hazard_weights.get(inst.label, 3.0)
        # Apply pose_risk_factor to seated persons in heatmap too
        effective_w = base_w * inst.pose_risk_factor
        d_px   = depth_metric[inst.mask]
        danger = np.clip(effective_w / (d_px + 0.5), 0.0, effective_w)
        heatmap[inst.mask] = np.maximum(heatmap[inst.mask], danger)
    heatmap = np.clip(heatmap / max_w, 0.0, 1.0)
    heatmap[ground_mask] = 0.0
    return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Perception  (main public class)
# ─────────────────────────────────────────────────────────────────────────────

class Perception:
    """
    Synchronous perception pipeline.

    Ghost suppression:    StableTracker with confirmation gate
    Missed person fix:    Lower conf threshold, per-class IoU, immediate-confirm
    Seated person fix:    PostureClassifier tags persons with pose_risk_factor
    """

    def __init__(self, cfg, width: int, height: int):
        self.cfg    = cfg
        self.width  = width
        self.height = height

        self.device = self._select_device()
        log.info(f"[Perception] Device: {self.device}")

        self._depth_est  = DepthEstimator(cfg.depth.model_type, self.device)
        self._seg_det    = SegmentationDetector(cfg, self.device)
        self._ground_det = GroundPlaneDetector(cfg)
        self._smoother   = DepthSmoother(
            cfg.depth.sg_window_len,
            cfg.depth.sg_poly_order,
            cfg.depth.ema_alpha,
        )
        self._tracker  = StableTracker()
        self._posture  = PostureClassifier()

        from config import CFG
        self._calibrator = MetricCalibrator(
            default_scale   = cfg.depth.default_scale,
            default_shift   = cfg.depth.default_shift,
            calib_ema_alpha = getattr(cfg.depth, "calib_ema_alpha", 0.12),
            camera_height_m = cfg.camera.chest_height_m,
            fy              = CFG.fy if CFG.fy > 0 else 500.0,
        )
        self._frame_id = 0

    def process(self, frame: np.ndarray) -> PerceptionOutput:
        t0   = time.perf_counter()
        h, w = frame.shape[:2]

        from config import CFG
        if CFG.fy > 0:
            self._calibrator.set_fy(CFG.fy)

        # 1. Depth
        scale = self.cfg.pipeline.depth_frame_scale
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        depth_raw = cv2.resize(
            self._depth_est.estimate(small), (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

        # 2. Ground plane
        ground_mask, horizon_y, roll_deg = self._ground_det.detect(frame, depth_raw)

        # 3. Metric calibration
        scale_val, shift_val, depth_metric = self._calibrator.calibrate(
            depth_raw, ground_mask, horizon_y, CFG.cy
        )

        # 4. Temporal smoothing
        depth_smooth = self._smoother.update(depth_metric)

        # 5. Segmentation (raw detections, unconfirmed)
        raw_masks = self._seg_det.detect(frame)

        # 6. Stable tracking + confirmation gate (FIX-MISS-2/3/4)
        confirmed = self._tracker.update(raw_masks, depth_smooth, w)

        # 7. Posture classification (FIX-POSE-1)
        for inst in confirmed:
            seated, prf = self._posture.classify(
                inst, depth_smooth, CFG.fy, horizon_y, h, w
            )
            inst.is_seated        = seated
            inst.pose_risk_factor = prf
            if seated:
                inst.label = "person_seated"

        # 8. Risk heatmap
        heatmap = _build_risk_heatmap(
            depth_metric, confirmed, ground_mask,
            self.cfg.risk.HAZARD_WEIGHTS,
        )

        self._frame_id += 1
        return PerceptionOutput(
            frame          = frame,
            depth_raw      = depth_raw,
            depth_metric   = depth_metric,
            depth_smooth   = depth_smooth,
            ground_mask    = ground_mask,
            obstacle_masks = confirmed,
            risk_heatmap   = heatmap,
            horizon_y      = horizon_y,
            roll_deg       = roll_deg,
            depth_scale    = scale_val,
            depth_shift    = shift_val,
            frame_id       = self._frame_id,
            inference_ms   = (time.perf_counter() - t0) * 1000.0,
        )

    def stop(self) -> None:
        pass

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

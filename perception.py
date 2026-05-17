"""
perception.py — SoundVision V3
================================
Unified perception class that runs:
  1. YOLOv11-seg  → semantic instance masks
  2. MiDaS        → monocular depth map (metric-calibrated)
  3. Risk Heatmap → depth map masked by obstacle pixels only

─────────────────────────────────────────────────────────────
GHOST DETECTION FIX  (this version)
─────────────────────────────────────────────────────────────
Root cause of ghost detections:
  track_id was set to `i` (enumerate loop index) from YOLO.
  On every new frame, YOLO re-numbers boxes from 0 upward.
  If a real object disappears and a new one appears in the
  same frame slot, it inherits the OLD object's depth/velocity
  history from SpatialAnalyzerV3, producing fabricated TTC
  values and spurious risk scores ("ghosts").

Fixes applied:
  FIX-GHOST-1  IoU-based identity matching between frames so
               track_id is stable across frames, not re-indexed.
  FIX-GHOST-2  Confirmation gate: a detection must appear in
               MIN_CONFIRM_FRAMES consecutive frames before it
               is passed downstream to the risk engine.
               Single-frame blips (shadows, reflections, YOLO
               noise) are silently dropped.
  FIX-GHOST-3  Raised per-class confidence thresholds above the
               global floor for classes that ghost most often
               (person, backpack) where YOLO hallucinates on
               texture/shadow.
  FIX-GHOST-4  Mask sanity check: masks smaller than 0.05% of
               frame area are discarded (sub-pixel noise masks).
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
# Tuning constants for ghost suppression
# ─────────────────────────────────────────────────────────────────────────────

# How many consecutive frames a detection must appear before it's "real"
MIN_CONFIRM_FRAMES: int = 3

# IoU threshold to consider two boxes the same object across frames
TRACK_IOU_THRESH: float = 0.25

# Per-class minimum confidence (overrides global cfg.seg.conf_thresh)
# Higher values for classes that hallucinate most on shadows/textures
PER_CLASS_CONF: Dict[str, float] = {
    "person":        0.55,   # raised — most common ghost class
    "bicycle":       0.50,
    "car":           0.50,
    "motorcycle":    0.50,
    "bus":           0.50,
    "truck":         0.50,
    "traffic light": 0.55,
    "stop sign":     0.55,
    "backpack":      0.60,   # raised — very common false positive
}

# Minimum mask area as a fraction of total frame pixels
MIN_MASK_AREA_FRAC: float = 0.0005   # 0.05% — kills sub-pixel ghost masks


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InstanceMask:
    """One detected obstacle with its segmentation mask and metadata."""
    label:    str
    cls_id:   int
    conf:     float
    bbox:     Tuple[int, int, int, int]   # x1, y1, x2, y2
    mask:     np.ndarray                  # bool H×W
    track_id: int = -1


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
# IoU-based Frame-to-Frame Tracker
# ─────────────────────────────────────────────────────────────────────────────

def _box_iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """Intersection-over-Union of two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


@dataclass
class _TrackEntry:
    """Internal tracker state for one persistent object."""
    track_id:     int
    label:        str
    bbox:         Tuple[int, int, int, int]
    hits:         int = 1          # consecutive frames seen
    missed:       int = 0          # consecutive frames missed
    confirmed:    bool = False     # True once hits >= MIN_CONFIRM_FRAMES


class StableTracker:
    """
    Lightweight IoU-based tracker that gives each physical object a
    stable integer track_id across frames.

    FIX-GHOST-1: Prevents track_id recycling by matching detections
    to existing tracks via IoU before assigning IDs.

    FIX-GHOST-2: Only returns detections once they have been seen in
    MIN_CONFIRM_FRAMES consecutive frames, eliminating single-frame
    hallucinations.
    """

    MAX_MISSED = 5   # frames before a track is deleted

    def __init__(self):
        self._tracks:  Dict[int, _TrackEntry] = {}
        self._next_id: int = 0

    def update(
        self,
        raw_detections: List[InstanceMask],
    ) -> List[InstanceMask]:
        """
        Match raw_detections to existing tracks.
        Returns only CONFIRMED detections with stable track_ids.
        """
        # ── Step 1: age all tracks (assume missed this frame) ─────────────
        for t in self._tracks.values():
            t.missed += 1

        # ── Step 2: greedy IoU matching (highest IoU pair first) ──────────
        matched_track_ids: set = set()
        matched_det_idxs:  set = set()

        track_ids  = list(self._tracks.keys())

        for _ in range(len(raw_detections)):
            best_iou  = TRACK_IOU_THRESH
            best_ti   = -1
            best_di   = -1

            for ti, tid in enumerate(track_ids):
                if tid in matched_track_ids:
                    continue
                t = self._tracks[tid]
                for di, det in enumerate(raw_detections):
                    if di in matched_det_idxs:
                        continue
                    if det.label != t.label:
                        continue
                    iou = _box_iou(det.bbox, t.bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_ti  = ti
                        best_di  = di

            if best_di == -1:
                break   # no more matchable pairs above threshold

            tid = track_ids[best_ti]
            det = raw_detections[best_di]
            t   = self._tracks[tid]

            t.bbox   = det.bbox
            t.hits  += 1
            t.missed  = 0
            t.confirmed = t.confirmed or (t.hits >= MIN_CONFIRM_FRAMES)

            matched_track_ids.add(tid)
            matched_det_idxs.add(best_di)

        # ── Step 3: unmatched detections → new tracks ─────────────────────
        for di, det in enumerate(raw_detections):
            if di not in matched_det_idxs:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _TrackEntry(
                    track_id = tid,
                    label    = det.label,
                    bbox     = det.bbox,
                    hits     = 1,
                )

        # ── Step 4: delete stale tracks ───────────────────────────────────
        stale = [tid for tid, t in self._tracks.items()
                 if t.missed > self.MAX_MISSED]
        for tid in stale:
            del self._tracks[tid]

        # ── Step 5: build output — only confirmed, non-missed tracks ──────
        # Build a map from bbox→track for confirmed tracks seen this frame
        confirmed_out: List[InstanceMask] = []
        for tid, t in self._tracks.items():
            if not t.confirmed or t.missed > 0:
                continue
            # Find the matching raw detection to get mask/conf
            for det in raw_detections:
                if det.label == t.label and _box_iou(det.bbox, t.bbox) >= TRACK_IOU_THRESH:
                    det.track_id = tid
                    confirmed_out.append(det)
                    break

        return confirmed_out


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
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        self._infer(dummy)
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
    """
    Converts MiDaS inverse-relative depth d̃ to metric depth Z (metres).
    Model:  Z = scale / (d̃ + shift)
    scale is estimated per-frame using ground-plane geometry.
    """

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
    """
    FIX-GHOST-3: Per-class confidence thresholds (higher for ghost-prone classes).
    FIX-GHOST-4: Mask area sanity check rejects sub-pixel noise masks.
    """

    def __init__(self, cfg, device: torch.device):
        from ultralytics import YOLO
        self._obstacle_classes: Dict[int, str] = cfg.seg.OBSTACLE_CLASSES
        self._conf   = cfg.seg.conf_thresh      # global floor
        self._iou    = cfg.seg.iou_thresh
        self._device = str(device)

        log.info(f"[SegDet] Loading [{cfg.seg.model_path}]…")
        try:
            self.model = YOLO(cfg.seg.model_path)
        except Exception:
            log.warning("[SegDet] Fallback to yolov8n-seg.pt")
            self.model = YOLO("yolov8n-seg.pt")

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._run(dummy)
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

            # FIX-GHOST-3: enforce per-class minimum confidence
            min_conf = PER_CLASS_CONF.get(label, self._conf)
            if conf_val < min_conf:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            raw_mask     = masks_data[i]
            mask_resized = cv2.resize(
                raw_mask, (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            # FIX-GHOST-4: reject tiny/noisy masks
            mask_area = int(mask_resized.sum())
            if mask_area < frame_pixels * MIN_MASK_AREA_FRAC:
                continue

            out.append(InstanceMask(
                label    = label,
                cls_id   = cls_id,
                conf     = conf_val,
                bbox     = (x1, y1, x2, y2),
                mask     = mask_resized,
                track_id = -1,   # assigned by StableTracker, not here
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
        w_val  = hazard_weights.get(inst.label, 3.0)
        d_px   = depth_metric[inst.mask]
        danger = np.clip(w_val / (d_px + 0.5), 0.0, w_val)
        heatmap[inst.mask] = np.maximum(heatmap[inst.mask], danger)
    heatmap = np.clip(heatmap / max_w, 0.0, 1.0)
    heatmap[ground_mask] = 0.0
    return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Perception  (main public class)
# ─────────────────────────────────────────────────────────────────────────────

class Perception:
    """
    Synchronous perception pipeline with ghost-detection suppression.

    Ghost suppression layers (in order):
      1. Per-class confidence threshold  (SegmentationDetector)
      2. Minimum mask area check         (SegmentationDetector)
      3. IoU-based frame-to-frame tracking + confirmation gate (StableTracker)
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

        # FIX-GHOST-1/2: stable IoU tracker with confirmation gate
        self._tracker = StableTracker()

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

        # 5. Segmentation (raw detections)
        raw_masks = self._seg_det.detect(frame)

        # 6. FIX-GHOST: stable tracking + confirmation gate
        obstacle_masks = self._tracker.update(raw_masks)

        # 7. Risk heatmap
        heatmap = _build_risk_heatmap(
            depth_metric, obstacle_masks, ground_mask,
            self.cfg.risk.HAZARD_WEIGHTS,
        )

        self._frame_id += 1
        return PerceptionOutput(
            frame          = frame,
            depth_raw      = depth_raw,
            depth_metric   = depth_metric,
            depth_smooth   = depth_smooth,
            ground_mask    = ground_mask,
            obstacle_masks = obstacle_masks,
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

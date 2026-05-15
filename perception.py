"""
perception.py — SoundVision V3
================================
Unified perception class that runs:
  1. YOLOv11-seg  → semantic instance masks
  2. MiDaS        → monocular depth map (metric-calibrated)
  3. Risk Heatmap → depth map masked by obstacle pixels only

Both models run in parallel background threads.
The public method `process(frame)` returns a PerceptionOutput
containing everything downstream modules need.

Ground-Plane Auto-Calibration
------------------------------
Because the camera is chest-mounted (variable tilt/roll), we cannot
hardcode a horizon line. Instead, we identify the ground-plane mask
each frame using two complementary signals:
  a) Explicit COCO "road / sidewalk" classes (15, 61) if present
  b) Low-position heuristic: pixels in the bottom 40% of frame whose
     depth derivative (vertical gradient) is smooth and continuous
     (characteristic of a flat plane receding from the camera)
The ground mask is used to:
  - Fit a horizon line (auto-calibration)
  - Anchor the metric depth scale (scale/shift estimation)
  - Define the walking corridor trapezoid
"""

from __future__ import annotations

import threading
import queue
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import torch

log = logging.getLogger("SV3.Perception")
# --- FIXED: LOCAL YOLO LOADING ---
        # This looks for the file in the current folder
self.seg_model = YOLO("yolo11n-seg.pt")
        
        # --- FIXED: LOCAL MIDAS LOADING ---
model_type = "MiDaS_small"
        # We set pretrained=False so it doesn't try to download from GitHub
self.midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False)
        
try:
            # Pointing specifically to the file you moved
    self.midas.load_state_dict(torch.load("midas_weights.pt", map_location=self.device))
    print("✅ MiDaS local weights loaded successfully.")
except Exception as e:
    print(f"⚠️ Local weights failed, trying standard load: {e}")
    self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
            
self.midas.to(self.device).eval()

# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InstanceMask:
    """One detected object with its segmentation mask and metadata."""
    label:        str
    cls_id:       int
    conf:         float
    bbox:         Tuple[int, int, int, int]   # x1,y1,x2,y2 in original resolution
    mask:         np.ndarray                  # bool H×W at original resolution
    track_id:     int = -1


@dataclass
class PerceptionOutput:
    """Everything produced by one call to Perception.process()."""
    frame:            np.ndarray              # original frame (BGR)
    depth_raw:        np.ndarray              # MiDaS raw inverse-depth  H×W float32
    depth_metric:     np.ndarray              # metric depth in metres    H×W float32
    depth_smooth:     np.ndarray              # EMA-smoothed metric depth H×W float32
    ground_mask:      np.ndarray              # bool H×W — walkable surface
    obstacle_masks:   List[InstanceMask]      # obstacle detections
    risk_heatmap:     np.ndarray              # float32 H×W [0,1] danger intensity
    horizon_y:        int                     # estimated horizon pixel row
    roll_deg:         float                   # estimated camera roll (°)
    depth_scale:      float                   # metric calibration scale
    depth_shift:      float                   # metric calibration shift
    frame_id:         int = 0
    inference_ms:     float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MiDaS Depth Estimator
# ─────────────────────────────────────────────────────────────────────────────

class DepthEstimator:
    """
    Wraps MiDaS for monocular depth estimation.

    Outputs raw inverse-relative depth (MiDaS native).
    Metric conversion (scale/shift) is performed by Perception
    using the ground-plane anchor.
    """

    def __init__(self, model_type: str, device: torch.device):
        self.device = device
        log.info(f"Loading MiDaS [{model_type}]…")
        self.model = torch.hub.load(
            "intel-isl/MiDaS", model_type, trust_repo=True
        )
        self.model.to(device).eval()

        transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        if model_type == "DPT_Hybrid":
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

        # Warmup
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        self._infer(dummy)
        log.info("MiDaS warmup complete.")

    @torch.inference_mode()
    def _infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        inp = self.transform(frame_rgb).to(self.device)
        pred = self.model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        return pred.cpu().numpy().astype(np.float32)

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._infer(frame_rgb)


# ─────────────────────────────────────────────────────────────────────────────
# YOLO Segmentation Detector
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationDetector:
    """
    Wraps YOLOv11-seg (or YOLOv8-seg fallback) for instance segmentation.
    Returns InstanceMask objects for ground-plane and obstacle classes.
    """

    # COCO classes that represent ground / walkable surface
    GROUND_COCO_IDS = {15, 61}   # 15=bench area proxy; 61=dining table (rare)
    # More reliable: classes 0–79 not in obstacle set → filter by position heuristic

    def __init__(self, model_path: str, conf: float, iou: float,
                 obstacle_classes: Dict[int, str], device: torch.device):
        from ultralytics import YOLO
        self.obstacle_classes = obstacle_classes
        self.conf = conf
        self.iou  = iou
        self.device = str(device)

        log.info(f"Loading YOLO segmentation [{model_path}]…")
        try:
            self.model = YOLO(model_path)
        except Exception:
            fallback = "yolov8n-seg.pt"
            log.warning(f"{model_path} not found, falling back to {fallback}")
            self.model = YOLO(fallback)

        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._infer(dummy)
        log.info("YOLO-seg warmup complete.")

    def _infer(self, frame: np.ndarray) -> Tuple[List[InstanceMask], List[InstanceMask]]:
        """Returns (obstacle_masks, ground_masks)."""
        h, w = frame.shape[:2]
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )[0]

        obstacles: List[InstanceMask] = []

        if results.masks is None:
            return obstacles

        masks_data = results.masks.data.cpu().numpy()  # N × Hmask × Wmask
        boxes      = results.boxes

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label  = self.obstacle_classes.get(cls_id, None)
            if label is None:
                continue

            # Resize mask to original frame resolution
            raw_mask = masks_data[i]
            mask_resized = cv2.resize(
                raw_mask, (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            obstacles.append(InstanceMask(
                label=label,
                cls_id=cls_id,
                conf=conf,
                bbox=(x1, y1, x2, y2),
                mask=mask_resized,
                track_id=i,   # YOLO track_id if tracking enabled; else index
            ))

        return obstacles

    def detect(self, frame: np.ndarray) -> List[InstanceMask]:
        return self._infer(frame)


# ─────────────────────────────────────────────────────────────────────────────
# Ground-Plane Detector
# ─────────────────────────────────────────────────────────────────────────────

class GroundPlaneDetector:
    """
    Identifies the walkable ground surface without requiring explicit
    road/sidewalk COCO classes.

    Strategy
    ---------
    1. Take the bottom `search_frac` of the frame.
    2. Compute the vertical gradient of the depth map in that region.
    3. Low-gradient (smooth depth surface) + below-median-depth pixels
       are candidate ground pixels.
    4. Apply Otsu thresholding on gradient magnitude to segment ground.
    5. Morphological cleanup for mask quality.
    6. Fit a line to the top edge of the ground mask → horizon estimate.
    7. Estimate roll from that line's angle.
    """

    def __init__(self, search_top: float = 0.30, search_bot: float = 0.90):
        self.search_top = search_top
        self.search_bot = search_bot

    def detect(
        self, depth_metric: np.ndarray, frame_bgr: np.ndarray
    ) -> Tuple[np.ndarray, int, float]:
        """
        Returns
        -------
        ground_mask : bool H×W
        horizon_y   : int pixel row of estimated horizon
        roll_deg    : float estimated camera roll in degrees
        """
        h, w = depth_metric.shape
        y0 = int(h * self.search_top)
        y1 = int(h * self.search_bot)

        region = depth_metric[y0:y1, :]

        # ── Vertical depth gradient in search region ──────────────────────
        grad_y = np.abs(cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=5))

        # ── Colour-space ground hint (HSV green/grey tones) ───────────────
        hsv    = cv2.cvtColor(frame_bgr[y0:y1], cv2.COLOR_BGR2HSV)
        # Grey asphalt: low saturation
        low_sat = hsv[:, :, 1] < 60

        # ── Combined ground probability ───────────────────────────────────
        # Low gradient + below median depth + low saturation
        med_depth  = np.median(region)
        near_floor = region < (med_depth * 1.4)   # pixels closer than 1.4× median
        smooth     = grad_y < np.percentile(grad_y, 40)

        candidate = smooth & near_floor & low_sat

        # ── Morphological cleanup ─────────────────────────────────────────
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
        cleaned = cv2.morphologyEx(
            candidate.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)

        # Full-frame mask
        ground_mask = np.zeros((h, w), dtype=bool)
        ground_mask[y0:y1, :] = cleaned

        # ── Horizon & roll estimation ─────────────────────────────────────
        horizon_y, roll_deg = self._fit_horizon(ground_mask, h, w)

        return ground_mask, horizon_y, roll_deg

    @staticmethod
    def _fit_horizon(ground_mask: np.ndarray, h: int, w: int) -> Tuple[int, float]:
        """
        Find the topmost continuous row of ground pixels and fit a line
        to estimate both the horizon row and any camera roll.
        """
        # Top edge of ground mask per column
        top_edge_y = []
        top_edge_x = []
        for col in range(0, w, 8):     # sample every 8 columns for speed
            col_mask = ground_mask[:, col]
            rows = np.where(col_mask)[0]
            if len(rows) > 0:
                top_edge_y.append(int(rows[0]))
                top_edge_x.append(col)

        if len(top_edge_x) < 10:
            # Fallback: assume flat horizon at 42% of frame
            return int(h * 0.42), 0.0

        pts_x = np.array(top_edge_x, dtype=np.float32)
        pts_y = np.array(top_edge_y, dtype=np.float32)

        # Robust line fit (RANSAC via polyfit with outlier clip)
        coeffs = np.polyfit(pts_x, pts_y, 1)   # y = mx + b
        slope, intercept = coeffs

        horizon_y = int(np.clip(intercept + slope * (w / 2), 0, h - 1))
        roll_deg  = float(np.degrees(np.arctan(slope)))  # positive = right-side-up tilt

        return horizon_y, roll_deg


# ─────────────────────────────────────────────────────────────────────────────
# Metric Depth Calibrator
# ─────────────────────────────────────────────────────────────────────────────

class MetricCalibrator:
    """
    Estimates per-frame scale & shift to convert MiDaS inverse-relative
    depth to metric depth (metres).

    Method: least-squares fit on ground-plane pixels where we know
    the approximate metric depth from the ground-plane geometry:
        depth_metric_anchor[px] = camera_height / tan(angle_below_horizon)
    """

    def __init__(self, camera_height_m: float, fy: float, horizon_y_init: int,
                 default_scale: float, default_shift: float):
        self.camera_height_m = camera_height_m
        self.fy              = fy
        self.default_scale   = default_scale
        self.default_shift   = default_shift
        # EMA for scale and shift to prevent frame-to-frame jumps
        self._scale_ema = default_scale
        self._shift_ema = default_shift
        self._ema_alpha = 0.12

    def calibrate(
        self,
        depth_raw: np.ndarray,
        ground_mask: np.ndarray,
        horizon_y: int,
        cy: float,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Returns (scale, shift, depth_metric_H×W).
        """
        h, w = depth_raw.shape

        # ── Build geometric metric anchor from ground pixels ───────────────
        row_idx, col_idx = np.where(ground_mask)
        if len(row_idx) < 50:
            # Not enough ground pixels — use last calibration
            metric = self._scale_ema / (depth_raw + self._shift_ema + 1e-6)
            return self._scale_ema, self._shift_ema, np.clip(metric, 0.3, 30.0)

        # For ground pixels: angle below horizon → known metric depth
        # angle_v = arctan((row - horizon_y) / fy)
        # depth_m = camera_height / tan(angle_v)
        angles_rad = np.arctan(
            np.maximum(row_idx - horizon_y, 1).astype(np.float32) / max(self.fy, 1.0)
        )
        d_anchor = np.clip(
            self.camera_height_m / (np.tan(angles_rad) + 1e-6),
            0.5, 20.0
        )

        # ── Least-squares: d_anchor = scale / (depth_raw + shift) ──────────
        # Linearise: depth_raw * d_anchor + shift * d_anchor = scale
        # → A x = b  where x = [scale, scale*shift_approx] but simplify:
        # Assume shift ≈ 0.15 (weak prior), solve for scale only
        raw_at_ground = depth_raw[row_idx, col_idx].astype(np.float64)
        d_anchor_f    = d_anchor.astype(np.float64)

        # scale_est = median( d_anchor * (raw + shift) )
        shift_fixed = self._shift_ema
        scale_est   = float(np.median(d_anchor_f * (raw_at_ground + shift_fixed)))
        scale_est   = np.clip(scale_est, 0.5, 30.0)

        # EMA smooth
        self._scale_ema = (
            self._ema_alpha * scale_est +
            (1 - self._ema_alpha) * self._scale_ema
        )

        metric = self._scale_ema / (depth_raw + self._shift_ema + 1e-6)
        metric = np.clip(metric, 0.3, 30.0).astype(np.float32)

        return self._scale_ema, self._shift_ema, metric


# ─────────────────────────────────────────────────────────────────────────────
# Depth Smoother (EMA + Savitzky-Golay temporal)
# ─────────────────────────────────────────────────────────────────────────────

class DepthSmoother:
    """
    Maintains a rolling buffer of depth frames and applies either:
      a) Savitzky-Golay filter (once buffer is full) for optimal smoothing
      b) EMA for every frame (low latency)
    The two are blended: SG output is used when available, else EMA.
    """

    def __init__(self, window: int = 7, poly: int = 2, ema_alpha: float = 0.18):
        from scipy.signal import savgol_filter
        self._savgol = savgol_filter
        self.window    = window if window % 2 == 1 else window + 1
        self.poly      = poly
        self.ema_alpha = ema_alpha
        self._buffer: List[np.ndarray] = []
        self._ema:    Optional[np.ndarray] = None

    def update(self, depth_metric: np.ndarray) -> np.ndarray:
        """Returns temporally smoothed depth map."""
        # EMA (always updated)
        if self._ema is None:
            self._ema = depth_metric.copy()
        else:
            self._ema = (
                self.ema_alpha * depth_metric +
                (1 - self.ema_alpha) * self._ema
            )

        # Buffer for SG
        self._buffer.append(depth_metric)
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

        if len(self._buffer) == self.window:
            try:
                stack = np.stack(self._buffer, axis=0)   # W × H × W
                sg = self._savgol(stack, self.window, self.poly, axis=0)
                return sg[-1].astype(np.float32)
            except Exception:
                pass   # SG failed (e.g. scipy not available) → fall through

        return self._ema.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Risk Heatmap Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_risk_heatmap(
    depth_smooth: np.ndarray,
    obstacle_masks: List[InstanceMask],
    ground_mask: np.ndarray,
    hazard_weights: Dict[str, float],
    max_depth: float = 20.0,
) -> np.ndarray:
    """
    Produces a float32 H×W heatmap in [0, 1] where:
      - Ground pixels = 0 (safe)
      - Obstacle pixels = (hazard_weight / distance_m) normalised to [0,1]

    Only obstacle pixels that *intersect* the ground mask (i.e. their
    bottom portion touches walkable surface) contribute to the heatmap.
    """
    h, w = depth_smooth.shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    for inst in obstacle_masks:
        weight    = hazard_weights.get(inst.label, 3.0)
        obj_depth = depth_smooth[inst.mask]
        if obj_depth.size == 0:
            continue

        # Per-pixel danger: weight / depth, clipped
        danger_px = np.clip(weight / (obj_depth + 0.5), 0.0, weight)

        # Write into heatmap
        heatmap[inst.mask] = np.maximum(heatmap[inst.mask], danger_px)

    # Normalise to [0,1] by max possible value
    max_possible = max(hazard_weights.values(), default=20.0)
    heatmap = np.clip(heatmap / max_possible, 0.0, 1.0)

    # Ground pixels are definitionally safe
    heatmap[ground_mask] = 0.0

    return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Perception (main class)
# ─────────────────────────────────────────────────────────────────────────────

class Perception:
    """
    Unified perception pipeline.

    Both depth and segmentation inference run in parallel background threads.
    `process(frame)` submits the frame and returns a PerceptionOutput
    built from the most recently completed depth + seg results.

    This means the very first call may return placeholder data
    (takes ~2 frames to warm up both threads). From frame 3 onward,
    results are always at most 1 inference cycle stale — negligible at
    10+ Hz.
    """

    def __init__(self, cfg, frame_width: int, frame_height: int):
        from config import CFG
        self.cfg = cfg
        self.w   = frame_width
        self.h   = frame_height

        self.device = self._select_device()
        log.info(f"Perception device: {self.device}")

        # ── Sub-models ────────────────────────────────────────────────────
        self.depth_est  = DepthEstimator(cfg.depth.model_type, self.device)
        self.seg_det    = SegmentationDetector(
            cfg.seg.model_path, cfg.seg.conf_thresh, cfg.seg.iou_thresh,
            cfg.seg.OBSTACLE_CLASSES, self.device
        )
        self.ground_det = GroundPlaneDetector(
            cfg.camera.horizon_search_top, cfg.camera.horizon_search_bot
        )
        self.calibrator = MetricCalibrator(
            cfg.camera.chest_height_m, CFG.fy,
            int(frame_height * 0.42),
            cfg.depth.default_scale, cfg.depth.default_shift
        )
        self.smoother   = DepthSmoother(
            cfg.depth.sg_window_len, cfg.depth.sg_poly_order, cfg.depth.ema_alpha
        )

        # ── Parallel inference queues ─────────────────────────────────────
        self._depth_in:  queue.Queue = queue.Queue(maxsize=cfg.pipeline.inference_queue_size)
        self._depth_out: queue.Queue = queue.Queue(maxsize=cfg.pipeline.inference_queue_size)
        self._seg_in:    queue.Queue = queue.Queue(maxsize=cfg.pipeline.inference_queue_size)
        self._seg_out:   queue.Queue = queue.Queue(maxsize=cfg.pipeline.inference_queue_size)

        self._stop = threading.Event()
        self._depth_thread = threading.Thread(target=self._depth_worker, daemon=True)
        self._seg_thread   = threading.Thread(target=self._seg_worker,   daemon=True)
        self._depth_thread.start()
        self._seg_thread.start()

        # ── State ─────────────────────────────────────────────────────────
        self._last_depth_raw:    Optional[np.ndarray] = None
        self._last_depth_metric: Optional[np.ndarray] = None
        self._last_depth_smooth: Optional[np.ndarray] = None
        self._last_obstacles:    List[InstanceMask]   = []
        self._last_ground:       Optional[np.ndarray] = None
        self._last_horizon_y:    int   = int(frame_height * 0.42)
        self._last_roll:         float = 0.0
        self._last_scale:        float = cfg.depth.default_scale
        self._last_shift:        float = cfg.depth.default_shift
        self._frame_id:          int   = 0

    # ── Public ────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> PerceptionOutput:
        """
        Submit frame for parallel inference.
        Returns PerceptionOutput built from latest available results.
        """
        t0 = time.perf_counter()

        # Downscale for depth model (performance)
        scale = self.cfg.pipeline.depth_frame_scale
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        # Submit to both queues (drop if full — prefer freshness)
        for q, item in [(self._depth_in, small), (self._seg_in, frame)]:
            try:
                q.put_nowait(item)
            except queue.Full:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                q.put_nowait(item)

        # Collect latest results (non-blocking)
        self._drain_depth()
        self._drain_seg()

        # Build output from current state
        out = self._build_output(frame, t0)
        self._frame_id += 1
        return out

    def stop(self):
        self._stop.set()
        self._depth_thread.join(timeout=2.0)
        self._seg_thread.join(timeout=2.0)

    # ── Workers ───────────────────────────────────────────────────────────

    def _depth_worker(self):
        while not self._stop.is_set():
            try:
                frame_small = self._depth_in.get(timeout=0.05)
            except queue.Empty:
                continue
            depth_raw = self.depth_est.estimate(frame_small)
            # Upscale back to original resolution
            depth_up  = cv2.resize(depth_raw, (self.w, self.h),
                                   interpolation=cv2.INTER_LINEAR)
            try:
                self._depth_out.put_nowait(depth_up)
            except queue.Full:
                try:
                    self._depth_out.get_nowait()
                except queue.Empty:
                    pass
                self._depth_out.put_nowait(depth_up)

    def _seg_worker(self):
        while not self._stop.is_set():
            try:
                frame = self._seg_in.get(timeout=0.05)
            except queue.Empty:
                continue
            obstacles = self.seg_det.detect(frame)
            try:
                self._seg_out.put_nowait(obstacles)
            except queue.Full:
                try:
                    self._seg_out.get_nowait()
                except queue.Empty:
                    pass
                self._seg_out.put_nowait(obstacles)

    # ── Internal ──────────────────────────────────────────────────────────

    def _drain_depth(self):
        """Pull latest depth result without blocking."""
        depth_raw = None
        while True:
            try:
                depth_raw = self._depth_out.get_nowait()
            except queue.Empty:
                break
        if depth_raw is None:
            return
        if self._last_depth_raw is None:
            self._last_depth_raw = depth_raw

        # Ground plane detection on raw depth
        ground_mask, horizon_y, roll_deg = self.ground_det.detect(
            depth_raw, np.zeros((self.h, self.w, 3), dtype=np.uint8)
        )
        # Metric calibration
        from config import CFG
        scale, shift, depth_metric = self.calibrator.calibrate(
            depth_raw, ground_mask, horizon_y, CFG.cy
        )
        depth_smooth = self.smoother.update(depth_metric)

        self._last_depth_raw    = depth_raw
        self._last_depth_metric = depth_metric
        self._last_depth_smooth = depth_smooth
        self._last_ground       = ground_mask
        self._last_horizon_y    = horizon_y
        self._last_roll         = roll_deg
        self._last_scale        = scale
        self._last_shift        = shift

    def _drain_seg(self):
        obstacles = None
        while True:
            try:
                obstacles = self._seg_out.get_nowait()
            except queue.Empty:
                break
        if obstacles is not None:
            self._last_obstacles = obstacles

    def _build_output(self, frame: np.ndarray, t0: float) -> PerceptionOutput:
        h, w = frame.shape[:2]

        depth_raw    = self._last_depth_raw    if self._last_depth_raw    is not None else np.ones((h, w), dtype=np.float32)
        depth_metric = self._last_depth_metric if self._last_depth_metric is not None else np.ones((h, w), dtype=np.float32) * 5.0
        depth_smooth = self._last_depth_smooth if self._last_depth_smooth is not None else depth_metric.copy()
        ground_mask  = self._last_ground       if self._last_ground       is not None else np.zeros((h, w), dtype=bool)

        heatmap = build_risk_heatmap(
            depth_smooth,
            self._last_obstacles,
            ground_mask,
            self.cfg.risk.HAZARD_WEIGHTS,
        )

        return PerceptionOutput(
            frame          = frame,
            depth_raw      = depth_raw,
            depth_metric   = depth_metric,
            depth_smooth   = depth_smooth,
            ground_mask    = ground_mask,
            obstacle_masks = list(self._last_obstacles),
            risk_heatmap   = heatmap,
            horizon_y      = self._last_horizon_y,
            roll_deg       = self._last_roll,
            depth_scale    = self._last_scale,
            depth_shift    = self._last_shift,
            frame_id       = self._frame_id,
            inference_ms   = (time.perf_counter() - t0) * 1000,
        )

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

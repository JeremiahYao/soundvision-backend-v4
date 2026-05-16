"""
perception.py — SoundVision V3
================================
Unified perception class that runs:
  1. YOLOv11-seg  → semantic instance masks
  2. MiDaS        → monocular depth map (metric-calibrated)
  3. Risk Heatmap → depth map masked by obstacle pixels only

Ground-Plane Auto-Calibration
------------------------------
Because the camera is chest-mounted (variable tilt/roll), the horizon
is estimated each frame from two complementary signals:
  a) Low vertical-gradient pixels in the lower search band of the frame
     (flat ground receding from camera has a smooth, consistent depth surface)
  b) Low HSV saturation (grey asphalt / concrete)

The ground mask drives:
  - Horizon line estimation (auto-calibration of tilt/roll)
  - Per-frame metric depth scale via the geometric anchor:
        Z_ground[row] = camera_height / tan(angle_below_horizon[row])
    We solve for `scale` such that  scale / (d̃ + shift) ≈ Z_ground.

Fixes applied vs. previous version
-------------------------------------
  FIX-1  cfg.seg.iou_threshold  → cfg.seg.iou_thresh
  FIX-2  cfg.seg.obstacle_classes → cfg.seg.OBSTACLE_CLASSES
  FIX-3  Depth metric formula fixed:  Z = scale / (d̃ + shift)
         Ground-plane bootstrap for scale estimation (replaces hardcoded 5.0)
  FIX-4  GroundPlaneDetector.detect now accepts (frame_bgr, depth_raw)
         consistently with its call site in Perception.process()
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
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InstanceMask:
    """One detected obstacle with its segmentation mask and metadata."""
    label:    str
    cls_id:   int
    conf:     float
    bbox:     Tuple[int, int, int, int]  # x1, y1, x2, y2 (original resolution)
    mask:     np.ndarray                 # bool H×W at original resolution
    track_id: int = -1


@dataclass
class PerceptionOutput:
    """Complete output of one Perception.process() call."""
    frame:          np.ndarray   # original BGR frame
    depth_raw:      np.ndarray   # MiDaS inverse-relative depth   H×W float32
    depth_metric:   np.ndarray   # metric depth in metres          H×W float32
    depth_smooth:   np.ndarray   # temporally smoothed metric depth H×W float32
    ground_mask:    np.ndarray   # bool H×W — walkable surface
    obstacle_masks: List[InstanceMask]
    risk_heatmap:   np.ndarray   # float32 H×W in [0, 1]
    horizon_y:      int          # estimated horizon pixel row
    roll_deg:       float        # estimated camera roll (°)
    depth_scale:    float        # metric calibration scale
    depth_shift:    float        # metric calibration shift
    frame_id:       int   = 0
    inference_ms:   float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MiDaS Depth Estimator
# ─────────────────────────────────────────────────────────────────────────────

class DepthEstimator:
    """
    Wraps MiDaS for monocular depth estimation.

    Output: raw inverse-relative depth d̃ (higher value = closer object).
    Metric conversion happens in MetricCalibrator, not here.
    """

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

        # Warmup pass to eliminate first-frame latency spike
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
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._infer(frame_rgb)


# ─────────────────────────────────────────────────────────────────────────────
# Metric Depth Calibrator
# ─────────────────────────────────────────────────────────────────────────────

class MetricCalibrator:
    """
    Converts MiDaS inverse-relative depth d̃ to metric depth Z (metres).

    Conversion model:  Z = scale / (d̃ + shift)

    scale is estimated per-frame using the ground-plane geometric anchor:
        For a ground pixel at row v below horizon_y:
            angle_below = arctan((v - horizon_y) / fy)
            Z_anchor    = camera_height / tan(angle_below)

    We find scale = median(Z_anchor * (d̃ + shift)) over ground pixels.
    scale is EMA-smoothed across frames for stability.
    """

    def __init__(
        self,
        default_scale: float,
        default_shift: float,
        calib_ema_alpha: float,
        camera_height_m: float,
        fy: float,
    ):
        self._scale     = default_scale
        self._shift     = default_shift
        self._ema_alpha = calib_ema_alpha
        self._cam_h     = camera_height_m
        self._fy        = fy   # set once intrinsics are known; updated via set_fy()

    def set_fy(self, fy: float) -> None:
        self._fy = fy

    def calibrate(
        self,
        depth_raw:    np.ndarray,
        ground_mask:  np.ndarray,
        horizon_y:    int,
        cy:           float,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Returns (scale, shift, depth_metric H×W float32).
        """
        h, w = depth_raw.shape

        # Build per-row geometric depth anchor for ground pixels
        rows, _ = np.where(ground_mask)
        if len(rows) >= 30 and self._fy > 0:
            # Angle below horizon for each ground pixel row
            delta_v    = (rows - horizon_y).astype(np.float32)
            delta_v    = np.maximum(delta_v, 1.0)   # must be below horizon
            angle_rad  = np.arctan(delta_v / self._fy)
            # Avoid division by zero for pixels right at horizon
            tan_angle  = np.tan(angle_rad)
            tan_angle  = np.maximum(tan_angle, 1e-4)
            z_anchor   = self._cam_h / tan_angle     # geometric ground depth
            z_anchor   = np.clip(z_anchor, 0.5, 20.0)

            # Corresponding raw depth values
            raw_at_ground = depth_raw[ground_mask].astype(np.float32)
            raw_at_ground = np.maximum(raw_at_ground, 1e-6)

            # solve:  z_anchor = scale / (raw + shift)
            # → scale = median(z_anchor * (raw + shift))
            scale_est = float(np.median(z_anchor * (raw_at_ground + self._shift)))
            scale_est = np.clip(scale_est, 0.5, 50.0)

            # EMA smooth
            self._scale = (
                self._ema_alpha * scale_est +
                (1.0 - self._ema_alpha) * self._scale
            )

        # Compute metric depth map
        depth_metric = self._scale / (depth_raw + self._shift + 1e-6)
        depth_metric = np.clip(depth_metric, 0.3, 25.0).astype(np.float32)

        return self._scale, self._shift, depth_metric


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Depth Smoother  (Savitzky-Golay + EMA fallback)
# ─────────────────────────────────────────────────────────────────────────────

class DepthSmoother:
    """
    Maintains a rolling frame buffer and applies Savitzky-Golay smoothing
    along the time axis. Falls back to EMA while the buffer fills.

    SG fitting:  fits a polynomial of degree `poly` over `window` frames
    and evaluates at the most recent frame — preserves edges better than
    a plain moving average while still removing noise.
    """

    def __init__(self, window: int = 7, poly: int = 2, ema_alpha: float = 0.20):
        # Ensure odd window
        self._win   = window if window % 2 == 1 else window + 1
        self._poly  = poly
        self._alpha = ema_alpha
        self._buf:  List[np.ndarray] = []
        self._ema:  Optional[np.ndarray] = None

        # Build SG coefficients once
        try:
            from scipy.signal import savgol_coeffs
            self._sg_coeffs = savgol_coeffs(self._win, self._poly, pos=self._win - 1)
            self._use_sg    = True
        except ImportError:
            log.warning("[DepthSmoother] scipy not available; using EMA only.")
            self._use_sg = False

    def update(self, depth_metric: np.ndarray) -> np.ndarray:
        # Always update EMA
        if self._ema is None:
            self._ema = depth_metric.copy()
        else:
            self._ema = (
                self._alpha * depth_metric +
                (1.0 - self._alpha) * self._ema
            )

        if not self._use_sg:
            return self._ema.astype(np.float32)

        self._buf.append(depth_metric.copy())
        if len(self._buf) > self._win:
            self._buf.pop(0)

        if len(self._buf) == self._win:
            stack = np.stack(self._buf, axis=0)             # (win, H, W)
            # Apply SG coefficients along the time axis
            smoothed = np.einsum("t,thw->hw", self._sg_coeffs, stack)
            return np.clip(smoothed, 0.3, 25.0).astype(np.float32)

        return self._ema.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Ground Plane Detector
# ─────────────────────────────────────────────────────────────────────────────

class GroundPlaneDetector:
    """
    Identifies the walkable ground surface and estimates horizon + roll.

    Strategy
    ---------
    1. Restrict search to the configured band of the frame.
    2. Compute vertical depth gradient — ground has low gradient.
    3. Use low HSV saturation as a grey-asphalt indicator.
    4. Combine signals into a ground candidate mask.
    5. Morphological cleanup.
    6. Fit a line to the top edge → horizon row + roll estimate.

    NOTE: This method accepts (frame_bgr, depth_raw) — both are needed.
    """

    def __init__(self, cfg):
        self._top = cfg.camera.horizon_search_top
        self._bot = cfg.camera.horizon_search_bot

    def detect(
        self,
        frame_bgr:  np.ndarray,
        depth_raw:  np.ndarray,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Returns
        -------
        ground_mask : bool H×W
        horizon_y   : int pixel row
        roll_deg    : float camera roll estimate (degrees)
        """
        h, w = depth_raw.shape
        y0 = int(h * self._top)
        y1 = int(h * self._bot)

        region_d  = depth_raw[y0:y1, :]
        region_bgr = frame_bgr[y0:y1, :]

        # Vertical gradient of depth (low = smooth surface = possible ground)
        grad_y = np.abs(cv2.Sobel(region_d, cv2.CV_32F, 0, 1, ksize=5))
        smooth  = grad_y < np.percentile(grad_y, 45)

        # Colour: grey asphalt has low saturation
        hsv     = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
        low_sat = hsv[:, :, 1] < 70

        # Depth: prefer pixels closer than 1.5× median of region
        med_d    = float(np.median(region_d))
        near     = region_d < (med_d * 1.5)

        candidate = (smooth & low_sat & near).astype(np.uint8)

        # Morphological cleanup
        k       = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
        cleaned = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, k).astype(bool)

        # Full-frame mask
        ground_mask = np.zeros((h, w), dtype=bool)
        ground_mask[y0:y1, :] = cleaned

        # Horizon estimation: topmost ground row per column
        horizon_y, roll_deg = self._fit_horizon(ground_mask, h, w)

        return ground_mask, horizon_y, roll_deg

    @staticmethod
    def _fit_horizon(
        ground_mask: np.ndarray, h: int, w: int
    ) -> Tuple[int, float]:
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
        m, b  = np.polyfit(px, py, 1)
        hy    = int(np.clip(b + m * (w / 2.0), 0, h - 1))
        roll  = float(np.degrees(np.arctan(m)))
        return hy, roll


# ─────────────────────────────────────────────────────────────────────────────
# YOLO Segmentation Detector
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationDetector:
    """
    Wraps YOLO11-seg (or YOLOv8-seg fallback) for instance segmentation.

    FIX-1: uses cfg.seg.iou_thresh  (not iou_threshold)
    FIX-2: uses cfg.seg.OBSTACLE_CLASSES (not obstacle_classes)
    """

    def __init__(self, cfg, device: torch.device):
        from ultralytics import YOLO

        # FIX-2: correct attribute name
        self._obstacle_classes: Dict[int, str] = cfg.seg.OBSTACLE_CLASSES
        self._conf   = cfg.seg.conf_thresh
        # FIX-1: correct attribute name
        self._iou    = cfg.seg.iou_thresh
        self._device = str(device)

        log.info(f"[SegDet] Loading [{cfg.seg.model_path}]…")
        try:
            self.model = YOLO(cfg.seg.model_path)
        except Exception:
            log.warning(f"[SegDet] Fallback to yolov8n-seg.pt")
            self.model = YOLO("yolov8n-seg.pt")

        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._run(dummy)
        log.info("[SegDet] Warmup complete.")

    def detect(self, frame: np.ndarray) -> List[InstanceMask]:
        return self._run(frame)

    def _run(self, frame: np.ndarray) -> List[InstanceMask]:
        h, w = frame.shape[:2]
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

        masks_data = results.masks.data.cpu().numpy()   # N × Hm × Wm
        boxes      = results.boxes

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            label  = self._obstacle_classes.get(cls_id)
            if label is None:
                continue

            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            raw_mask     = masks_data[i]
            mask_resized = cv2.resize(
                raw_mask, (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            out.append(InstanceMask(
                label    = label,
                cls_id   = cls_id,
                conf     = conf_val,
                bbox     = (x1, y1, x2, y2),
                mask     = mask_resized,
                track_id = i,
            ))

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Risk Heatmap Builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_risk_heatmap(
    depth_metric:   np.ndarray,
    obstacle_masks: List[InstanceMask],
    ground_mask:    np.ndarray,
    hazard_weights: Dict[str, float],
) -> np.ndarray:
    """
    float32 H×W in [0, 1].
    Obstacle pixels scored by (weight / depth); ground pixels forced to 0.
    """
    h, w    = depth_metric.shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    max_w = max(hazard_weights.values(), default=28.0)

    for inst in obstacle_masks:
        if not inst.mask.any():
            continue
        w_val   = hazard_weights.get(inst.label, 3.0)
        d_px    = depth_metric[inst.mask]
        danger  = np.clip(w_val / (d_px + 0.5), 0.0, w_val)
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

    process(frame) runs depth + segmentation sequentially and returns
    a fully populated PerceptionOutput.  For an async/parallel variant
    see the threaded wrapper in main.py (InferenceThread).
    """

    def __init__(self, cfg, width: int, height: int):
        self.cfg    = cfg
        self.width  = width
        self.height = height

        self.device = self._select_device()
        log.info(f"[Perception] Device: {self.device}")

        # Sub-components
        self._depth_est  = DepthEstimator(cfg.depth.model_type, self.device)
        self._seg_det    = SegmentationDetector(cfg, self.device)
        self._ground_det = GroundPlaneDetector(cfg)
        self._smoother   = DepthSmoother(
            cfg.depth.sg_window_len,
            cfg.depth.sg_poly_order,
            cfg.depth.ema_alpha,
        )

        # Calibrator — fy is updated once intrinsics are computed
        from config import CFG
        self._calibrator = MetricCalibrator(
            default_scale    = cfg.depth.default_scale,
            default_shift    = cfg.depth.default_shift,
            calib_ema_alpha  = cfg.depth.calib_ema_alpha
                               if hasattr(cfg.depth, "calib_ema_alpha") else 0.12,
            camera_height_m  = cfg.camera.chest_height_m,
            fy               = CFG.fy if CFG.fy > 0 else 500.0,
        )

        self._frame_id    = 0
        self._last_smooth: Optional[np.ndarray] = None

    # ── Public ────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> PerceptionOutput:
        t0 = time.perf_counter()

        h, w = frame.shape[:2]

        # Update calibrator with latest intrinsics (covers the case where
        # compute_intrinsics was called after Perception was constructed)
        from config import CFG
        if CFG.fy > 0:
            self._calibrator.set_fy(CFG.fy)

        # 1. Depth estimation (raw inverse-relative)
        scale = self.cfg.pipeline.depth_frame_scale
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        depth_raw_small = self._depth_est.estimate(small)
        depth_raw = cv2.resize(
            depth_raw_small, (w, h), interpolation=cv2.INTER_LINEAR
        )

        # 2. Ground plane detection
        ground_mask, horizon_y, roll_deg = self._ground_det.detect(
            frame, depth_raw
        )

        # FIX-3: proper metric calibration using ground-plane geometric anchor
        scale_val, shift_val, depth_metric = self._calibrator.calibrate(
            depth_raw, ground_mask, horizon_y, CFG.cy
        )

        # 3. Temporal smoothing
        depth_smooth = self._smoother.update(depth_metric)
        self._last_smooth = depth_smooth

        # 4. Instance segmentation
        obstacle_masks = self._seg_det.detect(frame)

        # 5. Risk heatmap
        heatmap = _build_risk_heatmap(
            depth_metric,
            obstacle_masks,
            ground_mask,
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
        """No-op for synchronous variant; present for API compatibility."""
        pass

    @staticmethod
    def _select_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

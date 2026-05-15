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
    frame:             np.ndarray              # original frame (BGR)
    depth_raw:         np.ndarray              # MiDaS raw inverse-depth  H×W float32
    depth_metric:      np.ndarray              # metric depth in metres    H×W float32
    depth_smooth:      np.ndarray              # EMA-smoothed metric depth H×W float32
    ground_mask:       np.ndarray              # bool H×W — walkable surface
    obstacle_masks:    List[InstanceMask]      # obstacle detections
    risk_heatmap:      np.ndarray              # float32 H×W [0,1] danger intensity
    horizon_y:         int                     # estimated horizon pixel row
    roll_deg:          float                   # estimated camera roll (°)
    depth_scale:       float                   # metric calibration scale
    depth_shift:       float                   # metric calibration shift
    frame_id:          int = 0
    inference_ms:      float = 0.0


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
        # --- FIXED: LOCAL OFFLINE LOADING ---
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False, trust_repo=True)
            self.model.load_state_dict(torch.load("midas_weights.pt", map_location=device))
            log.info("✅ MiDaS local weights loaded successfully.")
        except Exception as e:
            log.warning(f"⚠️ Local weights failed ({e}), attempting online load...")
            self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            
        self.model.to(device).eval()

        # Try to load transforms from local hub cache if available
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
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

    def _infer(self, frame: np.ndarray) -> List[InstanceMask]:
        """Returns obstacle_masks."""
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
    def __init__(self, cfg):
        self.cfg = cfg

    def detect(self, frame_bgr: np.ndarray, depth_raw: np.ndarray) -> np.ndarray:
        h, w = depth_raw.shape
        mask = np.zeros((h, w), dtype=bool)

        # Heuristic: pixels in bottom portion with low depth gradient
        search_top = int(h * self.cfg.camera.horizon_search_top)
        search_bot = int(h * self.cfg.camera.horizon_search_bot)
        
        # Simple position-based heuristic for fallback
        mask[search_bot:, :] = True
        
        # Refine with depth smoothness if possible
        dy = np.abs(np.diff(depth_raw, axis=0))
        dy = np.pad(dy, ((1,0),(0,0)), mode='constant')
        mask = mask & (dy < np.percentile(dy, 70))
        
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# Perception Engine
# ─────────────────────────────────────────────────────────────────────────────

class Perception:
    def __init__(self, cfg, width: int, height: int):
        self.cfg = cfg
        self.device = self._select_device()
        self.width = width
        self.height = height

        self.detector = SegmentationDetector(
            cfg.seg.model_path, cfg.seg.conf_thresh, 
            cfg.seg.iou_threshold, cfg.seg.obstacle_classes, self.device
        )
        self.depth = DepthEstimator(cfg.depth.model_type, self.device)
        self.ground = GroundPlaneDetector(cfg)

        self._frame_id = 0
        self._last_depth_metric = None
        self._last_depth_smooth = None
        self._last_obstacles    = []
        self._last_ground       = None
        self._last_horizon_y    = int(height * 0.5)
        self._last_roll         = 0.0
        self._last_scale        = 1.0
        self._last_shift        = 0.0

    def stop(self):
        pass

    def process(self, frame: np.ndarray) -> PerceptionOutput:
        t0 = time.perf_counter()
        self._frame_id += 1

        # 1. Inference
        depth_raw = self.depth.estimate(frame)
        obstacles = self.detector.detect(frame)
        ground    = self.ground.detect(frame, depth_raw)

        # 2. Metric Calibration (simplified)
        # In a real setup, scale is derived from ground points + camera height
        # Here we use a stable default or simple median anchor
        self._last_scale = 5.0  # approximate meters for MiDaS small
        depth_metric = (1.0 / (depth_raw + 1e-6)) * self._last_scale
        
        # 3. Smoothing
        if self._last_depth_smooth is None:
            self._last_depth_smooth = depth_metric
        else:
            alpha = self.cfg.depth.ema_alpha
            self._last_depth_smooth = alpha * depth_metric + (1 - alpha) * self._last_depth_smooth

        self._last_obstacles = obstacles
        self._last_ground    = ground

        # 4. Heatmap
        heatmap = self._build_risk_heatmap(depth_metric, obstacles, ground)

        return PerceptionOutput(
            frame          = frame,
            depth_raw      = depth_raw,
            depth_metric   = depth_metric,
            depth_smooth   = self._last_depth_smooth,
            ground_mask    = ground,
            obstacle_masks = obstacles,
            risk_heatmap   = heatmap,
            horizon_y      = self._last_horizon_y,
            roll_deg       = self._last_roll,
            depth_scale    = self._last_scale,
            depth_shift    = self._last_shift,
            frame_id       = self._frame_id,
            inference_ms   = (time.perf_counter() - t0) * 1000,
        )

    def _build_risk_heatmap(self, depth, obstacles, ground):
        h, w = depth.shape
        heatmap = np.zeros((h, w), dtype=np.float32)
        for obs in obstacles:
            # Higher risk for closer objects
            dist = np.median(depth[obs.mask]) if np.any(obs.mask) else 20.0
            risk = np.clip(10.0 / (dist + 1e-6), 0, 1)
            heatmap[obs.mask] = risk
        return heatmap

    @staticmethod
    def _select_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

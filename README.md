# SoundVision V3 — Technical Documentation

> **Semantic Segmentation + Monocular Depth for Pedestrian Collision Avoidance**  
> A chest-mount spatial awareness system that tells visually impaired users what's in their path — before it's a problem.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    THREAD TOPOLOGY                          │
├──────────────┬──────────────────────────┬───────────────────┤
│  CAPTURE     │  INFERENCE               │  RENDER (main)    │
│  Thread      │  Thread                  │  Thread           │
│              │                          │                   │
│  cv2.Video   │  ┌─────────────────┐    │  HUDRenderer      │
│  Capture()   │  │  Perception     │    │  ↓                │
│  ↓           │  │  ├─DepthThread  │    │  cv2.VideoWriter  │
│  Queue[2]    │  │  └─SegThread    │    │  ↓                │
│              │  └────────┬────────┘    │  cv2.imshow()     │
│              │           │             │                   │
│              │  SpatialAnalyzerV3      │                   │
│              │           │             │                   │
│              │  RiskEngineV3           │                   │
│              │           │             │                   │
│              │  GuidanceSystem → TTS   │                   │
│              │           │             │                   │
│              │  PipelineState (locked) │                   │
└──────────────┴──────────────────────────┴───────────────────┘
```

All heavy AI inference is isolated in background threads. The main thread **only** reads the latest shared state and renders — guaranteeing zero-lag video output regardless of GPU speed.

---

## File Structure

| File | Role |
|---|---|
| `config.py` | All tunable parameters — FOV, depths, weights, cooldowns |
| `perception.py` | Parallel YOLOv11-seg + MiDaS → depth maps, instance masks, risk heatmap |
| `spatial_v3.py` | 2D masks → 3D world coordinates, corridor, TTC, velocity |
| `risk_engine_v3.py` | Vector-intersection risk scoring, stationary suppression, trend analysis |
| `main.py` | Full async pipeline, HUD renderer, TTS, CLI |
| `README.md` | This document |

---

## The 3D Ground-Plane Projection Math

### 1. Pinhole Camera Model

Every pixel `(u, v)` in the image corresponds to a ray in 3D space. With known intrinsics:

```
fx = (W/2) / tan(HFOV/2)     # horizontal focal length in pixels
fy = (H/2) / tan(VFOV/2)     # vertical focal length in pixels
cx = W/2                      # principal point (assumed centred)
cy = H/2
```

Given metric depth `d` (metres from camera) for pixel `(u, v)`, the **camera-frame 3D point** is:

```
X_cam = (u - cx) * d / fx
Y_cam = (v - cy) * d / fy
Z_cam = d
```

### 2. Camera-Tilt Correction (Chest Mount)

The camera is mounted on the user's chest, tilted downward by angle `θ` (nominal 12°, auto-refined by ground mask). We rotate around the X-axis to go from camera frame to a **world frame** where `Y=0` is the ground:

```
X_w =  X_cam
Y_w =  Y_cam·cos(θ) − Z_cam·sin(θ)
Z_w =  Y_cam·sin(θ) + Z_cam·cos(θ)
```

After this rotation:
- `Z_w` = **forward distance** (metres ahead of user)
- `X_w` = **lateral offset** (metres left/right; positive = right)
- `Y_w` = **vertical offset** (metres; ground = `−camera_height`)

Ground pixels satisfy: `Y_w ≈ −H` where `H = 1.2 m` (chest height).

### 3. Auto-Horizon Calibration

Because tilt and roll vary (posture, terrain), we **don't hardcode** the horizon. Instead, each frame:

1. Take the bottom 60% of the frame.
2. Compute vertical depth gradient (Sobel Y).
3. Find pixels where: gradient is **low** (flat surface), depth is **near median**, and colour saturation is **low** (grey asphalt).
4. These are ground candidates.
5. Fit a **RANSAC line** to the top edge of the ground mask.
6. The line gives: `horizon_y` (pixel row) and `roll_deg` (slope angle).

### 4. Metric Depth Calibration (Scale/Shift)

MiDaS outputs **inverse-relative disparity** — not metric depth. We convert via:

```
depth_metric = scale / (depth_raw + shift)
```

`scale` and `shift` are estimated each frame using the ground plane as a **free anchor**:

For each ground pixel at row `v`, the geometric metric depth is:
```
angle_below_horizon = arctan((v - horizon_y) / fy)
depth_anchor = camera_height / tan(angle_below_horizon)
```

We then solve for `scale` via:
```
scale = median(depth_anchor × (depth_raw + shift))
```

`shift` is held at a stable prior (EMA-smoothed) to keep the system robust when ground pixels are sparse. This gives **centimetre-accurate depth** near the camera without any calibration target.

### 5. Walking Corridor (Dynamic Trapezoid)

The danger zone is a trapezoid in the `(X_w, Z_w)` ground plane:

```
Near edge (Z_w = 0):    width = 0.80 m   (shoulder width)
Far edge  (Z_w = 30 m): width = 1.60 m   (expands with perspective)
```

A 3D point is inside the corridor with probability:

```
half_w(Z) = near_half + (far_half − near_half) × (Z / Z_max)

excess = |X_w| − half_w(Z)

if excess ≤ 0:       P = 1.0                    (fully inside)
if excess ≤ margin:  P = sigmoid(−excess/margin × 5)   (soft edge)
if excess > margin:  P = 0.0                    (outside)
```

The soft sigmoid boundary prevents the risk score from jumping abruptly when an object crosses the corridor edge.

### 6. Projecting the Corridor to Pixels

We project the 4 world-frame corners back to image space:

```
# World → Camera (inverse tilt rotation)
X_cam =  X_w
Y_cam =  Y_w·cos(θ) + Z_w·sin(θ)
Z_cam = −Y_w·sin(θ) + Z_w·cos(θ)

# Camera → Pixel
u = fx × X_cam / Z_cam + cx
v = fy × Y_cam / Z_cam + cy
```

Roll correction is applied by rotating all 4 pixel corners around the frame centre by `−roll_deg`.

---

## Time-to-Collision (TTC) — Depth Delta Method

Unlike bounding-box TTC (which fails when perspective distorts box size), we compute TTC from the **actual depth of the closest mask pixel**:

```
history = [d₁, d₂, d₃, …, dₙ]  (metric depths over last N AI frames)

slope = linear_regression_slope(history)   # metres per AI-frame

rate_m_s = slope × AI_fps                  # metres per second

TTC = −current_depth / rate_m_s            # seconds
```

- If `slope ≥ 0`: object is receding → `TTC = ∞`
- If `slope < 0`: object is approaching → `TTC = current_depth / |rate_m_s|`

Using **linear regression** over 12 frames (rather than frame-to-frame delta) makes TTC robust to single-frame depth noise.

---

## Stationary Object Suppression

Moving obstacles are dangerous; parked cars are not (for a pedestrian). We detect stationarity via:

```
depth_std = std(depth_history[-6 frames])
speed_3d  = ||velocity_vector||

if depth_std < 0.08 m  AND  speed_3d < 0.05 m/frame:
    stationary_counter += 1
else:
    stationary_counter = max(0, counter − 1)

is_stationary = (stationary_counter ≥ 5)
```

When `is_stationary = True`, the risk score is multiplied by a per-frame decay factor (`0.92`), asymptotically reducing parked-car alerts to near zero after ~10 frames. The score recovers instantly if the object starts moving.

---

## Risk Formula

```
R_raw = (mass_weight × velocity_effective × path_intersection)
        / distance_m
        × TTC_multiplier(ttc_s)
        × exp(max(0, 6 − distance_m) × 0.35)   ← proximity boost
        × (1 + mask_area_fraction × 8)           ← size factor

R_smooth = EMA(R_raw, α=0.35)    ← higher α when score rising
R_final  = R_smooth × stationary_decay_factor
```

`velocity_effective` weights closing velocity 3× more than lateral:
```
velocity_effective = 0.5 × |vel_total| + 1.5 × max(−vz, 0)
```

---

## Depth Smoothing Pipeline

Three layers of smoothing prevent jitter in TTC and risk scores:

```
Raw MiDaS output
    ↓
Per-frame metric calibration (scale/shift EMA, α=0.12)
    ↓
Temporal Savitzky-Golay filter (window=7, poly=2)
    [falls back to EMA α=0.18 when buffer not full]
    ↓
Per-track depth history (12 frames) → linear regression for TTC
    ↓
Risk score EMA (α=0.35) with asymmetric attack/decay
```

The Savitzky-Golay filter fits a polynomial to the last 7 depth frames and evaluates it at the current frame — this preserves edges (sudden real depth changes) while suppressing noise, unlike a plain moving average.

---

## Installation

```bash
pip install ultralytics torch torchvision opencv-python numpy scipy pyttsx3
```

**Models downloaded automatically on first run:**
- `yolo11n-seg.pt` (YOLOv11 nano segmentation, ~6 MB)
- `MiDaS_small` (via torch.hub, ~100 MB)

---

## Usage

```bash
# Process a video file
python main.py video.mp4 output_name

# Live webcam with display
python main.py 0 live_output --show

# Headless, no TTS (server/Colab)
python main.py video.mp4 result --no-tts

# All overlays disabled (clean output)
python main.py video.mp4 clean --no-heatmap --no-corridor --no-depth-overlay
```

---

## Tuning Guide

| Parameter | Location | Effect |
|---|---|---|
| `chest_height_m` | `CameraConfig` | Metric calibration anchor — measure your actual mount height |
| `mount_tilt_deg` | `CameraConfig` | Initial tilt estimate (auto-refined per frame) |
| `hfov_deg / vfov_deg` | `CameraConfig` | Match your camera's actual FOV |
| `corridor_width_near_m` | `RiskConfig` | How wide a path to protect (body width) |
| `target_ai_fps` | `PipelineConfig` | Lower on slow hardware (2–5 Hz still works) |
| `ema_alpha` (depth) | `DepthConfig` | Higher = more responsive but jittery guidance |
| `stationary_decay` | `RiskConfig` | Lower = faster suppression of parked objects |
| `TIER_CRITICAL` | `RiskConfig` | Raise if getting too many false STOP alerts |

---

## Known Limitations & Future Work

- **Stairs**: Ground-plane detector may misclassify stair risers as obstacles. A separate stair-specific model would improve this.
- **Glass / reflective surfaces**: MiDaS struggles with reflective floors. Adding a specular highlight detector as a ground-mask veto would help.
- **Night / low light**: YOLO confidence drops; consider a low-light enhancement pre-pass.
- **GPS fusion**: Absolute position would allow static obstacle mapping (parked cars always at same GPS coords = never alert).

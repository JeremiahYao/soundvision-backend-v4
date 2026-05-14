"""
main.py — SoundVision V3
==========================
High-performance asynchronous pipeline:

  Frame Capture Thread  →  Shared Frame Buffer
                                   ↓
  Perception Thread (depth + seg parallel inside Perception class)
                                   ↓
  Spatial Analysis + Risk Engine   (same thread as perception consumer)
                                   ↓
  Guidance + TTS Thread            (separate, non-blocking)
                                   ↓
  HUD Render + Write               (main thread, every frame)

This architecture guarantees:
  - Frame capture never stalls on AI inference
  - Audio never stalls on video rendering
  - AI results are always as fresh as possible

Usage
------
  python main.py  --video  path/to/video.mp4  --output  result
  python main.py  --video  0                  --output  live  --show
  python main.py  --video  path.mp4  --no-tts --no-depth-overlay
"""

from __future__ import annotations

import argparse
import logging
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ── SoundVision modules ───────────────────────────────────────────────────────
from config import CFG, Config
from perception import Perception, PerceptionOutput
from spatial_v3 import SpatialAnalyzerV3, Object3D, CorridorTrapezoid
from risk_engine_v3 import RiskEngineV3, ThreatRecord, Severity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SV3.Main")


# ─────────────────────────────────────────────────────────────────────────────
# Guidance System
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES = {
    Severity.CRITICAL: [
        "Stop! {label} {direction}!",
        "Danger! {label} {direction}, {dist:.1f} metres. Stop now.",
        "Collision risk! {label} {direction}, TTC {ttc:.0f} seconds.",
    ],
    Severity.HIGH: [
        "Warning: {label} approaching from {direction}, {dist:.1f} metres.",
        "Caution — {label} {direction}. {dist:.1f} metres, TTC {ttc:.0f} seconds.",
        "Watch out: {label} closing in from {direction}.",
    ],
    Severity.MEDIUM: [
        "{label} on your {direction}, {dist:.1f} metres away.",
        "Heads up — {label} {direction}, {dist:.1f} metres.",
    ],
    Severity.LOW: [
        "{label} detected {direction}, {dist:.1f} metres.",
    ],
}

SEVERITY_ICON = {
    Severity.CRITICAL: "⛔",
    Severity.HIGH:     "⚠️",
    Severity.MEDIUM:   "🔔",
    Severity.LOW:      "ℹ️",
    Severity.CLEAR:    "✅",
}

SEVERITY_BGR = {
    Severity.CRITICAL: (0,   0,   255),
    Severity.HIGH:     (0,   100, 255),
    Severity.MEDIUM:   (0,   200, 255),
    Severity.LOW:      (0,   220, 180),
    Severity.CLEAR:    (60,  220,  60),
}


class GuidanceSystem:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._cooldown:   dict = {}   # track_id → last spoken time
        self._tpl_idx:    dict = {}   # track_id → template rotation index
        self._last_sev:   dict = {}
        self._spoke_clear       = True
        self._last_clear_t      = 0.0
        self._prev_top_sev      = Severity.CLEAR

    def generate_speak(self, top: Optional[ThreatRecord]) -> Optional[str]:
        """Return TTS string or None (silence this frame)."""
        now = time.monotonic()
        if top is None:
            if (not self._spoke_clear and
                    self._prev_top_sev not in (Severity.CLEAR, Severity.LOW) and
                    now - self._last_clear_t > self.cfg.guidance.clear_delay_s):
                self._spoke_clear   = True
                self._last_clear_t  = now
                return self.cfg.guidance.clear_msg
            return None

        self._spoke_clear = False
        sev = top.severity
        tid = top.obj.track_id

        # Escalation override
        prev = self._last_sev.get(tid, Severity.CLEAR)
        escalated = (sev == Severity.CRITICAL and
                     prev not in (Severity.CRITICAL, Severity.HIGH))
        self._last_sev[tid] = sev

        cooldown = self.cfg.guidance.COOLDOWN_S.get(sev, 5.0)
        last_t   = self._cooldown.get(tid, 0.0)
        if not escalated and (now - last_t) < cooldown:
            return None

        self._cooldown[tid] = now
        self._prev_top_sev  = sev

        templates = TEMPLATES[sev]
        idx = self._tpl_idx.get(tid, 0) % len(templates)
        self._tpl_idx[tid] = idx + 1

        ttc_display = min(top.ttc_s, 99.0)
        text = templates[idx].format(
            label=top.obj.label,
            direction=top.direction,
            dist=top.distance_m,
            ttc=ttc_display,
        )
        return text

    def hud_text(self, top: Optional[ThreatRecord]) -> str:
        if top is None:
            return f"{SEVERITY_ICON[Severity.CLEAR]} Path clear."
        icon = SEVERITY_ICON.get(top.severity, "")
        ttc_str = f"  TTC {top.ttc_s:.1f}s" if top.ttc_s < 60 else ""
        trend_str = "↑" if top.trend > 2 else ("↓" if top.trend < -2 else "")
        return (
            f"{icon} {top.obj.label.upper()} {top.direction}"
            f"  |  {top.distance_m:.1f} m{ttc_str}"
            f"  |  risk {top.score:.0f} {trend_str}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TTS Engine
# ─────────────────────────────────────────────────────────────────────────────

class TTSEngine:
    def __init__(self, rate: int = 160):
        self._q: queue.Queue = queue.Queue(maxsize=3)
        self._engine = None
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
            log.info("TTS: pyttsx3 ready.")
        except Exception:
            log.warning("TTS: pyttsx3 unavailable — silent mode.")
        threading.Thread(target=self._run, daemon=True).start()

    def speak(self, text: str):
        if not text:
            return
        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass

    def _run(self):
        while True:
            text = self._q.get()
            if self._engine:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception:
                    pass
            else:
                print(f"[AUDIO] {text}")


# ─────────────────────────────────────────────────────────────────────────────
# HUD Renderer
# ─────────────────────────────────────────────────────────────────────────────

class HUDRenderer:
    def __init__(self, cfg: Config, show_depth: bool = True,
                 show_heatmap: bool = True, show_corridor: bool = True):
        self.cfg            = cfg
        self.show_depth     = show_depth
        self.show_heatmap   = show_heatmap
        self.show_corridor  = show_corridor

    def render(
        self,
        frame: np.ndarray,
        perc:  PerceptionOutput,
        objects: List[Object3D],
        corridor: Optional[CorridorTrapezoid],
        threats:  List[ThreatRecord],
        hud_text: str,
        fps: float,
        frame_id: int,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # ── 1. Depth overlay (bottom-right mini-map) ──────────────────────
        if self.show_depth and perc.depth_smooth is not None:
            self._overlay_depth(out, perc.depth_smooth, w, h)

        # ── 2. Risk heatmap overlay ────────────────────────────────────────
        if self.show_heatmap and perc.risk_heatmap is not None:
            self._overlay_heatmap(out, perc.risk_heatmap)

        # ── 3. Ground mask (semi-transparent green) ───────────────────────
        if perc.ground_mask is not None:
            gm_color = np.zeros_like(out)
            gm_color[perc.ground_mask] = (0, 80, 0)
            cv2.addWeighted(out, 1.0, gm_color, 0.25, 0, out)

        # ── 4. Walking corridor ────────────────────────────────────────────
        if self.show_corridor and corridor is not None:
            self._draw_corridor(out, corridor)

        # ── 5. Obstacle bounding boxes + masks ────────────────────────────
        for threat in threats:
            self._draw_obstacle(out, threat)

        # ── 6. Horizon line ────────────────────────────────────────────────
        hy = perc.horizon_y
        cv2.line(out, (0, hy), (w, hy), (200, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(out, f"horizon  roll {perc.roll_deg:+.1f}°",
                    (8, hy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (200, 200, 0), 1, cv2.LINE_AA)

        # ── 7. Top banner ─────────────────────────────────────────────────
        top_sev = threats[0].severity if threats else Severity.CLEAR
        banner_col = SEVERITY_BGR.get(top_sev, (60, 60, 60))
        self._draw_banner(out, hud_text, banner_col, w)

        # ── 8. Stats strip (bottom) ───────────────────────────────────────
        self._draw_stats(out, fps, frame_id, perc, threats, w, h)

        return out

    # ── Sub-renders ───────────────────────────────────────────────────────

    def _overlay_depth(self, frame, depth, w, h):
        mini_h, mini_w = 120, 180
        d_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
        d_small = cv2.resize(d_color, (mini_w, mini_h))
        x0, y0 = w - mini_w - 8, h - mini_h - 8
        roi = frame[y0:y0+mini_h, x0:x0+mini_w]
        cv2.addWeighted(roi, 0.3, d_small, 0.7, 0, roi)
        frame[y0:y0+mini_h, x0:x0+mini_w] = roi
        cv2.rectangle(frame, (x0, y0), (x0+mini_w, y0+mini_h), (120,120,120), 1)
        cv2.putText(frame, "depth", (x0+4, y0+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)

    def _overlay_heatmap(self, frame, heatmap):
        hm_u8 = (heatmap * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_HOT)
        # Only blend where heatmap is significant
        mask = heatmap > 0.1
        blend = frame.copy()
        cv2.addWeighted(frame, 0.65, hm_color, 0.35, 0, blend)
        frame[mask] = blend[mask]

    def _draw_corridor(self, frame, corridor: CorridorTrapezoid):
        pts = np.array(corridor.pixel_corners, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 140, 255))
        cv2.addWeighted(frame, 0.82, overlay, 0.18, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 180, 255), thickness=2)

    def _draw_obstacle(self, frame, threat: ThreatRecord):
        obj    = threat.obj
        colour = SEVERITY_BGR.get(threat.severity, (180, 180, 180))
        x1, y1, x2, y2 = obj.inst.bbox
        thick  = 3 if threat.severity in (Severity.CRITICAL, Severity.HIGH) else 2

        # Filled mask tint
        tint = np.zeros_like(frame)
        tint[obj.inst.mask] = colour
        cv2.addWeighted(frame, 1.0, tint, 0.30, 0, frame)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)

        # Label tag
        lines = [
            f"{obj.label}  {obj.distance_m:.1f}m",
            f"TTC {obj.ttc_s:.1f}s  risk {threat.score:.0f}",
        ]
        tag_h = 18
        for li, txt in enumerate(lines):
            ty = y1 - (len(lines) - li) * tag_h
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, ty-1), (x1+tw+4, ty+tag_h-2), colour, -1)
            cv2.putText(frame, txt, (x1+2, ty+tag_h-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

        # Path intersection bar
        bar_w = int((x2 - x1) * threat.obj.path_intersection)
        cv2.rectangle(frame, (x1, y2+2), (x1+bar_w, y2+7), colour, -1)

    def _draw_banner(self, frame, text, colour, w):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 58), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, text, (12, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.78, colour, 1, cv2.LINE_AA)

    def _draw_stats(self, frame, fps, frame_id, perc, threats, w, h):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-36), (w, h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

        stats = (
            f"frame {frame_id}"
            f"  |  fps {fps:.1f}"
            f"  |  threats {len(threats)}"
            f"  |  horizon {perc.horizon_y}px"
            f"  |  depth_scale {perc.depth_scale:.2f}"
            f"  |  inference {perc.inference_ms:.0f}ms"
        )
        cv2.putText(frame, stats, (8, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Frame Capture Thread
# ─────────────────────────────────────────────────────────────────────────────

class FrameCapture:
    """
    Runs cv2.VideoCapture in a background thread.
    Main thread always gets the *latest* frame without blocking on I/O.
    """

    def __init__(self, source):
        self._cap  = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = self._cap.get(cv2.CAP_PROP_FPS) or 20.0
        self.total  = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._q    = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._done = threading.Event()
        self._t    = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def read(self) -> Optional[np.ndarray]:
        try:
            return self._q.get(timeout=0.5)
        except queue.Empty:
            return None

    def done(self) -> bool:
        return self._done.is_set()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2.0)
        self._cap.release()

    def _run(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._done.set()
                break
            # Drop old frame if consumer is behind
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put(frame)


# ─────────────────────────────────────────────────────────────────────────────
# Inference + Guidance Thread
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    """Shared state between inference thread and render thread."""
    threats:   List[ThreatRecord] = None
    objects:   List[Object3D]     = None
    corridor:  Optional[CorridorTrapezoid] = None
    perc:      Optional[PerceptionOutput]  = None
    hud_text:  str = "Initialising…"
    lock:      threading.Lock = None

    def __post_init__(self):
        self.lock    = threading.Lock()
        self.threats = []
        self.objects = []


class InferenceThread:
    """
    Runs: Perception → Spatial → RiskEngine → Guidance.
    Pushes results to PipelineState for the render thread to read.
    """

    def __init__(self, cfg, width, height, tts: TTSEngine, state: PipelineState):
        self.cfg     = cfg
        self.state   = state
        self.tts     = tts
        CFG.compute_intrinsics(width, height)

        self.perc    = Perception(cfg, width, height)
        self.spatial = SpatialAnalyzerV3(cfg, height, width)
        self.engine  = RiskEngineV3(cfg)
        self.guide   = GuidanceSystem(cfg)

        self._stop   = threading.Event()
        self._t      = threading.Thread(target=self._run, daemon=True)

    def start(self, frame_queue: queue.Queue):
        self._fq = frame_queue
        self._t.start()

    def stop(self):
        self._stop.set()
        self.perc.stop()
        self._t.join(timeout=3.0)

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self._fq.get(timeout=0.05)
            except queue.Empty:
                continue

            perc_out             = self.perc.process(frame)
            objects, corridor    = self.spatial.analyze(perc_out)
            threats              = self.engine.evaluate_all(objects)
            top                  = threats[0] if threats else None
            speak                = self.guide.generate_speak(top)
            hud                  = self.guide.hud_text(top)

            if speak:
                self.tts.speak(speak)

            with self.state.lock:
                self.state.perc     = perc_out
                self.state.objects  = objects
                self.state.corridor = corridor
                self.state.threats  = threats
                self.state.hud_text = hud


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(
    video_path: str,
    output_name: str,
    show_window: bool       = False,
    tts_enabled: bool       = True,
    show_depth: bool        = True,
    show_heatmap: bool      = True,
    show_corridor: bool     = True,
):
    log.info("=" * 60)
    log.info("SoundVision V3 starting")
    log.info(f"  Source : {video_path}")
    log.info(f"  Output : {output_name}")
    log.info("=" * 60)

    # Ensure output directory exists
    out_dir = Path(CFG.pipeline.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap     = FrameCapture(0 if video_path == "0" else video_path)
    W, H    = cap.width, cap.height
    src_fps = cap.fps

    log.info(f"Video: {W}×{H} @ {src_fps:.1f} fps  ({cap.total} frames)")

    # Output writer
    out_path = str(out_dir / f"{output_name}.{CFG.pipeline.output_ext}")
    fourcc   = cv2.VideoWriter_fourcc(*CFG.pipeline.output_fourcc)
    writer   = cv2.VideoWriter(out_path, fourcc, src_fps, (W, H))

    tts     = TTSEngine(CFG.guidance.tts_rate) if tts_enabled else TTSEngine.__new__(TTSEngine)
    state   = PipelineState()
    hud     = HUDRenderer(CFG, show_depth, show_heatmap, show_corridor)

    # Inference thread has its own frame queue
    infer_q = queue.Queue(maxsize=2)
    inf_t   = InferenceThread(CFG, W, H, tts, state)
    inf_t.start(infer_q)
    
    # --- CRITICAL FIX: AI WARMUP ---
    log.info("⏳ [Main] Waiting for AI models to initialize (Warmup)...")
    max_init_wait = 120  
    start_init_t = time.time()
    
    while True:
        with state.lock:
            # Check if the perception object has processed at least one frame
            is_ready = state.perc is not None
        
        if is_ready:
            log.info("🚀 [Main] AI Engine is ONLINE. Starting video processing.")
            break
            
        if time.time() - start_init_t > max_init_wait:
            log.error("❌ [Main] AI Warmup timed out. Check internet/GPU.")
            inf_t.stop()
            cap.stop()
            return

        time.sleep(1.0)
        if int(time.time() - start_init_t) % 5 == 0:
            log.info(f"   ...loading models ({int(time.time() - start_init_t)}s elapsed)...")

    # --- PROCESSING LOOP ---
    frame_id      = 0
    fps_smooth    = 0.0
    t_last        = time.perf_counter()
    ai_skip       = max(1, int(src_fps / CFG.pipeline.target_ai_fps))

    log.info(f"Pipeline live. AI every {ai_skip} frames. Press Q to quit.")

    try:
        # Loop until video ends AND all AI tasks are finished
        while not (cap.done() and infer_q.empty()):
            frame = cap.read()
            if frame is None:
                if cap.done(): break
                continue

            # Feed inference thread (every ai_skip frames)
            if frame_id % ai_skip == 0:
                try:
                    infer_q.put_nowait(frame.copy())
                except queue.Full:
                    pass

            # Snapshot current state
            with state.lock:
                perc     = state.perc
                objects  = list(state.objects)
                corridor = state.corridor
                threats  = list(state.threats)
                hud_text = state.hud_text

            # FPS calculation
            now       = time.perf_counter()
            dt        = now - t_last or 1e-6
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)
            t_last    = now

            # HUD render
            if perc is not None:
                rendered = hud.render(
                    frame, perc, objects, corridor,
                    threats, hud_text, fps_smooth, frame_id
                )
            else:
                rendered = frame

            writer.write(rendered)

            if show_window:
                cv2.imshow("SoundVision V3", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1
            if frame_id % 150 == 0:
                pct = f"{frame_id/max(cap.total,1)*100:.1f}%" if cap.total > 0 else f"f{frame_id}"
                log.info(f"  [{pct}]  fps={fps_smooth:.1f}  threats={len(threats)}")

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        inf_t.stop()
        cap.stop()
        writer.release()
        if show_window:
            cv2.destroyAllWindows()
        log.info(f"Done. Saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="SoundVision V3 — Semantic Segmentation + Monocular Depth pedestrian safety system"
    )
    p.add_argument("video",  help="Path to input video or '0' for webcam")
    p.add_argument("output", help="Output file base name (saved to /content/)")
    p.add_argument("--show",             action="store_true", help="Show live OpenCV window")
    p.add_argument("--no-tts",           action="store_true", help="Disable TTS audio")
    p.add_argument("--no-depth-overlay", action="store_true", help="Hide depth mini-map")
    p.add_argument("--no-heatmap",       action="store_true", help="Hide risk heatmap overlay")
    p.add_argument("--no-corridor",      action="store_true", help="Hide corridor trapezoid")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        video_path     = args.video,
        output_name    = args.output,
        show_window    = args.show,
        tts_enabled    = not args.no_tts,
        show_depth     = not args.no_depth_overlay,
        show_heatmap   = not args.no_heatmap,
        show_corridor  = not args.no_corridor,
    )

"""
main.py — SoundVision V3
==========================
High-performance pipeline with correct video output timing.

─────────────────────────────────────────────────────────────
VIDEO SPEED FIX  (this version)
─────────────────────────────────────────────────────────────
Root cause of 10× speed-up in output video:

  The old FrameCapture thread used a queue(maxsize=2) with
  drop-oldest policy. The main loop read frames as fast as
  Python could execute — no wall-clock throttle.

  Because the main loop ran faster than real-time (GPU inference
  on a video file is not limited by frame rate), the capture
  thread dropped most frames. The VideoWriter received only
  every ~10th frame but was told fps=30, so 30 seconds of
  content compressed into ~3 seconds of output.

  Combined with ai_skip (only process every Nth frame), the
  multiplicative effect produced ~10× speed.

Fix applied — FIX-SPEED:
  Replaced FrameCapture thread + drop-oldest queue with a
  simple sequential reader.  The main loop:
    1. Reads EVERY frame from the source in order (no dropping).
    2. Writes EVERY frame to the output VideoWriter.
    3. For AI inference, sends only every ai_skip-th frame to
       the InferenceThread (unchanged — this is for performance,
       not for writing).
    4. The last known AI result is reused on non-AI frames for
       the HUD overlay. This keeps the video smooth and correct
       speed while AI runs at its own pace.

  Result: output video has exactly the same frame count and
  duration as the input. Playback speed is correct.

─────────────────────────────────────────────────────────────
GHOST DETECTION FIX (in perception.py)
─────────────────────────────────────────────────────────────
  See perception.py for the full explanation. Summary:
  - StableTracker gives each physical object a persistent ID
  - MIN_CONFIRM_FRAMES gate drops single-frame blips
  - Per-class confidence thresholds raised for ghost-prone classes
  - Minimum mask area check discards sub-pixel noise
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

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
    Severity.CRITICAL: "[STOP]",
    Severity.HIGH:     "[WARN]",
    Severity.MEDIUM:   "[NOTE]",
    Severity.LOW:      "[INFO]",
    Severity.CLEAR:    "[OK]  ",
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
        self.cfg           = cfg
        self._cooldown:    dict = {}
        self._tpl_idx:     dict = {}
        self._last_sev:    dict = {}
        self._spoke_clear        = True
        self._last_clear_t       = 0.0
        self._prev_top_sev       = Severity.CLEAR

    def generate_speak(self, top: Optional[ThreatRecord]) -> Optional[str]:
        now = time.monotonic()
        if top is None:
            if (
                not self._spoke_clear
                and self._prev_top_sev not in (Severity.CLEAR, Severity.LOW)
                and now - self._last_clear_t > self.cfg.guidance.clear_delay_s
            ):
                self._spoke_clear  = True
                self._last_clear_t = now
                return self.cfg.guidance.clear_msg
            return None

        self._spoke_clear = False
        sev = top.severity
        tid = top.obj.track_id
        prev      = self._last_sev.get(tid, Severity.CLEAR)
        escalated = (sev == Severity.CRITICAL
                     and prev not in (Severity.CRITICAL, Severity.HIGH))
        self._last_sev[tid] = sev

        cooldown = self.cfg.guidance.COOLDOWN_S.get(sev, 5.0)
        if not escalated and (now - self._cooldown.get(tid, 0.0)) < cooldown:
            return None

        self._cooldown[tid] = now
        self._prev_top_sev  = sev
        templates = TEMPLATES[sev]
        idx = self._tpl_idx.get(tid, 0) % len(templates)
        self._tpl_idx[tid] = idx + 1

        return templates[idx].format(
            label=top.obj.label,
            direction=top.direction,
            dist=top.distance_m,
            ttc=min(top.ttc_s, 99.0),
        )

    def hud_text(self, top: Optional[ThreatRecord]) -> str:
        if top is None:
            return f"{SEVERITY_ICON[Severity.CLEAR]} Path clear."
        icon      = SEVERITY_ICON.get(top.severity, "")
        ttc_str   = f"  TTC {top.ttc_s:.1f}s" if top.ttc_s < 60 else ""
        trend_str = "^" if top.trend > 2 else ("v" if top.trend < -2 else "")
        return (
            f"{icon} {top.obj.label.upper()} {top.direction}"
            f"  |  {top.distance_m:.1f}m{ttc_str}"
            f"  |  risk {top.score:.0f} {trend_str}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TTS Engine
# ─────────────────────────────────────────────────────────────────────────────

class _NullTTS:
    """Stub for when TTS is disabled — speak() just prints."""
    def speak(self, text: str) -> None:
        if text:
            print(f"[AUDIO] {text}")


class TTSEngine:
    def __init__(self, rate: int = 160):
        self._q: queue.Queue = queue.Queue(maxsize=3)
        self._engine = None
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
            log.info("[TTS] pyttsx3 ready.")
        except Exception:
            log.warning("[TTS] pyttsx3 unavailable — printing mode.")
        threading.Thread(target=self._run, daemon=True).start()

    def speak(self, text: str) -> None:
        if not text:
            return
        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass

    def _run(self) -> None:
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


def _make_tts(enabled: bool, rate: int):
    return TTSEngine(rate) if enabled else _NullTTS()


# ─────────────────────────────────────────────────────────────────────────────
# HUD Renderer
# ─────────────────────────────────────────────────────────────────────────────

class HUDRenderer:
    def __init__(self, cfg: Config, show_depth=True, show_heatmap=True,
                 show_corridor=True):
        self.cfg           = cfg
        self.show_depth    = show_depth
        self.show_heatmap  = show_heatmap
        self.show_corridor = show_corridor

    def render(self, frame, perc, objects, corridor, threats,
               hud_text, fps, frame_id):
        out = frame.copy()
        h, w = out.shape[:2]

        if self.show_depth and perc.depth_smooth is not None:
            self._overlay_depth(out, perc.depth_smooth, w, h)

        if self.show_heatmap and perc.risk_heatmap is not None:
            self._overlay_heatmap(out, perc.risk_heatmap)

        if perc.ground_mask is not None and perc.ground_mask.any():
            gm = np.zeros_like(out)
            gm[perc.ground_mask] = (0, 80, 0)
            cv2.addWeighted(out, 1.0, gm, 0.25, 0, out)

        if self.show_corridor and corridor is not None:
            self._draw_corridor(out, corridor)

        for threat in threats:
            self._draw_obstacle(out, threat)

        hy = perc.horizon_y
        cv2.line(out, (0, hy), (w, hy), (200, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(out, f"horizon  roll {perc.roll_deg:+.1f}deg",
                    (8, max(hy - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 0), 1, cv2.LINE_AA)

        top_sev    = threats[0].severity if threats else Severity.CLEAR
        self._draw_banner(out, hud_text, SEVERITY_BGR.get(top_sev, (60,60,60)), w)
        self._draw_stats(out, fps, frame_id, perc, threats, w, h)
        return out

    def _overlay_depth(self, frame, depth, w, h):
        mh, mw = 120, 180
        d_norm  = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
        d_small = cv2.resize(d_color, (mw, mh))
        x0, y0  = w - mw - 8, h - mh - 8
        roi = frame[y0:y0+mh, x0:x0+mw]
        cv2.addWeighted(roi, 0.3, d_small, 0.7, 0, roi)
        frame[y0:y0+mh, x0:x0+mw] = roi
        cv2.rectangle(frame, (x0,y0),(x0+mw,y0+mh),(120,120,120),1)
        cv2.putText(frame,"depth",(x0+4,y0+14),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1)

    def _overlay_heatmap(self, frame, heatmap):
        hm_u8    = (heatmap * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_HOT)
        mask     = heatmap > 0.1
        blend    = frame.copy()
        cv2.addWeighted(frame, 0.65, hm_color, 0.35, 0, blend)
        frame[mask] = blend[mask]

    def _draw_corridor(self, frame, corridor):
        pts = np.array(corridor.pixel_corners, dtype=np.int32)
        ov  = frame.copy()
        cv2.fillPoly(ov, [pts], (0, 140, 255))
        cv2.addWeighted(frame, 0.82, ov, 0.18, 0, frame)
        cv2.polylines(frame, [pts], True, (0, 180, 255), 2)

    def _draw_obstacle(self, frame, threat: ThreatRecord):
        obj    = threat.obj
        colour = SEVERITY_BGR.get(threat.severity, (180, 180, 180))
        x1, y1, x2, y2 = obj.inst.bbox
        thick  = 3 if threat.severity in (Severity.CRITICAL, Severity.HIGH) else 2

        tint = np.zeros_like(frame)
        tint[obj.inst.mask] = colour
        cv2.addWeighted(frame, 1.0, tint, 0.30, 0, frame)
        cv2.rectangle(frame, (x1,y1), (x2,y2), colour, thick)

        lines = [
            f"{obj.label}  {obj.distance_m:.1f}m",
            f"TTC {obj.ttc_s:.1f}s  risk {threat.score:.0f}",
        ]
        tag_h = 18
        for li, txt in enumerate(lines):
            ty = y1 - (len(lines) - li) * tag_h
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame,(x1,ty-1),(x1+tw+4,ty+tag_h-2),colour,-1)
            cv2.putText(frame,txt,(x1+2,ty+tag_h-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,0),1,cv2.LINE_AA)

        bar_w = int((x2-x1) * max(obj.path_intersection, 0))
        if bar_w > 0:
            cv2.rectangle(frame,(x1,y2+2),(x1+bar_w,y2+7),colour,-1)

    def _draw_banner(self, frame, text, colour, w):
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w,58), (0,0,0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, text, (12,40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.78, colour, 1, cv2.LINE_AA)

    def _draw_stats(self, frame, fps, frame_id, perc, threats, w, h):
        ov = frame.copy()
        cv2.rectangle(ov, (0,h-36), (w,h), (0,0,0), -1)
        cv2.addWeighted(ov, 0.50, frame, 0.50, 0, frame)
        stats = (
            f"frame {frame_id}"
            f"  |  fps {fps:.1f}"
            f"  |  threats {len(threats)}"
            f"  |  horizon {perc.horizon_y}px"
            f"  |  scale {perc.depth_scale:.2f}"
            f"  |  infer {perc.inference_ms:.0f}ms"
        )
        cv2.putText(frame, stats, (8, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Pipeline State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    threats:  List[ThreatRecord]          = field(default_factory=list)
    objects:  List[Object3D]              = field(default_factory=list)
    corridor: Optional[CorridorTrapezoid] = None
    perc:     Optional[PerceptionOutput]  = None
    hud_text: str                         = "Initialising…"
    lock:     threading.Lock              = field(default_factory=threading.Lock)


# ─────────────────────────────────────────────────────────────────────────────
# Inference Thread  (AI only — does NOT touch the video frames)
# ─────────────────────────────────────────────────────────────────────────────

class InferenceThread:
    """
    Receives frames from an input queue, runs the full AI pipeline,
    and writes results to PipelineState.

    Decoupled from the main render loop so AI latency never affects
    the output video frame rate.
    """

    def __init__(self, cfg, width, height, tts, state: PipelineState):
        self.cfg   = cfg
        self.state = state
        self.tts   = tts

        CFG.compute_intrinsics(width, height)
        self.perc    = Perception(cfg, width, height)
        self.spatial = SpatialAnalyzerV3(cfg, height, width)
        self.engine  = RiskEngineV3(cfg)
        self.guide   = GuidanceSystem(cfg)

        self._stop = threading.Event()
        self._t    = threading.Thread(target=self._run, daemon=True)

    def start(self, frame_queue: queue.Queue) -> None:
        self._fq = frame_queue
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        self.perc.stop()
        self._t.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._fq.get(timeout=0.05)
            except queue.Empty:
                continue

            perc_out          = self.perc.process(frame)
            objects, corridor = self.spatial.analyze(perc_out)
            threats           = self.engine.evaluate_all(objects)
            top               = threats[0] if threats else None
            speak             = self.guide.generate_speak(top)
            hud               = self.guide.hud_text(top)

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
    video_path:    str,
    output_name:   str,
    show_window:   bool = False,
    tts_enabled:   bool = True,
    show_depth:    bool = True,
    show_heatmap:  bool = True,
    show_corridor: bool = True,
) -> None:

    log.info("=" * 60)
    log.info("SoundVision V3")
    log.info(f"  Source : {video_path}")
    log.info(f"  Output : {output_name}")
    log.info("=" * 60)

    out_dir = Path(CFG.pipeline.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Open video source ─────────────────────────────────────────────────
    src = 0 if video_path == "0" else video_path
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info(f"Video: {W}x{H} @ {src_fps:.1f} fps  ({total} frames)")

    # ── Output writer ─────────────────────────────────────────────────────
    out_path = str(out_dir / f"{output_name}.{CFG.pipeline.output_ext}")
    fourcc   = cv2.VideoWriter_fourcc(*CFG.pipeline.output_fourcc)
    writer   = cv2.VideoWriter(out_path, fourcc, src_fps, (W, H))

    tts   = _make_tts(tts_enabled, CFG.guidance.tts_rate)
    state = PipelineState()
    hud   = HUDRenderer(CFG, show_depth, show_heatmap, show_corridor)

    # ── Inference thread ──────────────────────────────────────────────────
    # FIX-SPEED: queue maxsize is large enough that it never needs to drop.
    # AI frames are only submitted every ai_skip-th frame, so the queue
    # won't grow unboundedly even if AI is slow.
    infer_q = queue.Queue(maxsize=4)
    inf_t   = InferenceThread(CFG, W, H, tts, state)
    inf_t.start(infer_q)

    # ── Warmup ────────────────────────────────────────────────────────────
    log.info("Feeding first frame — waiting for AI models to initialise…")
    ret, first_frame = cap.read()
    if not ret:
        log.error("Could not read first frame.")
        inf_t.stop()
        cap.release()
        writer.release()
        return

    infer_q.put(first_frame.copy())

    max_wait = 120
    start_t  = time.time()
    last_log = -1

    while True:
        with state.lock:
            ready = state.perc is not None
        if ready:
            log.info("AI Engine online — starting main loop.")
            break
        elapsed = int(time.time() - start_t)
        if elapsed >= max_wait:
            log.error("AI warmup timed out.")
            inf_t.stop()
            cap.release()
            writer.release()
            return
        if elapsed % 5 == 0 and elapsed != last_log:
            last_log = elapsed
            log.info(f"  ...loading models ({elapsed}s elapsed)…")
        time.sleep(0.25)

    # ── FIX-SPEED: write the first frame we already read ──────────────────
    # We process it with whatever HUD state exists (just initialised above).
    with state.lock:
        perc     = state.perc
        objects  = list(state.objects)
        corridor = state.corridor
        threats  = list(state.threats)
        hud_text = state.hud_text

    if perc is not None:
        rendered = hud.render(first_frame, perc, objects, corridor,
                              threats, hud_text, 0.0, 0)
    else:
        rendered = first_frame.copy()
    writer.write(rendered)

    # ── Main render loop ──────────────────────────────────────────────────
    #
    # FIX-SPEED: The loop reads frames sequentially with cap.read() — every
    # single frame is read and written in order.  No dropping.
    # AI inference is submitted every ai_skip frames, but the RENDER path
    # touches every frame.  The last AI result is reused on non-AI frames.
    #
    ai_skip    = max(1, int(src_fps / CFG.pipeline.target_ai_fps))
    frame_id   = 1    # 0 was the first frame already written above
    fps_smooth = 0.0
    t_last     = time.perf_counter()

    log.info(f"Processing. AI every {ai_skip} frames. Ctrl-C to abort.")

    try:
        while True:
            # FIX-SPEED: blocking sequential read — preserves every frame
            ret, frame = cap.read()
            if not ret:
                break

            # Submit to AI thread every ai_skip frames
            # Use put_nowait + try/except to never block the render loop
            if frame_id % ai_skip == 0:
                try:
                    infer_q.put_nowait(frame.copy())
                except queue.Full:
                    pass   # AI is behind; reuse last result (safe)

            # Snapshot latest AI results (lock-free copy, non-blocking)
            with state.lock:
                perc     = state.perc
                objects  = list(state.objects)
                corridor = state.corridor
                threats  = list(state.threats)
                hud_text = state.hud_text

            # FPS display (wall-clock, reflects render speed for diagnostics)
            now        = time.perf_counter()
            dt         = max(now - t_last, 0.001)
            fps_smooth = 0.9 * fps_smooth + 0.1 / dt
            t_last     = now

            # Render HUD onto frame
            if perc is not None:
                rendered = hud.render(frame, perc, objects, corridor,
                                      threats, hud_text, fps_smooth, frame_id)
            else:
                rendered = frame.copy()
                cv2.putText(rendered, "Initialising…", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2)

            # FIX-SPEED: write EVERY frame in order
            writer.write(rendered)

            if show_window:
                cv2.imshow("SoundVision V3", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1
            if frame_id % 200 == 0:
                pct = f"{frame_id/max(total,1)*100:.1f}%" if total > 0 else f"f{frame_id}"
                log.info(
                    f"  [{pct}]  render_fps={fps_smooth:.1f}"
                    f"  threats={len(threats)}"
                    f"  ai_skip={ai_skip}"
                )

    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        inf_t.stop()
        cap.release()
        writer.release()
        if show_window:
            cv2.destroyAllWindows()
        log.info(f"Done — saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="SoundVision V3 — Segmentation + Depth pedestrian safety"
    )
    p.add_argument("video",              help="Input video path or '0' for webcam")
    p.add_argument("output",             help="Output file base name")
    p.add_argument("--show",             action="store_true")
    p.add_argument("--no-tts",           action="store_true")
    p.add_argument("--no-depth-overlay", action="store_true")
    p.add_argument("--no-heatmap",       action="store_true")
    p.add_argument("--no-corridor",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        video_path    = args.video,
        output_name   = args.output,
        show_window   = args.show,
        tts_enabled   = not args.no_tts,
        show_depth    = not args.no_depth_overlay,
        show_heatmap  = not args.no_heatmap,
        show_corridor = not args.no_corridor,
    )

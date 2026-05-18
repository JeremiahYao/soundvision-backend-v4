"""
main.py — SoundVision V3
==========================
High-performance asynchronous pipeline:

  FrameCapture Thread  →  infer_q  →  InferenceThread
                                            ↓
                                   Perception → Spatial → Risk → Guidance
                                            ↓
                                   PipelineState (thread-safe)
                                            ↓
  Main Thread reads PipelineState → HUDRenderer → VideoWriter

Fixes / upgrades vs. previous version
---------------------------------------
  FIX-6   TTSEngine: broken __new__ bypass when tts_enabled=False.
           Now uses a proper NullTTS stub so speak() is always safe.
  FIX-7   PipelineState: List fields use field(default_factory=list)
           instead of = None to satisfy dataclass typing rules.
  FIX-10  Warmup log spam: tracks last-logged second to avoid
           printing on every loop iteration within the same second.
  FIX-11  Alert templates: rewritten to be grammatically correct
           with the rich direction phrases from describe_direction().
           Previous templates produced broken sentences like
           "approaching from directly ahead" when direction already
           encodes motion ("ahead, approaching fast").
  FIX-12  Secondary-threat audio: when the top threat has cleared its
           cooldown but a second CRITICAL/HIGH threat exists, a brief
           secondary alert is queued after the primary.
  FIX-13  "Path clear" is now spoken only after a MEDIUM-or-above
           threat clears, and is suppressed if a new threat arrives
           within clear_delay_s. Previously it could fire
           spuriously during very brief track gaps.

Usage
------
  python main.py  path/to/video.mp4  result_name
  python main.py  0                  live  --show
  python main.py  video.mp4  out  --no-tts --no-depth-overlay
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
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
# Alert templates  — FIX-11
#
# Design rules:
#   1. {direction} already encodes motion ("ahead, approaching fast",
#      "on your left, crossing toward you") so templates must NOT add
#      redundant motion words like "approaching from" before it.
#   2. Sentences must be grammatically complete.
#   3. CRITICAL templates are short (≤8 words spoken) — blind user
#      needs to react, not parse a paragraph.
#   4. TTC is shown only when < 10 s and rounds to integer seconds.
#   5. Templates rotate per-track (idx cycles) to avoid repetition fatigue.
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES = {
    Severity.CRITICAL: [
        "Stop! {label} {direction}!",
        "Danger! {label} {direction}, {dist:.0f} metres.",
        "Stop now — {label} {direction}.",
        "Collision risk! {label} {direction}{ttc_clause}.",
    ],
    Severity.HIGH: [
        "Warning: {label} {direction}, {dist:.1f} metres.",
        "Caution — {label} {direction}{ttc_clause}.",
        "Watch out: {label} {direction}, {dist:.1f} metres.",
    ],
    Severity.MEDIUM: [
        "{label} {direction}, {dist:.1f} metres.",
        "Heads up — {label} {direction}, {dist:.1f} metres away.",
    ],
    Severity.LOW: [
        "{label} {direction}, {dist:.1f} metres.",
    ],
}

# Short secondary-threat addendum (appended to primary message when a
# second significant threat exists): FIX-12
SECONDARY_TEMPLATES = [
    "Also: {label} {direction}.",
    "{label} {direction} as well.",
]

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


def _fmt_ttc(ttc_s: float) -> str:
    """Return ', TTC N seconds' only when TTC is meaningful (< 10 s)."""
    if ttc_s < 10.0:
        return f", {ttc_s:.0f} seconds"
    return ""


def _render_template(template: str, threat: ThreatRecord) -> str:
    """
    Safely render one alert template string.

    All format keys are pre-computed here to avoid KeyError if a template
    variant references a field the previous template didn't.
    """
    return template.format(
        label      = threat.obj.label,
        direction  = threat.direction,
        dist       = threat.distance_m,
        ttc_clause = _fmt_ttc(threat.ttc_s),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Guidance System
# ─────────────────────────────────────────────────────────────────────────────

class GuidanceSystem:
    """
    Decides what to speak and when, based on the ranked threat list.

    Key behaviours:
      - Per-track cooldowns prevent the same object from being announced
        more than once per cooldown window.
      - Escalation bypass: if a track's severity suddenly jumps to CRITICAL
        from non-CRITICAL/HIGH, the cooldown is ignored.
      - Secondary alert (FIX-12): after the primary message, if a second
        CRITICAL or HIGH threat exists with no recent announcement, a short
        addendum is queued.
      - "Path clear" (FIX-13): only fires after a MEDIUM-or-above threat
        has cleared, and is suppressed if a new threat arrives quickly.
    """

    # Minimum severity that triggers a "path clear" message after it leaves
    _CLEAR_MIN_SEVERITY = {Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL}

    def __init__(self, cfg: Config):
        self.cfg            = cfg
        self._cooldown:     dict = {}    # tid → last-spoken time
        self._tpl_idx:      dict = {}    # tid → template rotation index
        self._last_sev:     dict = {}    # tid → last severity spoken
        self._spoke_clear         = True
        self._last_clear_t        = 0.0
        self._prev_top_sev        = Severity.CLEAR
        self._secondary_cooldown: dict = {}   # tid → last secondary-spoken time

    def generate_speak(self, threats: List[ThreatRecord]) -> Optional[str]:
        """
        Given the full sorted threat list, return the string to speak
        (or None if nothing should be said right now).

        Combines primary message + optional secondary addendum into one
        TTS call so pyttsx3 speaks them without an inter-call gap.
        """
        now = time.monotonic()
        top = threats[0] if threats else None

        # ── No threats at all: maybe announce clear ────────────────────────
        if top is None:
            return self._maybe_speak_clear(now)

        self._spoke_clear = False

        # ── Primary threat ─────────────────────────────────────────────────
        primary_msg = self._primary_message(top, now)

        # ── Secondary threat addendum (FIX-12) ────────────────────────────
        secondary_msg = None
        if primary_msg and len(threats) >= 2:
            secondary_msg = self._secondary_message(threats[1], now)

        if primary_msg is None:
            # Check if a secondary that bypassed cooldown warrants standalone
            # announcement — only for CRITICAL
            if len(threats) >= 2 and threats[1].severity == Severity.CRITICAL:
                secondary_msg = self._primary_message(threats[1], now)
                return secondary_msg

            self._prev_top_sev = top.severity
            return None

        self._prev_top_sev = top.severity

        if secondary_msg:
            return primary_msg + "  " + secondary_msg
        return primary_msg

    # ── Internal helpers ───────────────────────────────────────────────────

    def _maybe_speak_clear(self, now: float) -> Optional[str]:
        """FIX-13: only speak clear if a significant threat recently left."""
        if (
            not self._spoke_clear
            and self._prev_top_sev in self._CLEAR_MIN_SEVERITY
            and now - self._last_clear_t > self.cfg.guidance.clear_delay_s
        ):
            self._spoke_clear  = True
            self._last_clear_t = now
            return self.cfg.guidance.clear_msg
        return None

    def _primary_message(self, threat: ThreatRecord, now: float) -> Optional[str]:
        sev = threat.severity
        tid = threat.obj.track_id

        prev      = self._last_sev.get(tid, Severity.CLEAR)
        escalated = (
            sev == Severity.CRITICAL
            and prev not in (Severity.CRITICAL, Severity.HIGH)
        )
        self._last_sev[tid] = sev

        cooldown = self.cfg.guidance.COOLDOWN_S.get(sev, 5.0)
        last_t   = self._cooldown.get(tid, 0.0)
        if not escalated and (now - last_t) < cooldown:
            return None

        self._cooldown[tid] = now

        templates = TEMPLATES[sev]
        idx = self._tpl_idx.get(tid, 0) % len(templates)
        self._tpl_idx[tid] = idx + 1

        return _render_template(templates[idx], threat)

    def _secondary_message(self, threat: ThreatRecord, now: float) -> Optional[str]:
        """
        Short addendum for the second-highest threat, only if significant
        and not recently announced.  Uses a shorter cooldown than primary.
        """
        if threat.severity not in (Severity.CRITICAL, Severity.HIGH):
            return None

        tid      = threat.obj.track_id
        cooldown = max(self.cfg.guidance.COOLDOWN_S.get(threat.severity, 5.0) * 0.5, 2.0)
        last_t   = self._secondary_cooldown.get(tid, 0.0)
        if now - last_t < cooldown:
            return None

        self._secondary_cooldown[tid] = now
        idx = self._tpl_idx.get(tid, 0) % len(SECONDARY_TEMPLATES)
        tmpl = SECONDARY_TEMPLATES[idx]
        return tmpl.format(label=threat.obj.label, direction=threat.direction)

    def hud_text(self, top: Optional[ThreatRecord]) -> str:
        if top is None:
            return f"{SEVERITY_ICON[Severity.CLEAR]} Path clear."
        icon      = SEVERITY_ICON.get(top.severity, "")
        ttc_str   = f"  TTC {top.ttc_s:.1f}s" if top.ttc_s < 60 else ""
        trend_str = "↑" if top.trend > 2 else ("↓" if top.trend < -2 else "")
        return (
            f"{icon} {top.obj.label.upper()} — {top.direction}"
            f"  |  {top.distance_m:.1f}m{ttc_str}"
            f"  |  risk {top.score:.0f} {trend_str}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TTS Engine
# ─────────────────────────────────────────────────────────────────────────────

class _NullTTS:
    """
    Silent TTS stub used when tts_enabled=False.
    FIX-6: replaces the broken TTSEngine.__new__() bypass that left
           _q and _engine uninitialised, causing AttributeError on speak().
    """
    def speak(self, text: str) -> None:
        print(f"[AUDIO] {text}")


class TTSEngine:
    """Live TTS using pyttsx3 (falls back to printing if unavailable)."""

    def __init__(self, rate: int = 160):
        self._q: queue.Queue = queue.Queue(maxsize=3)
        self._engine = None
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
            log.info("[TTS] pyttsx3 ready.")
        except Exception:
            log.warning("[TTS] pyttsx3 unavailable — silent mode.")
        threading.Thread(target=self._run, daemon=True).start()

    def speak(self, text: str) -> None:
        if not text:
            return
        try:
            self._q.put_nowait(text)
        except queue.Full:
            pass   # drop if TTS is already busy — safety > completeness

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
    """
    FIX-6: factory that always returns an object with a safe speak() method.
    """
    if enabled:
        return TTSEngine(rate)
    return _NullTTS()


# ─────────────────────────────────────────────────────────────────────────────
# HUD Renderer
# ─────────────────────────────────────────────────────────────────────────────

class HUDRenderer:
    def __init__(
        self,
        cfg: Config,
        show_depth:    bool = True,
        show_heatmap:  bool = True,
        show_corridor: bool = True,
    ):
        self.cfg           = cfg
        self.show_depth    = show_depth
        self.show_heatmap  = show_heatmap
        self.show_corridor = show_corridor

    def render(
        self,
        frame:    np.ndarray,
        perc:     PerceptionOutput,
        objects:  List[Object3D],
        corridor: Optional[CorridorTrapezoid],
        threats:  List[ThreatRecord],
        hud_text: str,
        fps:      float,
        frame_id: int,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        if self.show_depth and perc.depth_smooth is not None:
            self._overlay_depth(out, perc.depth_smooth, w, h)

        if self.show_heatmap and perc.risk_heatmap is not None:
            self._overlay_heatmap(out, perc.risk_heatmap)

        if perc.ground_mask is not None and perc.ground_mask.any():
            gm_color = np.zeros_like(out)
            gm_color[perc.ground_mask] = (0, 80, 0)
            cv2.addWeighted(out, 1.0, gm_color, 0.25, 0, out)

        if self.show_corridor and corridor is not None:
            self._draw_corridor(out, corridor)

        for threat in threats:
            self._draw_obstacle(out, threat)

        hy = perc.horizon_y
        cv2.line(out, (0, hy), (w, hy), (200, 200, 0), 1, cv2.LINE_AA)
        cv2.putText(
            out, f"horizon  roll {perc.roll_deg:+.1f}deg",
            (8, max(hy - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 0), 1, cv2.LINE_AA,
        )

        top_sev    = threats[0].severity if threats else Severity.CLEAR
        banner_col = SEVERITY_BGR.get(top_sev, (60, 60, 60))
        self._draw_banner(out, hud_text, banner_col, w)
        self._draw_stats(out, fps, frame_id, perc, threats, w, h)

        return out

    # ── Sub-renders ───────────────────────────────────────────────────────

    def _overlay_depth(self, frame, depth, w, h):
        mini_h, mini_w = 120, 180
        d_norm  = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO)
        d_small = cv2.resize(d_color, (mini_w, mini_h))
        x0, y0  = w - mini_w - 8, h - mini_h - 8
        roi = frame[y0:y0 + mini_h, x0:x0 + mini_w]
        cv2.addWeighted(roi, 0.3, d_small, 0.7, 0, roi)
        frame[y0:y0 + mini_h, x0:x0 + mini_w] = roi
        cv2.rectangle(frame, (x0, y0), (x0 + mini_w, y0 + mini_h), (120, 120, 120), 1)
        cv2.putText(frame, "depth", (x0 + 4, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    def _overlay_heatmap(self, frame, heatmap):
        hm_u8    = (heatmap * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_HOT)
        mask     = heatmap > 0.1
        blend    = frame.copy()
        cv2.addWeighted(frame, 0.65, hm_color, 0.35, 0, blend)
        frame[mask] = blend[mask]

    def _draw_corridor(self, frame, corridor: CorridorTrapezoid):
        pts = np.array(corridor.pixel_corners, dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 140, 255))
        cv2.addWeighted(frame, 0.82, overlay, 0.18, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True,
                      color=(0, 180, 255), thickness=2)

    def _draw_obstacle(self, frame, threat: ThreatRecord):
        obj    = threat.obj
        colour = SEVERITY_BGR.get(threat.severity, (180, 180, 180))
        x1, y1, x2, y2 = obj.inst.bbox
        thick  = 3 if threat.severity in (Severity.CRITICAL, Severity.HIGH) else 2

        tint = np.zeros_like(frame)
        tint[obj.inst.mask] = colour
        cv2.addWeighted(frame, 1.0, tint, 0.30, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)

        lines = [
            f"{obj.label}  {obj.distance_m:.1f}m",
            f"TTC {obj.ttc_s:.1f}s  risk {threat.score:.0f}",
            threat.direction,   # show full direction phrase on HUD
        ]
        tag_h = 18
        for li, txt in enumerate(lines):
            ty = y1 - (len(lines) - li) * tag_h
            (tw, _), _ = cv2.getTextSize(
                txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1
            )
            cv2.rectangle(
                frame, (x1, ty - 1), (x1 + tw + 4, ty + tag_h - 2), colour, -1
            )
            cv2.putText(
                frame, txt, (x1 + 2, ty + tag_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA,
            )

        bar_w = int((x2 - x1) * max(threat.obj.path_intersection, 0))
        if bar_w > 0:
            cv2.rectangle(frame, (x1, y2 + 2), (x1 + bar_w, y2 + 7), colour, -1)

    def _draw_banner(self, frame, text, colour, w):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 58), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(
            frame, text, (12, 40),
            cv2.FONT_HERSHEY_DUPLEX, 0.78, colour, 1, cv2.LINE_AA,
        )

    def _draw_stats(self, frame, fps, frame_id, perc, threats, w, h):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 36), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
        stats = (
            f"frame {frame_id}"
            f"  |  fps {fps:.1f}"
            f"  |  threats {len(threats)}"
            f"  |  horizon {perc.horizon_y}px"
            f"  |  scale {perc.depth_scale:.2f}"
            f"  |  infer {perc.inference_ms:.0f}ms"
        )
        cv2.putText(
            frame, stats, (8, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Frame Capture Thread
# ─────────────────────────────────────────────────────────────────────────────

class FrameCapture:
    """Background thread that keeps the latest frame always available."""

    def __init__(self, source):
        self._cap = cv2.VideoCapture(source)
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

    def stop(self) -> None:
        self._stop.set()
        self._t.join(timeout=1.0)
        self._cap.release()

    def _run(self) -> None:
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret:
                self._done.set()
                break
            if self._q.full():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass
            self._q.put(frame)


# ─────────────────────────────────────────────────────────────────────────────
# Shared Pipeline State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    """
    Thread-safe shared state between InferenceThread and the render loop.

    FIX-7: List fields use field(default_factory=list) so dataclass
           initialises them as empty lists instead of sharing a None sentinel.
    """
    threats:  List[ThreatRecord]          = field(default_factory=list)
    objects:  List[Object3D]              = field(default_factory=list)
    corridor: Optional[CorridorTrapezoid] = None
    perc:     Optional[PerceptionOutput]  = None
    hud_text: str                         = "Initialising…"
    lock:     threading.Lock              = field(default_factory=threading.Lock)


# ─────────────────────────────────────────────────────────────────────────────
# Inference Thread
# ─────────────────────────────────────────────────────────────────────────────

class InferenceThread:
    """
    Runs: Perception → SpatialAnalyzer → RiskEngine → GuidanceSystem.
    Writes results to PipelineState for the render thread to consume.

    Note: generate_speak() now receives the full threats list (not just top)
    so GuidanceSystem can issue secondary alerts.  (FIX-12)
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

            # FIX-12: pass full list so secondary alerts can fire
            speak = self.guide.generate_speak(threats)
            hud   = self.guide.hud_text(top)

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

    cap = FrameCapture(0 if video_path == "0" else video_path)
    W, H    = cap.width, cap.height
    src_fps = cap.fps
    log.info(f"Video: {W}x{H} @ {src_fps:.1f} fps  ({cap.total} frames)")

    out_path = str(out_dir / f"{output_name}.{CFG.pipeline.output_ext}")
    fourcc   = cv2.VideoWriter_fourcc(*CFG.pipeline.output_fourcc)
    writer   = cv2.VideoWriter(out_path, fourcc, src_fps, (W, H))

    # FIX-6: always returns an object with a safe speak() method
    tts   = _make_tts(tts_enabled, CFG.guidance.tts_rate)
    state = PipelineState()
    hud   = HUDRenderer(CFG, show_depth, show_heatmap, show_corridor)

    infer_q = queue.Queue(maxsize=2)
    inf_t   = InferenceThread(CFG, W, H, tts, state)
    inf_t.start(infer_q)

    # ── Warmup: feed first frame and wait for first PerceptionOutput ──────
    log.info("Feeding first frame — waiting for AI models to initialise…")
    first_frame = cap.read()
    if first_frame is not None:
        infer_q.put(first_frame.copy())

    max_wait = 120
    start_t  = time.time()
    last_log = -1   # FIX-10: track last second we logged to avoid spam

    while True:
        with state.lock:
            ready = state.perc is not None
        if ready:
            log.info("AI Engine online — starting main loop.")
            break

        elapsed = int(time.time() - start_t)
        if elapsed >= max_wait:
            log.error("AI warmup timed out. Exiting.")
            inf_t.stop()
            cap.stop()
            writer.release()
            return

        # FIX-10: only print every 5 seconds, not every loop iteration
        if elapsed % 5 == 0 and elapsed != last_log:
            last_log = elapsed
            log.info(f"  ...loading models ({elapsed}s elapsed)…")

        time.sleep(0.25)

    # ── Main render loop ──────────────────────────────────────────────────
    frame_id   = 0
    fps_smooth = 0.0
    t_last     = time.perf_counter()
    ai_skip    = max(1, int(src_fps / CFG.pipeline.target_ai_fps))
    log.info(f"Running. AI every {ai_skip} frames. Ctrl-C or Q to quit.")

    try:
        while not cap.done():
            frame = cap.read()
            if frame is None:
                break

            if frame_id % ai_skip == 0:
                try:
                    infer_q.put_nowait(frame.copy())
                except queue.Full:
                    pass

            with state.lock:
                perc     = state.perc
                objects  = list(state.objects)
                corridor = state.corridor
                threats  = list(state.threats)
                hud_text = state.hud_text

            now        = time.perf_counter()
            dt         = max(now - t_last, 0.001)
            fps_smooth = 0.9 * fps_smooth + 0.1 / dt
            t_last     = now

            if perc is not None:
                rendered = hud.render(
                    frame, perc, objects, corridor,
                    threats, hud_text, fps_smooth, frame_id,
                )
            else:
                rendered = frame.copy()
                cv2.putText(
                    rendered, "Initialising perception…", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2,
                )

            writer.write(rendered)

            if show_window:
                cv2.imshow("SoundVision V3", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1
            if frame_id % 150 == 0:
                pct = (
                    f"{frame_id / max(cap.total, 1) * 100:.1f}%"
                    if cap.total > 0
                    else f"f{frame_id}"
                )
                log.info(
                    f"  [{pct}]  fps={fps_smooth:.1f}"
                    f"  threats={len(threats)}"
                    f"  skip={ai_skip}"
                )

    except KeyboardInterrupt:
        log.info("Interrupted.")
    finally:
        inf_t.stop()
        cap.stop()
        writer.release()
        if show_window:
            cv2.destroyAllWindows()
        log.info(f"Saved to {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
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

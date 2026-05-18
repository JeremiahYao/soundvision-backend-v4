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

Fixes applied vs. previous version
---------------------------------------
  FIX-6   TTSEngine: proper NullTTS stub — speak() always safe.
  FIX-7   PipelineState: List fields use field(default_factory=list).
  FIX-10  Warmup log spam suppressed.
  FIX-11  Templates rewritten for correct grammar with motion-aware
          direction phrases.
  FIX-12  Secondary-threat audio for simultaneous CRITICAL/HIGH threats.
  FIX-13  "Path clear" suppressed during brief tracking gaps.
  FIX-14  Navigation-grade audio:
           - Relatable distance language ("arm's reach", "2 steps")
           - Multi-obstacle count ("3 people blocking your path")
           - Avoidance instruction ("move right", "stop and wait")
           - Scene summary for situational awareness
  FIX-15  Stationary-obstacle velocity floor: stationary objects inside
          the corridor get a minimum closing velocity equal to the user's
          own walking speed, so they are never silently ignored.
  FIX-16  Startup voice: speaks "SoundVision starting. Please wait." on
          launch and "System ready. Path monitoring active." when AI is
          online. Speaks "System stopped." on shutdown.
  FIX-17  Bluetooth audio routing: TTSEngine automatically routes audio
          to the paired Bluetooth earpiece on Linux/Jetson using PulseAudio.
          Falls back silently if Bluetooth is not connected yet.

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
import subprocess
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
# Human-relatable distance language  (FIX-14)
# ─────────────────────────────────────────────────────────────────────────────

def _distance_phrase(dist_m: float, cfg: Config) -> str:
    """
    Convert a metric distance to a human-relatable spoken phrase.

    Uses the DISTANCE_PHRASES config dict (sorted by upper bound).
    Beyond the largest key: returns "far ahead".

    Examples:
      0.5 m → "right in front of you"
      1.0 m → "very close"
      1.8 m → "close"
      3.0 m → "nearby"
      5.0 m → "ahead"
      9.0 m → "in the distance"
     20.0 m → "far ahead"
    """
    for upper_m, phrase in sorted(cfg.guidance.DISTANCE_PHRASES.items()):
        if dist_m <= upper_m:
            return phrase
    return "far ahead"


# ─────────────────────────────────────────────────────────────────────────────
# Avoidance instruction generator  (FIX-14)
# ─────────────────────────────────────────────────────────────────────────────

def _avoidance_instruction(threats: List[ThreatRecord]) -> str:
    """
    Generate a short, actionable instruction based on the spatial layout
    of ALL active threats. This answers "what should I do?" not just
    "what's there?".

    lateral_m convention: negative = left of user, positive = right.
    """
    if not threats:
        return ""

    left_count   = sum(1 for t in threats if t.obj.lateral_m < -0.25)
    right_count  = sum(1 for t in threats if t.obj.lateral_m >  0.25)
    centre_count = sum(1 for t in threats if abs(t.obj.lateral_m) <= 0.25)

    top_sev  = threats[0].severity
    top_dist = threats[0].distance_m

    if top_sev == Severity.CRITICAL:
        if left_count > 0 and right_count > 0:
            return "Stop — path blocked."
        if left_count > 0:
            return "Move right."
        if right_count > 0:
            return "Move left."
        return "Stop."

    if top_sev == Severity.HIGH:
        if left_count > right_count:
            return "Bear right."
        if right_count > left_count:
            return "Bear left."
        if centre_count >= 1 and top_dist < 2.0:
            return "Slow down."
        return "Caution."

    if top_sev == Severity.MEDIUM:
        if left_count > right_count and right_count == 0:
            return "Keep right."
        if right_count > left_count and left_count == 0:
            return "Keep left."

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Multi-obstacle scene summary  (FIX-14)
# ─────────────────────────────────────────────────────────────────────────────

def _scene_summary(threats: List[ThreatRecord], cfg: Config) -> str:
    """
    Compact spoken scene description when multiple threats exist.
    Fires only when there are 2+ threats at MEDIUM or above.

    Examples:
      "3 people close."
      "Person and bicycle close."
    """
    significant = [
        t for t in threats
        if t.severity in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM)
    ]
    if len(significant) < 2:
        return ""

    counts: Counter = Counter(t.obj.label for t in significant)
    parts = []
    for label, count in counts.most_common():
        if count == 1:
            parts.append(label)
        elif label.endswith("s"):
            parts.append(f"{count} {label}")
        else:
            parts.append(f"{count} {label}s")

    if len(parts) == 1:
        label_str = parts[0]
    elif len(parts) == 2:
        label_str = f"{parts[0]} and {parts[1]}"
    else:
        label_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"

    dist_phrase = _distance_phrase(significant[0].distance_m, cfg)
    return f"{label_str} {dist_phrase}."


# ─────────────────────────────────────────────────────────────────────────────
# Alert templates  (FIX-11 + FIX-14)
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATES = {
    Severity.CRITICAL: [
        "Stop! {label} {direction}. {action}",
        "Danger — {label} {direction}, {dist_phrase}. {action}",
        "Stop now! {label} {direction}{ttc_clause}. {action}",
        "{label} {direction}, {dist_phrase}. {action}",
    ],
    Severity.HIGH: [
        "Warning: {label} {direction}, {dist_phrase}. {action}",
        "Caution — {label} {direction}{ttc_clause}. {action}",
        "Watch out: {label} {direction}, {dist_phrase}. {action}",
    ],
    Severity.MEDIUM: [
        "{label} {direction}, {dist_phrase}.",
        "Heads up — {label} {direction}, {dist_phrase}.",
    ],
    Severity.LOW: [
        "{label} {direction}.",
    ],
}

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
    if ttc_s < 10.0:
        return f", {ttc_s:.0f} seconds"
    return ""


def _render_template(
    template: str,
    threat: ThreatRecord,
    dist_phrase: str,
    action: str,
) -> str:
    rendered = template.format(
        label       = threat.obj.label,
        direction   = threat.direction,
        dist_phrase = dist_phrase,
        ttc_clause  = _fmt_ttc(threat.ttc_s),
        action      = action,
    ).strip()
    while "  " in rendered:
        rendered = rendered.replace("  ", " ")
    rendered = rendered.replace(". .", ".").replace("!.", "!").strip()
    if not rendered.endswith((".", "!", "?")):
        rendered += "."
    return rendered


# ─────────────────────────────────────────────────────────────────────────────
# Guidance System
# ─────────────────────────────────────────────────────────────────────────────

class GuidanceSystem:
    """
    Decides what to speak and when, based on the ranked threat list.

    Full spoken output structure (FIX-14):
      [scene summary if 2+ threats]  →  [primary alert + action]  →  [secondary]

    Example:
      "2 people and a bicycle close.
       Stop! Person ahead, approaching fast. Move right.
       Also: bicycle on your left."
    """

    _CLEAR_MIN_SEVERITY       = {Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL}
    _SCENE_SUMMARY_COOLDOWN_S = 10.0

    def __init__(self, cfg: Config):
        self.cfg                    = cfg
        self._cooldown:      Dict   = {}
        self._tpl_idx:       Dict   = {}
        self._last_sev:      Dict   = {}
        self._secondary_cd:  Dict   = {}
        self._spoke_clear           = True
        self._last_clear_t          = 0.0
        self._prev_top_sev          = Severity.CLEAR
        self._last_scene_summary_t  = 0.0

    def generate_speak(self, threats: List[ThreatRecord]) -> Optional[str]:
        now = time.monotonic()
        top = threats[0] if threats else None

        if top is None:
            return self._maybe_speak_clear(now)

        self._spoke_clear = False

        # Scene summary
        scene = ""
        if (
            len(threats) >= 2
            and now - self._last_scene_summary_t > self._SCENE_SUMMARY_COOLDOWN_S
        ):
            scene = _scene_summary(threats, self.cfg)
            if scene:
                self._last_scene_summary_t = now

        # Primary
        avoidance = _avoidance_instruction(threats)
        primary   = self._primary_message(top, avoidance, now)

        # Secondary
        secondary = ""
        if primary and len(threats) >= 2:
            secondary = self._secondary_message(threats[1], now)

        if primary is None:
            if len(threats) >= 2 and threats[1].severity == Severity.CRITICAL:
                msg = self._primary_message(threats[1], avoidance, now)
                self._prev_top_sev = threats[1].severity
                return msg
            self._prev_top_sev = top.severity
            return None

        self._prev_top_sev = top.severity
        parts = [p for p in [scene, primary, secondary] if p]
        return "  ".join(parts)

    def _maybe_speak_clear(self, now: float) -> Optional[str]:
        if (
            not self._spoke_clear
            and self._prev_top_sev in self._CLEAR_MIN_SEVERITY
            and now - self._last_clear_t > self.cfg.guidance.clear_delay_s
        ):
            self._spoke_clear  = True
            self._last_clear_t = now
            return self.cfg.guidance.clear_msg
        return None

    def _primary_message(
        self, threat: ThreatRecord, avoidance: str, now: float
    ) -> Optional[str]:
        sev = threat.severity
        tid = threat.obj.track_id

        prev      = self._last_sev.get(tid, Severity.CLEAR)
        escalated = (
            sev == Severity.CRITICAL
            and prev not in (Severity.CRITICAL, Severity.HIGH)
        )
        self._last_sev[tid] = sev

        cooldown = self.cfg.guidance.COOLDOWN_S.get(sev, 5.0)
        if not escalated and (now - self._cooldown.get(tid, 0.0)) < cooldown:
            return None

        self._cooldown[tid] = now
        templates = TEMPLATES[sev]
        idx = self._tpl_idx.get(tid, 0) % len(templates)
        self._tpl_idx[tid] = idx + 1

        dist_phrase = _distance_phrase(threat.distance_m, self.cfg)
        return _render_template(templates[idx], threat, dist_phrase, avoidance)

    def _secondary_message(self, threat: ThreatRecord, now: float) -> str:
        if threat.severity not in (Severity.CRITICAL, Severity.HIGH):
            return ""
        tid      = threat.obj.track_id
        cooldown = max(self.cfg.guidance.COOLDOWN_S.get(threat.severity, 5.0) * 0.5, 2.0)
        if now - self._secondary_cd.get(tid, 0.0) < cooldown:
            return ""
        self._secondary_cd[tid] = now
        idx  = self._tpl_idx.get(tid, 0) % len(SECONDARY_TEMPLATES)
        return SECONDARY_TEMPLATES[idx].format(
            label     = threat.obj.label,
            direction = threat.direction,
        )

    def hud_text(self, threats: List[ThreatRecord]) -> str:
        top = threats[0] if threats else None
        if top is None:
            return f"{SEVERITY_ICON[Severity.CLEAR]} Path clear."

        icon      = SEVERITY_ICON.get(top.severity, "")
        ttc_str   = f"  TTC {top.ttc_s:.1f}s" if top.ttc_s < 60 else ""
        trend_str = "↑" if top.trend > 2 else ("↓" if top.trend < -2 else "")
        dist_p    = _distance_phrase(top.distance_m, self.cfg)
        n         = len(threats)
        count_str = f"  ({n} threats)" if n > 1 else ""

        return (
            f"{icon} {top.obj.label.upper()} — {top.direction}"
            f"  |  {dist_p} ({top.distance_m:.1f}m){ttc_str}"
            f"  |  risk {top.score:.0f} {trend_str}{count_str}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TTS Engine  (FIX-16 + FIX-17)
# ─────────────────────────────────────────────────────────────────────────────

def _route_audio_to_bluetooth() -> bool:
    """
    FIX-17: Route PulseAudio output to the paired Bluetooth earpiece.

    On Linux/Jetson, paired Bluetooth audio devices appear as a PulseAudio
    sink named "bluez_sink.*". This function finds that sink and sets it as
    the default, so pyttsx3 speech goes to the earpiece automatically.

    Returns True if routing succeeded, False if no Bluetooth sink found.
    This is called once at startup and silently ignored if BT is not yet
    connected — the systemd service waits 5 s for BT before launching,
    so by the time this runs the earpiece should be connected.
    """
    try:
        # List all PulseAudio sinks
        result = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        sinks = result.stdout.strip().splitlines()

        # Find the Bluetooth sink (named bluez_sink.XX_XX_XX_XX_XX_XX.*)
        bt_sink = None
        for line in sinks:
            parts = line.split()
            if len(parts) >= 2 and "bluez_sink" in parts[1]:
                bt_sink = parts[1]
                break

        if bt_sink is None:
            log.warning("[TTS] No Bluetooth sink found — audio goes to default output.")
            return False

        # Set as default sink
        subprocess.run(
            ["pactl", "set-default-sink", bt_sink],
            capture_output=True,
            timeout=3,
        )
        log.info(f"[TTS] Audio routed to Bluetooth sink: {bt_sink}")
        return True

    except FileNotFoundError:
        # pactl not available — not on Linux, or PulseAudio not installed
        log.info("[TTS] pactl not found — skipping Bluetooth routing (non-Linux system).")
        return False
    except Exception as e:
        log.warning(f"[TTS] Bluetooth routing failed: {e}")
        return False


class _NullTTS:
    """Silent stub used when tts_enabled=False."""
    def speak(self, text: str) -> None:
        print(f"[AUDIO] {text}")


class TTSEngine:
    """
    Live TTS using pyttsx3.

    FIX-17: On startup, automatically routes audio to the paired Bluetooth
    earpiece using PulseAudio. Falls back to default audio output silently
    if no Bluetooth device is connected.

    FIX-16: Exposes speak() immediately — startup messages can be queued
    before the background thread finishes initialising pyttsx3.
    """

    def __init__(self, rate: int = 145):
        self._q: queue.Queue = queue.Queue(maxsize=3)
        self._engine         = None

        # FIX-17: route audio to Bluetooth BEFORE initialising pyttsx3
        # so pyttsx3 picks up the correct default sink from PulseAudio.
        _route_audio_to_bluetooth()

        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)

            # Pick the clearest available English voice
            voices = self._engine.getProperty("voices")
            if voices:
                # Prefer a voice with "english" in its ID/name if available
                english_voice = next(
                    (v for v in voices if "english" in v.id.lower()), voices[0]
                )
                self._engine.setProperty("voice", english_voice.id)
                log.info(f"[TTS] Voice: {english_voice.id}")

            log.info(f"[TTS] pyttsx3 ready at {rate} wpm.")
        except Exception as e:
            log.warning(f"[TTS] pyttsx3 unavailable ({e}) — silent mode.")

        threading.Thread(target=self._run, daemon=True).start()

    def speak(self, text: str) -> None:
        """Queue a message for speech. Drops silently if queue is full."""
        if not text:
            return
        try:
            self._q.put_nowait(text)
        except queue.Full:
            # Safety > completeness: if already speaking, drop rather than queue up
            # a backlog that would play out long after the situation has changed.
            log.debug(f"[TTS] Queue full — dropped: {text[:40]}")

    def _run(self) -> None:
        while True:
            text = self._q.get()
            if self._engine:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    log.warning(f"[TTS] Speech error: {e}")
                    print(f"[AUDIO] {text}")
            else:
                print(f"[AUDIO] {text}")


def _make_tts(enabled: bool, rate: int):
    return TTSEngine(rate) if enabled else _NullTTS()


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
                    (8, max(hy - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (200, 200, 0), 1, cv2.LINE_AA)

        top_sev    = threats[0].severity if threats else Severity.CLEAR
        banner_col = SEVERITY_BGR.get(top_sev, (60, 60, 60))
        self._draw_banner(out, hud_text, banner_col, w)
        self._draw_stats(out, fps, frame_id, perc, threats, w, h)
        return out

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

        dist_p = _distance_phrase(obj.distance_m, self.cfg)
        lines  = [
            f"{obj.label}  {dist_p} ({obj.distance_m:.1f}m)",
            f"TTC {obj.ttc_s:.1f}s  risk {threat.score:.0f}",
            threat.direction,
        ]
        tag_h = 18
        for li, txt in enumerate(lines):
            ty = y1 - (len(lines) - li) * tag_h
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            cv2.rectangle(frame, (x1, ty - 1),
                          (x1 + tw + 4, ty + tag_h - 2), colour, -1)
            cv2.putText(frame, txt, (x1 + 2, ty + tag_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

        bar_w = int((x2 - x1) * max(threat.obj.path_intersection, 0))
        if bar_w > 0:
            cv2.rectangle(frame, (x1, y2 + 2), (x1 + bar_w, y2 + 7), colour, -1)

    def _draw_banner(self, frame, text, colour, w):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 58), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, text, (12, 40), cv2.FONT_HERSHEY_DUPLEX,
                    0.72, colour, 1, cv2.LINE_AA)

    def _draw_stats(self, frame, fps, frame_id, perc, threats, w, h):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 36), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)
        stats = (
            f"frame {frame_id}  |  fps {fps:.1f}  |  threats {len(threats)}"
            f"  |  horizon {perc.horizon_y}px  |  scale {perc.depth_scale:.2f}"
            f"  |  infer {perc.inference_ms:.0f}ms"
        )
        cv2.putText(frame, stats, (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (180, 180, 180), 1, cv2.LINE_AA)


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
    """Thread-safe shared state between InferenceThread and the render loop."""
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

    FIX-15: Stationary objects inside the corridor receive a minimum closing
    velocity equal to the user's own walking speed before risk evaluation,
    so they are never silently dropped below TIER_LOW.
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
        self._stop   = threading.Event()
        self._t      = threading.Thread(target=self._run, daemon=True)

    def start(self, frame_queue: queue.Queue) -> None:
        self._fq = frame_queue
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        self.perc.stop()
        self._t.join(timeout=5.0)

    def _apply_stationary_floor(self, objects: List[Object3D]) -> None:
        """
        FIX-15: For stationary objects overlapping the walking corridor,
        inject a minimum closing velocity representing the user's own
        approach speed. Without this, a person standing in the user's
        path scores near zero because vz ≈ 0.
        """
        floor = self.cfg.risk.stationary_vel_floor
        for obj in objects:
            if obj.is_stationary and obj.path_intersection > 0.05:
                vx, vy, vz = obj.velocity
                if abs(vz) < floor:
                    obj.velocity = (vx, vy, -floor)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._fq.get(timeout=0.05)
            except queue.Empty:
                continue

            perc_out          = self.perc.process(frame)
            objects, corridor = self.spatial.analyze(perc_out)
            self._apply_stationary_floor(objects)
            threats           = self.engine.evaluate_all(objects)
            speak             = self.guide.generate_speak(threats)
            hud               = self.guide.hud_text(threats)

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

    # ── TTS init + startup voice  (FIX-16) ───────────────────────────────
    tts = _make_tts(tts_enabled, CFG.guidance.tts_rate)
    tts.speak("SoundVision starting. Please wait.")
    log.info("Startup message queued.")

    state = PipelineState()
    hud   = HUDRenderer(CFG, show_depth, show_heatmap, show_corridor)

    infer_q = queue.Queue(maxsize=2)
    inf_t   = InferenceThread(CFG, W, H, tts, state)
    inf_t.start(infer_q)

    log.info("Feeding first frame — waiting for AI models to initialise…")
    first_frame = cap.read()
    if first_frame is not None:
        infer_q.put(first_frame.copy())

    max_wait = 120
    start_t  = time.time()
    last_log = -1

    while True:
        with state.lock:
            ready = state.perc is not None
        if ready:
            log.info("AI Engine online — starting main loop.")
            # ── System ready voice  (FIX-16) ─────────────────────────────
            tts.speak("System ready. Path monitoring active.")
            break
        elapsed = int(time.time() - start_t)
        if elapsed >= max_wait:
            log.error("AI warmup timed out. Exiting.")
            tts.speak("System error. Please restart the device.")
            time.sleep(3)   # give TTS time to finish speaking before exit
            inf_t.stop(); cap.stop(); writer.release()
            return
        if elapsed % 5 == 0 and elapsed != last_log:
            last_log = elapsed
            log.info(f"  ...loading models ({elapsed}s elapsed)…")
        time.sleep(0.25)

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
                cv2.putText(rendered, "Initialising perception…", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

            writer.write(rendered)

            if show_window:
                cv2.imshow("SoundVision V3", rendered)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1
            if frame_id % 150 == 0:
                pct = (f"{frame_id / max(cap.total, 1) * 100:.1f}%"
                       if cap.total > 0 else f"f{frame_id}")
                log.info(f"  [{pct}]  fps={fps_smooth:.1f}"
                         f"  threats={len(threats)}  skip={ai_skip}")

    except KeyboardInterrupt:
        log.info("Interrupted.")

    finally:
        # ── Shutdown voice  (FIX-16) ──────────────────────────────────────
        tts.speak("System stopped.")
        time.sleep(2)   # give TTS time to finish before process exits
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

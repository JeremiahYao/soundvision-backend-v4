"""
voice_control.py — SoundVision V3
===================================
Handles voice command input from the user.

How it works:
  1. User presses F1 (laptop) — a physical button will replace this on the Jetson.
  2. Device speaks "Listening." and plays a short beep.
  3. Whisper records 4 seconds of audio and transcribes it locally (no internet needed).
  4. The transcribed text is matched to a known command and executed.

Commands supported:
  "read this" / "what does this say"   → reads text in the camera frame
  "repeat" / "say again"               → repeats the last obstacle alert
  "help" / "what can you do"           → lists available commands
  "battery" / "how much battery"       → announces battery percentage
  "stop" / "cancel"                    → cancels the current operation
  "navigate to [place]"                → (ready for when navigation.py is built)

Laptop testing:
  Press F1 to trigger voice input.
  Make sure your microphone is working before running.

Jetson (hardware):
  Replace the F1 keyboard hook with a GPIO button interrupt.
  Everything else stays the same.

First run:
  Whisper downloads ~150 MB of model weights on first use ("tiny" model).
  Subsequent runs use cached models.

Dependencies:
  pip install openai-whisper sounddevice keyboard psutil
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Callable

import numpy as np

log = logging.getLogger("SV3.VoiceControl")

# ── Optional dependency guards ─────────────────────────────────────────────

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False
    log.warning("[Voice] sounddevice not installed. Run: pip install sounddevice")

try:
    import whisper as _whisper_lib
    _WHISPER_AVAILABLE = True
except ImportError:
    _WHISPER_AVAILABLE = False
    log.warning("[Voice] openai-whisper not installed. Run: pip install openai-whisper")

try:
    import keyboard as _keyboard_lib
    _KEYBOARD_AVAILABLE = True
except ImportError:
    _KEYBOARD_AVAILABLE = False
    log.warning("[Voice] keyboard not installed. Run: pip install keyboard")

try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import torch as _torch
    _CUDA = _torch.cuda.is_available()
except ImportError:
    _CUDA = False


# ── Command keyword table ─────────────────────────────────────────────────
#
# Maps a command name → list of phrases Whisper might transcribe for it.
# Matching is substring-based: if ANY phrase appears in the transcription,
# the command fires. Add more phrases if recognition is inconsistent.

_COMMAND_KEYWORDS: dict[str, list[str]] = {
    "read":     ["read", "what does this say", "what does it say",
                 "read this", "read the sign", "read text"],
    "repeat":   ["repeat", "say again", "what did you say", "again"],
    "help":     ["help", "what can you do", "commands", "what are the commands"],
    "battery":  ["battery", "battery level", "how much battery", "charge"],
    "stop":     ["stop", "cancel", "never mind", "forget it"],
    "navigate": ["navigate", "take me to", "go to", "directions to",
                 "how do i get to"],
}


class VoiceController:
    """
    Listens for F1 keypress, records a voice command, transcribes it with
    Whisper, and executes the matching command.

    Parameters
    ----------
    cfg         : Config object (uses cfg.voice.* settings)
    tts         : TTS engine with .speak(text) method
    text_reader : TextReader instance for "read this" command
    state       : PipelineState for accessing current_frame and last_alert
    on_navigate : Optional callback for navigation command: fn(destination: str)
    """

    def __init__(
        self,
        cfg,
        tts,
        text_reader,
        state,
        on_navigate: Optional[Callable[[str], None]] = None,
    ):
        self._cfg          = cfg
        self._tts          = tts
        self._text_reader  = text_reader
        self._state        = state
        self._on_navigate  = on_navigate
        self._model        = None
        self._model_ready  = False
        self._recording    = False
        self._stop_evt     = threading.Event()

    # ── Public ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the voice controller (non-blocking). Call once in run()."""
        if not (_SD_AVAILABLE and _WHISPER_AVAILABLE):
            log.warning("[Voice] Missing dependencies — voice control disabled.")
            self._tts.speak(
                "Voice control is unavailable. Install whisper and sounddevice."
            )
            return

        # Load Whisper model in background
        threading.Thread(target=self._load_whisper, daemon=True).start()

        # Register F1 hotkey
        if _KEYBOARD_AVAILABLE:
            try:
                _keyboard_lib.add_hotkey(
                    self._cfg.voice.trigger_key,
                    self._on_trigger,
                )
                log.info(
                    f"[Voice] Ready. Press {self._cfg.voice.trigger_key.upper()} "
                    f"to give a voice command."
                )
            except Exception as e:
                log.warning(f"[Voice] Could not register hotkey: {e}")
        else:
            log.warning("[Voice] keyboard library unavailable — hotkey disabled.")

    def stop(self) -> None:
        self._stop_evt.set()
        if _KEYBOARD_AVAILABLE:
            try:
                _keyboard_lib.remove_hotkey(self._cfg.voice.trigger_key)
            except Exception:
                pass

    # ── Hotkey callback (fires on F1 press) ───────────────────────────────

    def _on_trigger(self) -> None:
        """Called when the trigger key is pressed. Spawns recording thread."""
        if self._recording:
            return   # Ignore if already processing
        if not self._model_ready:
            self._tts.speak("Voice control is still loading. Please wait.")
            return
        threading.Thread(target=self._listen_and_execute, daemon=True).start()

    # ── Record → transcribe → execute ─────────────────────────────────────

    def _listen_and_execute(self) -> None:
        self._recording = True
        try:
            self._tts.speak("Listening.")
            time.sleep(0.5)   # Let TTS finish before recording starts

            audio = self._record_audio()
            if audio is None:
                self._tts.speak("Could not access microphone.")
                return

            self._tts.speak("Processing.")
            text = self._transcribe(audio)
            if not text:
                self._tts.speak("Could not understand. Please try again.")
                return

            log.info(f"[Voice] Heard: '{text}'")
            self._execute(text)

        except Exception as e:
            log.error(f"[Voice] Error: {e}")
            self._tts.speak("Voice command failed. Please try again.")
        finally:
            self._recording = False

    def _record_audio(self) -> Optional[np.ndarray]:
        """Record audio for the configured duration. Returns float32 numpy array."""
        try:
            vc = self._cfg.voice
            audio = sd.rec(
                int(vc.record_seconds * vc.sample_rate),
                samplerate=vc.sample_rate,
                channels=1,
                dtype=np.float32,
            )
            sd.wait()
            return audio.flatten()
        except Exception as e:
            log.error(f"[Voice] Recording failed: {e}")
            return None

    def _transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using Whisper. Returns lowercase text."""
        try:
            result = self._model.transcribe(
                audio,
                language="en",
                fp16=_CUDA,                      # fp16 only on GPU
                condition_on_previous_text=False,
                temperature=0.0,                 # deterministic
            )
            return result["text"].lower().strip()
        except Exception as e:
            log.error(f"[Voice] Transcription failed: {e}")
            return ""

    # ── Command execution ─────────────────────────────────────────────────

    def _execute(self, text: str) -> None:
        """Match transcribed text to a command and run it."""
        cmd = self._match_command(text)

        if cmd == "read":
            self._cmd_read()
        elif cmd == "repeat":
            self._cmd_repeat()
        elif cmd == "help":
            self._cmd_help()
        elif cmd == "battery":
            self._cmd_battery()
        elif cmd == "stop":
            self._cmd_stop()
        elif cmd == "navigate":
            self._cmd_navigate(text)
        else:
            self._tts.speak(
                "Command not recognised. Say 'help' to hear available commands."
            )

    def _match_command(self, text: str) -> Optional[str]:
        """Return the command name whose keywords best match the text."""
        for cmd, keywords in _COMMAND_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return cmd
        return None

    # ── Individual commands ────────────────────────────────────────────────

    def _cmd_read(self) -> None:
        """Trigger OCR on the current camera frame."""
        frame = None
        with self._state.lock:
            if self._state.current_frame is not None:
                frame = self._state.current_frame.copy()

        if frame is None:
            self._tts.speak("No camera frame available yet.")
        else:
            self._text_reader.read_frame_async(frame)

    def _cmd_repeat(self) -> None:
        """Repeat the last obstacle alert."""
        with self._state.lock:
            last = self._state.last_alert

        if last:
            self._tts.speak(f"Repeating last alert: {last}")
        else:
            self._tts.speak("No recent alert to repeat.")

    def _cmd_help(self) -> None:
        self._tts.speak(
            "Available commands: "
            "say 'read this' to read text in view. "
            "Say 'repeat' to hear the last obstacle alert again. "
            "Say 'battery' for battery level. "
            "Say 'stop' to cancel. "
            "Say 'navigate to' followed by a place name for directions."
        )

    def _cmd_battery(self) -> None:
        msg = self._battery_status()
        self._tts.speak(msg)

    def _cmd_stop(self) -> None:
        self._tts.speak("Okay.")

    def _cmd_navigate(self, text: str) -> None:
        """Extract destination and call the navigation callback."""
        if self._on_navigate is None:
            self._tts.speak(
                "Navigation is not set up yet. "
                "It will be available in the next update."
            )
            return

        # Extract destination: everything after the trigger phrase
        destination = text
        for trigger in ["navigate to", "take me to", "go to",
                        "directions to", "how do i get to"]:
            if trigger in text:
                destination = text.split(trigger, 1)[-1].strip()
                break

        if not destination:
            self._tts.speak("Please say a destination after 'navigate to'.")
            return

        self._tts.speak(f"Finding route to {destination}.")
        self._on_navigate(destination)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _load_whisper(self) -> None:
        """Load Whisper model in background on startup."""
        try:
            log.info(
                f"[Voice] Loading Whisper '{self._cfg.voice.whisper_model}' "
                f"(first run downloads ~150 MB)…"
            )
            device = "cuda" if _CUDA else "cpu"
            self._model = _whisper_lib.load_model(
                self._cfg.voice.whisper_model,
                device=device,
            )
            self._model_ready = True
            log.info(f"[Voice] Whisper ready. Device={device}. "
                     f"Press {self._cfg.voice.trigger_key.upper()} to speak.")
        except Exception as e:
            log.error(f"[Voice] Failed to load Whisper: {e}")

    @staticmethod
    def _battery_status() -> str:
        """Return a human-readable battery status string."""
        if not _PSUTIL_AVAILABLE:
            return "Battery status unavailable. Install psutil to enable."
        try:
            bat = _psutil.sensors_battery()
            if bat is None:
                return "Battery status unavailable on this device."
            pct      = int(bat.percent)
            charging = "charging" if bat.power_plugged else "not charging"
            return f"Battery at {pct} percent, {charging}."
        except Exception:
            return "Battery status unavailable."

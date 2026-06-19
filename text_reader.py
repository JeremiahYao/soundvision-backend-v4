"""
text_reader.py — SoundVision V3
================================
Reads visible text from the current camera frame using EasyOCR.

Triggered by voice command ("read this") or F1 key press.

Behaviour:
  - Runs OCR in a background thread so obstacle detection is never blocked.
  - Announces: "It says: Clementi MRT Station, Exit A"
  - If nothing readable found: "No text found in view."
  - If already busy: "Still reading, please wait."

Language support:
  English + Simplified Chinese — covers virtually all Singapore public signage.

First run:
  EasyOCR downloads ~200 MB of model weights on first use.
  Subsequent runs use the cached models.

GPU:
  Automatically uses CUDA GPU if available (Jetson), otherwise CPU (laptop).
  CPU is slower (~5–10 s per frame) but works fine for a triggered command.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("SV3.TextReader")

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False
    log.warning(
        "[TextReader] EasyOCR not installed — text reading disabled.\n"
        "  Fix: pip install easyocr"
    )

try:
    import torch
    _CUDA = torch.cuda.is_available()
except ImportError:
    _CUDA = False


class TextReader:
    """
    OCR wrapper that reads text from a camera frame and speaks it.

    Usage:
        reader = TextReader(cfg, tts)
        reader.read_frame_async(frame)   # non-blocking, speaks result
    """

    def __init__(self, cfg, tts):
        self._cfg   = cfg
        self._tts   = tts
        self._ocr   = None          # lazy-loaded — avoids 10 s startup delay
        self._busy  = threading.Event()
        self._lock  = threading.Lock()
        self._ready = False

        if _EASYOCR_AVAILABLE:
            # Load models in background so device startup isn't delayed
            threading.Thread(target=self._load, daemon=True).start()
        else:
            log.warning("[TextReader] Disabled — install easyocr to enable.")

    # ── Public ────────────────────────────────────────────────────────────

    def read_frame_async(self, frame: np.ndarray) -> None:
        """
        Trigger OCR on the given frame. Non-blocking — runs in a thread.
        Speaks the result through TTS when done.
        """
        if not _EASYOCR_AVAILABLE:
            self._tts.speak("Text reading is not available. Please install EasyOCR.")
            return

        if self._busy.is_set():
            self._tts.speak("Still reading, please wait.")
            return

        if not self._ready:
            self._tts.speak("Text reader is still loading. Please try again in a moment.")
            return

        threading.Thread(
            target=self._run_ocr,
            args=(frame.copy(),),
            daemon=True,
        ).start()

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Internal ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load EasyOCR models (runs once, in background on startup)."""
        try:
            log.info("[TextReader] Loading EasyOCR models (first run downloads ~200 MB)…")
            tr_cfg = self._cfg.text_reader
            self._ocr   = easyocr.Reader(
                list(tr_cfg.languages),
                gpu=_CUDA,
                verbose=False,
            )
            self._ready = True
            log.info(f"[TextReader] Ready. GPU={_CUDA}, languages={tr_cfg.languages}")
        except Exception as e:
            log.error(f"[TextReader] Failed to load: {e}")

    def _run_ocr(self, frame: np.ndarray) -> None:
        """Run OCR synchronously in a worker thread."""
        self._busy.set()
        try:
            self._tts.speak("Reading.")

            tr_cfg  = self._cfg.text_reader
            results = self._ocr.readtext(frame)

            # Filter by confidence and minimum length
            texts = [
                text.strip()
                for (_, text, conf) in results
                if conf   >= tr_cfg.min_confidence
                and len(text.strip()) >= tr_cfg.min_text_length
            ]

            # Deduplicate while preserving order
            seen, unique = set(), []
            for t in texts:
                t_lower = t.lower()
                if t_lower not in seen:
                    seen.add(t_lower)
                    unique.append(t)

            if unique:
                combined = ". ".join(unique)
                log.info(f"[TextReader] Found: {combined}")
                self._tts.speak(f"It says: {combined}")
            else:
                self._tts.speak("No text found in view.")

        except Exception as e:
            log.error(f"[TextReader] OCR error: {e}")
            self._tts.speak("Could not read text.")
        finally:
            self._busy.clear()

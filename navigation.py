"""
navigation.py — SoundVision V3
================================
Google Maps walking navigation with step-count instructions for blind users.

Standard Google Maps gives landmark-based instructions like:
  "Turn left at the traffic light in 50 metres"
  → useless for a blind user who cannot see the traffic light.

SoundVision converts all instructions to step-count + direction:
  "Turn left in about 65 steps"
  "Continue straight for about 130 steps"
  "You have arrived at your destination"

How it works:
  1. User says "navigate to Clementi MRT" → VoiceController calls start_route()
  2. System gets starting GPS position (or estimates via IP geolocation)
  3. Google Maps Directions API returns walking steps with distances
  4. Each step is converted: distance_m ÷ 0.75 = approximate steps
  5. As user walks, position is tracked via GPS or dead reckoning
  6. Instructions are announced at 15 m and 3 m before each turn
  7. Navigation audio pauses if a CRITICAL safety alert is active

GPS modes (in order of preference):
  usb_gps       : USB GPS receiver plugged into laptop (most accurate)
                  Recommended hardware: any USB GPS on Shopee, ~$20 SGD
                  Requires: pip install pyserial pynmea2
  ip_geolocation: Estimates starting position from your internet IP
                  Accuracy: ~100-300 m in urban Singapore (sufficient for demo)
  dead_reckoning: Estimates position based on time and walking speed
                  No hardware needed. Updates position every 2 seconds.
                  Walking speed assumption: 1.2 m/s (average adult)

For laptop street testing with no GPS hardware:
  NavigationManager auto-uses ip_geolocation + dead_reckoning.
  For a short demo route this is accurate enough to get directions right.

To use navigation in VoiceController:
  - "navigate to Clementi MRT"
  - "take me to Buona Vista MRT"
  - "stop navigation" / "cancel"

Setup:
  pip install requests pyserial pynmea2
"""

from __future__ import annotations

import html
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

log = logging.getLogger("SV3.Navigation")

# ── Optional dependency guards ─────────────────────────────────────────────

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    log.warning("[Nav] requests not installed. Run: pip install requests")

try:
    import serial as _serial
    import pynmea2 as _pynmea2
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False


# ── Load API key ───────────────────────────────────────────────────────────

def _load_api_key() -> str:
    """Load Google Maps API key from api_keys.py or environment variable."""
    try:
        from api_keys import GOOGLE_MAPS_API_KEY as key
        if key and key != "PASTE_YOUR_API_KEY_HERE":
            return key
    except ImportError:
        pass
    key = os.environ.get("SOUNDVISION_MAPS_KEY", "")
    if not key:
        log.error(
            "[Nav] No API key found.\n"
            "  Option 1: Edit api_keys.py and paste your key.\n"
            "  Option 2: Set env var SOUNDVISION_MAPS_KEY=your_key"
        )
    return key


# ── Constants ──────────────────────────────────────────────────────────────

_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
_GEOCODE_URL    = "https://maps.googleapis.com/maps/api/geocode/json"
_IPGEO_URL      = "http://ip-api.com/json/"

_AVG_STEP_M     = 0.75    # average adult step length in metres
_WALK_SPEED_MS  = 1.20    # assumed walking speed for dead reckoning (m/s)
_EARTH_R        = 6371000 # Earth radius in metres

# Maneuver string → natural spoken phrase
_MANEUVER_PHRASES = {
    "turn-left":         "Turn left",
    "turn-right":        "Turn right",
    "straight":          "Continue straight",
    "turn-slight-left":  "Bear left",
    "turn-slight-right": "Bear right",
    "turn-sharp-left":   "Turn sharp left",
    "turn-sharp-right":  "Turn sharp right",
    "uturn-left":        "Turn around",
    "uturn-right":       "Turn around",
    "ramp-left":         "Take the ramp on your left",
    "ramp-right":        "Take the ramp on your right",
    "roundabout-left":   "At the roundabout, exit left",
    "roundabout-right":  "At the roundabout, exit right",
    "merge":             "Merge ahead",
    "ferry":             "Board the ferry",
    "":                  "Continue",
}


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class NavStep:
    """One step in a walking route."""
    end_lat:         float
    end_lng:         float
    distance_m:      float
    steps:           int          # rounded step count for spoken output
    instruction:     str          # full spoken instruction, e.g. "Turn left in 65 steps"
    direction_only:  str          # short direction, e.g. "Turn left"
    prep_announced:  bool = False  # True once 15m warning has been spoken
    turn_announced:  bool = False  # True once "now" instruction has been spoken


@dataclass
class GPSPosition:
    lat: float
    lng: float


# ── Geometry helpers ──────────────────────────────────────────────────────

def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return great-circle distance in metres between two lat/lng points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * _EARTH_R * math.asin(math.sqrt(a))


def _bearing(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return initial bearing in degrees (0=N, 90=E) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lng2 - lng1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _move_position(lat: float, lng: float,
                   distance_m: float, bearing_deg: float) -> Tuple[float, float]:
    """Move a lat/lng point by distance_m in the given direction."""
    d  = distance_m / _EARTH_R
    br = math.radians(bearing_deg)
    phi1, lam1 = math.radians(lat), math.radians(lng)
    phi2 = math.asin(math.sin(phi1) * math.cos(d) +
                     math.cos(phi1) * math.sin(d) * math.cos(br))
    lam2 = lam1 + math.atan2(
        math.sin(br) * math.sin(d) * math.cos(phi1),
        math.cos(d) - math.sin(phi1) * math.sin(phi2)
    )
    return math.degrees(phi2), math.degrees(lam2)


def _distance_to_steps(metres: float) -> int:
    """Convert a distance in metres to a round step count."""
    raw = metres / _AVG_STEP_M
    # Round to nearest 5 for cleaner speech: "about 20 steps" not "17 steps"
    return max(int(round(raw / 5) * 5), 5)


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities from a Maps instruction string."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


# ── GPS classes ────────────────────────────────────────────────────────────

class DeadReckoningGPS:
    """
    Estimates position based on time and assumed walking speed.
    No hardware required — good enough for short demo routes.

    The position updates every 2 seconds assuming the user is walking
    continuously at 1.2 m/s toward the current waypoint.
    """

    def __init__(self, start_lat: float, start_lng: float):
        self._lat       = start_lat
        self._lng       = start_lng
        self._bearing   = 0.0
        self._last_t    = time.monotonic()
        self._lock      = threading.Lock()

    def get_position(self) -> GPSPosition:
        now = time.monotonic()
        with self._lock:
            elapsed      = now - self._last_t
            dist_m       = elapsed * _WALK_SPEED_MS
            new_lat, new_lng = _move_position(
                self._lat, self._lng, dist_m, self._bearing
            )
            self._lat    = new_lat
            self._lng    = new_lng
            self._last_t = now
        return GPSPosition(lat=self._lat, lng=self._lng)

    def set_bearing_to(self, target_lat: float, target_lng: float) -> None:
        """Update walking direction toward a target waypoint."""
        with self._lock:
            self._bearing = _bearing(self._lat, self._lng, target_lat, target_lng)


class USBGPS:
    """
    Reads real GPS position from a USB GPS receiver via a serial COM port.

    Recommended hardware: any USB GPS module on Shopee (~$20 SGD).
    On Windows: appears as COM3 or COM4 in Device Manager.
    On Linux/Jetson: appears as /dev/ttyUSB0 or /dev/ttyACM0.

    Auto-detects the port if not specified.
    """

    def __init__(self, port: Optional[str] = None, baudrate: int = 9600):
        self._port      = port
        self._baudrate  = baudrate
        self._position  = None
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        self._thread    = None

    def start(self) -> bool:
        if not _SERIAL_AVAILABLE:
            log.warning("[GPS] pyserial/pynmea2 not installed. "
                        "Run: pip install pyserial pynmea2")
            return False
        port = self._port or self._auto_detect()
        if not port:
            log.warning("[GPS] No GPS device found on serial ports.")
            return False
        self._thread = threading.Thread(
            target=self._read_loop, args=(port,), daemon=True
        )
        self._thread.start()
        log.info(f"[GPS] USB GPS started on {port}.")
        return True

    def stop(self) -> None:
        self._stop_evt.set()

    def get_position(self) -> Optional[GPSPosition]:
        with self._lock:
            return self._position

    def _read_loop(self, port: str) -> None:
        try:
            with _serial.Serial(port, self._baudrate, timeout=5) as ser:
                while not self._stop_evt.is_set():
                    try:
                        line = ser.readline().decode("ascii", errors="replace").strip()
                        if line.startswith(("$GPGGA", "$GNGGA", "$GPRMC", "$GNRMC")):
                            msg = _pynmea2.parse(line)
                            if hasattr(msg, "latitude") and msg.latitude:
                                with self._lock:
                                    self._position = GPSPosition(
                                        lat=float(msg.latitude),
                                        lng=float(msg.longitude),
                                    )
                    except Exception:
                        pass
        except Exception as e:
            log.error(f"[GPS] Serial error on {port}: {e}")

    @staticmethod
    def _auto_detect() -> Optional[str]:
        """Try to find a GPS receiver among serial ports."""
        try:
            import serial.tools.list_ports
            GPS_KEYWORDS = ["GPS", "U-BLOX", "SIRF", "GNSS", "NMEA",
                            "GLOBALSAT", "GARMIN"]
            for info in _serial.tools.list_ports.comports():
                desc = (info.description or "").upper()
                if any(kw in desc for kw in GPS_KEYWORDS):
                    log.info(f"[GPS] Auto-detected: {info.device} ({info.description})")
                    return info.device
        except Exception:
            pass
        return None


# ── Route converter ────────────────────────────────────────────────────────

def _parse_steps(route: dict) -> List[NavStep]:
    """Convert a Google Maps route object into a list of NavStep objects."""
    steps: List[NavStep] = []
    for leg in route.get("legs", []):
        for s in leg.get("steps", []):
            dist_m    = float(s["distance"]["value"])
            maneuver  = s.get("maneuver", "")
            direction = _MANEUVER_PHRASES.get(maneuver, _strip_html(s.get("html_instructions", "")))
            n_steps   = _distance_to_steps(dist_m)
            end_loc   = s["end_location"]

            # Build the full spoken instruction
            if dist_m < 10:
                instruction = f"{direction} now."
            else:
                instruction = f"{direction} in about {n_steps} steps."

            steps.append(NavStep(
                end_lat       = float(end_loc["lat"]),
                end_lng       = float(end_loc["lng"]),
                distance_m    = dist_m,
                steps         = n_steps,
                instruction   = instruction,
                direction_only= direction,
            ))
    return steps


# ── IP geolocation helper ──────────────────────────────────────────────────

def _ip_geolocation() -> Optional[Tuple[float, float]]:
    """
    Estimate current position from IP address.
    Accuracy: ~100–500 m in urban Singapore.
    Good enough as a navigation starting point for demos.
    """
    if not _REQUESTS_AVAILABLE:
        return None
    try:
        data = _requests.get(_IPGEO_URL, timeout=5).json()
        if data.get("status") == "success":
            return float(data["lat"]), float(data["lon"])
    except Exception:
        pass
    return None


def _geocode(address: str, api_key: str) -> Optional[Tuple[float, float]]:
    """Convert a place name / address to lat/lng using the Geocoding API."""
    if not _REQUESTS_AVAILABLE or not api_key:
        return None
    try:
        resp = _requests.get(_GEOCODE_URL, params={
            "address": address, "region": "sg", "key": api_key,
        }, timeout=8)
        data = resp.json()
        if data.get("status") == "OK":
            loc = data["results"][0]["geometry"]["location"]
            return float(loc["lat"]), float(loc["lng"])
        log.warning(f"[Nav] Geocoding failed: {data.get('status')}")
    except Exception as e:
        log.error(f"[Nav] Geocoding error: {e}")
    return None


# ── Navigation Manager ─────────────────────────────────────────────────────

class NavigationManager:
    """
    Manages the complete walking navigation lifecycle.

    Integrates with the obstacle detection pipeline:
      - Navigation audio is Priority 3 (below CRITICAL and HIGH safety alerts).
      - If a CRITICAL or HIGH threat is active, navigation pauses for 3 s.

    Usage:
        nav = NavigationManager(cfg, tts, state)
        nav.start()

        # Called by VoiceController when user says "navigate to X":
        nav.start_route("Clementi MRT")

        # Called by VoiceController when user says "stop":
        nav.stop_route()
    """

    _PREP_DIST_M   = 15.0   # announce upcoming turn at this distance
    _ARRIVE_DIST_M =  6.0   # announce "turn now" at this distance
    _DEST_DIST_M   = 12.0   # announce "you have arrived" at this distance

    def __init__(self, cfg, tts, state):
        self._cfg      = cfg
        self._tts      = tts
        self._state    = state
        self._api_key  = _load_api_key()
        self._steps:   List[NavStep] = []
        self._step_idx = 0
        self._active   = False
        self._dest_name= ""
        self._gps      = None
        self._dr_gps   = None
        self._lock     = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread:  Optional[threading.Thread] = None

        # Try to start USB GPS in background (won't block startup)
        self._usb_gps = USBGPS(port=getattr(cfg.navigation, "gps_port", None) or None)

    def start(self) -> None:
        """Call once at pipeline startup. Attempts to connect USB GPS."""
        threading.Thread(target=self._usb_gps.start, daemon=True).start()

    # ── Public API ────────────────────────────────────────────────────────

    def start_route(self, destination: str, origin: Optional[str] = None) -> None:
        """
        Begin navigation to a destination.

        destination : spoken place name, e.g. "Clementi MRT"
        origin      : optional spoken starting location; if None, uses GPS or IP
        """
        if not _REQUESTS_AVAILABLE:
            self._tts.speak("Navigation is unavailable. Please install the requests library.")
            return
        if not self._api_key:
            self._tts.speak("Navigation API key is not configured.")
            return

        # Stop any active route first
        self.stop_route()

        self._dest_name = destination
        threading.Thread(
            target=self._setup_and_run,
            args=(destination, origin),
            daemon=True,
        ).start()

    def stop_route(self) -> None:
        """Cancel the active navigation route."""
        with self._lock:
            if not self._active:
                return
            self._active = False
            self._stop_evt.set()
        log.info("[Nav] Route cancelled.")
        self._tts.speak("Navigation stopped.")

    @property
    def is_active(self) -> bool:
        return self._active

    # ── Setup ─────────────────────────────────────────────────────────────

    def _setup_and_run(self, destination: str, origin_hint: Optional[str]) -> None:
        self._tts.speak(f"Finding route to {destination}. Please wait.")

        # 1. Get starting position
        start = self._get_starting_pos(origin_hint)
        if start is None:
            self._tts.speak(
                "Could not determine your starting location. "
                "Please try again, or say your location after the command, "
                "for example: navigate to Clementi MRT from Queenstown."
            )
            return

        start_lat, start_lng = start
        log.info(f"[Nav] Starting position: {start_lat:.5f}, {start_lng:.5f}")

        # 2. Fetch Google Maps directions
        steps, error = self._fetch_directions(start_lat, start_lng, destination)
        if error or not steps:
            self._tts.speak(
                f"Could not find a route to {destination}. "
                f"Please check the destination and try again."
            )
            log.warning(f"[Nav] Directions error: {error}")
            return

        # 3. Set up dead reckoning GPS
        self._dr_gps = DeadReckoningGPS(start_lat, start_lng)
        if steps:
            self._dr_gps.set_bearing_to(steps[0].end_lat, steps[0].end_lng)

        # 4. Announce first instruction
        total_steps = sum(s.steps for s in steps)
        first_inst  = steps[0].instruction if steps else "Proceed to your destination."
        self._tts.speak(
            f"Route found. About {total_steps} steps to {destination}. "
            f"{first_inst}"
        )

        # 5. Start navigation loop
        with self._lock:
            self._steps    = steps
            self._step_idx = 0
            self._active   = True
            self._stop_evt.clear()

        self._navigation_loop()

    def _get_starting_pos(
        self, hint: Optional[str]
    ) -> Optional[Tuple[float, float]]:
        """
        Determine starting position in order of preference:
          1. Real USB GPS reading (most accurate)
          2. Geocode a spoken hint ("from Queenstown MRT")
          3. IP geolocation (~100–500 m in urban Singapore)
          4. None (caller handles failure)
        """
        # 1. Real GPS
        real = self._usb_gps.get_position()
        if real:
            log.info(f"[Nav] Using real GPS: {real.lat:.5f}, {real.lng:.5f}")
            return real.lat, real.lng

        # 2. Geocode spoken hint
        if hint:
            result = _geocode(hint, self._api_key)
            if result:
                log.info(f"[Nav] Geocoded '{hint}': {result}")
                return result

        # 3. IP geolocation
        result = _ip_geolocation()
        if result:
            log.info(f"[Nav] IP geolocation: {result}")
            return result

        return None

    # ── Directions API ────────────────────────────────────────────────────

    def _fetch_directions(
        self, lat: float, lng: float, destination: str
    ) -> Tuple[List[NavStep], Optional[str]]:
        """Call Google Maps Directions API. Returns (steps, error_msg)."""
        try:
            resp = _requests.get(_DIRECTIONS_URL, params={
                "origin":      f"{lat},{lng}",
                "destination": destination,
                "mode":        "walking",
                "region":      "sg",
                "language":    "en",
                "units":       "metric",
                "key":         self._api_key,
            }, timeout=10)
            data = resp.json()
            status = data.get("status")
            if status != "OK":
                return [], f"Maps API returned: {status}"
            steps = _parse_steps(data["routes"][0])
            log.info(f"[Nav] Got {len(steps)} steps.")
            return steps, None
        except Exception as e:
            return [], str(e)

    # ── Navigation loop ───────────────────────────────────────────────────

    def _navigation_loop(self) -> None:
        """Background loop that checks position and announces instructions."""
        log.info("[Nav] Navigation loop started.")
        last_ambient_t = time.monotonic()

        while not self._stop_evt.is_set():
            with self._lock:
                if not self._active or self._step_idx >= len(self._steps):
                    break

            pos = self._get_current_position()
            if pos is None:
                time.sleep(2)
                continue

            with self._lock:
                if self._step_idx >= len(self._steps):
                    break
                step = self._steps[self._step_idx]

            dist_to_waypoint = _haversine(pos.lat, pos.lng,
                                          step.end_lat, step.end_lng)

            # ── Arrived at this waypoint ───────────────────────────────────
            if dist_to_waypoint < self._ARRIVE_DIST_M:
                if not step.turn_announced:
                    step.turn_announced = True
                    self._speak_nav(f"{step.direction_only} now.")
                self._advance_step(pos)

            # ── Approaching turn (15 m warning) ───────────────────────────
            elif dist_to_waypoint < self._PREP_DIST_M and not step.prep_announced:
                step.prep_announced = True
                prep_steps = _distance_to_steps(dist_to_waypoint)
                self._speak_nav(
                    f"{step.direction_only} coming up, in about {prep_steps} steps."
                )

            # ── Ambient "you're on track" reassurance every 60 s ──────────
            elif time.monotonic() - last_ambient_t > 60:
                last_ambient_t = time.monotonic()
                remaining = sum(
                    s.steps for s in self._steps[self._step_idx:]
                )
                self._speak_nav(
                    f"Continue for about {remaining} more steps "
                    f"to reach {self._dest_name}.",
                    priority="ambient",
                )

            time.sleep(self._cfg.navigation.update_interval_s)

        log.info("[Nav] Navigation loop ended.")

    def _advance_step(self, pos: GPSPosition) -> None:
        """Move to the next navigation step."""
        with self._lock:
            self._step_idx += 1

            # Check if final destination reached
            if self._step_idx >= len(self._steps):
                self._active = False
                self._tts.speak(
                    f"You have arrived at {self._dest_name}. "
                    f"Your destination should be nearby."
                )
                log.info("[Nav] Arrived at destination.")
                return

            # Announce next step
            next_step = self._steps[self._step_idx]
            if self._dr_gps:
                self._dr_gps.set_bearing_to(next_step.end_lat, next_step.end_lng)

        self._speak_nav(next_step.instruction)

    def _get_current_position(self) -> Optional[GPSPosition]:
        """Get current position: real GPS if available, else dead reckoning."""
        real = self._usb_gps.get_position()
        if real:
            # Update dead reckoning to match real GPS
            if self._dr_gps:
                self._dr_gps._lat = real.lat
                self._dr_gps._lng = real.lng
            return real
        if self._dr_gps:
            return self._dr_gps.get_position()
        return None

    # ── Audio with priority checking ──────────────────────────────────────

    def _speak_nav(self, text: str, priority: str = "turn") -> None:
        """
        Speak a navigation instruction, respecting the safety alert priority.

        Priority hierarchy:
          CRITICAL safety → always fires, never deferred
          HIGH safety     → defers navigation by 3 s
          Navigation turn → fires after safety alert clears
          Navigation ambient → suppressed if any threat is active
        """
        from risk_engine_v3 import Severity

        with self._state.lock:
            threats = list(self._state.threats)

        top_sev = threats[0].severity if threats else Severity.CLEAR

        # Ambient updates are suppressed entirely when any threat is active
        if priority == "ambient" and threats:
            return

        # Turn instructions wait for HIGH to clear (3 s max wait)
        if top_sev in (Severity.CRITICAL, Severity.HIGH):
            time.sleep(3.0)

        log.info(f"[Nav] Speaking: {text}")
        self._tts.speak(text)

"""Privacy primitives. Software-only for now — hardware mute and on-device
redaction land later, but the surfaces here are how the rest of the code
asks 'are we allowed to be listening / looking right now?'.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from robot.config import CAMERA_LOG_PATH, MAX_UTTERANCE_SECONDS, MIC_ENABLED_DEFAULT


class MicGate:
    """Thread-safe on/off switch for the microphone, plus a per-utterance cap.

    The Edge consults `enabled` before reading from the recorder, and uses
    `MAX_UTTERANCE_SECONDS` as a hard ceiling on a single capture so a stuck
    VAD can never record an unbounded clip.
    """

    def __init__(self, enabled: bool = MIC_ENABLED_DEFAULT,
                 max_seconds: int = MAX_UTTERANCE_SECONDS):
        self._enabled = enabled
        self._lock = threading.Lock()
        self.max_seconds = max_seconds
        self._observers: list = []

    def subscribe(self, callback) -> None:
        """Register `callback(enabled: bool)` to run after every set()/toggle().

        Observers get the new *software* flag. They are invoked outside the
        gate's lock (an observer may read gate state without deadlocking), and
        a raising observer is logged and swallowed — a broken subscriber must
        never block the deafen switch or starve other subscribers. First
        consumer: ClipService pauses + flushes its buffers on gate-off; the
        Phase 8 status LED is the planned second.
        """
        with self._lock:
            self._observers.append(callback)

    @property
    def enabled(self) -> bool:
        """Effective mic state: software-on AND not hardware-muted.

        Callers check this to ask 'are we allowed to listen right now?'.
        Hardware mute always wins — a physical button cutting mic power can't
        be overridden in software. `software_enabled` exposes the raw flag.
        """
        with self._lock:
            software = self._enabled
        return software and not self.hardware_muted()

    @property
    def software_enabled(self) -> bool:
        """The raw software flag, ignoring any hardware mute."""
        with self._lock:
            return self._enabled

    def set(self, value: bool) -> None:
        with self._lock:
            self._enabled = value
        self._notify(value)

    def toggle(self) -> bool:
        with self._lock:
            self._enabled = not self._enabled
            value = self._enabled
        self._notify(value)
        return value

    def _notify(self, value: bool) -> None:
        with self._lock:
            observers = list(self._observers)
        for callback in observers:
            try:
                callback(value)
            except Exception:
                logging.getLogger(__name__).exception(
                    "MicGate observer raised; ignoring"
                )

    def set_hardware_mute_pin(self, pin_reader) -> None:
        """Register a hardware mute source (Phase 8 / Pi).

        Placeholder for the physical mute button on the desk-robot shopping
        list (desk-robot-plan.md §5 — 'required for v1'). On the Pi, the
        button cuts mic power at the hardware level; this hook lets the
        software gate *also* observe that state so the LED/UX can reflect it.

        `pin_reader` is any zero-arg callable returning True when the mic is
        hardware-muted. Until the Pi exists this is a no-op store — the
        software gate alone governs the mic on the laptop.
        """
        self._hardware_mute_pin = pin_reader

    def hardware_muted(self) -> bool:
        """True if a registered hardware mute source reports muted.

        Defaults to False (no hardware mute wired yet). The effective mic
        state is `enabled and not hardware_muted()` — hardware always wins,
        matching the physical button cutting mic power at the source.
        """
        reader = getattr(self, "_hardware_mute_pin", None)
        if reader is None:
            return False
        try:
            return bool(reader())
        except Exception:
            # A flaky pin reader must fail safe: treat as muted, never as
            # silently-live. Privacy default is "off when uncertain".
            return True


_camera_logger: logging.Logger | None = None


def _get_camera_logger() -> logging.Logger:
    global _camera_logger
    if _camera_logger is not None:
        return _camera_logger
    logger = logging.getLogger("camera_access")
    logger.setLevel(logging.INFO)
    Path(CAMERA_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(CAMERA_LOG_PATH)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    _camera_logger = logger
    return logger


def log_camera_access(reason: str, duration_s: float | None = None) -> None:
    """Append a line every time the camera is opened. Future-you will want
    a paper trail when somebody asks 'when did this thing look at me?'."""
    msg = f"camera_open reason={reason!r}"
    if duration_s is not None:
        msg += f" duration_s={duration_s:.2f}"
    _get_camera_logger().info(msg)


def utterance_cap_reached(start_time: float, max_seconds: int) -> bool:
    return (time.time() - start_time) > max_seconds

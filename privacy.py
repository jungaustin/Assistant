"""Privacy primitives. Software-only for now — hardware mute and on-device
redaction land later, but the surfaces here are how the rest of the code
asks 'are we allowed to be listening / looking right now?'.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from config import CAMERA_LOG_PATH, MAX_UTTERANCE_SECONDS, MIC_ENABLED_DEFAULT


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

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set(self, value: bool) -> None:
        with self._lock:
            self._enabled = value

    def toggle(self) -> bool:
        with self._lock:
            self._enabled = not self._enabled
            return self._enabled


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

"""Latency probe. Three marks per turn: stt_finish, brain_first_token,
tts_first_audio. Prints deltas the moment first audio fires.
"""

from __future__ import annotations

import time
from typing import Optional


class _Probe:
    def __init__(self) -> None:
        self.stt_finish: Optional[float] = None
        self.brain_first_token: Optional[float] = None
        self.tts_first_audio: Optional[float] = None

    def reset(self) -> None:
        self.stt_finish = None
        self.brain_first_token = None
        self.tts_first_audio = None

    def mark_stt_finish(self) -> None:
        self.reset()
        self.stt_finish = time.perf_counter()

    def mark_brain_first_token(self) -> None:
        if self.brain_first_token is None:
            self.brain_first_token = time.perf_counter()

    def mark_tts_first_audio(self) -> None:
        if self.tts_first_audio is not None:
            return
        self.tts_first_audio = time.perf_counter()
        self._emit()

    def _emit(self) -> None:
        if self.stt_finish is None or self.tts_first_audio is None:
            return
        stt_to_token = (
            (self.brain_first_token - self.stt_finish) * 1000
            if self.brain_first_token is not None
            else None
        )
        token_to_audio = (
            (self.tts_first_audio - self.brain_first_token) * 1000
            if self.brain_first_token is not None
            else None
        )
        total = (self.tts_first_audio - self.stt_finish) * 1000
        parts = [f"latency: total={total:.0f}ms"]
        if stt_to_token is not None:
            parts.append(f"stt→token={stt_to_token:.0f}ms")
        if token_to_audio is not None:
            parts.append(f"token→audio={token_to_audio:.0f}ms")
        print(" ".join(parts))


probe = _Probe()

"""Piper TTS engine. Local, fast, no network hop.

Loads an ONNX voice from disk and synthesizes audio chunks. Plays via
sounddevice. Accepts a string or a token-stream iterator; with an iterator,
audio is generated/played per chunk so first-audio latency is bounded by
the first chunk, not the full response.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Union

import numpy as np
import sounddevice as sd
from piper import PiperVoice

from latency import probe


TextLike = Union[str, Iterable[str], Iterator[str]]


class PiperEngine:
    def __init__(self, voice_path: str):
        path = Path(voice_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Piper voice not found at {voice_path!r}. "
                f"Set PIPER_VOICE_PATH in .env to a valid .onnx file."
            )
        self.voice = PiperVoice.load(str(path))
        self.sample_rate = self.voice.config.sample_rate

    def speak(self, text_or_generator: TextLike) -> None:
        """Synthesize and play audio. With a generator, plays per chunk."""
        if isinstance(text_or_generator, str):
            self._speak_text(text_or_generator)
            return
        buffer = ""
        for token in text_or_generator:
            if not token:
                continue
            buffer += token
            if any(buffer.endswith(p) for p in (". ", "! ", "? ", "\n")):
                self._speak_text(buffer.strip())
                buffer = ""
        if buffer.strip():
            self._speak_text(buffer.strip())

    def _speak_text(self, text: str) -> None:
        if not text:
            return
        chunks = []
        for audio_chunk in self.voice.synthesize(text):
            samples = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
            chunks.append(samples)
        if not chunks:
            return
        audio = np.concatenate(chunks)
        probe.mark_tts_first_audio()
        sd.play(audio, self.sample_rate)
        sd.wait()

    def shutdown(self) -> None:
        pass

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

from robot.latency import probe


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
        """Synthesize and play audio.

        With a generator, each item is treated as ALREADY a phrase-sized
        chunk and synthesized immediately. The sentence chunker upstream
        (robot.core.chunker) is the single source of phrasing truth; this
        engine is a dumb synth-and-play sink. Earlier versions buffered
        again here on `. ! ? \\n`, which silently discarded the chunker's
        comma boundaries and force-flushes — wasting the latency win.
        """
        if isinstance(text_or_generator, str):
            self._speak_text(text_or_generator)
            return
        for chunk in text_or_generator:
            if chunk and chunk.strip():
                self._speak_text(chunk.strip())

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

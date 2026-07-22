"""Piper TTS engine. Local, fast, no network hop.

Loads an ONNX voice from disk and synthesizes audio chunks. Plays via
sounddevice. Accepts a string or a token-stream iterator; with an iterator,
audio is generated/played per chunk so first-audio latency is bounded by
the first chunk, not the full response.
"""

from __future__ import annotations

import threading
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
        # Playback state for barge-in (see stop()). _speaking is set for the
        # duration of a speak() call; _stop is raised by stop() to cut the
        # current utterance short. Cross-thread: stop() is called from the
        # hotkey thread while speak() runs on the Edge's TTS thread.
        self._stop = threading.Event()
        self._speaking = threading.Event()

    @property
    def is_speaking(self) -> bool:
        """True while a speak() call is in flight (audio playing or pending)."""
        return self._speaking.is_set()

    def speak(self, text_or_generator: TextLike) -> None:
        """Synthesize and play audio.

        With a generator, each item is treated as ALREADY a phrase-sized
        chunk and synthesized immediately. The sentence chunker upstream
        (robot.core.chunker) is the single source of phrasing truth; this
        engine is a dumb synth-and-play sink. Earlier versions buffered
        again here on `. ! ? \\n`, which silently discarded the chunker's
        comma boundaries and force-flushes — wasting the latency win.

        stop() can interrupt this mid-stream (barge-in). Once stopped we stop
        *playing* but keep pulling from the generator to the end: the upstream
        pump in main._speak_stream feeds a bounded queue, so a consumer that
        quit early would wedge the producer on a full queue.
        """
        self._stop.clear()
        self._speaking.set()
        try:
            if isinstance(text_or_generator, str):
                self._speak_text(text_or_generator)
                return
            for chunk in text_or_generator:
                if self._stop.is_set():
                    continue  # barged in: drain the generator, play nothing
                if chunk and chunk.strip():
                    self._speak_text(chunk.strip())
        finally:
            self._speaking.clear()

    def _speak_text(self, text: str) -> None:
        if not text or self._stop.is_set():
            return
        chunks = []
        for audio_chunk in self.voice.synthesize(text):
            samples = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
            chunks.append(samples)
        if not chunks or self._stop.is_set():
            return
        audio = np.concatenate(chunks)
        probe.mark_tts_first_audio()
        sd.play(audio, self.sample_rate)
        # Close the stop/play race: if stop() landed between the check above
        # and sd.play(), its sd.stop() hit a stream that hadn't started yet and
        # this chunk would play to the end. The flag is set before stop() calls
        # sd.stop(), so re-checking here catches that ordering; a stop() any
        # later unblocks the wait below directly.
        if self._stop.is_set():
            sd.stop()
        # sd.stop() from stop() unblocks this wait early, ending the utterance.
        sd.wait()

    def stop(self) -> None:
        """Interrupt playback now (barge-in). Safe to call when idle.

        Raises the stop flag so speak()'s loop stops synthesizing further
        chunks, and stops the sounddevice stream so the in-flight chunk's
        sd.wait() returns immediately instead of playing to the end.
        """
        self._stop.set()
        try:
            sd.stop()
        except Exception:
            pass

    def shutdown(self) -> None:
        pass

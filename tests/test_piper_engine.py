"""Tests for PiperEngine's speak() contract.

These tests don't load the ONNX model. They verify the speak() flow:
each chunk from the iterator gets synthesized as-is, with no internal
buffering. The chunker upstream is the only source of phrasing.
"""

from __future__ import annotations

import threading
from typing import List

import pytest

from robot.voice.engines.piper_engine import PiperEngine


class _FakeEngine(PiperEngine):
    """PiperEngine with __init__ skipped (no ONNX load) and _speak_text
    replaced with a recorder. Drives the same speak() method under test."""

    def __init__(self) -> None:
        # Skip super().__init__: we never call _speak_text's body. But speak()
        # and stop() use the playback flags, so wire them up as the real
        # __init__ would.
        self.spoken: List[str] = []
        self._stop = threading.Event()
        self._speaking = threading.Event()

    def _speak_text(self, text: str) -> None:
        self.spoken.append(text)


# ---------- String input ----------


def test_speak_str_synthesizes_once():
    eng = _FakeEngine()
    eng.speak("Hello world.")
    assert eng.spoken == ["Hello world."]


# ---------- Iterator input ----------


def test_each_chunk_synthesized_immediately():
    """The whole point of the change: comma-terminated phrases must
    synthesize immediately, not get buffered until the next period."""
    eng = _FakeEngine()
    eng.speak(["Sure, ", "I can help. "])
    # OLD behavior was to buffer "Sure, " and only fire on "Sure, I can help."
    # NEW behavior: two synth calls, in order.
    assert eng.spoken == ["Sure,", "I can help."]


def test_chunks_with_question_mark_synthesize_per_chunk():
    eng = _FakeEngine()
    eng.speak(["How are you? ", "I'm fine. ", "Thanks."])
    assert eng.spoken == ["How are you?", "I'm fine.", "Thanks."]


def test_force_flushed_chunks_with_no_terminal_punct_synthesize():
    """When the chunker force-flushes at 80 chars (no boundary), the
    engine must still synthesize the chunk."""
    eng = _FakeEngine()
    eng.speak(["A long run with no punctuation here ", "and more here. "])
    assert eng.spoken == [
        "A long run with no punctuation here",
        "and more here.",
    ]


def test_empty_and_whitespace_only_chunks_skipped():
    eng = _FakeEngine()
    eng.speak(["", "   ", "Hi.", "\n", "  "])
    assert eng.spoken == ["Hi."]


def test_generator_input_is_consumed_lazily():
    """Pass an actual generator (not a list) to make sure iteration works
    and the engine doesn't try to convert it to a list first."""
    eng = _FakeEngine()

    def gen():
        yield "First. "
        yield "Second."

    eng.speak(gen())
    assert eng.spoken == ["First.", "Second."]


def test_empty_iterator_yields_no_synth_calls():
    eng = _FakeEngine()
    eng.speak(iter([]))
    assert eng.spoken == []


# ---------- Barge-in / stop ----------


def test_not_speaking_when_idle():
    assert _FakeEngine().is_speaking is False


def test_is_speaking_true_during_speak():
    eng = _FakeEngine()
    seen: List[bool] = []

    def record(text: str) -> None:
        seen.append(eng.is_speaking)

    eng._speak_text = record
    eng.speak(["Hi."])
    assert seen == [True]           # set while a chunk is being spoken
    assert eng.is_speaking is False  # cleared afterward


def test_stop_mid_stream_drains_without_speaking(monkeypatch):
    """After a barge-in mid-utterance, no further chunks play, but the whole
    generator is still consumed so the upstream bounded queue can't wedge."""
    import robot.voice.engines.piper_engine as mod

    monkeypatch.setattr(mod.sd, "stop", lambda: None)  # no audio device in tests

    eng = _FakeEngine()
    consumed: List[str] = []

    def gen():
        for c in ["One. ", "Two. ", "Three. "]:
            consumed.append(c)
            yield c

    # Barge in while the first chunk is being spoken.
    orig = eng._speak_text

    def speak_then_stop(text: str) -> None:
        orig(text)
        eng.stop()

    eng._speak_text = speak_then_stop
    eng.speak(gen())

    assert eng.spoken == ["One."]                       # only the first played
    assert consumed == ["One. ", "Two. ", "Three. "]    # generator fully drained
    assert eng.is_speaking is False


def test_stop_before_speak_is_reset_on_next_utterance(monkeypatch):
    """A stale stop from a previous turn must not suppress the next speak()."""
    import robot.voice.engines.piper_engine as mod

    monkeypatch.setattr(mod.sd, "stop", lambda: None)

    eng = _FakeEngine()
    eng.stop()               # left the flag set
    eng.speak(["Fresh."])    # speak() should clear it and play normally
    assert eng.spoken == ["Fresh."]


def test_stop_when_idle_is_safe(monkeypatch):
    import robot.voice.engines.piper_engine as mod

    monkeypatch.setattr(mod.sd, "stop", lambda: None)
    _FakeEngine().stop()  # must not raise


def test_stop_racing_play_start_still_cuts_audio(monkeypatch):
    """stop() landing between the pre-play check and sd.play() must still cut
    the chunk: the engine re-checks the flag right after play and stops the
    just-started stream itself instead of letting the phrase play out."""
    import robot.voice.engines.piper_engine as mod

    calls: List[str] = []

    class _Chunk:
        audio_int16_bytes = b"\x00\x01" * 8

    class _Voice:
        def synthesize(self, text):
            return [_Chunk()]

    eng = _FakeEngine()
    eng.voice = _Voice()
    eng.sample_rate = 16000
    # Restore the real _speak_text (the fake replaced it with a recorder).
    eng._speak_text = lambda text: PiperEngine._speak_text(eng, text)

    def fake_play(audio, rate):
        calls.append("play")
        # Simulate the race: stop() fires while play is starting — the flag is
        # already set by the time play returns, but its sd.stop() hit nothing.
        eng._stop.set()

    monkeypatch.setattr(mod.sd, "play", fake_play)
    monkeypatch.setattr(mod.sd, "stop", lambda: calls.append("stop"))
    monkeypatch.setattr(mod.sd, "wait", lambda: calls.append("wait"))
    monkeypatch.setattr(mod.probe, "mark_tts_first_audio", lambda: None)

    eng._speak_text("Hello there.")
    # The post-play re-check must have stopped the stream before waiting.
    assert calls == ["play", "stop", "wait"]

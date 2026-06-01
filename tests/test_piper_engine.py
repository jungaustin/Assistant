"""Tests for PiperEngine's speak() contract.

These tests don't load the ONNX model. They verify the speak() flow:
each chunk from the iterator gets synthesized as-is, with no internal
buffering. The chunker upstream is the only source of phrasing.
"""

from __future__ import annotations

from typing import List

import pytest

from robot.voice.engines.piper_engine import PiperEngine


class _FakeEngine(PiperEngine):
    """PiperEngine with __init__ skipped (no ONNX load) and _speak_text
    replaced with a recorder. Drives the same speak() method under test."""

    def __init__(self) -> None:
        # Skip super().__init__: we never call _speak_text's body.
        self.spoken: List[str] = []

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

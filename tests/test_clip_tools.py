"""Tests for the clip-that wiring (eng plan T5): the save_clip tool, the
keyword fast-path regex, and the Edge's background-save behavior."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from robot.core.clip import ClipError, is_clip_command
from robot.tools.inner.clip_tools import ClipTools


def wait_until(cond, timeout: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.005)
    return False


# --- fast-path regex (decision 9A) ----------------------------------------
# The false-positive cases here are the eng review's mandated enumeration:
# a mid-sentence "clip that" must NOT trigger a save.

FAST_PATH_HITS = [
    "clip that",
    "Clip that.",
    "clip this",
    "clip it!",
    "save that clip",
    "save the clip",
    "save this clip",
    "nemo, clip that",
    "Nemo clip that",
    "hey nemo, clip that",
    "ok nemo clip it",
    "please clip that",
    "nemo, please save that clip",
    "clip that, please",
    "clip that please.",
    "  clip that  ",
]

FAST_PATH_MISSES = [
    "the clip that fell off",
    "I watched a clip that was funny",
    "can you clip that for me",
    "save that clip for later",
    "clip that video from yesterday",
    "she said clip that during the stream",
    "paperclip that",
    "eclipse that",
    "clip",
    "that clip",
    "save the clipboard",
    "record the last minute",  # paraphrase — belongs to the save_clip tool
    "",
]


@pytest.mark.parametrize("utterance", FAST_PATH_HITS)
def test_fast_path_matches(utterance):
    assert is_clip_command(utterance) is True


@pytest.mark.parametrize("utterance", FAST_PATH_MISSES)
def test_fast_path_rejects(utterance):
    assert is_clip_command(utterance) is False


# --- save_clip tool --------------------------------------------------------


class FakeClipService:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def save(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def test_save_clip_success_names_the_file():
    tools = ClipTools(FakeClipService(result=Path("/x/clip-20260719-101500.mp4")))
    out = tools.save_clip()
    assert "clip-20260719-101500.mp4" in out
    assert out.startswith("Clip saved")


def test_save_clip_relays_the_spoken_failure():
    tools = ClipTools(FakeClipService(error=ClipError("My mic gate was off.")))
    assert tools.save_clip() == "My mic gate was off."


def test_save_clip_unexpected_error_returns_generic_spoken_string():
    tools = ClipTools(FakeClipService(error=RuntimeError("boom")))
    out = tools.save_clip()
    assert out == "Something went wrong saving the clip."


def test_save_clip_without_service_is_unavailable():
    tools = ClipTools(None)
    assert tools.available is False
    assert tools.save_clip() == "Clipping isn't available right now."


def test_create_save_clip_tool_shape():
    tools = ClipTools(FakeClipService(result=Path("/x/c.mp4")))
    tool = tools.create_save_clip_tool()
    assert tool.name == "save_clip"
    assert "last" in tool.description.lower()
    assert "Clip saved" in tool.func()


# --- Edge fast-path save (background thread) -------------------------------


class FakeTTS:
    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text):
        self.spoken.append(text)


class FakeSTT:
    def listen(self):  # pragma: no cover — never driven in these tests
        return ""

    def shutdown(self):
        pass


def make_edge(clip_service):
    from robot.main import Edge

    return Edge(
        transport=object(),
        speech_to_text=FakeSTT(),
        text_to_speech=FakeTTS(),
        clip_service=clip_service,
    )


def test_start_clip_save_success_is_silent(capsys):
    service = FakeClipService(result=Path("/x/clip.mp4"))
    edge = make_edge(service)
    edge._start_clip_save()
    assert wait_until(lambda: service.calls == 1)
    assert wait_until(lambda: "clip saved" in capsys.readouterr().out)
    assert edge.text_to_speech.spoken == []  # the ack already covered it


def test_start_clip_save_failure_is_spoken(capsys):
    service = FakeClipService(error=ClipError("I couldn't put the clip file together."))
    edge = make_edge(service)
    edge._start_clip_save()
    assert wait_until(
        lambda: edge.text_to_speech.spoken == ["I couldn't put the clip file together."]
    )


def test_start_clip_save_unexpected_error_stays_quiet():
    service = FakeClipService(error=RuntimeError("boom"))
    edge = make_edge(service)
    edge._start_clip_save()
    assert wait_until(lambda: service.calls == 1)
    time.sleep(0.05)
    assert edge.text_to_speech.spoken == []  # logged, not spoken

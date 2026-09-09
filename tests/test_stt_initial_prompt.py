"""Whisper must be biased toward this robot's logging vocabulary.

Measured 2026-08-25: "Can you long for me 240 calories for tofu." (what
Whisper produced for "log") got 0/5 real tool calls, while the same sentence
with "log" got 3/5. The brain cannot reliably repair a verb that arrived
wrong, so the fix belongs at the transcription step.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _build_recorder(monkeypatch, **env):
    """Call make_stt_recorder with AudioToTextRecorder mocked out."""
    fake_cls = MagicMock(return_value="recorder")
    fake_module = SimpleNamespace(AudioToTextRecorder=fake_cls)
    monkeypatch.setitem(sys.modules, "RealtimeSTT", fake_module)

    import importlib

    import robot.config as config

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    config = importlib.reload(config)
    with patch.object(config, "resolve_input_device_index", return_value=None):
        config.make_stt_recorder()
    return fake_cls.call_args.kwargs, config


def test_initial_prompt_is_passed_to_whisper(monkeypatch):
    kwargs, config = _build_recorder(monkeypatch)
    assert kwargs["initial_prompt"] == config.STT_INITIAL_PROMPT
    assert kwargs["initial_prompt_realtime"] == config.STT_INITIAL_PROMPT


def test_default_prompt_teaches_the_word_log(monkeypatch):
    _, config = _build_recorder(monkeypatch)
    prompt = config.STT_INITIAL_PROMPT.lower()
    assert "log" in prompt
    assert "calories" in prompt
    # Short on purpose: a long initial_prompt makes Whisper echo it back on
    # near-silent audio.
    assert len(config.STT_INITIAL_PROMPT) < 300


def test_empty_env_value_disables_the_bias(monkeypatch):
    kwargs, _ = _build_recorder(monkeypatch, STT_INITIAL_PROMPT="")
    assert kwargs["initial_prompt"] is None
    assert kwargs["initial_prompt_realtime"] is None


def test_custom_prompt_is_honoured(monkeypatch):
    kwargs, _ = _build_recorder(monkeypatch, STT_INITIAL_PROMPT="Log my weight.")
    assert kwargs["initial_prompt"] == "Log my weight."

"""Tests for the follow-up echo/junk filter.

The follow-up window listens without a wake word, so it can capture (a) the
tail of Nemo's own TTS as a "user" turn and (b) ambient noise the recorder
transcribes as fragments. Both must be discarded, or Nemo answers itself and
spins a self-talk loop. Real commands — including short ones — must survive.
"""

from __future__ import annotations

from robot.main import _is_echo_or_junk


def test_exact_echo_is_discarded():
    spoken = "The last food log has been deleted."
    assert _is_echo_or_junk("The last food log has been deleted.", spoken)


def test_partial_echo_is_discarded():
    spoken = "The last food log has been deleted."
    assert _is_echo_or_junk("food log has been deleted", spoken)


def test_punctuation_noise_is_discarded():
    assert _is_echo_or_junk(". . .", "anything at all")


def test_bare_number_is_discarded():
    assert _is_echo_or_junk("3.", "anything at all")


def test_empty_is_discarded():
    assert _is_echo_or_junk("   ", "anything at all")


def test_real_command_is_kept():
    spoken = "The last food log has been deleted."
    assert not _is_echo_or_junk("log 500 calories for pork belly", spoken)


def test_short_command_not_treated_as_echo():
    # "stop" is a substring of the spoken reply, but it's under the 3-word
    # echo floor, so it must still be honored as a command.
    assert not _is_echo_or_junk("stop", "Okay, I will stop the music now.")


def test_short_lexical_utterance_kept_when_nothing_spoken():
    assert not _is_echo_or_junk("delete that", "")


def test_new_command_resembling_confirmation_is_kept():
    # The app's core pattern: a real command issued right after Nemo confirms a
    # near-identical one. It must NOT be mistaken for an echo and dropped.
    spoken = "I've logged 350 calories for rice."
    assert not _is_echo_or_junk("log 350 calories for rice", spoken)


def test_full_verbatim_echo_still_discarded():
    # A genuine echo of (essentially) the whole reply is still caught.
    spoken = "I've logged 350 calories for rice."
    assert _is_echo_or_junk("I've logged 350 calories for rice", spoken)

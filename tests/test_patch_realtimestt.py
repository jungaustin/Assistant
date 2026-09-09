"""Tests for the RealtimeSTT patch script.

Two fixes are applied to the vendored library: poll_connection's infinite loop
(a closed pipe must end the worker, not spin on EOFError) and abort()'s
unbounded was_interrupted.wait() (which hung forever, left the recording
running, and wedged every hotkey behind the controller's action lock). These
tests drive the pure patch_text() transform against a representative source
snippet — they do NOT touch the real installed file.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_realtimestt.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("patch_realtimestt", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


patch_mod = _load_module()


# A minimal but faithful copy of the upstream poll_connection, indented as a
# class method so the patched result can be parsed as real Python.
_SAMPLE = '''\
import logging
import time

TIME_SLEEP = 0.02


class TranscriptionWorker:
    def poll_connection(self):
        while not self.shutdown_event.is_set():
            try:
                # Use a longer timeout to reduce polling frequency
                if self.conn.poll(0.01):  # Increased from 0.01 to 0.5 seconds
                    data = self.conn.recv()
                    self.queue.put(data)
                else:
                    # Sleep only if no data, but use a shorter sleep
                    time.sleep(TIME_SLEEP)
            except Exception as e:
                logging.error(f"Error receiving data from connection: {e}", exc_info=True)
                time.sleep(TIME_SLEEP)


class AudioToTextRecorder:
    def abort(self):
        state = self.state
        self.start_recording_on_voice_activity = False
        self.stop_recording_on_voice_deactivity = False
        self.interrupt_stop_event.set()
        if self.state != "inactive": # if inactive, was_interrupted will never be set
            self.was_interrupted.wait()
            self._set_state("transcribing")
        self.was_interrupted.clear()
        if self.is_recording: # if recording, make sure to stop the recorder
            self.stop()
'''


def test_patches_the_except_block():
    new_src, status = patch_mod.patch_text(_SAMPLE)
    assert status.startswith("patched")
    assert patch_mod.MARKER in new_src
    # The pipe-closed family is now caught before the generic handler.
    assert "except (EOFError, BrokenPipeError, OSError):" in new_src
    assert "self.shutdown_event.set()" in new_src
    # Generic handler is preserved for other (transient) errors.
    assert 'logging.error(f"Error receiving data from connection' in new_src


def test_patched_source_is_valid_python():
    new_src, _ = patch_mod.patch_text(_SAMPLE)
    ast.parse(new_src)  # raises SyntaxError if indentation/structure is off


def test_pipe_closed_handler_precedes_generic():
    new_src, _ = patch_mod.patch_text(_SAMPLE)
    assert new_src.index("except (EOFError") < new_src.index("except Exception as e")


def test_idempotent():
    once, _ = patch_mod.patch_text(_SAMPLE)
    twice, status = patch_mod.patch_text(once)
    assert status == "already"
    assert twice == once


def test_raises_when_block_absent():
    with pytest.raises(patch_mod.PatchError):
        patch_mod.patch_text("print('no poll_connection here')\n")


def test_installed_library_is_patched():
    """The live venv should already carry the patch (applied via just sync)."""
    import RealtimeSTT.audio_recorder as m

    assert patch_mod.MARKER in Path(m.__file__).read_text()


# --- abort() must not be able to block forever -----------------------------


def test_abort_wait_is_given_a_timeout():
    """Unbounded, abort() never returned: the recording was never stopped (the
    self.stop() after the wait is unreachable) and HotkeyController held its
    action lock for good, so every Page Up / Page Down logged only
    'hotkey_ignored_busy'. Seen 2026-09-07."""
    new_src, _ = patch_mod.patch_text(_SAMPLE)
    assert "self.was_interrupted.wait()\n" not in new_src, "bare wait must be gone"
    assert "self.was_interrupted.wait(timeout=2.0)" in new_src


def test_abort_still_reaches_stop_after_a_timeout():
    new_src, _ = patch_mod.patch_text(_SAMPLE)
    abort_src = new_src[new_src.index("    def abort(self):"):]
    # Match the executable line, not the words "self.stop()" inside the patch's
    # own explanatory comment — which is what this assertion caught first.
    stop_line = "\n            self.stop()"
    assert stop_line in abort_src
    assert abort_src.index("wait(timeout=2.0)") < abort_src.index(stop_line)


def test_patched_abort_is_valid_python():
    new_src, _ = patch_mod.patch_text(_SAMPLE)
    ast.parse(new_src)


def test_both_patches_are_applied_and_named():
    _, status = patch_mod.patch_text(_SAMPLE)
    assert "poll_connection" in status and "abort" in status, status


def test_applying_twice_is_a_no_op():
    once, _ = patch_mod.patch_text(_SAMPLE)
    twice, status = patch_mod.patch_text(once)
    assert status == "already"
    assert twice == once


def test_a_missing_target_is_reported_not_silently_skipped():
    """If upstream changes the code we patch, fail loudly — a silently skipped
    patch means the hang comes back without anyone noticing."""
    with pytest.raises(patch_mod.PatchError):
        patch_mod.patch_text("class AudioToTextRecorder:\n    pass\n")

"""Tests for the RealtimeSTT poll_connection patch script.

The script rewrites the vendored library's infinite-loop except block so a
closed pipe ends the worker instead of spinning on EOFError. These tests drive
the pure patch_text() transform against a representative source snippet — they
do NOT touch the real installed file.
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
'''


def test_patches_the_except_block():
    new_src, status = patch_mod.patch_text(_SAMPLE)
    assert status == "patched"
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

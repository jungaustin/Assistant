"""Patch RealtimeSTT's poll_connection infinite-loop bug in the installed venv.

The bug
-------
RealtimeSTT runs its faster-whisper transcription in a *spawned* child process
(AudioToTextRecorder sets mp start method to "spawn"). Inside that child,
TranscriptionWorker.poll_connection reads the command pipe from the parent in a
loop bounded only by a shared shutdown_event:

    while not self.shutdown_event.is_set():
        try:
            if self.conn.poll(0.01):
                data = self.conn.recv()
                ...
        except Exception as e:
            logging.error("Error receiving data from connection: ...", exc_info=True)
            time.sleep(TIME_SLEEP)

When the parent's end of the pipe closes but shutdown_event is never set — a
parent crash, a `kill -9`, a power-off, or any teardown that doesn't cleanly
signal the child — conn.recv() raises EOFError every iteration. A closed pipe
never reopens, so the generic `except` logs the same traceback and sleeps,
forever. That's the runaway EOFError spam (and a child that never dies).

Why patch on disk instead of monkeypatching
--------------------------------------------
The loop runs in a *spawned* process, which re-imports RealtimeSTT.audio_recorder
from disk. A monkeypatch applied in the parent process is not inherited by the
child under spawn, so the only fix that reaches the child is one written to the
installed source file.

The fix
-------
Treat a closed pipe (EOFError / BrokenPipeError / OSError) as terminal: set
shutdown_event and break. That stops this thread AND lets the worker's own main
loop (`while not self.shutdown_event.is_set()`) exit, so an orphaned child
terminates instead of spinning. Any other exception keeps the original
log-and-retry behavior.

Second bug: abort() can hang forever
------------------------------------
AudioToTextRecorder.abort() waits on an event with no timeout:

    if self.state != "inactive":
        self.was_interrupted.wait()      # unbounded
        self._set_state("transcribing")
    if self.is_recording:
        self.stop()

If the recording loop never sets was_interrupted, abort() never returns. Two
consequences, both seen on 2026-09-07: the recording is never stopped (the
self.stop() on the last line is unreachable), and HotkeyController holds its
action lock for the whole call, so every later Page Up / Page Down is dropped
as "hotkey_ignored_busy" — the robot's physical buttons go dead until restart.
Bounded here so abort() always reaches stop().

Idempotent: re-running is a no-op once patched (detected via the PATCH(robot)
marker). uv sync can reinstall RealtimeSTT and wipe the patches, so this runs
from the `just sync` recipe after every sync.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "PATCH(robot)"

ORIGINAL = (
    "            except Exception as e:\n"
    "                logging.error(f\"Error receiving data from connection: {e}\", exc_info=True)\n"
    "                time.sleep(TIME_SLEEP)\n"
)

REPLACEMENT = (
    "            except (EOFError, BrokenPipeError, OSError):\n"
    "                # PATCH(robot): the parent closed the pipe (clean shutdown,\n"
    "                # crash, or hard-kill). A closed pipe never reopens, so the\n"
    "                # original `except Exception: log; sleep` spun here forever\n"
    "                # logging EOFError. Treat it as terminal: signal shutdown so\n"
    "                # this thread and the worker's main loop both exit and an\n"
    "                # orphaned child dies instead of spinning.\n"
    "                self.shutdown_event.set()\n"
    "                break\n"
    "            except Exception as e:\n"
    "                logging.error(f\"Error receiving data from connection: {e}\", exc_info=True)\n"
    "                time.sleep(TIME_SLEEP)\n"
)


ABORT_ORIGINAL = (
    '        if self.state != "inactive": # if inactive, was_interrupted will never be set\n'
    "            self.was_interrupted.wait()\n"
    '            self._set_state("transcribing")\n'
)

ABORT_REPLACEMENT = (
    '        if self.state != "inactive": # if inactive, was_interrupted will never be set\n'
    "            # PATCH(robot): bound this wait. It had no timeout, so when the\n"
    "            # recording loop never set was_interrupted, abort() blocked\n"
    "            # forever: the recording was never stopped (self.stop() below is\n"
    "            # unreachable from here) and the caller's lock was held for good,\n"
    "            # which killed every hotkey until restart. Timing out and\n"
    "            # continuing still reaches stop(), which is the point of abort().\n"
    "            if not self.was_interrupted.wait(timeout=2.0):\n"
    "                logging.warning(\n"
    "                    \"PATCH(robot) abort(): was_interrupted not set within 2s; \"\n"
    "                    \"stopping the recorder anyway\"\n"
    "                )\n"
    '            self._set_state("transcribing")\n'
)


PATCHES = (
    ("poll_connection", ORIGINAL, REPLACEMENT),
    ("abort", ABORT_ORIGINAL, ABORT_REPLACEMENT),
)


class PatchError(RuntimeError):
    """Raised when the source doesn't look like what we expect to patch."""


def patch_text(src: str) -> tuple[str, str]:
    """Return (new_source, status) after applying every patch in PATCHES.

    status is "already" (all present, unchanged) or "patched: a, b". Raises
    PatchError if a target block isn't found exactly once, which means
    RealtimeSTT changed upstream and the patch needs review rather than
    silently doing the wrong thing.
    """
    applied = []
    for name, original, replacement in PATCHES:
        if replacement in src:
            continue  # this one is already in place
        count = src.count(original)
        if count != 1:
            raise PatchError(
                f"expected exactly 1 {name} block to patch, found {count}"
            )
        src = src.replace(original, replacement, 1)
        applied.append(name)

    if not applied:
        return src, "already"
    return src, "patched: " + ", ".join(applied)


def _target_file() -> Path:
    """Locate the installed audio_recorder.py via the active interpreter."""
    import RealtimeSTT.audio_recorder as m  # noqa: WPS433 — runtime import on purpose

    return Path(m.__file__)


def main() -> int:
    try:
        path = _target_file()
    except Exception as e:  # pragma: no cover - environment problem, surface it
        print(f"patch_realtimestt: cannot locate RealtimeSTT: {e}", file=sys.stderr)
        return 1

    try:
        new_src, status = patch_text(path.read_text())
    except PatchError as e:
        print(
            f"patch_realtimestt: {e} in {path}. RealtimeSTT may have changed "
            "upstream — review scripts/patch_realtimestt.py against the new source.",
            file=sys.stderr,
        )
        return 1

    if status == "already":
        print(f"patch_realtimestt: already patched ({path})")
        return 0

    path.write_text(new_src)
    print(f"patch_realtimestt: {status} ({path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

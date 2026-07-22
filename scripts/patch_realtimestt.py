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

Idempotent: re-running is a no-op once patched (detected via the PATCH(robot)
marker). uv sync can reinstall RealtimeSTT and wipe the patch, so this runs
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


class PatchError(RuntimeError):
    """Raised when the source doesn't look like what we expect to patch."""


def patch_text(src: str) -> tuple[str, str]:
    """Return (new_source, status) for a RealtimeSTT source string.

    status is "already" (marker present, unchanged) or "patched". Raises
    PatchError if the poll_connection except block isn't found exactly once,
    which means RealtimeSTT changed upstream and the patch needs review rather
    than silently doing the wrong thing.
    """
    if MARKER in src:
        return src, "already"
    count = src.count(ORIGINAL)
    if count != 1:
        raise PatchError(
            f"expected exactly 1 poll_connection except block, found {count}"
        )
    return src.replace(ORIGINAL, REPLACEMENT, 1), "patched"


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
    print(f"patch_realtimestt: patched poll_connection ({path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

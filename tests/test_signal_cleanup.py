"""Tests for _install_signal_cleanup: SIGTERM/SIGHUP must run amain's finally.

A plain `kill` (or VS Code's stop button) sends SIGTERM, which by default
terminates Python without running `finally` blocks. That skips
edge.speech_to_text.shutdown(), orphaning the RealtimeSTT subprocesses — their
poll_connection loop then spins forever logging EOFError because the pipe to
the dead parent closed but their shutdown_event was never set. The handler
converts the signal into a cancellation of the main task so the same cleanup
path runs as on Ctrl-C.
"""

from __future__ import annotations

import asyncio
import os
import signal

import pytest

from robot.main import _install_signal_cleanup


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGHUP])
def test_signal_cancels_task_and_runs_finally(sig):
    # Mirrors amain's shape: install the handlers, block "running", and
    # record whether the finally (the STT shutdown slot) actually executed.
    # Note: the signal is sent to our own process — if the handler were not
    # installed, the default disposition would terminate the test run itself,
    # which is exactly the production failure this guards against.
    cleaned_up = []

    async def fake_amain():
        loop = asyncio.get_running_loop()
        _install_signal_cleanup(loop, asyncio.current_task())
        try:
            os.kill(os.getpid(), sig)
            await asyncio.sleep(30)  # must be interrupted by the signal
        finally:
            cleaned_up.append(True)
            # Restore default dispositions so this test can't leak handlers
            # into other tests sharing the process.
            for s in (signal.SIGTERM, signal.SIGHUP):
                loop.remove_signal_handler(s)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(fake_amain())

    assert cleaned_up == [True]

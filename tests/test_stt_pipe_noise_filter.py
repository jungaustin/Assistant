"""Tests for RealtimeSTTPipeNoiseFilter.

RealtimeSTT's poll_connection thread logs a burst of EOFError tracebacks to
the root logger when the transcription child's pipe closes during a deliberate
shutdown()/restart(). The filter drops exactly that teardown noise while
letting any genuinely novel error from the same code path through.
"""

from __future__ import annotations

import logging

from robot.core.logging import RealtimeSTTPipeNoiseFilter


_MSG = "Error receiving data from connection: "


def _record(msg: str, exc_type: type[BaseException] | None) -> logging.LogRecord:
    exc_info = None
    if exc_type is not None:
        try:
            raise exc_type("boom")
        except exc_type:
            import sys

            exc_info = sys.exc_info()
    return logging.LogRecord(
        name="root", level=logging.ERROR, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=exc_info,
    )


def test_drops_eoferror_pipe_close_burst():
    f = RealtimeSTTPipeNoiseFilter()
    assert f.filter(_record(_MSG, EOFError)) is False


def test_drops_broken_pipe():
    f = RealtimeSTTPipeNoiseFilter()
    assert f.filter(_record(_MSG, BrokenPipeError)) is False


def test_keeps_same_message_with_unrelated_exception():
    # A ValueError from that path is not the teardown race — let it through.
    f = RealtimeSTTPipeNoiseFilter()
    assert f.filter(_record(_MSG, ValueError)) is True


def test_keeps_same_message_with_no_exc_info():
    f = RealtimeSTTPipeNoiseFilter()
    assert f.filter(_record(_MSG, None)) is True


def test_keeps_unrelated_messages_even_with_eoferror():
    f = RealtimeSTTPipeNoiseFilter()
    assert f.filter(_record("something else entirely", EOFError)) is True


def test_integration_filter_suppresses_via_handler():
    """Wired onto a real handler, the EOFError record is suppressed end to end
    while a genuine error on the same logger still emits."""
    emitted: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            emitted.append(record.getMessage())

    handler = _Capture()
    handler.addFilter(RealtimeSTTPipeNoiseFilter())
    logger = logging.getLogger("test.stt.pipe")
    logger.propagate = False
    logger.setLevel(logging.ERROR)
    logger.addHandler(handler)
    try:
        try:
            raise EOFError
        except EOFError:
            logger.error("Error receiving data from connection: ", exc_info=True)
        logger.error("a real error")
    finally:
        logger.removeHandler(handler)

    assert emitted == ["a real error"]

"""structlog configuration.

Two output formats, auto-selected:

- **Pretty/colored** when stdout is a terminal — what you see during ``just run``.
- **JSON lines** when stdout is piped or redirected — grep-friendly,
  parseable by log aggregators.

LOG_LEVEL env var sets the level (default INFO). Set LOG_FORMAT=json to
force JSON output even on a TTY, or LOG_FORMAT=human to force pretty
output even when piped.

Hooks stdlib logging into structlog too, so the modules that use plain
``logging.getLogger(__name__)`` (core/bus.py, core/heartbeat.py,
brain/agent.py) emit through the same pipeline as the explicit
structlog callers in main.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


class RealtimeSTTPipeNoiseFilter(logging.Filter):
    """Drop RealtimeSTT's teardown pipe-close spam from the log.

    RealtimeSTT's poll_connection thread reads a multiprocessing pipe in a
    loop bounded by a shutdown_event. On shutdown()/restart() the transcription
    child closes its end of the pipe *before* that thread notices the event, so
    conn.recv() raises EOFError — which its bare ``except Exception`` logs to
    the root logger (with a full traceback) every TIME_SLEEP until the flag is
    finally seen. The result is a burst of identical "Error receiving data from
    connection" tracebacks that mean nothing: we tore the recorder down on
    purpose (see ear/realtimestt.py shutdown()/restart()).

    We suppress only the pipe-closed family (EOFError/BrokenPipeError) carrying
    that specific message. Any other exception from that code path — or that
    message without a pipe-closed cause — still passes through, so a real pipe
    fault during normal operation is not hidden.
    """

    _NEEDLE = "Error receiving data from connection"
    _PIPE_CLOSED = (EOFError, BrokenPipeError)

    def filter(self, record: logging.LogRecord) -> bool:
        if self._NEEDLE not in record.getMessage():
            return True
        exc_type = record.exc_info[0] if record.exc_info else None
        if exc_type is not None and issubclass(exc_type, self._PIPE_CLOSED):
            return False  # drop the teardown-race noise
        return True


def _resolve_format() -> str:
    """`json` or `human`. Env var wins; otherwise auto-detect from TTY."""
    forced = os.getenv("LOG_FORMAT", "").lower()
    if forced in ("json", "human"):
        return forced
    return "human" if sys.stdout.isatty() else "json"


def configure_logging(level: str | None = None) -> None:
    """Configure structlog + stdlib logging. Idempotent — calling twice
    just reconfigures with the new args.

    Call this once at process startup, before any logging happens.
    `level` overrides LOG_LEVEL env var.
    """
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt = _resolve_format()

    # Stdlib root config so plain logging.getLogger(__name__) callers
    # are captured. The handler writes to stderr by default; structlog's
    # ProcessorFormatter takes over the formatting.
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "json":
        final_processor = structlog.processors.JSONRenderer()
    else:
        # Pretty, with ANSI colors. Strips the ANSI when not a TTY (which
        # shouldn't happen here since we picked human FOR a TTY, but be safe).
        final_processor = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            final_processor,
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    # Silence RealtimeSTT's EOFError pipe-close burst on recorder teardown.
    handler.addFilter(RealtimeSTTPipeNoiseFilter())

    root = logging.getLogger()
    # Idempotency: replace handlers we installed earlier.
    root.handlers = [handler]
    root.setLevel(log_level)

    # Third-party loggers that spam at INFO level. Each is pinned to
    # WARNING so their INFO chatter doesn't flood the terminal during
    # recording. Comment any line out below to see that library's
    # messages again (useful when debugging that specific layer).
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)  # every audio chunk
    logging.getLogger("httpx").setLevel(logging.WARNING)           # every HTTP call
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Convenience wrapper. Prefer this in new code over
    ``logging.getLogger(__name__)`` for the structlog API."""
    return structlog.get_logger(name)

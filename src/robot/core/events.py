"""Typed events. The wire format for Phase 8's WebSocket transport, and
the message format for the in-process bus today (when components actually
get refactored onto it).

All events are pydantic models with a `type` discriminator and a `ts`
timestamp (UTC). Deserialize a JSON payload into the right subtype with
``Event.model_validate(payload)`` — pydantic uses the ``type`` field to
pick the class.

This module exists so Phase 8 has somewhere to plug into. The bus and
existing components don't publish these yet (per the cut-down Phase 5
scope); they'll start when components move onto the bus.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class _Base(BaseModel):
    """Common fields every event carries. `ts` defaults to wall-clock UTC at
    construction; `source` is the component that emitted the event."""

    ts: datetime = Field(default_factory=_now)
    source: str = Field(
        description="Component that emitted this event (e.g. 'ear', 'brain')."
    )


# ---------- Edge → Brain ----------


class WakeDetected(_Base):
    """The wake word fired. Edge is about to start capturing an utterance."""

    type: Literal["wake_detected"] = "wake_detected"


class TranscriptReady(_Base):
    """STT finished. Carries the transcribed utterance for the Brain."""

    type: Literal["transcript_ready"] = "transcript_ready"
    text: str


# ---------- Brain → Edge ----------


class BrainToken(_Base):
    """A single token from the streaming LLM. Pre-chunker."""

    type: Literal["brain_token"] = "brain_token"
    text: str


class BrainToolCall(_Base):
    """The Brain is invoking a tool. `id` matches the tool result back to this
    call (LLM tool-calling convention)."""

    type: Literal["brain_tool_call"] = "brain_tool_call"
    id: str
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class SpeakChunk(_Base):
    """A phrase-sized chunk ready for TTS. Post-chunker (see core/chunker.py)."""

    type: Literal["speak_chunk"] = "speak_chunk"
    text: str


# ---------- Operational ----------


class Heartbeat(_Base):
    """Component liveness ping. Phase 8 uses missing heartbeats to detect a
    dropped WebSocket and trigger reconnect."""

    type: Literal["heartbeat"] = "heartbeat"
    interval_s: float = Field(
        default=5.0,
        description="Expected seconds between heartbeats from this source.",
    )


class Error(_Base):
    """Component failure. `details` is free-form for stack traces, retry
    counts, anything the consumer wants to log."""

    type: Literal["error"] = "error"
    message: str
    details: Optional[Dict[str, Any]] = None


# ---------- Discriminated union ----------

Event = Annotated[
    Union[
        WakeDetected,
        TranscriptReady,
        BrainToken,
        BrainToolCall,
        SpeakChunk,
        Heartbeat,
        Error,
    ],
    Field(discriminator="type"),
]


__all__ = [
    "Event",
    "WakeDetected",
    "TranscriptReady",
    "BrainToken",
    "BrainToolCall",
    "SpeakChunk",
    "Heartbeat",
    "Error",
]

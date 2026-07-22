"""save_clip tool — the Brain-side path for clip requests.

The Edge's keyword fast-path (main.py, clip plan decision 9A) already
handles the canonical phrasings ("clip that", "save that clip") without an
LLM round-trip; this tool exists so paraphrases still work ("record the
last minute", "save what just happened on screen").

Failure contract: every failure returns its SPOKEN string (ClipError
carries it), never raises — the model relays the string as the reply.
"""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool, StructuredTool

from robot.core.clip import ClipError

logger = logging.getLogger(__name__)


class ClipTools:
    """`clip_service` is injected by ToolManager (decision 4A). None means
    clipping is disabled/unavailable on this brain (CLIP_ENABLED=false, or a
    websocket brain whose Edge owns the capture) — ToolManager then skips
    registering the tool entirely so it never spends tool-budget tokens."""

    def __init__(self, clip_service=None):
        self.clip_service = clip_service

    @property
    def available(self) -> bool:
        return self.clip_service is not None

    def save_clip(self) -> str:
        if self.clip_service is None:
            return "Clipping isn't available right now."
        try:
            path = self.clip_service.save()
        except ClipError as exc:
            return exc.spoken
        except Exception:
            logger.exception("save_clip failed unexpectedly")
            return "Something went wrong saving the clip."
        return f"Clip saved — the last minute is in Nemo Clips as {path.name}."

    def create_save_clip_tool(self) -> BaseTool:
        return StructuredTool.from_function(
            func=self.save_clip,
            name="save_clip",
            description=(
                "Save the last ~60 seconds of the user's screen (with system "
                "audio and mic) as a video clip in their Nemo Clips folder. "
                "Use whenever the user asks to capture, record, or save what "
                "just happened on screen: 'record the last minute', 'save "
                "that moment', 'capture what just happened', 'I want a clip "
                "of that'. Takes no arguments — the footage is already "
                "buffered; this just saves it. Returns a short confirmation "
                "or the reason it failed; relay that to the user naturally. "
                "Do NOT use it for future recording ('start recording') — "
                "it only saves the minute that already happened."
            ),
        )

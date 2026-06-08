"""Edge runtime: mic + speaker (eventually camera) on one side, a Brain on
the other, with a Transport in between. Runs in-process today; swap the
transport for a WebSocket version in Phase 5 without touching this file.
"""

import asyncio

from robot.brain import Agent
from robot.config import FOLLOWUP_WINDOW_SECONDS
from robot.core.logging import configure_logging, get_logger
from robot.latency import probe
from robot.privacy import MicGate
from robot.ear import SpeechToText
from robot.transport import InProcessTransport
from robot.voice import TextToSpeech

log = get_logger(__name__)


class Edge:
    def __init__(self, transport, mic_gate: MicGate | None = None):
        self.transport = transport
        self.mic_gate = mic_gate or MicGate()
        self.speech_to_text = SpeechToText()
        self.text_to_speech = TextToSpeech()

    async def _listen_once(self) -> str | None:
        if not self.mic_gate.enabled:
            await asyncio.sleep(0.2)
            return None
        text = await asyncio.to_thread(self.speech_to_text.listen)
        if text:
            probe.mark_stt_finish()
        return text

    async def _listen_followup(self, timeout: float) -> str | None:
        """Listen for a follow-up utterance without requiring the wake word.
        Returns None on silence/timeout. MicGate still applies.
        """
        if not self.mic_gate.enabled:
            return None
        self.speech_to_text.set_wake_word_bypass(timeout)
        try:
            listen_task = asyncio.create_task(
                asyncio.to_thread(self.speech_to_text.listen)
            )
            try:
                text = await asyncio.wait_for(listen_task, timeout=timeout)
            except asyncio.TimeoutError:
                self.speech_to_text.abort()
                # CancelledError doesn't inherit from Exception in 3.8+;
                # catch both so the abort path doesn't propagate when the
                # listen task is cancelled by the timeout above.
                try:
                    await listen_task
                except (Exception, asyncio.CancelledError):
                    pass
                return None
        finally:
            self.speech_to_text.set_wake_word_bypass(0.0)
        if text:
            probe.mark_stt_finish()
        return text

    async def _speak_stream(self, token_iter):
        # Pipeline: Agent token stream → chunker → blocking queue → TTS engine.
        # The chunker emits phrase-sized strings so every engine (Piper,
        # OpenAI, Coqui) gets the same well-formed input. Engines that did
        # their own ad-hoc chunking now see chunks that already end at a
        # boundary, so the internal chunking is a no-op.
        import queue

        from robot.core.chunker import achunk_tokens

        q: queue.Queue = queue.Queue(maxsize=64)
        sentinel = object()

        def sync_iter():
            while True:
                item = q.get()
                if item is sentinel:
                    return
                yield item

        async def first_token_marker(aiter):
            # Wrap the token iterator so we can fire the latency probe on
            # the first token BEFORE the chunker buffers it. The chunker
            # holds tokens until a boundary; marking inside it would
            # mismeasure (we want brain-first-token, not chunker-first-flush).
            async for token in aiter:
                probe.mark_brain_first_token()
                yield token

        # Collect the full response text so we can print one `assistant: ...`
        # line after speech finishes — symmetric with the `user: ...` line
        # above. Helps triage which layer is misbehaving (Whisper mishearing,
        # LLM drift, TTS mispronunciation) without bag-of-print debugging.
        spoken_text: list[str] = []

        async def pump():
            async for chunk in achunk_tokens(first_token_marker(token_iter)):
                spoken_text.append(chunk)
                await asyncio.to_thread(q.put, chunk)
            await asyncio.to_thread(q.put, sentinel)

        pump_task = asyncio.create_task(pump())
        await asyncio.to_thread(self.text_to_speech.speak, sync_iter())
        await pump_task
        if spoken_text:
            print(f"assistant: {''.join(spoken_text).strip()}")

    async def run(self):
        log.info("edge_state", state="listening")
        while True:
            utterance = await self._listen_once()
            if not utterance:
                continue
            while utterance:
                # Conversation transcripts stay as plain prints — they're
                # for watching the robot live, and structlog metadata
                # (timestamp, level, logger name) would just be noise.
                # Pipe stdout to a file and switch to JSON mode for logs.
                print(f"user: {utterance}")
                tokens = self.transport.respond(utterance)
                await self._speak_stream(tokens)
                log.info(
                    "edge_state", state="follow_up", seconds=FOLLOWUP_WINDOW_SECONDS
                )
                utterance = await self._listen_followup(FOLLOWUP_WINDOW_SECONDS)
            log.info("edge_state", state="listening")


async def amain():
    configure_logging()
    log.info("edge_starting")
    brain = Agent()
    transport = InProcessTransport(brain)
    edge = Edge(transport)
    await edge.run()


if __name__ == "__main__":
    asyncio.run(amain())

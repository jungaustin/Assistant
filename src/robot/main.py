"""Edge runtime: mic + speaker (eventually camera) on one side, a Brain on
the other, with a Transport in between. Runs in-process today; swap the
transport for a WebSocket version in Phase 5 without touching this file.
"""

import asyncio

from robot.brain import Agent
from robot.config import FOLLOWUP_WINDOW_SECONDS
from robot.latency import probe
from robot.privacy import MicGate
from robot.ear import SpeechToText
from robot.transport import InProcessTransport
from robot.voice import TextToSpeech


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
                try:
                    await listen_task
                except Exception:
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

        async def pump():
            async for chunk in achunk_tokens(first_token_marker(token_iter)):
                await asyncio.to_thread(q.put, chunk)
            await asyncio.to_thread(q.put, sentinel)

        pump_task = asyncio.create_task(pump())
        await asyncio.to_thread(self.text_to_speech.speak, sync_iter())
        await pump_task

    async def run(self):
        print("listening")
        while True:
            utterance = await self._listen_once()
            if not utterance:
                continue
            while utterance:
                print(f"user: {utterance}")
                tokens = self.transport.respond(utterance)
                await self._speak_stream(tokens)
                print(f"follow-up window ({FOLLOWUP_WINDOW_SECONDS:.0f}s)")
                utterance = await self._listen_followup(FOLLOWUP_WINDOW_SECONDS)
            print("listening")


async def amain():
    brain = Agent()
    transport = InProcessTransport(brain)
    edge = Edge(transport)
    await edge.run()


if __name__ == "__main__":
    asyncio.run(amain())

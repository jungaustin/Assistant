"""Edge runtime: mic + speaker (eventually camera) on one side, a Brain on
the other, with a Transport in between. TRANSPORT=inproc (default) runs the
Brain in this process; TRANSPORT=websocket runs Edge-only against a brain
server (robot.brain_server) — loopback today, the Pi/Mac split at Phase 8.
"""

import asyncio
import re
import signal
import threading
import time
from difflib import SequenceMatcher

from robot.config import (
    BRAIN_WS_URL,
    CLIP_ENABLED,
    ECHO_SIMILARITY_THRESHOLD,
    FOLLOWUP_POST_SPEECH_DELAY,
    FOLLOWUP_WINDOW_SECONDS,
    MAX_UTTERANCE_SECONDS,
    STT_HEALTH_POLL_SECONDS,
    STT_LISTEN_HEARTBEAT_SECONDS,
    TRACKER_DB_PATH,
    TRANSPORT,
    TRANSPORT_TOKEN,
)
from robot.core.clip import ClipError, is_clip_command
from robot.core.logging import configure_logging, get_logger

# Disabled: daily reminder check-in.
# from robot.core.checkin import checkin_loop
# from robot.tools.inner.log import TrackerDB
from robot.keys import HotkeyController, HotkeyListener
from robot.latency import probe
from robot.privacy import MicGate
from robot.ear import SpeechToText
from robot.transport import InProcessTransport
from robot.voice import TextToSpeech
from robot.voice.beep import ready_beep

log = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    """Lowercase and reduce to space-joined word tokens for fuzzy comparison."""
    return " ".join(_WORD_RE.findall(text.lower()))


def _is_echo_or_junk(utterance: str, last_spoken: str) -> bool:
    """True if a follow-up transcript should be discarded rather than answered.

    Guards against two field-observed failure modes that drive the self-talk
    loop:

    - Junk: the recorder transcribes ambient noise or a speaker tail as
      fragments with no real words — ". . .", "3.". We drop anything with
      fewer than two alphabetic characters.
    - Echo: the mic captures Nemo's own TTS during the follow-up window, so the
      "user" turn is just (part of) what Nemo just said. Answering it opens a
      conversation with itself. We only echo-match utterances of three or more
      words, so genuinely short commands ("stop", "delete that") are never
      suppressed just because they happen to appear in the spoken reply.
    """
    cleaned = utterance.strip()
    if sum(c.isalpha() for c in cleaned) < 2:
        return True
    u = _normalize(utterance)
    s = _normalize(last_spoken)
    if not u:
        return True
    if not s or len(u.split()) < 3:
        return False
    # Verbatim (or a clean slice) of what we just said: a real echo.
    if u in s:
        return True
    # Fuzzy match catches STT-noisy echoes, but ONLY when the utterance spans
    # most of what we said. Without that length gate, a genuine new command that
    # merely rhymes with our confirmation — "log 350 calories for rice" right
    # after "I've logged 350 calories for rice" — scores ~0.86 and gets wrongly
    # dropped. A real full echo is ~as long as the reply; a fresh command isn't.
    if len(u) >= 0.8 * len(s):
        return SequenceMatcher(None, u, s).ratio() >= ECHO_SIMILARITY_THRESHOLD
    return False


class Edge:
    def __init__(
        self,
        transport,
        mic_gate: MicGate | None = None,
        speech_to_text=None,
        text_to_speech=None,
        clip_service=None,
    ):
        self.transport = transport
        self.mic_gate = mic_gate or MicGate()
        # Injectable so tests can drive the listen loop without real audio.
        self.speech_to_text = speech_to_text or SpeechToText()
        self.text_to_speech = text_to_speech or TextToSpeech()
        # Optional ClipService (clip plan 9A): when present, canonical
        # "clip that" phrasings save via the keyword fast-path below without
        # a Brain round-trip. None = clipping disabled.
        self.clip_service = clip_service

    @staticmethod
    def _set_mic_capture(stt, on: bool) -> None:
        """Mute/unmute the Ear's own audio capture, if it supports it.

        Best-effort: a fake/alternate Ear without set_microphone (tests,
        other engines) just keeps capturing continuously as before."""
        set_mic = getattr(stt, "set_microphone", None)
        if set_mic is None:
            return
        try:
            set_mic(on)
        except Exception:
            log.exception("stt set_microphone failed")

    @staticmethod
    def _force_stop(stt) -> None:
        """End an over-long recording, if the Ear supports it. Best-effort: a
        fake/alternate Ear without force_stop just keeps waiting on its own
        end-of-speech logic."""
        force_stop = getattr(stt, "force_stop", None)
        if force_stop is None:
            return
        try:
            force_stop()
        except Exception:
            log.exception("stt force_stop failed")

    async def _listen_once(self) -> str | None:
        if not self.mic_gate.enabled:
            await asyncio.sleep(0.2)
            return None
        text = await self._listen_with_watchdog()
        if text:
            probe.mark_stt_finish()
        return text

    async def _listen_with_watchdog(self) -> str | None:
        """Run a blocking wake-word listen() while watching recorder health.

        The wake-word listen blocks until RealtimeSTT's background capture
        thread signals speech — but that daemon thread can die silently, after
        which listen() never returns (the multi-minute hang we hit in the
        field). So we poll the recorder's health while blocked; if it goes dead,
        we abort the stuck listen, rebuild the recorder, and return None so the
        run loop simply listens again on a fresh recorder.

        `asyncio.shield` keeps the poll timeout from cancelling the in-flight
        listen, and awaiting the shielded task means a completed listen returns
        immediately — the poll interval adds no latency to the normal path. An
        Ear without `is_healthy`/`restart` (e.g. a test fake or another engine)
        degrades to a plain listen.
        """
        stt = self.speech_to_text
        is_healthy = getattr(stt, "is_healthy", None)
        restart = getattr(stt, "restart", None)
        t0 = time.monotonic()
        log.info("stt_listen_start")
        listen_task = asyncio.create_task(asyncio.to_thread(stt.listen))
        if is_healthy is None or restart is None:
            text = await listen_task
            log.info(
                "stt_listen_done",
                elapsed=round(time.monotonic() - t0, 2),
                chars=len(text or ""),
            )
            return text

        unhealthy_streak = 0
        next_heartbeat = t0 + STT_LISTEN_HEARTBEAT_SECONDS
        recording_since: float | None = None
        while True:
            try:
                text = await asyncio.wait_for(
                    asyncio.shield(listen_task), timeout=STT_HEALTH_POLL_SECONDS
                )
                log.info(
                    "stt_listen_done",
                    elapsed=round(time.monotonic() - t0, 2),
                    chars=len(text or ""),
                )
                return text
            except asyncio.TimeoutError:
                pass
            now = time.monotonic()
            healthy = is_healthy()
            # Cap a single capture: if the VAD never registers end-of-speech the
            # recorder records forever. Track how long it's been recording and
            # force-stop past the ceiling so listen() returns the partial
            # instead of hanging.
            if getattr(stt, "is_recording", False):
                if recording_since is None:
                    recording_since = now
                elif now - recording_since > MAX_UTTERANCE_SECONDS:
                    log.warning(
                        "stt_recording_cap_reached",
                        recorded=round(now - recording_since, 1),
                    )
                    self._force_stop(stt)
                    recording_since = None
            else:
                recording_since = None
            # Heartbeat so a long block is visibly alive (and shows recorder
            # health) instead of silent — regular ticks = waiting for the wake
            # word; ticks stopping or healthy=False = a wedge.
            if now >= next_heartbeat:
                log.info(
                    "stt_listen_heartbeat",
                    elapsed=round(now - t0, 1),
                    healthy=healthy,
                )
                next_heartbeat = now + STT_LISTEN_HEARTBEAT_SECONDS
            if healthy:
                unhealthy_streak = 0
                continue
            # Tolerate a single transient blip (e.g. mid-restart) before tearing
            # the recorder down — two consecutive dead reads means it's really
            # gone, not just briefly between states.
            unhealthy_streak += 1
            if unhealthy_streak < 2:
                continue
            log.warning(
                "stt_recorder_unhealthy_recovering",
                elapsed=round(time.monotonic() - t0, 2),
            )
            try:
                stt.abort()  # unblock the wedged listen() so the task can finish
            except Exception:
                log.exception("stt abort during recovery failed")
            try:
                await listen_task
            except (Exception, asyncio.CancelledError):
                pass
            try:
                restart()
            except Exception:
                log.exception("stt restart failed")
            log.info("stt_recorder_recovered", elapsed=round(time.monotonic() - t0, 2))
            return None

    async def _listen_followup(self, timeout: float) -> str | None:
        """Listen for a follow-up utterance without requiring the wake word.

        `timeout` bounds how long we wait for the user to *start* speaking —
        not how long they may speak. Once recording begins we drop the
        deadline and let the recorder end the utterance on its own silence
        detection, so a long or late-starting reply is never truncated. (The
        old behavior — an absolute `wait_for` on the whole listen — chopped
        active speech and discarded it.) Returns None on silence/timeout;
        MAX_UTTERANCE_SECONDS still caps a single capture. MicGate applies.
        """
        if not self.mic_gate.enabled:
            return None
        t0 = time.monotonic()
        log.info("followup_listen_start", window=timeout)
        self.speech_to_text.set_wake_word_bypass(timeout)
        try:
            listen_task = asyncio.create_task(
                asyncio.to_thread(self.speech_to_text.listen)
            )
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            # Phase 1: wait for speech to begin, or the listen to finish, or
            # the idle window to expire with nothing said.
            while not self.speech_to_text.is_recording and not listen_task.done():
                if loop.time() >= deadline:
                    self.speech_to_text.abort()
                    # CancelledError doesn't inherit from Exception in 3.8+;
                    # catch both so cancelling the listen doesn't propagate.
                    try:
                        await listen_task
                    except (Exception, asyncio.CancelledError):
                        pass
                    log.info(
                        "followup_listen_done",
                        elapsed=round(time.monotonic() - t0, 2),
                        chars=0,
                        reason="timeout",
                    )
                    return None
                await asyncio.sleep(0.05)
            # Phase 2: they're talking (or already done) — let it finish on the
            # recorder's own silence detection, but cap the total capture so a
            # VAD that never registers end-of-speech can't record forever.
            rec_deadline = loop.time() + MAX_UTTERANCE_SECONDS
            while not listen_task.done():
                if loop.time() >= rec_deadline:
                    log.warning(
                        "stt_recording_cap_reached",
                        recorded=MAX_UTTERANCE_SECONDS,
                        phase="followup",
                    )
                    self._force_stop(self.speech_to_text)
                    break
                await asyncio.sleep(0.1)
            text = await listen_task
        finally:
            self.speech_to_text.set_wake_word_bypass(0.0)
        if text:
            probe.mark_stt_finish()
        log.info(
            "followup_listen_done",
            elapsed=round(time.monotonic() - t0, 2),
            chars=len(text or ""),
        )
        return text

    async def _speak_stream(self, token_iter):
        # Pipeline: Agent token stream → chunker → blocking queue → TTS engine.
        # The chunker emits phrase-sized strings so every engine (Piper,
        # OpenAI, Coqui) gets the same well-formed input. Engines that did
        # their own ad-hoc chunking now see chunks that already end at a
        # boundary, so the internal chunking is a no-op.
        import queue

        from robot.core.chunker import achunk_tokens, sanitize_for_speech

        q: queue.Queue = queue.Queue(maxsize=64)
        sentinel = object()

        def sync_iter():
            while True:
                item = q.get()
                if item is sentinel:
                    return
                yield item

        t0 = time.monotonic()
        log.info("speak_start")
        first_token_seen = False

        async def first_token_marker(aiter):
            # Wrap the token iterator so we can fire the latency probe on
            # the first token BEFORE the chunker buffers it. The chunker
            # holds tokens until a boundary; marking inside it would
            # mismeasure (we want brain-first-token, not chunker-first-flush).
            nonlocal first_token_seen
            async for token in aiter:
                probe.mark_brain_first_token()
                if not first_token_seen:
                    # First token in hand: the brain produced output. A
                    # speak_start with no brain_first_token means the hang is
                    # upstream in the LLM/tools (cross-ref the agent's llm/tool
                    # logs); a brain_first_token with no speak_done means TTS
                    # playback stalled.
                    log.info(
                        "brain_first_token", elapsed=round(time.monotonic() - t0, 2)
                    )
                    first_token_seen = True
                yield token

        # Collect the full response text so we can print one `assistant: ...`
        # line after speech finishes — symmetric with the `user: ...` line
        # above. Helps triage which layer is misbehaving (Whisper mishearing,
        # LLM drift, TTS mispronunciation) without bag-of-print debugging.
        spoken_text: list[str] = []

        async def pump():
            async for chunk in achunk_tokens(first_token_marker(token_iter)):
                # Strip spoken-markup noise (*, `) before it reaches TTS or the
                # `assistant:` log, so both reflect what's actually spoken. A
                # chunk that was pure markup collapses to empty — skip it rather
                # than feed the engine a blank.
                chunk = sanitize_for_speech(chunk)
                if not chunk.strip():
                    continue
                spoken_text.append(chunk)
                await asyncio.to_thread(q.put, chunk)
            await asyncio.to_thread(q.put, sentinel)

        pump_task = asyncio.create_task(pump())
        # Mute the recorder's own capture for the duration of playback — its
        # wake-word/VAD thread runs continuously in the background (not just
        # during our listen() calls), so without this it can hear its own
        # TTS coming out of the speaker. Unmute unconditionally afterward:
        # if the mic was already off for another reason (deafened), the run
        # loop's mic_gate check upstream still keeps us from acting on
        # anything captured.
        self._set_mic_capture(self.speech_to_text, False)
        try:
            await asyncio.to_thread(self.text_to_speech.speak, sync_iter())
        finally:
            self._set_mic_capture(self.speech_to_text, True)
        await pump_task
        result = "".join(spoken_text).strip()
        log.info(
            "speak_done", elapsed=round(time.monotonic() - t0, 2), chars=len(result)
        )
        if result:
            print(f"assistant: {result}")
        return result

    async def _speak_text(self, text: str) -> str:
        """Speak a fixed string through the normal streaming pipeline, so it
        gets the same chunking/sanitizing/logging as a Brain reply."""

        async def one():
            yield text

        return await self._speak_stream(one())

    def _start_clip_save(self) -> None:
        """Kick the pending snapshot's save off-loop (fast-path, 9A).

        The spoken ack happens immediately; the save itself waits ≤6s for
        the boundary segment and remuxes, so it runs on its own thread. A
        failure is spoken when it surfaces — it may land during the
        follow-up window, which is acceptable for a rare error (same call
        we made for the check-in's interjections).
        """

        def worker():
            try:
                path = self.clip_service.save()
                print(f"clip saved: {path}")
            except ClipError as exc:
                print(f"assistant (clip): {exc.spoken}")
                try:
                    self.text_to_speech.speak(exc.spoken)
                except Exception:
                    log.exception("clip failure notice failed to speak")
            except Exception:
                log.exception("clip save failed unexpectedly")

        threading.Thread(target=worker, name="clip-save", daemon=True).start()

    async def run(self):
        # Audible "booted and listening" cue. By this point the Whisper and
        # TTS models are loaded, so the beep means the wake word will
        # actually be heard — not just that the process started.
        ready_beep()
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
                if self.clip_service is not None and is_clip_command(utterance):
                    # Keyword fast-path (9A): ack in <1s, save in the
                    # background. Paraphrases miss this regex on purpose and
                    # go to the Brain's save_clip tool instead.
                    self._start_clip_save()
                    last_spoken = await self._speak_text(
                        "Got it — clipping the last minute."
                    )
                else:
                    tokens = self.transport.respond(utterance)
                    last_spoken = await self._speak_stream(tokens)
                music_on = self.transport.music_active
                self.transport.clear_music_active()
                if music_on:
                    log.info(
                        "edge_state",
                        state="listening",
                        reason="music_active_skip_followup",
                    )
                    utterance = None
                else:
                    # Let the speaker's tail die down before reopening the mic so
                    # we don't capture our own voice as the next "user" turn.
                    if FOLLOWUP_POST_SPEECH_DELAY:
                        await asyncio.sleep(FOLLOWUP_POST_SPEECH_DELAY)
                    log.info(
                        "edge_state", state="follow_up", seconds=FOLLOWUP_WINDOW_SECONDS
                    )
                    utterance = await self._listen_followup(FOLLOWUP_WINDOW_SECONDS)
                    if utterance and _is_echo_or_junk(utterance, last_spoken):
                        # Heard our own echo or non-lexical noise — drop it and
                        # fall back to wake-word listening instead of replying to
                        # ourselves (the self-talk loop).
                        log.info(
                            "edge_state",
                            state="listening",
                            reason="echo_or_junk_discarded",
                        )
                        utterance = None
            log.info("edge_state", state="listening")


def make_transport(clip_service=None):
    """Build the Edge's transport per TRANSPORT. Imports are lazy on purpose:
    Edge-only mode (websocket) must not load the Agent's LangChain stack —
    that's the dependency split the Pi relies on at Phase 8.4."""
    if TRANSPORT == "websocket":
        from robot.transport.websocket import WebSocketTransport

        if not TRANSPORT_TOKEN:
            raise SystemExit(
                "TRANSPORT=websocket needs TRANSPORT_TOKEN set in .env "
                "(same value as the brain server)."
            )
        # clip_service stays Edge-side: the keyword fast-path still works,
        # but a remote brain has no save_clip tool (it can't reach this
        # process's buffers). Clip-over-transport is Phase 8.3 design work.
        log.info("transport_selected", kind="websocket", url=BRAIN_WS_URL)
        return WebSocketTransport(BRAIN_WS_URL, TRANSPORT_TOKEN)
    if TRANSPORT == "inproc":
        from robot.brain import Agent

        log.info("transport_selected", kind="inproc")
        return InProcessTransport(Agent(clip_service=clip_service))
    raise SystemExit(
        f"Unknown TRANSPORT={TRANSPORT!r}. Set TRANSPORT=inproc or "
        f"TRANSPORT=websocket in .env."
    )


def _install_signal_cleanup(loop, task) -> None:
    """Turn SIGTERM/SIGHUP into a cancellation of the main task.

    Python runs no `finally` blocks on an unhandled SIGTERM (VS Code's stop
    button, plain `kill`) or SIGHUP, so without this the STT subprocesses are
    orphaned and spin forever logging EOFError — the pipe to the dead parent
    closed but their shutdown_event was never set. Cancelling the task routes
    those signals through the same cleanup as Ctrl-C. SIGKILL can't be caught;
    `just run`/`just edge` sweep those orphans at next launch.
    """
    for sig in (signal.SIGTERM, signal.SIGHUP):
        loop.add_signal_handler(sig, task.cancel)


async def amain():
    configure_logging()
    log.info("edge_starting")

    # Clip-that (opt-in via CLIP_ENABLED). Built before the transport so the
    # inproc Agent can register the save_clip tool against the same instance
    # the Edge snapshots through. The speak callback closes over `edge`,
    # which exists by the time any watchdog notice fires.
    clip_service = None
    speech_to_text = None
    if CLIP_ENABLED:
        from robot.config import make_clip_service, make_stt_recorder

        def _clip_speak(message: str) -> None:
            print(f"assistant (clip): {message}")
            try:
                edge.text_to_speech.speak(message)
            except Exception:
                log.exception("clip notice failed to speak")

        clip_service = make_clip_service(speak=_clip_speak)
        # The snapshot hook rides the recorder factory (decisions 2A + 7A):
        # on_recording_start fires on wake-word, follow-up bypass, and
        # hotkey listens alike, and the factory survives watchdog restarts.
        speech_to_text = SpeechToText(
            recorder_factory=lambda: make_stt_recorder(
                on_recording_start=clip_service.take_snapshot
            )
        )

    transport = make_transport(clip_service)
    edge = Edge(transport, speech_to_text=speech_to_text, clip_service=clip_service)

    if clip_service is not None:
        # Gate off = instant pause + flush of all unsaved footage (3A).
        clip_service.attach_gate(edge.mic_gate)
        try:
            clip_service.start()
        except Exception:
            # Boot must survive a broken capture stack (missing helper
            # binary, no RAM disk perms). save_clip then answers with its
            # not-running phrase instead of the robot failing to start.
            log.exception("clip service failed to start; running without it")

    _install_signal_cleanup(asyncio.get_running_loop(), asyncio.current_task())

    # Physical buttons: PgUp = wake toggle (push-to-talk / cancel a false
    # trigger), PgDn = deafen. An event tap swallows both keys system-wide
    # while the robot runs (see keys.py for why hidutil remapping was a bust).
    # Best-effort — if Accessibility isn't granted the robot runs voice-only.
    hotkeys = HotkeyListener(
        HotkeyController(edge.mic_gate, edge.speech_to_text, edge.text_to_speech)
    )
    hotkeys.start()

    # Disabled: daily reminder check-in. Its own TrackerDB connection on
    # purpose — the once-a-day read mustn't contend with the brain's
    # tool-call connection. It speaks straight through the Edge's TTS; if
    # it ever fires mid-conversation the audio could overlap a reply,
    # which is acceptable for a once-a-day prompt until the Conductor
    # exists.
    # async def say(text: str) -> None:
    #     print(f"assistant (check-in): {text}")
    #     await asyncio.to_thread(edge.text_to_speech.speak, text)
    #
    # checkin_task = asyncio.create_task(checkin_loop(TrackerDB(TRACKER_DB_PATH), say))
    try:
        await edge.run()
    finally:
        # checkin_task.cancel()
        hotkeys.stop()
        if clip_service is not None:
            # Stops the capture helper and ejects the RAM disk — which also
            # destroys unsaved footage, the privacy rule. SIGKILL skips this;
            # `just _sweep-clip-ramdisk` wipes the orphan at next launch.
            clip_service.stop()
        # Shut the STT subprocesses down cleanly, otherwise a Ctrl-C/crash
        # orphans them and they spin forever logging EOFError.
        edge.speech_to_text.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(amain())
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Ctrl-C or a caught termination signal — cleanup already ran in
        # amain's finally; exit quietly instead of dumping a traceback.
        pass

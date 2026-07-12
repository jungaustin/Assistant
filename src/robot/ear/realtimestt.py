from robot.config import make_stt_recorder


class SpeechToText:
    def __init__(self, recorder=None):
        self.recorder = recorder if recorder is not None else make_stt_recorder()

    def listen(self):
        return self.recorder.text()

    @property
    def is_recording(self) -> bool:
        """True while the recorder is actively capturing speech.

        Lets the follow-up window tell "still waiting for you to start" apart
        from "you're talking" so it never times out mid-sentence.
        """
        return bool(getattr(self.recorder, "is_recording", False))

    def abort(self):
        """Stop the current listen() call."""
        self.recorder.abort()

    def force_stop(self):
        """End the in-progress recording now and transcribe what was captured.

        Unlike abort() (which discards the audio), this ends capture cleanly so
        the blocked listen() returns the partial transcription. It's the safety
        valve for the failure where the VAD never registers end-of-speech and
        the recorder would otherwise keep recording indefinitely. No-op if not
        currently recording; RealtimeSTT also enforces its own
        min_length_of_recording, so an over-eager stop can't truncate to nothing.
        """
        self.recorder.stop()

    def force_start(self) -> None:
        """Begin recording immediately, bypassing the wake word.

        Push-to-talk path: the recorder's start() flips it straight into
        recording, which also unblocks a listen() that was waiting for the
        wake word. RealtimeSTT ignores the call if it's too soon after the
        last recording stopped (min_gap_between_recordings), so a double-tap
        can't wedge anything.
        """
        self.recorder.start()

    def set_wake_word_bypass(self, seconds: float) -> None:
        """Allow voice without the wake word for `seconds` after the next
        listen() begins. Set to 0.0 to require the wake word again.
        """
        self.recorder.wake_word_activation_delay = seconds

    def is_healthy(self) -> bool:
        """Whether the recorder's background capture loop is still alive.

        RealtimeSTT runs wake-word + VAD in a daemon thread. That thread is the
        only thing that signals "speech started"; if it dies — it does so
        silently on an unhandled error — listen() blocks forever waiting for a
        signal that never comes. A False here is the watchdog's cue to rebuild
        the recorder. Unknown/not-yet-started state reports healthy so we never
        tear down a recorder that simply hasn't spun up its thread.
        """
        rec = self.recorder
        thread = getattr(rec, "recording_thread", None)
        if thread is None:
            return True
        running = getattr(rec, "is_running", True)
        return bool(running) and thread.is_alive()

    def shutdown(self) -> None:
        """Tear down the recorder's subprocesses on exit.

        RealtimeSTT runs its wake-word/VAD reader and the faster-whisper
        transcription in child processes. If the main process exits without
        this, those children are orphaned (reparented to launchd/init) and
        their poll_connection loop spins forever logging EOFError, because
        the pipe closed but their shutdown_event was never set. Best-effort:
        a wedged recorder may not close cleanly, but exit must not block on it.
        """
        try:
            self.recorder.shutdown()
        except Exception:
            pass

    def restart(self) -> None:
        """Tear down a wedged recorder and build a fresh one.

        Recovery-path only (the watchdog calls this when is_healthy() goes
        False), so the cost of reloading the STT model is acceptable — it beats
        an unbounded hang. Best-effort shutdown: a wedged recorder may not close
        cleanly, but we must not let that block the rebuild.
        """
        old = self.recorder
        try:
            old.shutdown()
        except Exception:
            pass
        self.recorder = make_stt_recorder()

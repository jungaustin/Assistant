from robot.config import make_stt_recorder


class SpeechToText:
    def __init__(self, recorder=None):
        self.recorder = recorder if recorder is not None else make_stt_recorder()

    def listen(self):
        return self.recorder.text()

    def abort(self):
        """Stop the current listen() call."""
        self.recorder.abort()

    def set_wake_word_bypass(self, seconds: float) -> None:
        """Allow voice without the wake word for `seconds` after the next
        listen() begins. Set to 0.0 to require the wake word again.
        """
        self.recorder.wake_word_activation_delay = seconds

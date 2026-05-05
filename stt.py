from config import make_stt_recorder


class SpeechToText:
    def __init__(self, recorder=None):
        self.recorder = recorder if recorder is not None else make_stt_recorder()

    def listen(self):
        return self.recorder.text()

    def wakeup(self):
        """Bypass the wake-word gate for the next utterance."""
        self.recorder.wakeup()

    def abort(self):
        """Stop the current listen() call."""
        self.recorder.abort()

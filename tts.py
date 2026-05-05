import atexit
from config import make_tts_engine


class TextToSpeech:
    """Thin façade over a TTS engine. Engines must expose `speak(text_or_iter)`
    and `shutdown()`. Engines that need streaming (RealtimeTTS-style) wrap
    `TextToAudioStream` internally and expose `.speak()` themselves."""

    def __init__(self, engine=None):
        self.engine = engine if engine is not None else make_tts_engine()
        atexit.register(self.shutdown_engine)

    def speak(self, text_or_generator):
        """Accepts a string or a token generator. With a generator, audio
        starts playing as soon as the first chunk is ready."""
        self.engine.speak(text_or_generator)

    def shutdown_engine(self):
        self.engine.shutdown()

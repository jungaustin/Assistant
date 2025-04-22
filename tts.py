from dotenv import load_dotenv
import os
load_dotenv()
#api_key = os.getenv("OPENAI_API_KEY")
#maybe swap to coqui engine later (it sounds good)
from RealtimeTTS import TextToAudioStream, OpenAIEngine
import atexit

class TextToSpeech:
    def __init__(self):
        self.engine = OpenAIEngine()
        self.stream = TextToAudioStream(self.engine)
        atexit.register(self.shutdown_engine)
    
    def speak(self, text_generator):
        self.stream.feed(text_generator).play()
    
    def shutdown_engine(self):
        self.engine.shutdown()
# Change to LangGraph Agent, and also create a tool that plays a playlist I own
# currently only working when something is already be ing played so handle pause/start playback
# on the test2 file, make it so it runs without the server being on using the refresh token

#Credit Sylvester Seo For helping


from llm import LLMInteractions
from tts import TextToSpeech
from stt import SpeechToText

class Assistant:
    def __init__(self):
        self.llm_interactions = LLMInteractions()
        self.speech_to_text = SpeechToText()
        self.text_to_speech = TextToSpeech()
    
    def process_text(self, text):
        print(text)
    
    def handle_input(self, text):
        ans = self.llm_interactions.ask_llm(text)
        self.process_text(ans)
    
    def run(self):
        while True:
            user_input = self.speech_to_text.listen()
            response = self.llm_interactions.run_agent(user_input)
            self.text_to_speech.speak(response)

if __name__ == '__main__':
    assistant = Assistant()
    assistant.run()


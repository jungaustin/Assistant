# Next change llm interactions to be an agent. I have a chatgpt page open called downlaod to venv only already asked
# on the test2 file, make it so it runs without the server being on using the refresh token

#Credit Sylvester Seo For helping

# Next to do, memory and summary are not working for some reason. I think i have to have the chain work with the memory as well invoking memory. do that next. Can also try to decrease delay but not mandatory its good enough. Next I will also have the assistant be able to play music

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


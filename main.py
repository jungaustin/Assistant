# Try again to use the better open wake word model cause its kinda bad
# Create play/pause feature

# Maybe make the opening of the app a different tool, so I can say "and" and have it do both. 

# currently only working when something is already being played so handle pause/start playback

#Credit Sylvester Seo For helping 



from llm import LLMInteractions
from tts import TextToSpeech
from stt import SpeechToText
from agent import Agent
from tool_manager import ToolManager

class Assistant:
    def __init__(self):
        self.tool_manager = ToolManager()
        self.agent = Agent()
        self.speech_to_text = SpeechToText()
        self.text_to_speech = TextToSpeech()
    
    def process_text(self, text):
        print(text)
    
    def handle_input(self, text):
        ans = self.agent.run(text)
        self.process_text(ans)
    
    def run(self):
        while True:
            print('listening')
            user_input = self.speech_to_text.listen()
            response = self.agent.run(user_input)
            self.text_to_speech.speak(response)

if __name__ == '__main__':
    assistant = Assistant()
    assistant.run()


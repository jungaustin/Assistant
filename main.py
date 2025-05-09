# Change to LangGraph Agent, and also create a tool that plays a playlist I own
# Currently two things that are wrong. Firstly, If the app is not open already, it opens it but it takes awhile so It gets a status error cause the app isnt open so it cant play. Secondly, The Text to Speech Highlights ALL of the thoughts of the agent, instead of just the final thought, so It repeats what I ask it to.
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
            user_input = self.speech_to_text.listen()
            response = self.agent.run(user_input)
            self.text_to_speech.speak(response)

if __name__ == '__main__':
    assistant = Assistant()
    assistant.run()


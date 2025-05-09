from agent import Agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
agent = Agent(llm)

input_text = "Play the song OMG by NewJeans"
for response in agent.run(input_text):
    print(response)

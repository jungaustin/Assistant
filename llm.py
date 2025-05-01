import getpass
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import initialize_agent, Tool, AgentType
from tool_manager import ToolManager

import tools.spotify_tools as spotify_tools
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


class LLMInteractions:
    def __init__(self):
        self.llm = llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.9,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=100,
            return_messages=True
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
            """You are Nemo, a friendly and capable AI assistant.
            You help the user with daily tasks, answer questions, and offer thoughtful advice.
            Keep your tone helpful and approachable, and your responses concise and accurate.
            There is no need to repeat the question back to me.
            Try your best to keep your responses short. More information is not necessary unless I ask.
            If I ask you to do a task, do your best to give 0 or 1 word answers. For example, if i ask you to play a song, there is no need for a response unless the song was not able to be played for some reason.

            Here are example questions and answers to guide your responses:

            User: Nemo, how big is an ant?
            Answer: 1.5 mm on average.
            
            User: Play Ditto by NewJeans.
            Answer: 
            
            User: Play Ferris Wheel by QWER.
            Answer: 

            User: Nemo, remind me to take out the trash every Wednesday night.
            Answer: Got it! I'll remind you.

            User: Nemo, I'm feeling burnt out lately. What should I do?
            Answer: It's okay to feel that way. Try to take short breaks during the day, get plenty of rest, and talk to someone you trust."""
            ),
            ("human", "{input}")
        ])
        self.tools_manager = ToolManager()
        self.agent = initialize_agent(
            tools = self.tools_manager.get_tools(),
            llm = self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory = self.memory,
            verbose=True,
        )
        
    # Define the summary check logic
    def summarize_conversation(self):
        """Summarize the conversation if needed"""
        conversation_history = self.memory.load_memory_variables({})["history"]
        if len(conversation_history) > 6:
            summary_message = "Summarize the conversation so far:"
            messages = conversation_history + [HumanMessage(content=summary_message)]
            summary = self.llm.invoke(messages)
            self.memory.save_context({"input": summary_message}, {"output": summary.content})
            return summary.content
        print(self.memory.load_memory_variables({})["history"])
        return None
    
    def run_agent(self, message):
        response = self.agent.run(message)
        yield response
        self.memory.save_context({"input" : message}, {"output": response})
        self.summarize_conversation
    
    
    
    # Ask the LLM and return the response
    # def ask_llm(self, input):
    #     return self.agent.run(input)
    # Old Streaming to Get the TTS Faster before I swapped to using an Agent
    # def stream_chain(self, message):
    #     output_chunks = []
    #     for chunk in self.chain.stream({"input": message}):
    #         content = getattr(chunk, 'content', None)
    #         if content:
    #             output_chunks.append(content)
    #             print()
    #             print("yield: " + content)
    #             print()
    #             yield content
        
    #     full_output = "".join(output_chunks)
        
    #     self.memory.save_context({"input": message}, {"output": full_output})
    #     self.summarize_conversation()

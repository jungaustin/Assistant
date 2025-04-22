import getpass
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.runnable import RunnablePassthrough

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM and memory
# llm = ChatOpenAI(model_name="gpt-4o",
#                  streaming=True,
#                  callbacks=[StreamingStdOutCallbackHandler()],
#                  temperature=0.9
# )
# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100, return_messages=True)

# # Define the prompt template
# prompt = ChatPromptTemplate.from_messages([
#     ("system",
#      "You are Nemo, a friendly and capable AI assistant. "
#      "You help the user with daily tasks, answer questions, and offer thoughtful advice. "
#      "Keep your tone helpful and approachable, and your responses concise and accurate."
#      "Try your best to keep your responses short. More information is not necessary unless I ask.\n\n"
     
#      "Here are example questions and answers to guide your responses:\n\n"
#      "User: Nemo, how big is an ant?\n"
#      "Answer: An ant is 1.5 mm on average.\n\n"
#      "User: Nemo, remind me to take out the trash every Wednesday night.\n"
#      "Answer: Got it! I'll remind you to take out the trash every Wednesday night.\n\n"
#      "User: Nemo, I'm feeling burnt out lately. What should I do?\n"
#      "Answer: It's okay to feel that way. Try to take short breaks during the day, get plenty of rest, and talk to someone you trust. Let me know if you'd like help planning your schedule."
#      ),
#     ("human", "{input}")
# ])

# # This chain invokes the prompt and the LLM to generate responses
# chain = prompt | llm

# # Define the summary check logic
# def summarize_conversation():
#     """Summarize the conversation if needed (every 6 messages or so)."""
#     conversation_history = memory.load_memory_variables({})["history"]
    
#     # Summarize conversation if there are more than 6 messages
#     if len(conversation_history) > 6:
#         summary_message = "Summarize the conversation so far:"
#         messages = conversation_history + [HumanMessage(content=summary_message)]
        
#         # Generate the summary using LLM
#         summary = llm.invoke(messages)
        
#         # Save the summary in memory
#         memory.save_context({"input": summary_message}, {"output": summary.content})
#         return summary.content
#     return None

# def stream_chain(message):
#     print("stream chain")
#     output_chunks = []
    
#     for chunk in chain.stream({"input": message}):
#         content = getattr(chunk, 'content', None)
#         if content:
#             output_chunks.append(content)
#             print()
#             print("yield: " + content)
#             print()
#             yield content
    
#     full_output = "".join(output_chunks)
    
#     memory.save_context({"input": message}, {"output": full_output})
#     summarize_conversation()
#     # result = chain.invoke({"input": message}).content
#     # #print(result)
    
#     # # Check if a summary is needed and trigger it
#     # summary = summarize_conversation()\
#     # # Save the current conversation input/output in memory
#     # memory.save_context({"input": message}, {"output": result})
#     # return result

# # Ask the LLM and return the response
# def ask_llm(input: str) -> str:
#     return "".join(stream_chain(input))



class LLMInteractions:
    def __init__(self):
        self.llm = llm = ChatOpenAI(model_name="gpt-4o",
                 streaming=True,
                 callbacks=[StreamingStdOutCallbackHandler()],
                 temperature=0.9
        )
        self.memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100, return_messages=True)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
            "You are Nemo, a friendly and capable AI assistant."
            "You help the user with daily tasks, answer questions, and offer thoughtful advice. "
            "Keep your tone helpful and approachable, and your responses concise and accurate."
            "There is no need to repeat the question back to me."
            "Try your best to keep your responses short. More information is not necessary unless I ask.\n\n"
            
            "Here are example questions and answers to guide your responses:\n\n"
            "User: Nemo, how big is an ant?\n"
            "Answer:1.5 mm on average.\n\n"
            "User: Nemo, remind me to take out the trash every Wednesday night.\n"
            "Answer: Got it! I'll remind you.\n\n"
            "User: Nemo, I'm feeling burnt out lately. What should I do?\n"
            "Answer: It's okay to feel that way. Try to take short breaks during the day, get plenty of rest, and talk to someone you trust."
            ),
            ("human", "{input}")
        ])
        self.chain = self.prompt | self.llm
        
    # Define the summary check logic
    def summarize_conversation(self):
        """Summarize the conversation if needed (every 6 messages or so)."""
        conversation_history = self.memory.load_memory_variables({})["history"]
        if len(conversation_history) > 6:
            summary_message = "Summarize the conversation so far:"
            messages = conversation_history + [HumanMessage(content=summary_message)]
            summary = llm.invoke(messages)
            self.memory.save_context({"input": summary_message}, {"output": summary.content})
            return summary.content
        return None
    
    # Comment on stream chain
    def stream_chain(self, message):
        output_chunks = []
        for chunk in self.chain.stream({"input": message}):
            content = getattr(chunk, 'content', None)
            if content:
                output_chunks.append(content)
                print()
                print("yield: " + content)
                print()
                yield content
        
        full_output = "".join(output_chunks)
        
        self.memory.save_context({"input": message}, {"output": full_output})
        self.summarize_conversation()
    
    # Ask the LLM and return the response
    def ask_llm(self, input):
        return "".join(self.stream_chain(input))

from typing import List

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from pydantic import BaseModel


# Define the system template
template = """Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool."""

# Function to format messages for the system
def get_messages_info(messages):
    return [SystemMessage(content=template)] + messages

# Define the PromptInstructions class
class PromptInstructions(BaseModel):
    """Instructions on how to prompt the LLM."""
    objective: str
    variables: List[str]
    constraints: List[str]
    requirements: List[str]

# Instantiate the Ollama-based model
llm = ChatOllama(model="qwen2.5:14b-instruct-q6_K", temperature=0)
llm_with_tool = llm.bind_tools([PromptInstructions])

# Define the function to process information
def info_chain(state):
    messages = get_messages_info(state["messages"])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

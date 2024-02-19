from langchain_openai import ChatOpenAI
from langchain.prompts import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import (
    ConversationBufferMemory,
    FileChatMessageHistory,
    ConversationSummaryMemory,
)
import argparse
import warnings

load_dotenv()

# Ignore all deprecation warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    default="return a list of numbers",
)
parser.add_argument(
    "--language",
    default="python",
)
args = parser.parse_args()


chat = ChatOpenAI(model="gpt-3.5-turbo", verbose=True)
# memory = ConversationBufferMemory(
#     input_key="human_message",
#     memory_key="messages",
#     return_messages=True,
#     chat_memory=FileChatMessageHistory("messages.json"),
# )

memory = ConversationSummaryMemory(
    input_key="human_message", memory_key="messages", return_messages=True, llm=chat
)

prompt = ChatPromptTemplate(
    input_variables=["human_message", "messages"],
    messages=[
        # SystemMessagePromptTemplate.from_template(
        #     "You are a chatbot which calculates mathematical results. Always use last answer as your base."
        # ),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{human_message}"),
    ],
)


chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)

while True:
    human_message = input(">> ")
    result = chain({"human_message": human_message, "messages": []})
    text = result["text"]
    print(f"test {text}")

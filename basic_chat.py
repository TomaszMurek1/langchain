from langchain_openai import ChatOpenAI
from langchain.prompts import (
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
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


chat = ChatOpenAI(model="gpt-3.5-turbo")
memory = ConversationBufferMemory(input_key="content", memory_key="messages", return_messages=True)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)


chain = LLMChain(llm=chat, prompt=prompt, memory=memory)

while True:
    content = input(">> ")
    result = chain({"content": content, "messages": []})
    text = result["text"]
    print(f"test {text}")

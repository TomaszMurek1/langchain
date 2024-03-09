from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import warnings

# Ignore all deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory="basic_chroma/emb",
            embedding_function=embeddings)

retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type(llm=chat,
                                    retriever=retriever,
                                    chain_type="stuff")

result = chain.run("What is an interesting fact about the English language?")

print(result)

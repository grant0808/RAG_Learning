from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent

llm = init_chat_model(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    model_provider="huggingface",
    temperature=0.7,
)



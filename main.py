from utils.RAGDataLoader import RAGDataLoader
from utils.RagEmbedder import RAGEmbedder
from utils.RAGRetriever import RAGRetrieverLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from utils.logger import logging
from utils.llm_setup import llm_Ollama, llm_OpenRouter, llm_Together, llm_OpenAI # to use this for llm based retrival 
from dotenv import load_dotenv
from utils.generation import RAGAnswerGenerator
import warnings
import os

warnings.filterwarnings('ignore', category=FutureWarning)
load_dotenv()

path = os.getenv('DATA_PATH')


loader = RAGDataLoader(root_dir=path)
vecStore = RAGEmbedder()

texts, metadatas = loader.load_documents(streaming=True)  # or streaming=True for large files
vec_db = vecStore.get_vectorstore()

ret = RAGRetrieverLoader(db=vec_db, texts=texts, metadatas=metadatas)
bm25 = ret.get_retriever(
    retriever_type = "bm25"
)

# llm_OpenRouter = ChatOpenAI(
#     model="meta-llama/llama-3.3-70b-instruct:free", #"mistralai/mistral-7b-instruct",            # Any OpenRouter-supported model
#     openai_api_key=os.getenv('OPENROUTER_API_KEY'),                    # OpenRouter key (not OpenAI)
#     openai_api_base="https://openrouter.ai/api/v1",   # IMPORTANT: direct to OpenRouter API
#     openai_organization="",                           # (optional)
#     # You can set temperature, max_tokens, etc. here as needed
# )

rag_answer_gen = RAGAnswerGenerator(
    llm=llm_OpenRouter,
    retriever=bm25
    )

answer = rag_answer_gen.generate_answer("Why did Scotland seek independence from the UK?")
print(answer)
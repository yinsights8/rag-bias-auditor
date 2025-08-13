
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_together import ChatTogether
import os
from dotenv import load_dotenv
import warnings
load_dotenv()

warnings.filterwarnings('ignore', category=FutureWarning)


# Open AI
llm_OpenAI = ChatOpenAI(
    model="gpt-3.5-turbo",      # or "gpt-4o" or your preferred OpenAI model
    openai_api_key=os.getenv("OPENROUTER_API_KEY")
)

# Together LLM (e.g., Llama-3-8B-Chat)
llm_Together = ChatTogether(
    model="meta-llama/Llama-3-8b-chat",  # Example: use a valid Together model name
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

# ---- OpenRouter LLM (uses ChatOpenAI) ----
OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
llm_OpenRouter = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct:free", #"mistralai/mistral-7b-instruct",            # Any OpenRouter-supported model
    openai_api_key=OPENROUTER_API_KEY,                    # OpenRouter key (not OpenAI)
    openai_api_base="https://openrouter.ai/api/v1",   # IMPORTANT: direct to OpenRouter API
    openai_organization="",                           # (optional)
    # You can set temperature, max_tokens, etc. here as needed
)

# Ollama LLM
llm_Ollama = ChatOllama(model="llama3")  # or "llama2", "mistral", etc.
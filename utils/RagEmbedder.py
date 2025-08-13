import os 
from utils.logger import logging
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from utils.RAGDataLoader import RAGDataLoader
# from utils.RAGRetriever import RAGRetrieverLoader
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

load_dotenv()


# path = os.getenv("VECTOR_DB_PATH")
# print(os.listdir(path))
data_loader = RAGDataLoader(root_dir=os.getenv("DATA_PATH"))
texts, metadatas = data_loader.load_documents()

class RAGEmbedder:
    """
    Loads a vector store (FAISS or Chroma) from a specified directory.
    Handles embedding initialization and storage type.
    """
    
    def __init__(self, 
                 vector_db_type: str = os.getenv('VECTOR_DB_TYPE'), 
                 embed_model: str = os.getenv('EMBED_MODEL'), 
                 index_dir: str = os.getenv('VECTOR_DB_PATH')):
        
        """
        Args:
            vector_db_type: 'faiss' or 'chroma'
            embed_model: Embedding model name (MiniLM, OpenAI, etc.)
            index_dir: Directory where the vector store is stored
        """
        self.vector_db_type = vector_db_type.lower()
        self.embed_model = embed_model
        self.index_dir = os.path.abspath(index_dir)
        self.vectorstore = None
        self.embedder = self._get_embedder()
        
        logging.info("Calling RagEmbedder For VectorStore...")
        
    def _get_embedder(self):
        """Initializes and returns the embedding model."""
        if self.embed_model.startswith("text-embedding-ada"):
            logging.info("OpenAI embedding model Initiated...")
            return OpenAIEmbeddings(model=self.embed_model)
        else:
            logging.info("HuggingFace embedding model Initiated...")
            return HuggingFaceEmbeddings(model_name=self.embed_model)
        
    def get_vectorstore(self):
        """Create a Vector store"""
        dbdir = os.path.join(self.index_dir, self.vector_db_type)
        if not os.path.exists(dbdir):
            os.makedirs(dbdir, exist_ok=True)
            db = FAISS.from_texts(texts, self.embedder, metadatas=metadatas)
            db.save_local(dbdir)
    
    def load_vectorstore(self):
        """Loads the vector store from the specified directory."""
        dbdir = os.path.join(self.index_dir, self.vector_db_type)
        if not os.path.exists(dbdir):
            logging.error(f"Vector store directory '{dbdir}' does not exist.")
            raise FileNotFoundError(f"Vector store directory '{dbdir}' does not exist.")
        
        if self.vector_db_type == "faiss":
            logging.info(f"vector store '{self.vector_db_type}' is Loaded...")
            self.vectorstore = FAISS.load_local(dbdir, self.embedder, allow_dangerous_deserialization=True)
        else:
            logging.error(f"Unknown vector DB type: {self.vector_db_type}")
            raise ValueError(f"Unknown vector DB type: {self.vector_db_type}")
        return self.vectorstore
        
        
    def get_vectorstore(self):
        """Returns the loaded vectorstore (or loads if not already loaded)."""
        if self.vectorstore is None:
            return self.load_vectorstore()
        logging.info(f"{self.vector_db_type} Vector Store Loded ")
        return self.vectorstore
    
        
        
if __name__ == "__main__":
    mode = RAGEmbedder()
    mode.get_vectorstore()
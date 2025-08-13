from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from utils.logger import logging
import os

load_dotenv()

class RAGRetrieverLoader:
    """
    Loads different types of retrievers from a vector store and text data.
    Supports: Dense, BM25, Hybrid, MMR, MultiQuery, Contextual Compression.
    """

    def __init__(self, 
                 db, 
                 texts=None, 
                 metadatas=None, 
                 embedder=None, 
                 llm=None):
        """
        Args:
            db: Loaded vectorstore (e.g., FAISS, Chroma)
            texts: List of text chunks (needed for BM25/hybrid)
            metadatas: List of chunk-level metadata
            embedder: Embedding function/model (optional for BM25)
            llm: LLM instance for advanced retrievers (optional)
        """
        self.db = db
        self.texts = texts
        self.metadatas = metadatas
        self.embedder = embedder
        self.llm = llm

    def get_retriever(self, retriever_type="dense", k=10, lambda_mult=0.5):
        """
        Returns a retriever of the given type.
        Supported: 'dense', 'bm25', 'hybrid', 'mmr', 'multiquery', 'compression'
        """
        retriever_type = retriever_type.lower()
        
        if retriever_type == "dense":
            logging.info(f"{retriever_type} Retriever selected...")
            return self.db.as_retriever(search_kwargs={"k": k})
        
        elif retriever_type == "bm25":
            if self.texts is None or self.metadatas is None:
                raise ValueError("BM25 retriever requires texts and metadatas.")
            retriever = BM25Retriever.from_texts(self.texts, metadatas=self.metadatas)
            retriever.k = k
            logging.info(f"{retriever_type} Retriever selected...")
            return retriever

        elif retriever_type == "hybrid":
            logging.info(f"{retriever_type} Retriever selected...")
            return self.db.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": k*2})
        
        elif retriever_type == "mmr":
            logging.info(f"{retriever_type} Retriever selected...")
            return self.db.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": k*2, "lambda_mult": lambda_mult})

        elif retriever_type == "multiquery":
            from langchain.retrievers.multi_query import MultiQueryRetriever
            if self.llm is None:
                raise ValueError("MultiQuery retriever requires an LLM instance.")
            base_retriever = self.db.as_retriever(search_kwargs={"k": k})
            logging.info(f"{retriever_type} Retriever selected...")
            return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=self.llm)
        
        elif retriever_type == "compression":
            from langchain.retrievers.document_compressors import LLMChainExtractor
            from langchain.retrievers import ContextualCompressionRetriever
            if self.llm is None:
                raise ValueError("ContextualCompressionRetriever requires an LLM instance.")
            compressor = LLMChainExtractor.from_llm(self.llm)
            base_retriever = self.db.as_retriever(search_kwargs={"k": k})
            logging.info(f"{retriever_type} Retriever selected...")
            return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
        
        else:
            logging.error(f"Retriever type '{retriever_type}' not implemented.")
            raise NotImplementedError(f"Retriever type '{retriever_type}' not implemented.")

# --- Example Usage ---

# if __name__ == "__main__":
    

    # rag_retriever = RAGRetrieverLoader(db=db, texts=texts, metadatas=metadatas, embedder=embedder, llm=llm_instance)
# bm25_retriever = rag_retriever.get_retriever("bm25", k=10)
# dense_retriever = rag_retriever.get_retriever("dense", k=10)
# hybrid_retriever = rag_retriever.get_retriever("hybrid", k=10)

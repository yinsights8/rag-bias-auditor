import os
import json
from utils.logger import logging
from dotenv import load_dotenv
from typing import List, Dict, Any, Generator
from langchain.text_splitter import RecursiveCharacterTextSplitter



load_dotenv()

class RAGDataLoader:
    """
    Loads and parses historical/cultural data for RAG pipelines.
    Expects JSON files in the given directory, each with fields: text, doc_id, title, person, gpe, loc, date.
    """

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        
    def split_documents_into_chunks(self,
        documents, 
        chunk_size=512, 
        chunk_overlap=64
    ):
        """
        Splits documents into chunks for embedding.
        Each document's text is split; metadata is duplicated for each chunk.

        Args:
            documents: List of dicts [{"text": ..., "metadata": {...}}, ...]
            chunk_size: Max length for each chunk (default: 512 chars)
            chunk_overlap: Overlap between chunks (default: 64 chars)

        Returns:
            texts: List[str] - the text chunks
            metadatas: List[dict] - the corresponding metadata for each chunk
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        texts = []
        metadatas = []
        for doc in documents:
            splits = splitter.split_text(doc["text"])
            texts.extend(splits)
            metadatas.extend([doc["metadata"]] * len(splits))
        logging.info("Text Chunking Operation is Completed...")
        return texts, metadatas

    def load_documents(self, streaming: bool = True) -> List[Dict[str, Any]]:
        """
        Loads all documents from JSON files in the directory.
        Supports normal and streaming (RAM-efficient) loading.

        Returns:
            List of dicts: [{"text": ..., "metadata": {...}}, ...]
        """
        json_files = [
            os.path.join(self.root_dir, f) 
            for f in os.listdir(self.root_dir) if f.endswith(".json")
        ]
        all_docs = []
        logging.info("Document Loading...")
        for file_path in json_files:
            if streaming:
                logging.info("Document Streaming...")
                all_docs.extend(list(self._load_documents_streaming(file_path)))
            else:
                logging.info("Streaming Off\nLoading complete data in Memory...")
                all_docs.extend(self._load_documents(file_path))
        logging.info("Document Loading completed...")
        
        text, metadata = self.split_documents_into_chunks(all_docs)
        return text, metadata
        # return all_docs
    
    

    def _load_documents(self, path: str) -> List[Dict[str, Any]]:
        """Loads documents from a JSON file (standard load)."""
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            {
                "text": entry.get("text", ""),
                "metadata": {
                    "doc_id": entry.get("doc_id", ""),
                    "title": entry.get("title", ""),
                    "person": entry.get("person", []),
                    "gpe": entry.get("gpe", []),
                    "loc": entry.get("loc", []),
                    "date": entry.get("date", [])
                }
            }
            for entry in data if isinstance(entry, dict)
        ]

    def _load_documents_streaming(self, path: str) -> Generator[Dict[str, Any], None, None]:
        """Loads documents from a JSON file using ijson for large files (streaming)."""
        import ijson
        with open(path, "r", encoding="utf-8") as f:
            for entry in ijson.items(f, 'item'):
                yield {
                    "text": entry.get("text", ""),
                    "metadata": {
                        "doc_id": entry.get("doc_id", ""),
                        "title": entry.get("title", ""),
                        "person": entry.get("person", []),
                        "gpe": entry.get("gpe", []),
                        "loc": entry.get("loc", []),
                        "date": entry.get("date", [])
                    }
                }


# if __name__ == "__main__":
    
#     path = os.getenv('DATA_PATH')

#     loader = RAGDataLoader(root_dir=path)
#     documents = loader.load_documents(streaming=True)  # or streaming=True for large files
#     texts, metadatas = loader.split_documents_into_chunks(documents)
#     logging.info("Data Loader suceess")
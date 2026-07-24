import os
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.config import settings
from src.embedding import EmbeddingEngine

class VectorDatabase:
    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embeddings = embedding_engine.get_embedding_model()
        self.vector_store: Optional[FAISS] = None
        self.index_path = str(settings.VECTOR_STORE_DIR)
        self.load_index()

    def load_index(self):
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            try:
                self.vector_store = FAISS.load_local(self.index_path, self.embeddings, allow_dangerous_deserialization=True)
            except Exception:
                self.vector_store = None

    def build_or_update_index(self, chunks: List[Dict[str, Any]]):
        documents = [Document(page_content=c["content"], metadata=c["metadata"]) for c in chunks]
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vector_store.add_documents(documents)
        self.vector_store.save_local(self.index_path)

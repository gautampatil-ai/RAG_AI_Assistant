from typing import List, Tuple
from langchain_core.documents import Document
from src.vector_database import VectorDatabase
from src.config import settings

class SimilarityRetriever:
    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db

    def retrieve(self, query: str, top_k: int = settings.TOP_K_RETRIEVAL) -> List[Tuple[Document, float]]:
        if not self.vector_db.vector_store:
            return []
        return self.vector_db.vector_store.similarity_search_with_score(query, k=top_k)

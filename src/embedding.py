from langchain_community.embeddings import HuggingFaceEmbeddings
from src.config import settings

class EmbeddingEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def get_embedding_model(self):
        return self.embeddings

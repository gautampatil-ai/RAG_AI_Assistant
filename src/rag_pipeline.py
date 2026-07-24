import time
from typing import Dict, Any
from langchain_core.messages import HumanMessage  # 👈 1. Added import

from src.loader import DocumentLoader
from src.chunker import TextChunker
from src.embedding import EmbeddingEngine
from src.vector_database import VectorDatabase
from src.retriever import SimilarityRetriever
from src.llm import GeminiLLM
from src.prompt_template import PROMPT
from src.utils import sanitize_filename, validate_file
from src.config import settings

class RAGPipeline:
    def __init__(self):
        self.chunker = TextChunker()
        self.embedding_engine = EmbeddingEngine()
        self.vector_db = VectorDatabase(self.embedding_engine)
        self.retriever = SimilarityRetriever(self.vector_db)
        self.llm_engine = GeminiLLM()
        self.total_files_uploaded = 0
        self.total_chunks_processed = 0

    def process_and_index_file(self, file_name: str, file_bytes: bytes) -> Dict[str, Any]:
        is_valid, msg = validate_file(file_name, len(file_bytes))
        if not is_valid: 
            raise ValueError(msg)

        safe_name = sanitize_filename(file_name)
        save_path = settings.UPLOAD_DIR / safe_name

        with open(save_path, "wb") as f:
            f.write(file_bytes)

        documents = DocumentLoader.load_document(save_path)
        chunks = self.chunker.split_documents(documents)
        self.vector_db.build_or_update_index(chunks)

        self.total_files_uploaded += 1
        self.total_chunks_processed += len(chunks)
        return {"filename": safe_name, "status": "Success", "chunks_added": len(chunks)}

    def answer_question(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        retrieved = self.retriever.retrieve(question)

        if not retrieved:
            return {
                "answer": "I couldn't find that information in the uploaded documents.",
                "citations": [],
                "elapsed_time": round(time.time() - start_time, 2)
            }

        context_str = ""
        citations = []
        for idx, (doc, score) in enumerate(retrieved):
            context_str += f"\n--- Context Snippet {idx + 1} ---\n{doc.page_content}\n"
            citations.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 1),
                "similarity_score": round(float(score), 4),
                "chunk_text": doc.page_content[:150] + "..."
            })

        formatted_prompt = PROMPT.format(context=context_str, question=question)
        
        # 👈 2. Updated invocation to wrap prompt in HumanMessage
        response = self.llm_engine.get_llm().invoke([HumanMessage(content=formatted_prompt)])

        return {
            "answer": response.content,
            "citations": citations,
            "elapsed_time": round(time.time() - start_time, 2)
        }

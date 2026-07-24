from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Any
from src.rag_pipeline import RAGPipeline
from src.chat_history import ChatHistoryTracker
from src.config import settings
from src.utils import logger

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

rag_pipeline = RAGPipeline()
history_tracker = ChatHistoryTracker()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    elapsed_time: float

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "vector_db_active": rag_pipeline.vector_db.vector_store is not None}

@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        return rag_pipeline.process_and_index_file(file.filename, content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        history_tracker.add_user_message(request.question)
        result = rag_pipeline.answer_question(request.question)
        history_tracker.add_assistant_message(result["answer"], result["citations"])
        history_tracker.log_query_metrics(result["elapsed_time"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_chat_history():
    return {"history": history_tracker.messages}

@app.delete("/history")
def clear_chat_history():
    history_tracker.clear()
    return {"message": "Chat history cleared successfully."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host=settings.HOST, port=settings.PORT, reload=True)

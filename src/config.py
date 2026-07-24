from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise RAG AI Assistant"
    VERSION: str = "1.0.0"
    GOOGLE_API_KEY: str = Field(default="", env="GOOGLE_API_KEY")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploaded_documents"
    VECTOR_STORE_DIR: Path = BASE_DIR / "data" / "vector_store"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_FILE_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt", ".csv"}
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    TOP_K_RETRIEVAL: int = 4
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash"
    TEMPERATURE: float = 0.1
    MAX_OUTPUT_TOKENS: int = 2048

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

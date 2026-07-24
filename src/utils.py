import re
import logging
from pathlib import Path
from typing import Tuple
from src.config import settings

def setup_logger(name: str = "RAG_Logger") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s]: %(message)s")
        file_handler = logging.FileHandler(settings.LOG_DIR / "app.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

logger = setup_logger()

def sanitize_filename(filename: str) -> str:
    filename = Path(filename).name
    return re.sub(r"[^\w\s\.-]", "", filename).strip().replace(" ", "_")

def validate_file(file_name: str, file_size_bytes: int) -> Tuple[bool, str]:
    ext = Path(file_name).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        return False, f"Unsupported format '{ext}'."
    if file_size_bytes > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, "File exceeds size threshold."
    return True, "Valid"

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()

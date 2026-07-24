from pathlib import Path
from typing import List, Dict, Any
import PyPDF2, docx, pandas as pd
from src.utils import logger, clean_text

class DocumentLoader:
    @staticmethod
    def load_pdf(file_path: Path) -> List[Dict[str, Any]]:
        docs = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for idx, page in enumerate(reader.pages):
                text = clean_text(page.extract_text() or "")
                if text:
                    docs.append({"content": text, "metadata": {"source": file_path.name, "page": idx + 1}})
        return docs

    @staticmethod
    def load_docx(file_path: Path) -> List[Dict[str, Any]]:
        doc = docx.Document(file_path)
        text = clean_text("\n".join([p.text for p in doc.paragraphs if p.text.strip()]))
        return [{"content": text, "metadata": {"source": file_path.name, "page": 1}}] if text else []

    @staticmethod
    def load_txt(file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = clean_text(f.read())
        return [{"content": text, "metadata": {"source": file_path.name, "page": 1}}] if text else []

    @staticmethod
    def load_csv(file_path: Path) -> List[Dict[str, Any]]:
        df = pd.read_csv(file_path)
        rows = [f"Row {idx + 1}: " + ", ".join([f"{c}: {v}" for c, v in row.items() if pd.notna(v)]) for idx, row in df.iterrows()]
        text = clean_text("\n".join(rows))
        return [{"content": text, "metadata": {"source": file_path.name, "page": 1}}] if text else []

    @classmethod
    def load_document(cls, file_path: Path) -> List[Dict[str, Any]]:
        ext = file_path.suffix.lower()
        if ext == ".pdf": return cls.load_pdf(file_path)
        elif ext == ".docx": return cls.load_docx(file_path)
        elif ext == ".txt": return cls.load_txt(file_path)
        elif ext == ".csv": return cls.load_csv(file_path)
        raise ValueError(f"Unsupported format: {ext}")

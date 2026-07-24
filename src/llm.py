from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings

class GeminiLLM:
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is missing in .env file.")
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=settings.TEMPERATURE,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS
        )

    def get_llm(self):
        return self.llm

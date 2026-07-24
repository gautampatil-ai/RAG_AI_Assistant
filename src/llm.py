import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings

class GeminiLLM:
    def __init__(self):
        # 1. Look for key in environment or Streamlit Secrets
        api_key = os.getenv("GOOGLE_API_KEY") or getattr(settings, "GOOGLE_API_KEY", "")
        
        if not api_key and hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]

        # Strip any accidental hidden whitespace or newlines
        if api_key:
            api_key = str(api_key).strip()

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing! "
                "Add it in Streamlit Cloud -> Manage App -> Settings -> Secrets."
            )

        # 2. Fallback to gemini-1.5-flash if needed
        model_name = getattr(settings, "GEMINI_MODEL_NAME", "gemini-1.5-flash")

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=settings.TEMPERATURE,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS
        )

    def get_llm(self) -> ChatGoogleGenerativeAI:
        return self.llm

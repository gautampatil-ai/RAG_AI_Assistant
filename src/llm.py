import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings

class GeminiLLM:
    def __init__(self):
        # 1. Fetch key from environment or Streamlit Secrets
        api_key = os.getenv("GOOGLE_API_KEY") or getattr(settings, "GOOGLE_API_KEY", "")
        
        if not api_key and hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]

        if not api_key:
            raise ValueError("GOOGLE_API_KEY missing! Add it in Streamlit Cloud -> Manage App -> Settings -> Secrets.")

        # 2. Instantiate LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL_NAME,
            google_api_key=api_key,
            temperature=settings.TEMPERATURE,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS
        )

    def get_llm(self) -> ChatGoogleGenerativeAI:
        return self.llm

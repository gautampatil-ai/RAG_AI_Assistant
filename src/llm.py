import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import settings

class GeminiLLM:
    def __init__(self):
        # 1. Look in config settings
        api_key = settings.GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY")
        
        # 2. Fall back to Streamlit Cloud Secrets if running on Streamlit Cloud
        if not api_key and hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]

        if not api_key:
            raise ValueError("GOOGLE_API_KEY is missing. Please set it in Streamlit Secrets or your .env file.")

        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL_NAME,
            google_api_key=api_key,
            temperature=settings.TEMPERATURE,
            max_output_tokens=settings.MAX_OUTPUT_TOKENS
        )

    def get_llm(self) -> ChatGoogleGenerativeAI:
        return self.llm

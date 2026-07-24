import streamlit as st
import json
from src.rag_pipeline import RAGPipeline
from src.chat_history import ChatHistoryTracker
from src.config import settings

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🤖", layout="wide")

@st.cache_resource
def initialize_pipeline():
    return RAGPipeline()

pipeline = initialize_pipeline()

if "history" not in st.session_state:
    st.session_state.history = ChatHistoryTracker()

with st.sidebar:
    st.title("🤖 RAG System Control")
    st.subheader("📁 Ingest Enterprise Data")
    uploaded_files = st.file_uploader("Upload PDF, DOCX, TXT, CSV", type=["pdf", "docx", "txt", "csv"], accept_multiple_files=True)
    
    if st.button("Process & Index Documents", type="primary"):
        if uploaded_files:
            progress_bar = st.progress(0)
            for idx, file in enumerate(uploaded_files):
                try:
                    pipeline.process_and_index_file(file.name, file.read())
                    st.success(f"Indexed: {file.name}")
                except Exception as e:
                    st.error(f"Error {file.name}: {str(e)}")
                progress_bar.progress((idx + 1) / len(uploaded_files))
            st.toast("Document processing completed!", icon="✅")

    st.markdown("---")
    st.subheader("📊 Analytics")
    st.metric("Files Ingested", pipeline.total_files_uploaded)
    st.metric("Total Chunks", pipeline.total_chunks_processed)
    st.metric("Avg Latency", f"{st.session_state.history.get_average_response_time()}s")
    
    if st.button("🗑️ Reset Chat Memory"):
        st.session_state.history.clear()
        st.rerun()

st.title("Enterprise Knowledge Assistant")

for msg in st.session_state.history.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your uploaded documents..."):
    st.session_state.history.add_user_message(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing context..."):
            result = pipeline.answer_question(prompt)
            st.markdown(result["answer"])
            st.session_state.history.add_assistant_message(result["answer"], result["citations"])
            st.session_state.history.log_query_metrics(result["elapsed_time"])

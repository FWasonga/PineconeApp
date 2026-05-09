# app.py
# ------------------------------------
# Streamlit web interface for your
# PDF RAG pipeline
# ------------------------------------

import streamlit as st
import tempfile
import os

from pdf_reader import load_pdf
from text_splitter import split_text
from embedder import embed_chunks, embed_query
from pinecone_store import create_index_if_not_exists, upsert_chunks, search
from query import generate_answer

# ------------------------------------
# Page configuration
# ------------------------------------
st.set_page_config(
    page_title="PDF Question Answering",
    page_icon="📄",
    layout="centered"
)

# ------------------------------------
# App title and description
# ------------------------------------
st.title("📄 PDF Question Answering")
st.markdown("Upload a PDF and ask questions about it. Powered by Pinecone + Groq.")

# ------------------------------------
# Session state
# keeps track of whether a PDF
# has already been ingested so we
# don't re-process it every time
# ------------------------------------
if "index" not in st.session_state:
    st.session_state.index = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------------------
# Sidebar — PDF upload section
# ------------------------------------
st.sidebar.header("📂 Upload Your PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file is not None:
    # Only re-process if a new PDF is uploaded
    if uploaded_file.name != st.session_state.pdf_name:
        
        st.sidebar.info(f"Processing: {uploaded_file.name}")
        
        # Progress bar to show user what's happening
        progress = st.sidebar.progress(0)
        status = st.sidebar.empty()

        # Save uploaded file to a temp location
        # (Streamlit gives us bytes, pypdf needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Step 1: Read PDF
        status.text("📖 Reading PDF...")
        progress.progress(20)
        text = load_pdf(tmp_path)

        # Step 2: Split text
        status.text("✂️ Splitting text into chunks...")
        progress.progress(40)
        chunks = split_text(text)

        # Step 3: Embed chunks
        status.text("🔢 Embedding chunks...")
        progress.progress(60)
        embeddings = embed_chunks(chunks)

        # Step 4: Store in Pinecone
        status.text("🌲 Storing in Pinecone...")
        progress.progress(80)
        index = create_index_if_not_exists()
        upsert_chunks(index, chunks, embeddings)

        # Save to session state
        st.session_state.index = index
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.chat_history = []  # reset chat for new PDF

        # Cleanup temp file
        os.unlink(tmp_path)

        progress.progress(100)
        status.text("✅ Ready!")
        st.sidebar.success(f"✅ '{uploaded_file.name}' loaded successfully!")

# ------------------------------------
# Main area — chat interface
# ------------------------------------
if st.session_state.index is not None:
    
    st.markdown(f"**Currently loaded:** `{st.session_state.pdf_name}`")
    st.divider()

    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])

    # Question input
    question = st.chat_input("Ask a question about your PDF...")

    if question:
        # Show user question immediately
        with st.chat_message("user"):
            st.write(question)

        # Generate answer with a spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Embed question
                query_vector = embed_query(question)
                
                # Search Pinecone
                relevant_chunks = search(st.session_state.index, query_vector)
                
                # Generate answer
                answer = generate_answer(question, relevant_chunks)
                
                st.write(answer)

        # Save to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

else:
    # Show instructions if no PDF uploaded yet
    st.info("👈 Upload a PDF from the sidebar to get started!")
    
    st.markdown("""
    ### How to use:
    1. Upload your PDF using the sidebar on the left
    2. Wait for it to process (takes about 30 seconds)
    3. Type your question in the chat box
    4. Get answers instantly!
    """)
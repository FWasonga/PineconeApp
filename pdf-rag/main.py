# main.py
# ------------------------------------
# Main entry point — runs the full
# PDF → Chunk → Embed → Store → Query
# RAG pipeline
# ------------------------------------

from pdf_reader import load_pdf
from text_splitter import split_text
from embedder import embed_chunks, embed_query
from pinecone_store import create_index_if_not_exists, upsert_chunks, search
from query import generate_answer

def ingest_pdf(filepath: str):
    """
    Reads a PDF, splits it, embeds it, and stores it in Pinecone.
    Run this once per PDF to load it into the system.
    
    Args:
        filepath: path to your PDF e.g. "documents/research.pdf"
    """
    print("\n" + "="*50)
    print("📄 INGESTING PDF")
    print("="*50)
    
    # Step 1: Read PDF
    text = load_pdf(filepath)
    
    # Step 2: Split into chunks
    chunks = split_text(text)
    
    # Step 3: Embed chunks
    embeddings = embed_chunks(chunks)
    
    # Step 4: Store in Pinecone
    index = create_index_if_not_exists()
    upsert_chunks(index, chunks, embeddings)
    
    print("\n✅ PDF ingested successfully!\n")
    return index


def ask_question(index, question: str):
    """
    Queries the stored PDF and generates an answer.
    
    Args:
        index: your Pinecone index object
        question: your question as a string
    """
    print("\n" + "="*50)
    print(f"🔍 QUESTION: {question}")
    print("="*50)
    
    # Step 1: Embed the question
    query_vector = embed_query(question)
    
    # Step 2: Find relevant chunks in Pinecone
    print("\n📌 Relevant chunks found:")
    relevant_chunks = search(index, query_vector)
    
    # Step 3: Generate an answer
    print("\n🤖 Generating answer...")
    answer = generate_answer(question, relevant_chunks)
    
    print(f"\n💡 ANSWER: {answer}\n")
    return answer


# ---- RUN IT ----
if __name__ == "__main__":
    
    # 1. Ingest your PDF (run once)
    index = ingest_pdf(r"C:\Users\user\PINECONE\.venv\Analog modulation schemes.pdf")  # ← change this to your PDF path
    
    # 2. Ask questions in a loop
    while True:
        question = input("\nAsk a question (or type 'quit' to exit): ")
        if question.lower() == "quit":
            break
        ask_question(index, question)
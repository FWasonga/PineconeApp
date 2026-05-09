# text_splitter.py
# ------------------------------------
# Splits a large text into smaller
# overlapping chunks ready for embedding
# ------------------------------------

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

def split_text(text: str) -> list[str]:
    """
    Splits text into chunks using RecursiveCharacterTextSplitter.
    
    It tries to split on paragraphs first, then sentences,
    then words — preserving natural language boundaries.
    
    Args:
        text: the full raw text from your PDF
    
    Returns:
        A list of text chunks e.g. ["chunk one...", "chunk two...", ...]
    """
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        
        # Tries these separators in order (paragraph → line → sentence → word)
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    
    print(f"✅ Text split into {len(chunks)} chunks")
    print(f"   Avg chunk size: "
          f"{sum(len(c) for c in chunks) // len(chunks)} characters")
    
    return chunks
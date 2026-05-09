# embedder.py
# ------------------------------------
# Loads a HuggingFace embedding model
# and converts text chunks into vectors
# ------------------------------------

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

# Load the model once at module level
# so it's not reloaded every time you call embed()
print(f"⏳ Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)
print(f"✅ Embedding model loaded")

def embed_chunks(chunks: list[str]):
    """
    Converts a list of text chunks into a list of vectors.
    
    Args:
        chunks: list of text strings
    
    Returns:
        numpy array of shape (num_chunks, 384)
        e.g. 120 chunks → shape (120, 384)
    """
    
    print(f"⏳ Embedding {len(chunks)} chunks...")
    
    embeddings = model.encode(
        chunks,
        show_progress_bar=True  # shows a loading bar
    )
    
    print(f"✅ Embeddings created: shape {embeddings.shape}")
    return embeddings

def embed_query(query: str):
    """
    Embeds a single query string for searching.
    Must use the same model as embed_chunks().
    
    Args:
        query: your search question as a string
    
    Returns:
        A single vector as a Python list (384 numbers)
    """
    return model.encode(query).tolist()
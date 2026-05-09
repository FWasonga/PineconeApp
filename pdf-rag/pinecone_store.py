# pinecone_store.py
# ------------------------------------
# Handles everything Pinecone-related:
# - Creating the index
# - Upserting vectors
# - Querying for similar vectors
# ------------------------------------

from pinecone import Pinecone, ServerlessSpec
from config import (
    PINECONE_API_KEY,
    INDEX_NAME,
    EMBEDDING_DIMENSION,
    TOP_K
)

# Connect to Pinecone using your API key
pc = Pinecone(api_key=PINECONE_API_KEY)

def create_index_if_not_exists():
    """
    Creates a Pinecone index if it doesn't already exist.
    Safe to call every time — won't overwrite existing data.
    """
    
    # Get list of existing index names
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    if INDEX_NAME not in existing_indexes:
        print(f"⏳ Creating Pinecone index: '{INDEX_NAME}'...")
        
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,  # must match your model
            metric="cosine",                # best for text similarity
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"          # free tier region
            )
        )
        print(f"✅ Index '{INDEX_NAME}' created")
    else:
        print(f"✅ Index '{INDEX_NAME}' already exists — skipping creation")
    
    # Return a reference to the index
    return pc.Index(INDEX_NAME)


def upsert_chunks(index, chunks: list[str], embeddings):
    """
    Stores all chunks and their embeddings in Pinecone.
    
    Each vector is stored with:
    - id: unique identifier e.g. "chunk-0"
    - values: the 384-dimensional vector
    - metadata: the original text (so we can retrieve it later)
    
    Args:
        index: your Pinecone index object
        chunks: list of text strings
        embeddings: numpy array of vectors matching the chunks
    """
    
    # Build the list of vectors to upload
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding.tolist(),   # convert numpy → Python list
            "metadata": {
                "text": chunk,              # original text for retrieval
                "chunk_index": i            # position in document
            }
        })
    
    # Upload in batches of 100 (Pinecone best practice)
    batch_size = 100
    total_batches = (len(vectors) + batch_size - 1) // batch_size
    
    print(f"⏳ Upserting {len(vectors)} vectors in "
          f"{total_batches} batches...")
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"   Batch {i // batch_size + 1}/{total_batches} done")
    
    print(f"✅ All chunks stored in Pinecone")


def search(index, query_vector: list, top_k: int = TOP_K) -> list[str]:
    """
    Searches Pinecone for chunks most similar to the query vector.
    
    Args:
        index: your Pinecone index object
        query_vector: embedded query as a Python list
        top_k: number of results to return
    
    Returns:
        List of the most relevant text chunks
    """
    
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True   # needed to get the text back
    )
    
    # Extract just the text from each result
    matched_chunks = []
    for match in results["matches"]:
        text = match["metadata"]["text"]
        score = match["score"]
        matched_chunks.append(text)
        print(f"   Score: {score:.4f} | {text[:60]}...")
    
    return matched_chunks
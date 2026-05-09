# config.py
# ------------------------------------
# Central config file — edit this file
# with your own values before running
# ------------------------------------

# Your Pinecone API key (get it from https://app.pinecone.io)
from dotenv import load_dotenv
load_dotenv()
import os
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "your-pinecone-api-key-here")


# Name of your Pinecone index
INDEX_NAME = "pdf-rag"

# Embedding model from HuggingFace
# This outputs 384-dimensional vectors
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Dimension must match the embedding model output
# all-MiniLM-L6-v2 → 384
EMBEDDING_DIMENSION = 384

# Chunk settings
CHUNK_SIZE = 500       # max characters per chunk
CHUNK_OVERLAP = 50     # characters shared between consecutive chunks

# How many chunks to retrieve when querying
TOP_K = 5

# LLM for generating answers
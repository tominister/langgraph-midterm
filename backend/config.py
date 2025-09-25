import os

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "user_docs")

# Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", 384))

# LLM
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "")  # e.g. https://api.groq.ai/v1/generate
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# Storage
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")

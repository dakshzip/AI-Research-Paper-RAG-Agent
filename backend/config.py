import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")

# Groq LLM
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Embeddings
DENSE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
EMBEDDING_DEVICE = "cpu"

# Reranker
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Retrieval
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "8"))
FINAL_K = int(os.getenv("FINAL_K", "5"))

# Document processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TEXT_PREVIEW_LENGTH = 200

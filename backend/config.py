import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
# Required for managed Qdrant Cloud clusters; empty for a local Docker instance.
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "") or None
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")

# Public demo mode (set PUBLIC_DEMO=1 on the hosted deployment): hides the API-key
# input, document upload, and RAGAS controls so anonymous visitors can only chat
# with the curated corpus.
PUBLIC_DEMO = os.getenv("PUBLIC_DEMO", "").lower() in ("1", "true", "yes")

# Groq LLM
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
# Optional: supply the key via .env so a deployment can auto-connect without the
# user pasting it into the sidebar each session.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Curated corpus folder for the admin seed script (scripts/seed_corpus.py).
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", str(PROJECT_ROOT / "data" / "corpus")))

# RAGAS judge — needs a stronger model than the chat LLM to emit reliable structured
# scores; the small instant model frequently returns unparseable judgments (NaN).
RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "openai/gpt-oss-120b")

# Embedding
DENSE_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_EMBEDDING_MODEL = "Qdrant/bm25"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


def _detect_device() -> str:
    """Pick the fastest available torch device: MPS (Apple) > CUDA > CPU."""
    override = os.getenv("EMBEDDING_DEVICE")
    if override:
        return override
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


EMBEDDING_DEVICE = _detect_device()

# Reranker
RERANKER_MODEL = "BAAI/bge-reranker-base"

# Retrieval
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "8"))
FINAL_K = int(os.getenv("FINAL_K", "5"))

# Document processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TEXT_PREVIEW_LENGTH = 100

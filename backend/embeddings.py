from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_qdrant import FastEmbedSparse

from backend import config


def get_dense_embeddings() -> HuggingFaceBgeEmbeddings:
    """Initialize BGE-large dense embedding model."""
    return HuggingFaceBgeEmbeddings(
        model_name=config.DENSE_EMBEDDING_MODEL,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_sparse_embeddings() -> FastEmbedSparse:
    """Initialize FastEmbed sparse (BM25) embedding model for hybrid search."""
    return FastEmbedSparse(model_name=config.SPARSE_EMBEDDING_MODEL)

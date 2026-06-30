from typing import List

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from backend import config


def get_indexed_sources() -> list[str]:
    """Return sorted list of unique source filenames stored in the Qdrant collection."""
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        names = [c.name for c in client.get_collections().collections]
        if config.QDRANT_COLLECTION not in names:
            return []
        sources: set[str] = set()
        offset = None
        while True:
            results, offset = client.scroll(
                collection_name=config.QDRANT_COLLECTION,
                with_payload=["source"],
                limit=100,
                offset=offset,
            )
            for pt in results:
                if pt.payload and "source" in pt.payload:
                    sources.add(pt.payload["source"])
            if offset is None:
                break
        return sorted(sources)
    except Exception:
        return []


def connect_existing_vectorstore(dense_embeddings, sparse_embeddings) -> QdrantVectorStore:
    """Reconnect to an already-indexed Qdrant collection without re-ingesting."""
    return QdrantVectorStore.from_existing_collection(
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        url=config.QDRANT_URL,
        collection_name=config.QDRANT_COLLECTION,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=config.DENSE_VECTOR_NAME,
        sparse_vector_name=config.SPARSE_VECTOR_NAME,
    )


def check_qdrant_connection() -> tuple[bool, str]:
    """Verify Qdrant is reachable before indexing."""
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        client.get_collections()
        return True, ""
    except Exception as exc:
        return False, (
            f"Cannot connect to Qdrant at {config.QDRANT_URL}. "
            f"Run `docker compose up -d` first. Error: {exc}"
        )


def _recreate_collection(client: QdrantClient) -> None:
    """Drop existing collection so re-processing starts fresh."""
    collections = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION in collections:
        client.delete_collection(config.QDRANT_COLLECTION)


def create_vectorstore(
    chunks: List[Document],
    dense_embeddings,
    sparse_embeddings,
) -> QdrantVectorStore:
    """Index document chunks into Qdrant with hybrid dense+sparse vectors."""
    ok, error = check_qdrant_connection()
    if not ok:
        raise ConnectionError(error)

    client = QdrantClient(url=config.QDRANT_URL)
    _recreate_collection(client)

    return QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        url=config.QDRANT_URL,
        collection_name=config.QDRANT_COLLECTION,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=config.DENSE_VECTOR_NAME,
        sparse_vector_name=config.SPARSE_VECTOR_NAME,
    )

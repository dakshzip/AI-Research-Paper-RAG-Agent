import uuid
from typing import List

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from backend import config


def collection_exists(client: QdrantClient) -> bool:
    """True when the configured collection already exists in Qdrant."""
    names = [c.name for c in client.get_collections().collections]
    return config.QDRANT_COLLECTION in names


def ensure_source_payload_index() -> None:
    """Create a keyword payload index on metadata.source (idempotent, best-effort).

    Lets Qdrant filter by document source efficiently when retrieval is scoped to
    a single paper (see retrieval.build_retriever). Re-creating an existing index
    is a no-op error we swallow; filtering works without the index too, just less
    optimally on large collections.
    """
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        if not collection_exists(client):
            return
        client.create_payload_index(
            collection_name=config.QDRANT_COLLECTION,
            field_name="metadata.source",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass


def _point_id(chunk: Document) -> str:
    """Deterministic point ID so re-indexing the same chunk upserts instead of duplicating."""
    source = chunk.metadata.get("source", "unknown")
    chunk_id = chunk.metadata.get("chunk_id", "")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_id}"))


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
            # QdrantVectorStore nests document metadata under a "metadata" payload
            # key, so the source filename lives at payload["metadata"]["source"].
            results, offset = client.scroll(
                collection_name=config.QDRANT_COLLECTION,
                with_payload=["metadata.source"],
                limit=100,
                offset=offset,
            )
            for pt in results:
                metadata = (pt.payload or {}).get("metadata") or {}
                source = metadata.get("source")
                if source:
                    sources.add(source)
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


def reset_collection(client: QdrantClient) -> None:
    """Drop the collection so the next ingest rebuilds from scratch (admin/opt-in only)."""
    if collection_exists(client):
        client.delete_collection(config.QDRANT_COLLECTION)


def upsert_documents(
    chunks: List[Document],
    dense_embeddings,
    sparse_embeddings,
) -> QdrantVectorStore:
    """Add chunks to the Qdrant collection, creating it on first use.

    Additive by design: existing documents are preserved so a curated corpus and
    later uploads accumulate in the same index. Deterministic point IDs make
    re-indexing the same chunk an idempotent upsert rather than a duplicate.
    """
    ok, error = check_qdrant_connection()
    if not ok:
        raise ConnectionError(error)

    ids = [_point_id(chunk) for chunk in chunks]
    client = QdrantClient(url=config.QDRANT_URL)

    if collection_exists(client):
        vectorstore = connect_existing_vectorstore(dense_embeddings, sparse_embeddings)
        vectorstore.add_documents(chunks, ids=ids)
        return vectorstore

    return QdrantVectorStore.from_documents(
        documents=chunks,
        ids=ids,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        url=config.QDRANT_URL,
        collection_name=config.QDRANT_COLLECTION,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=config.DENSE_VECTOR_NAME,
        sparse_vector_name=config.SPARSE_VECTOR_NAME,
    )

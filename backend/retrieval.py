from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models

from backend import config


def get_reranker_model() -> HuggingFaceCrossEncoder:
    """Load the cross-encoder reranker model (call once and cache at the app layer)."""
    return HuggingFaceCrossEncoder(
        model_name=config.RERANKER_MODEL,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
    )


def build_retriever(
    vectorstore: QdrantVectorStore,
    reranker_model: HuggingFaceCrossEncoder | None = None,
    source_filter: str | None = None,
) -> ContextualCompressionRetriever:
    """Build hybrid retriever with cross-encoder reranking.

    When source_filter is set, retrieval is scoped to a single document via a
    Qdrant payload filter on metadata.source. This is a pure narrowing: a filtered
    search is never slower than the unfiltered one, and keeping the whole corpus
    searchable is just source_filter=None (the default).
    """
    if reranker_model is None:
        reranker_model = get_reranker_model()
    search_kwargs = {"k": config.RETRIEVE_K}
    if source_filter:
        search_kwargs["filter"] = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=source_filter),
                )
            ]
        )
    base_retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    compressor = CrossEncoderReranker(model=reranker_model, top_n=config.FINAL_K)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

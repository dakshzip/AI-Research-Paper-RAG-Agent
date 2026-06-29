from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_qdrant import QdrantVectorStore

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
) -> ContextualCompressionRetriever:
    """Build hybrid retriever with cross-encoder reranking."""
    if reranker_model is None:
        reranker_model = get_reranker_model()
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVE_K})
    compressor = CrossEncoderReranker(model=reranker_model, top_n=config.FINAL_K)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

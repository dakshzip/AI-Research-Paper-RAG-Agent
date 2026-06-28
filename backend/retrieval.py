from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_qdrant import QdrantVectorStore

from backend import config


def build_retriever(vectorstore: QdrantVectorStore) -> ContextualCompressionRetriever:
    """Build hybrid retriever with cross-encoder reranking."""
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVE_K})
    reranker_model = HuggingFaceCrossEncoder(
        model_name=config.RERANKER_MODEL,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
    )
    compressor = CrossEncoderReranker(
        model=reranker_model,
        top_n=config.FINAL_K,
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )

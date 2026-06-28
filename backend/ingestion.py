import os
import tempfile
from typing import List

from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from backend import config


def _enrich_chunk_metadata(chunks: List[Document]) -> List[Document]:
    """Add citation-friendly metadata to each chunk."""
    enriched = []
    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        filename = os.path.basename(source) if source != "unknown" else "unknown"
        page = chunk.metadata.get("page", None)
        chunk.metadata["source"] = filename
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["text_preview"] = chunk.page_content[: config.TEXT_PREVIEW_LENGTH]
        enriched.append(chunk)
    return enriched


def load_and_split_documents(uploaded_files) -> List[Document]:
    """Load PDF/TXT uploads, split into chunks, and enrich metadata for citations."""
    documents: List[Document] = []

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(tmp_path)
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = uploaded_file.name
                documents.extend(loaded)
            elif file_extension == ".txt":
                loader = TextLoader(tmp_path)
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source"] = uploaded_file.name
                documents.extend(loaded)
            else:
                continue
        finally:
            os.remove(tmp_path)

    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return _enrich_chunk_metadata(chunks)

import os
import tempfile
from typing import List

from langchain_core.documents import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from backend import config


def _enrich_chunk_metadata(chunks: List[Document]) -> List[Document]:
    """Add citation-friendly metadata to each chunk.

    chunk_id is namespaced by source so IDs stay unique across documents, which
    keeps deterministic point IDs (see vectorstore._point_id) collision-free when
    the corpus is built up additively.
    """
    per_source_idx: dict[str, int] = {}
    enriched = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        filename = os.path.basename(source) if source != "unknown" else "unknown"
        idx = per_source_idx.get(filename, 0)
        per_source_idx[filename] = idx + 1
        chunk.metadata["source"] = filename
        chunk.metadata["chunk_id"] = f"{filename}:{idx}"
        chunk.metadata["text_preview"] = chunk.page_content[: config.TEXT_PREVIEW_LENGTH]
        enriched.append(chunk)
    return enriched


def _split_and_enrich(documents: List[Document]) -> List[Document]:
    """Split loaded documents into chunks and attach citation metadata."""
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    return _enrich_chunk_metadata(chunks)


def _load_file(path: str, source_name: str) -> List[Document]:
    """Load a single PDF/TXT file from disk, tagging each doc with its source name."""
    file_extension = os.path.splitext(path)[1].lower()
    if file_extension == ".pdf":
        loaded = PyPDFLoader(path).load()
    elif file_extension == ".txt":
        loaded = TextLoader(path).load()
    else:
        return []

    for doc in loaded:
        doc.metadata["source"] = source_name
    return loaded


def load_and_split_from_paths(paths: List[str]) -> List[Document]:
    """Load PDF/TXT files from disk paths, split, and enrich metadata for citations."""
    documents: List[Document] = []
    for path in paths:
        documents.extend(_load_file(path, os.path.basename(path)))
    return _split_and_enrich(documents)


def load_and_split_documents(uploaded_files) -> List[Document]:
    """Load PDF/TXT uploads, split into chunks, and enrich metadata for citations."""
    documents: List[Document] = []

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            documents.extend(_load_file(tmp_path, uploaded_file.name))
        finally:
            os.remove(tmp_path)

    return _split_and_enrich(documents)

# SCA using RAG

A document Q&A chatbot powered by **Retrieval-Augmented Generation (RAG)**. Upload PDF or TXT files, ask questions, and get answers grounded in your documents—with source citations, conversational follow-ups, and optional quality scoring.

## Features

| Feature | Description |
|---------|-------------|
| **Qdrant vector store** | Persistent hybrid-indexed storage via Docker (replaces in-memory FAISS) |
| **BGE-large embeddings** | Dense semantic vectors (`BAAI/bge-large-en-v1.5`) |
| **Hybrid search** | Combines dense (BGE) + sparse (BM25) retrieval in Qdrant |
| **Reranking** | Cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) refines top results |
| **Source citations** | Expandable source panel with filename, page, and snippet |
| **Conversation history** | History-aware query rewriting for follow-up questions |
| **RAGAS evaluation** | Optional faithfulness and answer-relevancy scores (sidebar toggle) |

## Architecture

```mermaid
flowchart TD
    upload[User uploads PDF/TXT] --> ingest[Ingestion: split + metadata]
    ingest --> embed[Dense BGE-large + sparse BM25 vectors]
    embed --> qdrant[(Qdrant Docker)]
    question[User question + chat history] --> contextualize[History-aware query rewrite]
    contextualize --> hybrid[Hybrid search dense+sparse]
    hybrid --> qdrant
    qdrant --> rerank[BGE reranker top-k]
    rerank --> llm[Groq Llama 3.1 answer]
    llm --> citations[UI: answer + source expander]
    llm --> ragasOpt{RAGAS toggle on?}
    ragasOpt -->|yes| ragas[RAGAS scores in sidebar]
    ragasOpt -->|no| done[Done]
    ragas --> done
```

## Project Structure

```
SCA-using-RAG/
├── README.md
├── requirements.txt
├── docker-compose.yml        # Local Qdrant
├── .env.example
├── prompts/
│   └── rag_system.txt        # System + contextualize prompts
├── frontend/
│   └── app.py                # Streamlit UI
└── backend/
    ├── config.py             # Models, retrieval knobs, env vars
    ├── embeddings.py         # BGE-large + FastEmbed sparse
    ├── ingestion.py          # Document loading and chunking
    ├── vectorstore.py        # Qdrant hybrid indexing
    ├── retrieval.py          # Hybrid retriever + reranker
    ├── rag_chain.py          # History-aware RAG chain
    └── evaluation/
        └── ragas_eval.py     # RAGAS metrics
```

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- [Groq API key](https://console.groq.com/)

## Quick Start

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd "SCA using RAG"
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env and set GROQ_API_KEY
   ```

4. **Start Qdrant**

   ```bash
   docker compose up -d
   ```

   Qdrant dashboard: http://localhost:6333/dashboard

5. **Run the app**

   ```bash
   streamlit run frontend/app.py
   ```

6. **Use the app**

   - Enter your Groq API key in the sidebar (or set `GROQ_API_KEY` in `.env`)
   - Upload PDF/TXT files
   - Click **Process Documents and Start Chat**
   - Ask questions; expand **Sources** under each answer
   - Optionally enable **RAGAS evaluation** for quality scores

## Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key for LLM and RAGAS judge |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | `rag_documents` | Collection name |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Groq model ID |
| `RETRIEVE_K` | `20` | Chunks retrieved before reranking |
| `FINAL_K` | `5` | Chunks sent to LLM after reranking |

Tune chunking and models in `backend/config.py`.

## How Each Enhancement Works

**Qdrant (vs FAISS)** — Vectors persist in Docker; collections support named dense and sparse vectors for hybrid search.

**BGE-large** — Stronger semantic embeddings than BGE-small; ~1.3 GB download on first run (CPU).

**Hybrid search** — Dense vectors capture meaning; BM25 sparse vectors match keywords. Qdrant fuses both via reciprocal rank fusion (RRF).

**Reranking** — Retrieves 20 candidates, then the BGE cross-encoder reranker picks the top 5 most relevant chunks for the LLM.

**Source citations** — Each chunk stores filename, page, and preview text; the UI shows deduplicated sources per answer.

**Conversation history** — Follow-ups like “tell me more about that” are rewritten into standalone questions before retrieval.

**RAGAS** — When enabled, each answer is scored for faithfulness (grounded in context) and answer relevancy (matches the question).

## Limitations

- First run downloads embedding and reranker models (several GB total).
- Embedding and reranking on CPU can be slow for large documents.
- Re-processing documents recreates the Qdrant collection (previous index is replaced).
- RAGAS adds ~3–8 seconds per query when enabled.

## Tech Stack

- **UI:** Streamlit
- **LLM:** Groq (Llama 3.1)
- **Vector DB:** Qdrant
- **Embeddings:** BGE-large (dense), FastEmbed BM25 (sparse)
- **Reranker:** BGE-reranker-v2-m3
- **Framework:** LangChain
- **Evaluation:** RAGAS

## License

MIT (adjust as needed for your repository).

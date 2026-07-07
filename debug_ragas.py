"""Throwaway repro to surface the real RAGAS judge failure.

Run:  GROQ_API_KEY=sk-... python3 debug_ragas.py
"""
import os

from backend.embeddings import get_dense_embeddings
from backend.evaluation.ragas_eval import evaluate_rag_response

key = os.environ["GROQ_API_KEY"]
emb = get_dense_embeddings()

out = evaluate_rag_response(
    groq_api_key=key,
    question="What is retrieval-augmented generation?",
    answer="RAG combines a retriever that fetches relevant documents with a "
    "generator LLM that conditions its answer on them.",
    contexts=[
        "Retrieval-augmented generation (RAG) augments an LLM with documents "
        "fetched from an external store at query time.",
    ],
    dense_embeddings=emb,
)
print(out)

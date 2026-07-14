# Hugging Face Spaces (Docker SDK) image for the RAG chatbot.
# The Space provides secrets as env vars: GROQ_API_KEY, QDRANT_URL,
# QDRANT_API_KEY, PUBLIC_DEMO=1.
FROM python:3.12-slim

# Spaces run the container as user 1000 with a writable /home/user only.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    EMBEDDING_DEVICE=cpu

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
USER user
# CPU-only torch first: the default wheel bundles ~6GB of CUDA libraries that a
# CPU Space never uses.
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

# Bake the embedding, reranker, and BM25 models into the image so a cold start
# serves immediately instead of downloading ~1.3GB first.
RUN python -c "\
from langchain_community.embeddings import HuggingFaceBgeEmbeddings; \
from langchain_community.cross_encoders import HuggingFaceCrossEncoder; \
from langchain_qdrant import FastEmbedSparse; \
HuggingFaceBgeEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}); \
HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-base', model_kwargs={'device': 'cpu'}); \
FastEmbedSparse(model_name='Qdrant/bm25')"

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port", "7860", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true"]

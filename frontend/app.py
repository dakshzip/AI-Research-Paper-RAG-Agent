import html
import sys
from pathlib import Path

import markdown as md
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from backend import config  # noqa: E402
from backend.embeddings import get_dense_embeddings, get_sparse_embeddings  # noqa: E402
from backend.evaluation.ragas_eval import evaluate_rag_response  # noqa: E402
from backend.ingestion import load_and_split_documents  # noqa: E402
from backend.rag_chain import create_rag_chain  # noqa: E402
from backend.retrieval import build_retriever, get_reranker_model  # noqa: E402
from backend.vectorstore import (  # noqa: E402
    check_qdrant_connection,
    connect_existing_vectorstore,
    ensure_source_payload_index,
    get_indexed_sources,
    upsert_documents,
)

# Sidebar "Focus" option that keeps the whole corpus searchable (no source filter).
ALL_PAPERS = "All papers"


# Marker the model emits when the documents don't answer the question (see prompts/rag_system.txt).
NOT_IN_DOCS_MARKER = "This cannot be answered from the information provided in your documents."


def _answer_grounded_in_docs(answer: str) -> bool:
    """False when the model declared the answer is not supported by the documents."""
    return NOT_IN_DOCS_MARKER.lower() not in answer.lower()


# Assistant answers use Markdown (comparison tables, headings, lists, code). Render it to
# HTML so it displays inside the styled chat bubble instead of as raw text.
_MD_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "nl2br"]


def _ai_html(text: str) -> str:
    """Convert an assistant Markdown answer to HTML for the chat bubble."""
    return md.markdown(text, extensions=_MD_EXTENSIONS)


def _format_sources(documents: list[Document]) -> list[dict]:
    """Deduplicate and format retrieved documents for citation UI."""
    seen: set[tuple] = set()
    sources: list[dict] = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": source,
                "page": page,
                "preview": doc.metadata.get("text_preview", doc.page_content[:200]),
            }
        )
    return sources


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for idx, src in enumerate(sources, start=1):
            page_label = f", page {src['page'] + 1}" if src.get("page") is not None else ""
            st.markdown(f"**{idx}. {src['source']}**{page_label}")
            st.caption(src["preview"])


def _chain_chat_history() -> list:
    """Return LangChain messages for RAG chain (exclude welcome message)."""
    history = []
    for msg in st.session_state.get("messages", []):
        if msg["role"] == "human":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai" and not msg.get("is_welcome"):
            history.append(AIMessage(content=msg["content"]))
    return history


def _friendly_request_error(exc: Exception, api_key: str = "") -> str:
    """Return a useful, bounded error message without exposing the API key."""
    message = " ".join(str(exc).split()) or "No additional details were provided."
    if api_key:
        message = message.replace(api_key, "[redacted]")

    lowered = message.lower()
    if "401" in lowered or "authentication" in lowered or "invalid_api_key" in lowered:
        return "Groq rejected the API key. Check that the key is complete and still active."
    if "429" in lowered or "rate limit" in lowered:
        return "Groq's rate limit was reached. Wait briefly, then try again."
    if "context_length" in lowered or "request too large" in lowered:
        return "The request contained too much text. Start a shorter chat or retrieve fewer chunks."
    if "timed out" in lowered or "timeout" in lowered:
        return "The request timed out. Check your connection and try again."
    if "connection" in lowered:
        return f"A service connection failed: {message[:420]}"
    return f"{type(exc).__name__}: {message[:500]}"


@st.cache_resource
def _load_dense_embeddings():
    return get_dense_embeddings()


@st.cache_resource
def _load_sparse_embeddings():
    return get_sparse_embeddings()


@st.cache_resource
def _load_reranker_model():
    return get_reranker_model()


@st.cache_data(ttl=300)
def _cached_sources() -> list[str]:
    """Cache the indexed-source list: get_indexed_sources() scrolls the whole
    collection (~130 round-trips at 13k points), far too slow to repeat on every
    Streamlit rerun — especially against a remote Qdrant. New uploads call
    _cached_sources.clear() to refresh immediately."""
    return get_indexed_sources()


def _build_chain(source_filter: str | None) -> None:
    """(Re)build the retriever + RAG chain from cached components in session state.

    Cheap enough to call on every focus change: the vectorstore, reranker, and LLM
    are already warm, so only the retriever wiring and prompt chain are rebuilt.
    Does not touch chat history, so switching focus mid-conversation is seamless.
    """
    retriever = build_retriever(
        st.session_state.vectorstore,
        st.session_state.reranker,
        source_filter=source_filter,
    )
    st.session_state.rag_chain = create_rag_chain(st.session_state.groq_llm, retriever)


def _init_rag_chain(groq_api_key: str, uploaded_files=None) -> str | None:
    """Build the RAG chain and store it in session state.

    When uploaded_files is provided, their chunks are added to the persistent
    index (additive — existing documents are kept). Otherwise the app connects to
    the already-indexed collection. Returns an error string on failure, else None.
    """
    dense = _load_dense_embeddings()
    sparse = _load_sparse_embeddings()
    reranker = _load_reranker_model()

    if uploaded_files:
        chunks = load_and_split_documents(uploaded_files)
        if not chunks:
            return "No supported documents were loaded."
        vectorstore = upsert_documents(chunks, dense, sparse)
        welcome = "Documents processed. Ask me anything about them!"
    else:
        vectorstore = connect_existing_vectorstore(dense, sparse)
        welcome = "Connected to indexed documents. Ask me anything!"

    # Enable efficient single-paper filtering for the Focus dropdown.
    ensure_source_payload_index()

    st.session_state.vectorstore = vectorstore
    st.session_state.reranker = reranker
    st.session_state.groq_llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=config.GROQ_MODEL,
        reasoning_effort="none",  # disable Qwen3 thinking mode (Groq API)
    )
    st.session_state.focus_label = ALL_PAPERS
    _build_chain(source_filter=None)

    st.session_state.groq_api_key = groq_api_key
    st.session_state.documents_processed = True
    st.session_state.messages = [
        {"role": "ai", "content": welcome, "sources": [], "is_welcome": True}
    ]
    st.session_state.last_ragas_scores = None
    return None


def main() -> None:
    st.set_page_config(page_title="AI Research Paper Query-er", layout="wide")

    st.markdown(
        """
        <style>
            :root {
                --ink: #e8edfb;
                --muted: #93a1c4;
                --blue: #3b82f6;
                --blue-bright: #60a5fa;
                --red: #f43f5e;
                --red-bright: #fb7185;
                --bg: #070b18;
                --surface: #0e1530;
                --surface-soft: #131c3d;
                --line: #24305c;
            }

            .stApp {
                color: var(--ink);
                color-scheme: dark;
                background:
                    radial-gradient(circle at 8% 4%, rgba(244, 63, 94, .16), transparent 28rem),
                    radial-gradient(circle at 95% 8%, rgba(59, 130, 246, .2), transparent 32rem),
                    radial-gradient(circle at 50% 110%, rgba(76, 29, 149, .25), transparent 40rem),
                    linear-gradient(145deg, #0a0714 0%, #070b18 45%, #060d20 100%);
            }

            [data-testid="stHeader"] { background: transparent; }

            [data-testid="stAppViewContainer"] > .main .block-container {
                max-width: 1120px;
                padding-top: 2.2rem;
                padding-bottom: 3rem;
            }

            [data-testid="stAppViewContainer"] > .main {
                overflow-y: scroll !important;
                scrollbar-color: var(--blue) #131c3d;
                scrollbar-width: auto;
            }

            [data-testid="stAppViewContainer"] > .main::-webkit-scrollbar,
            [data-testid="stVerticalBlockBorderWrapper"] ::-webkit-scrollbar {
                width: 12px;
            }

            [data-testid="stAppViewContainer"] > .main::-webkit-scrollbar-track,
            [data-testid="stVerticalBlockBorderWrapper"] ::-webkit-scrollbar-track {
                background: #131c3d;
            }

            [data-testid="stAppViewContainer"] > .main::-webkit-scrollbar-thumb,
            [data-testid="stVerticalBlockBorderWrapper"] ::-webkit-scrollbar-thumb {
                border: 3px solid #131c3d;
                border-radius: 999px;
                background: linear-gradient(180deg, var(--red), var(--blue));
            }

            h1, h2, h3, h4, h5, h6,
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stCaptionContainer"] {
                color: var(--ink);
            }

            .hero {
                position: relative;
                overflow: hidden;
                margin-bottom: 1.5rem;
                padding: 2.25rem 2.4rem;
                border: 1px solid rgba(96, 165, 250, .35);
                border-radius: 24px;
                color: #ffffff;
                background:
                    radial-gradient(circle at 85% 20%, rgba(96, 165, 250, .35), transparent 22rem),
                    linear-gradient(120deg, #4c0519 0%, #9f1239 28%, #312e81 65%, #172554 100%);
                box-shadow:
                    0 0 0 1px rgba(255, 255, 255, .04),
                    0 24px 60px rgba(2, 6, 23, .8),
                    0 0 80px rgba(244, 63, 94, .12);
            }

            .hero::after {
                content: "";
                position: absolute;
                width: 230px;
                height: 230px;
                right: -70px;
                top: -110px;
                border: 38px solid rgba(255, 255, 255, .08);
                border-radius: 50%;
            }

            .hero-kicker {
                margin-bottom: .7rem;
                font-size: .74rem;
                font-weight: 800;
                letter-spacing: .14em;
                text-transform: uppercase;
                background: linear-gradient(90deg, var(--red-bright), var(--blue-bright));
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
            }

            .hero h1 {
                margin: 0 0 .65rem;
                font-size: clamp(2rem, 5vw, 3.35rem);
                line-height: 1.05;
                letter-spacing: -.04em;
                color: #ffffff !important;
                text-shadow: 0 0 40px rgba(96, 165, 250, .35);
            }

            .hero p {
                max-width: 720px;
                margin: 0;
                font-size: 1.02rem;
                line-height: 1.65;
                color: #cbd7f5 !important;
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid #1e2a55;
                background: linear-gradient(180deg, #05091a 0%, #0a1233 55%, #26081a 100%);
            }

            [data-testid="stSidebar"] * { color: #e8edfb !important; }
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #93a1c4; }
            [data-testid="stSidebar"] hr { border-color: rgba(148, 163, 216, .18); }

            [data-testid="stSidebar"] input,
            [data-testid="stSidebar"] textarea {
                color: #e8edfb !important;
                background: #0e1530 !important;
                border: 1px solid #2c3a6e !important;
                -webkit-text-fill-color: #e8edfb !important;
            }

            [data-testid="stSidebar"] input::placeholder,
            [data-testid="stSidebar"] textarea::placeholder {
                color: #6c7a9e !important;
                -webkit-text-fill-color: #6c7a9e !important;
            }

            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
                color: var(--ink);
                border-color: #2c3a6e;
                background: rgba(14, 21, 48, .92);
            }

            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
                color: var(--ink) !important;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
                color: #ffffff !important;
                background: var(--blue) !important;
                -webkit-text-fill-color: #ffffff !important;
            }

            .stButton > button,
            [data-testid="stFormSubmitButton"] > button {
                min-height: 2.8rem;
                border: 0;
                border-radius: 12px;
                font-weight: 750;
                color: #ffffff !important;
                background: linear-gradient(105deg, #e11d48, #2563eb);
                box-shadow: 0 8px 24px rgba(37, 99, 235, .35), 0 0 32px rgba(225, 29, 72, .18);
                transition: transform .16s ease, box-shadow .16s ease;
            }

            .stButton > button:hover,
            [data-testid="stFormSubmitButton"] > button:hover {
                color: #ffffff !important;
                transform: translateY(-1px);
                box-shadow: 0 11px 30px rgba(37, 99, 235, .5), 0 0 40px rgba(225, 29, 72, .28);
            }

            .stButton > button *,
            [data-testid="stFormSubmitButton"] > button * {
                color: #ffffff !important;
            }

            .stButton > button:focus {
                outline: 3px solid rgba(96, 165, 250, .6);
                outline-offset: 2px;
            }

            [data-testid="stAlert"] {
                border: 1px solid #2c3a6e;
                border-left: 5px solid var(--blue);
                border-radius: 14px;
                background: rgba(14, 21, 48, .92);
                box-shadow: 0 8px 24px rgba(2, 6, 23, .5);
            }

            [data-testid="stAlert"] * { color: var(--ink) !important; }

            [data-testid="stVerticalBlockBorderWrapper"] {
                border-color: #24305c !important;
                border-radius: 20px !important;
                background: rgba(14, 21, 48, .6);
                box-shadow: 0 16px 42px rgba(2, 6, 23, .55);
            }

            .chat-message-user {
                max-width: 82%;
                margin: .8rem 0 .8rem auto;
                padding: .95rem 1.1rem;
                border: 1px solid rgba(96, 165, 250, .45);
                border-radius: 18px 18px 5px 18px;
                color: #ffffff;
                line-height: 1.55;
                background: linear-gradient(135deg, #1e3a8a, #2563eb);
                box-shadow: 0 8px 24px rgba(37, 99, 235, .3);
            }

            .chat-message-ai {
                max-width: 86%;
                margin: .8rem auto .8rem 0;
                padding: .95rem 1.1rem;
                border: 1px solid rgba(244, 63, 94, .35);
                border-radius: 18px 18px 18px 5px;
                color: var(--ink);
                line-height: 1.55;
                background: linear-gradient(135deg, #131c3d, #251129);
                box-shadow: 0 8px 24px rgba(159, 18, 57, .22);
            }

            .chat-message-user strong,
            .chat-message-ai strong {
                display: block;
                margin-bottom: .28rem;
                font-size: .72rem;
                letter-spacing: .08em;
                text-transform: uppercase;
                color: inherit;
            }

            /* Markdown rendered inside assistant bubbles */
            .chat-message-ai p { margin: 0 0 .6rem; }
            .chat-message-ai p:last-child { margin-bottom: 0; }
            .chat-message-ai h1, .chat-message-ai h2, .chat-message-ai h3 {
                margin: .6rem 0 .4rem;
                font-size: 1.02rem;
                color: var(--blue-bright);
            }
            .chat-message-ai ul, .chat-message-ai ol { margin: .2rem 0 .6rem 1.2rem; }
            .chat-message-ai li { margin: .15rem 0; }
            .chat-message-ai code {
                padding: .08rem .3rem;
                border-radius: 6px;
                font-size: .88em;
                color: #93c5fd;
                background: rgba(59, 130, 246, .16);
            }
            .chat-message-ai pre {
                padding: .7rem .9rem;
                border-radius: 10px;
                overflow-x: auto;
                border: 1px solid #24305c;
                background: #060a16;
            }
            .chat-message-ai pre code { background: transparent; color: #dbe6ff; }
            .chat-message-ai table {
                width: 100%;
                margin: .5rem 0;
                border-collapse: collapse;
                font-size: .92rem;
            }
            .chat-message-ai th, .chat-message-ai td {
                padding: .4rem .6rem;
                border: 1px solid #3a2144;
                text-align: left;
                vertical-align: top;
            }
            .chat-message-ai th { background: rgba(244, 63, 94, .14); font-weight: 700; }
            .chat-message-ai tbody tr:nth-child(even) td { background: rgba(255, 255, 255, .03); }

            [data-testid="stChatInput"] {
                border: 1px solid #2c3a6e;
                border-radius: 16px;
                color: var(--ink) !important;
                background: #0e1530 !important;
                box-shadow: 0 10px 28px rgba(2, 6, 23, .6), 0 0 24px rgba(59, 130, 246, .1);
                overflow: hidden;
            }

            [data-testid="stChatInput"]:focus-within {
                border-color: var(--blue);
                box-shadow: 0 10px 28px rgba(2, 6, 23, .6), 0 0 32px rgba(59, 130, 246, .25);
            }

            [data-testid="stChatInput"] > div,
            [data-testid="stChatInput"] textarea,
            [data-testid="stChatInputTextArea"] {
                color: var(--ink) !important;
                caret-color: var(--blue-bright) !important;
                background: #0e1530 !important;
                -webkit-text-fill-color: var(--ink) !important;
            }

            [data-testid="stChatInput"] textarea::placeholder,
            [data-testid="stChatInputTextArea"]::placeholder {
                color: #6c7a9e !important;
                opacity: 1;
                -webkit-text-fill-color: #6c7a9e !important;
            }

            [data-testid="stChatInputSubmitButton"] {
                color: var(--blue-bright) !important;
                background: #0e1530 !important;
            }

            [data-testid="stBottom"],
            [data-testid="stBottomBlockContainer"] {
                background: transparent !important;
                box-shadow: none !important;
            }

            [data-testid="stExpander"] {
                border-color: #24305c;
                border-radius: 12px;
                background: rgba(14, 21, 48, .8);
            }

            @media (max-width: 700px) {
                [data-testid="stAppViewContainer"] > .main .block-container { padding-top: 1rem; }
                .hero { padding: 1.6rem 1.35rem; border-radius: 18px; }
                .chat-message-user, .chat-message-ai { max-width: 94%; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">Document intelligence</div>
            <h1>SCA RAG Chatbot</h1>
            <p>Ask precise questions and get grounded answers from your own documents,
            complete with source citations and optional quality scoring.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "last_ragas_scores" not in st.session_state:
        st.session_state.last_ragas_scores = None
    if "auto_connect_attempted" not in st.session_state:
        st.session_state.auto_connect_attempted = False

    with st.sidebar:
        st.header("Setup")

        if config.PUBLIC_DEMO:
            # Hosted demo: the key stays server-side (never rendered into the
            # page), and visitors cannot upload into the shared index or spend
            # tokens on RAGAS runs.
            groq_api_key = config.GROQ_API_KEY
            uploaded_files = None
            enable_ragas = False
            st.caption("Public demo — chat with the curated corpus of AI research papers.")
        else:
            groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                value=config.GROQ_API_KEY,
                help="Get your key from the Groq console, or set GROQ_API_KEY in .env.",
            )

            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload PDF or TXT files",
                type=["pdf", "txt"],
                accept_multiple_files=True,
            )

            enable_ragas = st.checkbox(
                "Enable RAGAS evaluation",
                value=False,
                help="Score each answer for relevancy to the question (adds a few seconds).",
            )

        qdrant_ok, qdrant_error = check_qdrant_connection()
        if qdrant_ok:
            st.success("Qdrant connected")
        else:
            st.error(qdrant_error)

        existing_sources = _cached_sources() if qdrant_ok else []

        # Auto-connect to the persistent index on load so a served corpus is
        # queryable without any upload. Runs once per session when documents are
        # already indexed and a key is available (env or entered above).
        if (
            not st.session_state.documents_processed
            and not st.session_state.auto_connect_attempted
            and existing_sources
            and groq_api_key
        ):
            st.session_state.auto_connect_attempted = True
            with st.spinner("Connecting to indexed documents..."):
                error = _init_rag_chain(groq_api_key)
            if error:
                st.error(error)
            else:
                st.rerun()

        if not config.PUBLIC_DEMO and st.button(
            "Process Documents and Start Chat", use_container_width=True
        ):
            if not groq_api_key:
                st.error("Please enter your Groq API Key.")
            elif not uploaded_files:
                st.error("Please upload at least one file.")
            elif not qdrant_ok:
                st.error(qdrant_error)
            else:
                with st.spinner("Processing documents (embedding + indexing)..."):
                    try:
                        error = _init_rag_chain(groq_api_key, uploaded_files=uploaded_files)
                        if error:
                            st.error(error)
                        else:
                            _cached_sources.clear()
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Processing failed: {exc}")

        if existing_sources and not st.session_state.documents_processed:
            if st.button("Resume with existing documents", use_container_width=True):
                if not groq_api_key:
                    st.error("Please enter your Groq API Key.")
                else:
                    with st.spinner("Reconnecting to existing index..."):
                        error = _init_rag_chain(groq_api_key)
                    if error:
                        st.error(error)
                    else:
                        st.rerun()

        if st.session_state.last_ragas_scores:
            st.markdown("---")
            st.subheader("Latest RAGAS Score")
            scores = st.session_state.last_ragas_scores
            if scores.get("error"):
                st.warning(scores["error"])
            else:
                if scores.get("faithfulness") is not None:
                    st.metric("Faithfulness", f"{scores['faithfulness']:.2f}")
                if scores.get("answer_relevancy") is not None:
                    st.metric("Answer Relevancy", f"{scores['answer_relevancy']:.2f}")

        if st.session_state.documents_processed and existing_sources:
            st.markdown("---")
            st.subheader("Focus")
            options = [ALL_PAPERS] + existing_sources
            current = st.session_state.get("focus_label", ALL_PAPERS)
            if current not in options:
                current = ALL_PAPERS
            choice = st.selectbox(
                "Scope answers to one paper",
                options,
                index=options.index(current),
                help=(
                    "Restrict retrieval to a single paper for sharper answers during a "
                    "multi-question session. Leave on 'All papers' to search everything "
                    "(same behavior and speed as before)."
                ),
            )
            if choice != current:
                st.session_state.focus_label = choice
                _build_chain(source_filter=None if choice == ALL_PAPERS else choice)
                st.rerun()

        st.markdown("---")
        st.subheader("Indexed Documents")
        if existing_sources:
            st.caption(
                f"{len(existing_sources)} document(s) in the database — "
                "available from previous sessions, no need to re-upload."
            )
            for name in existing_sources:
                st.markdown(f"📄 {name}")
        else:
            st.caption("No documents indexed yet. Upload files to get started.")

    if not st.session_state.documents_processed:
        st.info(
            "Start Qdrant (`docker compose up -d`), then either upload documents and click "
            "**Process Documents and Start Chat**, or wait for auto-connect if documents are "
            "already indexed."
        )
        st.stop()

    st.success("Documents processed. You can now chat.")

    chat_container = st.container(height=500, border=True)
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "human":
                st.markdown(
                    f'<div class="chat-message-user"><strong>You</strong>{html.escape(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )
            elif msg["role"] == "ai":
                st.markdown(
                    f'<div class="chat-message-ai"><strong>Assistant</strong>{_ai_html(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )
                if msg.get("sources"):
                    _render_sources(msg["sources"])
        st.markdown('<div id="chat-scroll-anchor"></div>', unsafe_allow_html=True)

    with st.container():
        user_query = st.chat_input("Ask a question about your documents...")

    if st.session_state.pop("scroll_to_latest", False):
        components.html(
            """
            <script>
                window.requestAnimationFrame(() => {
                    const anchor = window.parent.document.getElementById("chat-scroll-anchor");
                    if (anchor) {
                        anchor.scrollIntoView({ behavior: "smooth", block: "end" });
                    }
                });
            </script>
            """,
            height=0,
        )

    if user_query and st.session_state.rag_chain:
        st.session_state.messages.append({"role": "human", "content": user_query})

        context_docs: list = []
        answer_parts: list[str] = []

        with chat_container:
            st.markdown(
                f'<div class="chat-message-user"><strong>You</strong>{html.escape(user_query)}</div>',
                unsafe_allow_html=True,
            )
            stream_placeholder = st.empty()
            try:
                for chunk in st.session_state.rag_chain.stream(
                    {
                        "input": user_query,
                        "chat_history": _chain_chat_history()[:-1],
                    }
                ):
                    if "context" in chunk:
                        context_docs = chunk["context"]
                    if chunk.get("answer"):
                        answer_parts.append(chunk["answer"])
                        stream_placeholder.markdown(
                            f'<div class="chat-message-ai"><strong>Assistant</strong>'
                            f'{_ai_html("".join(answer_parts))}</div>',
                            unsafe_allow_html=True,
                        )

                ai_answer = "".join(answer_parts)
                sources = (
                    _format_sources(context_docs)
                    if _answer_grounded_in_docs(ai_answer)
                    else []
                )
                st.session_state.messages.append(
                    {
                        "role": "ai",
                        "content": ai_answer,
                        "sources": sources,
                    }
                )

                if enable_ragas:
                    with st.spinner("Evaluating answer quality (RAGAS)..."):
                        contexts = [doc.page_content for doc in context_docs]
                        st.session_state.last_ragas_scores = evaluate_rag_response(
                            groq_api_key=st.session_state.groq_api_key,
                            question=user_query,
                            answer=ai_answer,
                            contexts=contexts,
                            dense_embeddings=_load_dense_embeddings(),
                        )
                else:
                    st.session_state.last_ragas_scores = None

                st.session_state.scroll_to_latest = True

            except Exception as exc:
                error_detail = _friendly_request_error(
                    exc,
                    st.session_state.get("groq_api_key", ""),
                )
                st.session_state.messages.append(
                    {
                        "role": "ai",
                        "content": f"I couldn't process that request. {error_detail}",
                        "sources": [],
                    }
                )
                st.session_state.scroll_to_latest = True

        st.rerun()


if __name__ == "__main__":
    main()

import html
import sys
from pathlib import Path

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
from backend.retrieval import build_retriever  # noqa: E402
from backend.vectorstore import check_qdrant_connection, create_vectorstore  # noqa: E402


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


def main() -> None:
    st.set_page_config(page_title="SCA RAG Chatbot", layout="wide")

    st.markdown(
        """
        <style>
            :root {
                --ink: #101b3d;
                --muted: #53617e;
                --blue: #155eef;
                --blue-dark: #0b2f88;
                --red: #e11d48;
                --surface: #ffffff;
                --surface-soft: #f4f7ff;
                --line: #d7def0;
            }

            .stApp {
                color: var(--ink);
                color-scheme: light;
                background:
                    radial-gradient(circle at 8% 4%, rgba(225, 29, 72, .14), transparent 26rem),
                    radial-gradient(circle at 95% 8%, rgba(21, 94, 239, .18), transparent 30rem),
                    linear-gradient(145deg, #fff8fa 0%, #f7f9ff 45%, #eef4ff 100%);
            }

            [data-testid="stHeader"] { background: transparent; }

            [data-testid="stAppViewContainer"] > .main .block-container {
                max-width: 1120px;
                padding-top: 2.2rem;
                padding-bottom: 3rem;
            }

            [data-testid="stAppViewContainer"] > .main {
                overflow-y: scroll !important;
                scrollbar-color: var(--blue) #e5eaf5;
                scrollbar-width: auto;
            }

            [data-testid="stAppViewContainer"] > .main::-webkit-scrollbar,
            [data-testid="stVerticalBlockBorderWrapper"] ::-webkit-scrollbar {
                width: 12px;
            }

            [data-testid="stAppViewContainer"] > .main::-webkit-scrollbar-track,
            [data-testid="stVerticalBlockBorderWrapper"] ::-webkit-scrollbar-track {
                background: #e5eaf5;
            }

            [data-testid="stAppViewContainer"] > .main::-webkit-scrollbar-thumb,
            [data-testid="stVerticalBlockBorderWrapper"] ::-webkit-scrollbar-thumb {
                border: 3px solid #e5eaf5;
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
                border: 1px solid rgba(255, 255, 255, .7);
                border-radius: 24px;
                color: #ffffff;
                background: linear-gradient(120deg, #9f1239 0%, #e11d48 30%, #4438ca 67%, #0b4cc7 100%);
                box-shadow: 0 20px 50px rgba(27, 45, 105, .22);
            }

            .hero::after {
                content: "";
                position: absolute;
                width: 230px;
                height: 230px;
                right: -70px;
                top: -110px;
                border: 38px solid rgba(255, 255, 255, .13);
                border-radius: 50%;
            }

            .hero-kicker {
                margin-bottom: .7rem;
                font-size: .74rem;
                font-weight: 800;
                letter-spacing: .14em;
                text-transform: uppercase;
                color: #ffffff;
            }

            .hero h1 {
                margin: 0 0 .65rem;
                font-size: clamp(2rem, 5vw, 3.35rem);
                line-height: 1.05;
                letter-spacing: -.04em;
                color: #ffffff !important;
            }

            .hero p {
                max-width: 720px;
                margin: 0;
                font-size: 1.02rem;
                line-height: 1.65;
                color: #f8faff !important;
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid #253d82;
                background: linear-gradient(180deg, #07132f 0%, #0d2255 55%, #40112a 100%);
            }

            [data-testid="stSidebar"] * { color: #f8faff !important; }
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #d5def5; }
            [data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, .18); }

            [data-testid="stSidebar"] input,
            [data-testid="stSidebar"] textarea {
                color: #101b3d !important;
                background: #ffffff !important;
                -webkit-text-fill-color: #101b3d !important;
            }

            [data-testid="stSidebar"] input::placeholder,
            [data-testid="stSidebar"] textarea::placeholder {
                color: #697590 !important;
                -webkit-text-fill-color: #697590 !important;
            }

            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
                color: var(--ink);
                border-color: #9eadd2;
                background: rgba(255, 255, 255, .96);
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
                background: linear-gradient(105deg, #e11d48, #155eef);
                box-shadow: 0 8px 20px rgba(21, 94, 239, .22);
                transition: transform .16s ease, box-shadow .16s ease;
            }

            .stButton > button:hover,
            [data-testid="stFormSubmitButton"] > button:hover {
                color: #ffffff !important;
                transform: translateY(-1px);
                box-shadow: 0 11px 24px rgba(21, 94, 239, .3);
            }

            .stButton > button *,
            [data-testid="stFormSubmitButton"] > button * {
                color: #ffffff !important;
            }

            .stButton > button:focus {
                outline: 3px solid rgba(255, 255, 255, .7);
                outline-offset: 2px;
            }

            [data-testid="stAlert"] {
                border: 1px solid #b8c6e6;
                border-left: 5px solid var(--blue);
                border-radius: 14px;
                background: rgba(255, 255, 255, .94);
                box-shadow: 0 8px 24px rgba(16, 27, 61, .08);
            }

            [data-testid="stAlert"] * { color: var(--ink) !important; }

            [data-testid="stVerticalBlockBorderWrapper"] {
                border-color: #c5cfe6 !important;
                border-radius: 20px !important;
                background: rgba(255, 255, 255, .72);
                box-shadow: 0 16px 42px rgba(29, 48, 99, .1);
            }

            .chat-message-user {
                max-width: 82%;
                margin: .8rem 0 .8rem auto;
                padding: .95rem 1.1rem;
                border: 1px solid #8fb2ff;
                border-radius: 18px 18px 5px 18px;
                color: #ffffff;
                line-height: 1.55;
                background: linear-gradient(135deg, #0d47b8, #155eef);
                box-shadow: 0 8px 20px rgba(21, 94, 239, .18);
            }

            .chat-message-ai {
                max-width: 86%;
                margin: .8rem auto .8rem 0;
                padding: .95rem 1.1rem;
                border: 1px solid #f2a9bb;
                border-radius: 18px 18px 18px 5px;
                color: #31101a;
                line-height: 1.55;
                background: linear-gradient(135deg, #ffffff, #fff1f4);
                box-shadow: 0 8px 20px rgba(159, 18, 57, .1);
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

            [data-testid="stChatInput"] {
                border: 1px solid #aebddd;
                border-radius: 16px;
                color: var(--ink) !important;
                background: #ffffff !important;
                box-shadow: 0 10px 28px rgba(25, 45, 96, .12);
                overflow: hidden;
            }

            [data-testid="stChatInput"] > div,
            [data-testid="stChatInput"] textarea,
            [data-testid="stChatInputTextArea"] {
                color: var(--ink) !important;
                caret-color: var(--blue) !important;
                background: #ffffff !important;
                -webkit-text-fill-color: var(--ink) !important;
            }

            [data-testid="stChatInput"] textarea::placeholder,
            [data-testid="stChatInputTextArea"]::placeholder {
                color: #66728d !important;
                opacity: 1;
                -webkit-text-fill-color: #66728d !important;
            }

            [data-testid="stChatInputSubmitButton"] {
                color: var(--blue) !important;
                background: #ffffff !important;
            }

            [data-testid="stBottom"],
            [data-testid="stBottomBlockContainer"] {
                background: transparent !important;
                box-shadow: none !important;
            }

            [data-testid="stExpander"] {
                border-color: #c5cfe6;
                border-radius: 12px;
                background: rgba(255, 255, 255, .85);
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

    with st.sidebar:
        st.header("Setup")

        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get your key from the Groq console.",
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
            help="Score each answer for faithfulness and relevancy (adds a few seconds).",
        )

        qdrant_ok, qdrant_error = check_qdrant_connection()
        if qdrant_ok:
            st.success("Qdrant connected")
        else:
            st.error(qdrant_error)

        if st.button("Process Documents and Start Chat", use_container_width=True):
            if not groq_api_key:
                st.error("Please enter your Groq API Key.")
            elif not uploaded_files:
                st.error("Please upload at least one file.")
            elif not qdrant_ok:
                st.error(qdrant_error)
            else:
                with st.spinner("Processing documents (embedding + indexing)..."):
                    try:
                        dense = _load_dense_embeddings()
                        sparse = _load_sparse_embeddings()
                        chunks = load_and_split_documents(uploaded_files)

                        if not chunks:
                            st.error("No supported documents were loaded.")
                        else:
                            vectorstore = create_vectorstore(chunks, dense, sparse)
                            retriever = build_retriever(vectorstore)
                            groq_llm = ChatGroq(
                                groq_api_key=groq_api_key,
                                model_name=config.GROQ_MODEL,
                            )
                            st.session_state.rag_chain = create_rag_chain(groq_llm, retriever)
                            st.session_state.groq_api_key = groq_api_key
                            st.session_state.documents_processed = True
                            st.session_state.messages = [
                                {
                                    "role": "ai",
                                    "content": "Documents processed. Ask me anything about them!",
                                    "sources": [],
                                    "is_welcome": True,
                                }
                            ]
                            st.session_state.last_ragas_scores = None
                            st.success(f"Indexed {len(chunks)} chunks.")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Processing failed: {exc}")

        if st.session_state.last_ragas_scores:
            st.markdown("---")
            st.subheader("Latest RAGAS Scores")
            scores = st.session_state.last_ragas_scores
            if scores.get("error"):
                st.warning(scores["error"])
            else:
                if scores.get("faithfulness") is not None:
                    st.metric("Faithfulness", f"{scores['faithfulness']:.2f}")
                if scores.get("answer_relevancy") is not None:
                    st.metric("Answer Relevancy", f"{scores['answer_relevancy']:.2f}")

        st.markdown("---")
        st.markdown(
            "**Stack:** Qdrant · BGE-large · BM25 · BGE reranker · Groq · RAGAS · Streamlit"
        )

    if not st.session_state.documents_processed:
        st.info(
            "Start Qdrant (`docker compose up -d`), upload documents, enter your Groq API key, "
            "then click **Process Documents and Start Chat**."
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
                    f'<div class="chat-message-ai"><strong>Assistant</strong>{html.escape(msg["content"])}</div>',
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

        with st.spinner("Generating answer..."):
            try:
                response = st.session_state.rag_chain.invoke(
                    {
                        "input": user_query,
                        "chat_history": _chain_chat_history()[:-1],
                    }
                )
                ai_answer = response["answer"]
                context_docs = response.get("context", [])
                sources = _format_sources(context_docs)

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

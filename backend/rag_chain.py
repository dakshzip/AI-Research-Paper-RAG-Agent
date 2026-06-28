from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from backend import config


def _load_prompts() -> dict[str, str]:
    """Load system and contextualize prompts from prompts/rag_system.txt."""
    content = (config.PROMPTS_DIR / "rag_system.txt").read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in content.splitlines():
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    return {
        "system": "\n".join(sections.get("SYSTEM", [])).strip(),
        "contextualize": "\n".join(sections.get("CONTEXTUALIZE", [])).strip(),
    }


def create_rag_chain(groq_llm, retriever):
    """Create a history-aware RAG chain with reranked hybrid retrieval."""
    prompts = _load_prompts()

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts["contextualize"]),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        groq_llm,
        retriever,
        contextualize_prompt,
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts["system"]),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    document_chain = create_stuff_documents_chain(groq_llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, document_chain)

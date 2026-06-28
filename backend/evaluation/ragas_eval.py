from typing import Any

from backend import config


def evaluate_rag_response(
    groq_api_key: str,
    question: str,
    answer: str,
    contexts: list[str],
    dense_embeddings,
) -> dict[str, Any]:
    """Run RAGAS metrics with Groq as judge and local BGE embeddings."""
    if not contexts:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "error": "No retrieved contexts available for evaluation.",
        }

    try:
        from datasets import Dataset
        from langchain_groq import ChatGroq
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as exc:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "error": (
                "RAGAS dependencies missing. Run: pip install ragas datasets"
            ),
        }

    try:
        judge_llm = ChatGroq(groq_api_key=groq_api_key, model_name=config.GROQ_MODEL)
        evaluator_llm = LangchainLLMWrapper(judge_llm)
        evaluator_embeddings = LangchainEmbeddingsWrapper(dense_embeddings)

        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
        )

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )

        row = result.to_pandas().iloc[0]
        faithfulness_score = row["faithfulness"]
        relevancy_score = row["answer_relevancy"]
        return {
            "faithfulness": (
                float(faithfulness_score)
                if faithfulness_score == faithfulness_score
                else None
            ),
            "answer_relevancy": (
                float(relevancy_score)
                if relevancy_score == relevancy_score
                else None
            ),
            "error": None,
        }
    except Exception as exc:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "error": f"RAGAS evaluation failed: {exc}",
        }

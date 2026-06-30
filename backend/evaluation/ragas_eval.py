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
            "answer_relevancy": None,
            "error": "No retrieved contexts available for evaluation.",
        }

    try:
        from langchain_groq import ChatGroq
        from ragas import EvaluationDataset, RunConfig, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy
    except ImportError as exc:
        return {
            "answer_relevancy": None,
            "error": (
                "RAGAS dependencies missing. Run: pip install ragas datasets"
            ),
        }

    try:
        judge_llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=config.RAGAS_JUDGE_MODEL,
            temperature=0,
        )
        evaluator_llm = LangchainLLMWrapper(judge_llm)
        evaluator_embeddings = LangchainEmbeddingsWrapper(dense_embeddings)

        # answer_relevancy defaults to strictness=3, which RAGAS implements by
        # requesting n=3 completions in one call. Groq rejects n>1 with a 400
        # ("'n' : number must be at most 1"), so force a single generation.
        answer_relevancy.strictness = 1

        # RAGAS >=0.2 schema: user_input / response / retrieved_contexts.
        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": contexts,
                }
            ]
        )

        # Cap retries/timeout and run metrics concurrently so a single answer
        # is scored in seconds rather than RAGAS's slow default backoff.
        run_config = RunConfig(timeout=60, max_retries=1, max_wait=10, max_workers=4)

        # raise_exceptions=True so a judge/API failure propagates as a real
        # exception instead of being silently coerced into a NaN score (which
        # otherwise surfaces only as an unhelpful "no usable score" message).
        result = evaluate(
            dataset=dataset,
            metrics=[answer_relevancy],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=run_config,
            raise_exceptions=True,
        )

        row = result.to_pandas().iloc[0]
        relevancy_score = row["answer_relevancy"]

        # x == x is False only for NaN, which RAGAS returns when the judge could
        # not produce a usable score.
        relevancy_val = (
            float(relevancy_score) if relevancy_score == relevancy_score else None
        )

        error = None
        if relevancy_val is None:
            error = (
                f"The judge model ({config.RAGAS_JUDGE_MODEL}) did not return a usable "
                "Answer Relevancy score. Try a different RAGAS_JUDGE_MODEL or re-run."
            )

        return {
            "answer_relevancy": relevancy_val,
            "error": error,
        }
    except Exception as exc:
        return {
            "answer_relevancy": None,
            "error": f"RAGAS evaluation failed: {exc}",
        }

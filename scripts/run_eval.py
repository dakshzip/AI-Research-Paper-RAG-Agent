"""Evaluation harness: retrieval quality, RAGAS answer quality, and latency.

Measures three metric families over the eval set in data/eval/eval_set.json:

1. Retrieval  — Recall@5 (hit rate) and MRR, with a hybrid-vs-dense and
                with/without-reranker ablation.
2. RAGAS      — faithfulness and answer relevancy, judged by RAGAS_JUDGE_MODEL.
3. Latency    — retrieval-only p50/p95 per config, and end-to-end p50/p95
                (retrieval + reranking + generation).

Usage:
    python3 scripts/run_eval.py [--limit N] [--skip-ragas] [--skip-generation]

Requires Qdrant running (docker compose up -d) and GROQ_API_KEY in .env.
Tip: EMBEDDING_DEVICE=cpu python3 scripts/run_eval.py avoids MPS memory issues.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config  # noqa: E402
from backend.embeddings import get_dense_embeddings, get_sparse_embeddings  # noqa: E402
from backend.retrieval import build_retriever, get_reranker_model  # noqa: E402
from backend.vectorstore import check_qdrant_connection, connect_existing_vectorstore  # noqa: E402

RESULTS_DIR = config.PROJECT_ROOT / "data" / "eval"
DEFAULT_EVAL_SET = RESULTS_DIR / "eval_set.json"


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


def score_retrieval(docs, expected_source: str) -> tuple[bool, float]:
    """Return (hit, reciprocal rank) for the expected source in retrieved docs."""
    for rank, doc in enumerate(docs, start=1):
        if doc.metadata.get("source") == expected_source:
            return True, 1.0 / rank
    return False, 0.0


def run_retrieval_config(name: str, retriever, eval_set: list[dict]) -> dict:
    hits, rrs, latencies = [], [], []
    for item in eval_set:
        start = time.perf_counter()
        docs = retriever.invoke(item["question"])
        latencies.append(time.perf_counter() - start)
        hit, rr = score_retrieval(docs, item["expected_source"])
        hits.append(hit)
        rrs.append(rr)
        marker = "+" if hit else "MISS"
        print(f"    [{marker}] {item['expected_source']}")
    return {
        "config": name,
        "recall_at_5": sum(hits) / len(hits),
        "mrr": sum(rrs) / len(rrs),
        "retrieval_latency_p50_s": round(statistics.median(latencies), 3),
        "retrieval_latency_p95_s": round(percentile(latencies, 0.95), 3),
        "misses": [
            item["expected_source"]
            for item, hit in zip(eval_set, hits)
            if not hit
        ],
    }


def _invoke_with_rate_limit_retry(chain, payload: dict, max_attempts: int = 8):
    """Retry on Groq 429s. The free tier allows 6000 tokens/min and one RAG
    request uses ~3400, so sustained runs inevitably hit the TPM ceiling and
    just need to wait for the per-minute budget to refill."""
    from groq import APIConnectionError, InternalServerError, RateLimitError

    for attempt in range(1, max_attempts + 1):
        try:
            return chain.invoke(payload)
        except (APIConnectionError, InternalServerError, RateLimitError) as exc:
            if attempt == max_attempts:
                raise
            print(f"      {type(exc).__name__}; waiting 20s (attempt {attempt})")
            time.sleep(20)


def run_generation(retriever, eval_set: list[dict]) -> tuple[list[dict], dict]:
    """Answer every eval question through the full RAG chain, timing end-to-end."""
    from langchain_groq import ChatGroq

    from backend.rag_chain import create_rag_chain

    llm = ChatGroq(
        groq_api_key=config.GROQ_API_KEY,
        model_name=config.GROQ_MODEL,
        temperature=0,
        reasoning_effort="none",
    )
    chain = create_rag_chain(llm, retriever)

    rows, latencies = [], []
    for i, item in enumerate(eval_set, start=1):
        start = time.perf_counter()
        result = _invoke_with_rate_limit_retry(
            chain, {"input": item["question"], "chat_history": []}
        )
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        rows.append(
            {
                "user_input": item["question"],
                "response": result["answer"],
                "retrieved_contexts": [d.page_content for d in result["context"]],
            }
        )
        print(f"    [{i}/{len(eval_set)}] {elapsed:.1f}s  {item['question'][:60]}")
        time.sleep(1)  # stay under Groq free-tier rate limits

    stats = {
        "e2e_latency_p50_s": round(statistics.median(latencies), 2),
        "e2e_latency_p95_s": round(percentile(latencies, 0.95), 2),
        "e2e_latency_mean_s": round(statistics.mean(latencies), 2),
    }

    # Persist answers so RAGAS can be re-run later (--ragas-only) without
    # paying for generation again.
    answers_path = RESULTS_DIR / f"answers_{datetime.now():%Y%m%d_%H%M%S}.json"
    answers_path.write_text(json.dumps(rows, indent=2))
    print(f"    answers saved to {answers_path}")

    return rows, stats


def run_ragas(rows: list[dict], dense_embeddings) -> dict:
    """Judge one answer at a time with pacing between rows.

    Faithfulness alone makes several long judge calls per answer, so batch
    parallelism blows through Groq's per-minute token budget and RAGAS coerces
    the resulting 429s into NaN scores. Serial + sleep trades speed for
    coverage.
    """
    from langchain_groq import ChatGroq
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, faithfulness

    judge = LangchainLLMWrapper(
        ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.RAGAS_JUDGE_MODEL,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(dense_embeddings)
    # Groq rejects n>1 generations, which strictness=3 would request.
    answer_relevancy.strictness = 1
    run_config = RunConfig(timeout=180, max_retries=3, max_wait=60, max_workers=1)

    per_metric: dict[str, list[float | None]] = {
        "faithfulness": [],
        "answer_relevancy": [],
    }
    for i, row in enumerate(rows, start=1):
        try:
            result = evaluate(
                dataset=EvaluationDataset.from_list([row]),
                metrics=[faithfulness, answer_relevancy],
                llm=judge,
                embeddings=embeddings,
                run_config=run_config,
            )
            scored = result.to_pandas().iloc[0]
            for metric in per_metric:
                value = scored[metric]
                # NaN (value != value) means the judge gave no usable score.
                per_metric[metric].append(
                    round(float(value), 3) if value == value else None
                )
        except Exception as exc:
            for metric in per_metric:
                per_metric[metric].append(None)
            print(f"    [{i}/{len(rows)}] judge failed: {exc}")
        else:
            print(f"    [{i}/{len(rows)}] "
                  f"faithfulness={per_metric['faithfulness'][-1]}  "
                  f"relevancy={per_metric['answer_relevancy'][-1]}")
        if i < len(rows):
            time.sleep(25)  # let the judge's per-minute token budget refill

    summary = {}
    for metric, values in per_metric.items():
        valid = [v for v in values if v is not None]
        summary[metric] = {
            "mean": round(statistics.mean(valid), 3) if valid else None,
            "scored": len(valid),
            "of": len(rows),
            "per_question": values,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N questions")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET,
                        help="Path to the eval question JSON (default: eval_set.json)")
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--ragas-only", type=Path, metavar="ANSWERS_JSON",
                        help="Skip retrieval/generation; judge a saved answers file")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Retrieval metrics only (no LLM calls at all)")
    args = parser.parse_args()

    if args.ragas_only:
        if not config.GROQ_API_KEY:
            sys.exit("GROQ_API_KEY missing from .env.")
        rows = json.loads(args.ragas_only.read_text())
        print(f"Judging {len(rows)} saved answers from {args.ragas_only.name} "
              f"(paced; ~25s per answer)\n")
        summary = run_ragas(rows, get_dense_embeddings())
        out_path = RESULTS_DIR / f"ragas_{datetime.now():%Y%m%d_%H%M%S}.json"
        out_path.write_text(json.dumps(summary, indent=2))
        for metric, s in summary.items():
            print(f"\n  {metric}: {s['mean']} (scored {s['scored']}/{s['of']})")
        print(f"\nSaved to {out_path}")
        return

    ok, error = check_qdrant_connection()
    if not ok:
        sys.exit(error)
    if not args.skip_generation and not config.GROQ_API_KEY:
        sys.exit("GROQ_API_KEY missing from .env (or pass --skip-generation).")

    eval_set = json.loads(args.eval_set.read_text())
    if args.limit:
        eval_set = eval_set[: args.limit]
    print(f"Evaluating {len(eval_set)} questions from {args.eval_set.name} "
          f"(k={config.RETRIEVE_K} retrieved, top {config.FINAL_K} after rerank)\n")

    print("Loading embedding + reranker models...")
    dense = get_dense_embeddings()
    sparse = get_sparse_embeddings()
    reranker = get_reranker_model()

    from langchain_qdrant import QdrantVectorStore, RetrievalMode

    hybrid_vs = connect_existing_vectorstore(dense, sparse)
    dense_vs = QdrantVectorStore.from_existing_collection(
        embedding=dense,
        url=config.QDRANT_URL,
        collection_name=config.QDRANT_COLLECTION,
        retrieval_mode=RetrievalMode.DENSE,
        vector_name=config.DENSE_VECTOR_NAME,
    )

    configs = {
        "hybrid + rerank (production)": build_retriever(hybrid_vs, reranker),
        "hybrid, no rerank": hybrid_vs.as_retriever(search_kwargs={"k": config.FINAL_K}),
        "dense only, no rerank": dense_vs.as_retriever(search_kwargs={"k": config.FINAL_K}),
    }

    results: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "eval_set": args.eval_set.name,
        "num_questions": len(eval_set),
        "chat_model": config.GROQ_MODEL,
        "judge_model": config.RAGAS_JUDGE_MODEL,
        "retrieval": [],
    }

    out_path = RESULTS_DIR / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"

    def save() -> None:
        out_path.write_text(json.dumps(results, indent=2))

    print("\n=== 1. Retrieval metrics (Recall@5, MRR) ===")
    for name, retriever in configs.items():
        print(f"\n  Config: {name}")
        results["retrieval"].append(run_retrieval_config(name, retriever, eval_set))
        save()

    if not args.skip_generation:
        print("\n=== 2. Generation (end-to-end latency) ===")
        rows, latency_stats = run_generation(
            configs["hybrid + rerank (production)"], eval_set
        )
        results["latency"] = latency_stats
        save()

        if not args.skip_ragas:
            print("\n=== 3. RAGAS (faithfulness, answer relevancy) ===")
            results["ragas"] = run_ragas(rows, dense)
            save()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results["retrieval"]:
        print(f"  {r['config']:32s} Recall@5={r['recall_at_5']:.1%}  "
              f"MRR={r['mrr']:.3f}  p50={r['retrieval_latency_p50_s']}s")
    if "latency" in results:
        lat = results["latency"]
        print(f"\n  End-to-end latency: p50={lat['e2e_latency_p50_s']}s  "
              f"p95={lat['e2e_latency_p95_s']}s")
    if "ragas" in results:
        for metric, s in results["ragas"].items():
            print(f"  {metric}: {s['mean']} (scored {s['scored']}/{s['of']})")
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()

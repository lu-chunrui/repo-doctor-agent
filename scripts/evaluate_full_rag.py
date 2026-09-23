import json
import time
from pathlib import Path

from agent.core import ToolCallingLLMClient
from config import settings
from rag.bm25 import BM25CodeRetriever
from rag.dense import DenseCodeRetriever
from rag.query_rewriter import (
    QueryRewriter,
    RewrittenHybridRetriever,
)
from rag.reranker import (
    CodeReranker,
    RerankedCodeSearch,
)
from scripts.evaluate_dense_rewrite import (
    calculate_summary,
    find_first_relevant_rank,
    load_queries,
)
from scripts.evaluate_hybrid_rewrite import (
    CachedQueryRewriter,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "retrieval"
    / "full_rag_results.json"
)

TOP_K = 5


def main():
    queries = load_queries()

    dense_retriever = DenseCodeRetriever()
    bm25_retriever = BM25CodeRetriever()

    print("正在建立 Dense 和 BM25 索引")

    index_start = time.perf_counter()

    dense_index_info = (
        dense_retriever.build_index(
            PROJECT_ROOT
        )
    )

    bm25_index_info = (
        bm25_retriever.build_index(
            PROJECT_ROOT
        )
    )

    index_seconds = (
        time.perf_counter()
        - index_start
    )

    llm_client = (
        ToolCallingLLMClient
        .from_environment()
    )

    base_rewriter = QueryRewriter(
        repository_path=PROJECT_ROOT,
        llm_client=llm_client,
    )

    cached_rewriter = CachedQueryRewriter(
        base_rewriter
    )

    rewritten_retriever = (
        RewrittenHybridRetriever(
            query_rewriter=(
                cached_rewriter
            ),
            dense_retriever=(
                dense_retriever
            ),
            bm25_retriever=(
                bm25_retriever
            ),
            rrf_k=settings.rrf_k,
            dense_weight=(
                settings.dense_weight
            ),
            bm25_weight=(
                settings.bm25_weight
            ),
        )
    )

    print("正在加载 Reranker")

    reranker_start = time.perf_counter()

    reranker = CodeReranker(
        model_name=(
            settings.reranker_model_name
        ),
        enabled=True,
        batch_size=(
            settings.reranker_batch_size
        ),
        max_length=(
            settings.reranker_max_length
        ),
        allow_fallback=(
            settings.reranker_allow_fallback
        ),
    )

    reranker_load_seconds = (
        time.perf_counter()
        - reranker_start
    )

    search_pipeline = RerankedCodeSearch(
        rewritten_retriever=(
            rewritten_retriever
        ),
        reranker=reranker,
    )

    rows = []

    for index, item in enumerate(
        queries,
        start=1,
    ):
        total_start = time.perf_counter()

        rewrite_start = time.perf_counter()

        rewrite_result = (
            cached_rewriter.prepare(
                item["query"]
            )
        )

        rewrite_latency_ms = (
            time.perf_counter()
            - rewrite_start
        ) * 1000

        retrieval_start = (
            time.perf_counter()
        )

        search_result = (
            search_pipeline.search(
                query=item["query"],
                final_k=TOP_K,
                recall_k=(
                    settings
                    .reranker_recall_k
                ),
                retrieval_candidate_k=max(
                    settings
                    .retrieval_candidate_k,
                    settings
                    .reranker_recall_k,
                ),
            )
        )

        retrieval_latency_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        total_latency_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        results = search_result["results"]

        first_relevant_rank = None

        if item["should_retrieve"]:
            first_relevant_rank = (
                find_first_relevant_rank(
                    results,
                    item["relevant_files"],
                )
            )

        row = {
            "id": item["id"],
            "query": item["query"],
            "query_type": (
                item["query_type"]
            ),
            "should_retrieve": (
                item["should_retrieve"]
            ),
            "relevant_files": (
                item["relevant_files"]
            ),
            "first_relevant_rank": (
                first_relevant_rank
            ),
            "rewrite": rewrite_result,
            "rewrite_source": (
                rewrite_result[
                    "rewrite_source"
                ]
            ),
            "rewrite_error": (
                rewrite_result[
                    "rewrite_error"
                ]
            ),
            "skipped": (
                search_result["skipped"]
            ),
            "skip_reason": (
                search_result.get(
                    "skip_reason"
                )
            ),
            "soft_fuse_triggered": (
                search_result.get(
                    "soft_fuse_triggered",
                    False,
                )
            ),
            "top_dense_similarity": (
                search_result.get(
                    "top_dense_similarity"
                )
            ),
            "reranker_used": (
                search_result[
                    "reranker_used"
                ]
            ),
            "reranker_model": (
                search_result.get(
                    "reranker_model"
                )
            ),
            "reranker_error": (
                search_result.get(
                    "reranker_error"
                )
            ),
            "rewrite_latency_ms": (
                rewrite_latency_ms
            ),
            "retrieval_latency_ms": (
                retrieval_latency_ms
            ),
            "total_latency_ms": (
                total_latency_ms
            ),
            "retrieved_count": len(
                results
            ),
            "retrieved_results": [
                {
                    "rank": rank,
                    "file": result["file"],
                    "start_line": (
                        result["start_line"]
                    ),
                    "end_line": (
                        result["end_line"]
                    ),
                    "rrf_score": (
                        result.get(
                            "rrf_score"
                        )
                    ),
                    "reranker_score": (
                        result.get(
                            "reranker_score"
                        )
                    ),
                    "matched_by": (
                        result.get(
                            "matched_by",
                            [],
                        )
                    ),
                }
                for rank, result
                in enumerate(
                    results,
                    start=1,
                )
            ],
        }

        rows.append(row)

        print(
            f"[{index}/{len(queries)}] "
            f"{item['id']} "
            f"rank={first_relevant_rank} "
            f"fused="
            f"{row['soft_fuse_triggered']} "
            f"reranker="
            f"{row['reranker_used']} "
            f"latency="
            f"{total_latency_ms:.2f}ms"
        )

    summary = calculate_summary(
        rows=rows,
        index_seconds=index_seconds,
    )

    summary["method"] = (
        "hybrid_query_rewrite_reranker"
    )

    summary["reranker_model"] = (
        settings.reranker_model_name
    )

    summary["reranker_load_seconds"] = (
        reranker_load_seconds
    )

    summary["reranker_available"] = (
        reranker.is_available()
    )

    summary["reranker_load_error"] = (
        reranker.load_error
    )

    summary["reranker_used_count"] = sum(
        row["reranker_used"]
        for row in rows
    )

    summary["reranker_error_count"] = sum(
        row["reranker_error"]
        is not None
        for row in rows
    )

    summary["soft_fuse_enabled"] = (
        settings.soft_fuse_enabled
    )

    summary["soft_fuse_threshold"] = (
        settings
        .soft_fuse_similarity_threshold
    )

    summary["soft_fuse_triggered_count"] = sum(
        row["soft_fuse_triggered"]
        for row in rows
    )

    summary[
        "soft_fuse_relevant_fused_count"
    ] = sum(
        row["soft_fuse_triggered"]
        and row["should_retrieve"]
        for row in rows
    )

    summary[
        "soft_fuse_irrelevant_fused_count"
    ] = sum(
        row["soft_fuse_triggered"]
        and not row["should_retrieve"]
        for row in rows
    )

    output = {
        "summary": summary,
        "index_info": {
            "dense": dense_index_info,
            "bm25": bm25_index_info,
        },
        "queries": rows,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "结果已保存到：",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
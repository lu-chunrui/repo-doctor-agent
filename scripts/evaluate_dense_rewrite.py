import json
import time
from pathlib import Path

from agent.core import (
    ToolCallingLLMClient,
)
from rag.dense import (
    DenseCodeRetriever,
)
from rag.query_rewriter import (
    QueryRewriter,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

QUERIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "retrieval"
    / "queries.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "retrieval"
    / "dense_rewrite_results.json"
)

TOP_K = 5


def load_queries():
    queries = []

    for line_number, line in enumerate(
        QUERIES_PATH.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            queries.append(
                json.loads(line)
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"第 {line_number} 行 JSON 错误："
                f"{error}"
            ) from error

    if not queries:
        raise ValueError(
            "评测查询不能为空"
        )

    return queries


def normalize_path(path):
    return str(path).replace(
        "\\",
        "/",
    )


def find_first_relevant_rank(
    results,
    relevant_files,
):
    relevant_files = {
        normalize_path(path)
        for path in relevant_files
    }

    for rank, result in enumerate(
        results,
        start=1,
    ):
        result_file = normalize_path(
            result["file"]
        )

        if result_file in relevant_files:
            return rank

    return None


def calculate_summary(
    rows,
    index_seconds,
):
    relevant_rows = [
        row
        for row in rows
        if row["should_retrieve"]
    ]

    irrelevant_rows = [
        row
        for row in rows
        if not row["should_retrieve"]
    ]

    relevant_count = len(
        relevant_rows
    )

    def hit_at(cutoff):
        if not relevant_count:
            return None

        return sum(
            row["first_relevant_rank"]
            is not None
            and row["first_relevant_rank"]
            <= cutoff
            for row in relevant_rows
        ) / relevant_count

    mrr = (
        sum(
            1 / row["first_relevant_rank"]
            if row["first_relevant_rank"]
            is not None
            else 0.0
            for row in relevant_rows
        )
        / relevant_count
        if relevant_count
        else None
    )

    false_positive_rate = (
        sum(
            row["retrieved_count"] > 0
            for row in irrelevant_rows
        )
        / len(irrelevant_rows)
        if irrelevant_rows
        else None
    )

    return {
        "method": "dense_query_rewrite",
        "query_count": len(rows),
        "relevant_query_count": (
            relevant_count
        ),
        "irrelevant_query_count": len(
            irrelevant_rows
        ),
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "mrr": mrr,
        "average_latency_ms": sum(
            row["total_latency_ms"]
            for row in rows
        ) / len(rows),
        "average_rewrite_latency_ms": sum(
            row["rewrite_latency_ms"]
            for row in rows
        ) / len(rows),
        "average_retrieval_latency_ms": sum(
            row["retrieval_latency_ms"]
            for row in rows
        ) / len(rows),
        "index_seconds": index_seconds,
        "irrelevant_false_positive_rate": (
            false_positive_rate
        ),
        "rewrite_failure_count": sum(
            row["rewrite_error"]
            is not None
            for row in rows
        ),
        "rewrite_fallback_count": sum(
            row["rewrite_source"]
            != "llm"
            for row in rows
        ),
        "skipped_query_count": sum(
            row["skipped"]
            for row in rows
        ),
    }


def main():
    queries = load_queries()

    print(
        f"读取到 {len(queries)} 条查询"
    )

    print("\n正在加载 Dense 模型")

    dense_retriever = (
        DenseCodeRetriever()
    )

    print("\n正在建立 Dense 索引")

    index_start = time.perf_counter()

    index_info = (
        dense_retriever.build_index(
            PROJECT_ROOT
        )
    )

    index_seconds = (
        time.perf_counter()
        - index_start
    )

    print("\n正在初始化 LLM")

    llm_client = (
        ToolCallingLLMClient
        .from_environment()
    )

    query_rewriter = QueryRewriter(
        repository_path=PROJECT_ROOT,
        llm_client=llm_client,
    )

    rows = []

    print("\n开始 Dense + Query Rewrite 评测")

    for index, item in enumerate(
        queries,
        start=1,
    ):
        total_start = time.perf_counter()

        rewrite_start = time.perf_counter()

        rewrite_result = (
            query_rewriter.rewrite(
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

        if rewrite_result["should_search"]:
            results = (
                dense_retriever.search(
                    query=rewrite_result[
                        "semantic_query"
                    ],
                    top_k=TOP_K,
                )
            )
            skipped = False
        else:
            results = []
            skipped = True

        retrieval_latency_ms = (
            time.perf_counter()
            - retrieval_start
        ) * 1000

        total_latency_ms = (
            time.perf_counter()
            - total_start
        ) * 1000

        first_relevant_rank = None

        if item["should_retrieve"]:
            first_relevant_rank = (
                find_first_relevant_rank(
                    results,
                    item[
                        "relevant_files"
                    ],
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
            "hit_at_1": (
                first_relevant_rank
                is not None
                and first_relevant_rank <= 1
            ),
            "hit_at_3": (
                first_relevant_rank
                is not None
                and first_relevant_rank <= 3
            ),
            "hit_at_5": (
                first_relevant_rank
                is not None
                and first_relevant_rank <= 5
            ),
            "reciprocal_rank": (
                1 / first_relevant_rank
                if first_relevant_rank
                is not None
                else 0.0
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
            "skipped": skipped,
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
                    "score": result["score"],
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
            f"search="
            f"{rewrite_result['should_search']} "
            f"rank={first_relevant_rank} "
            f"latency="
            f"{total_latency_ms:.2f}ms"
        )

    summary = calculate_summary(
        rows=rows,
        index_seconds=index_seconds,
    )

    output = {
        "summary": summary,
        "index_info": index_info,
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

    print("\n评测完成")

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
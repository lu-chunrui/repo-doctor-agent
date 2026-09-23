import json
import time
from collections import defaultdict
from pathlib import Path

from config import settings
from rag.bm25 import BM25CodeRetriever
from rag.dense import DenseCodeRetriever
from rag.hybrid import HybridCodeRetriever


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
    / "hybrid_results.json"
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
                f"queries.jsonl 第 "
                f"{line_number} 行格式错误："
                f"{error}"
            ) from error

    if not queries:
        raise ValueError(
            "评测数据不能为空"
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
    expected_files = {
        normalize_path(file_path)
        for file_path in relevant_files
    }

    for rank, result in enumerate(
        results,
        start=1,
    ):
        result_file = normalize_path(
            result["file"]
        )

        if result_file in expected_files:
            return rank

    return None


def calculate_metrics(rows):
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

    metrics = {
        "query_count": len(rows),
        "relevant_query_count": (
            relevant_count
        ),
        "irrelevant_query_count": len(
            irrelevant_rows
        ),
        "average_latency_ms": (
            sum(
                row["latency_ms"]
                for row in rows
            )
            / len(rows)
            if rows
            else 0.0
        ),
    }

    for cutoff in (1, 3, 5):
        metrics[f"hit_at_{cutoff}"] = (
            sum(
                row["first_relevant_rank"]
                is not None
                and row[
                    "first_relevant_rank"
                ] <= cutoff
                for row in relevant_rows
            )
            / relevant_count
            if relevant_count
            else None
        )

    metrics["mrr"] = (
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

    metrics[
        "irrelevant_false_positive_rate"
    ] = (
        sum(
            row["retrieved_count"] > 0
            for row in irrelevant_rows
        )
        / len(irrelevant_rows)
        if irrelevant_rows
        else None
    )

    return metrics


def main():
    queries = load_queries()

    print(
        f"读取到 {len(queries)} 条查询"
    )

    print("\n正在初始化 Dense")

    dense_retriever = (
        DenseCodeRetriever()
    )

    print("\n正在初始化 BM25")

    bm25_retriever = (
        BM25CodeRetriever()
    )

    print("\n正在建立索引")

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

    retriever = HybridCodeRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=settings.rrf_k,
        dense_weight=(
            settings.dense_weight
        ),
        bm25_weight=(
            settings.bm25_weight
        ),
    )

    rows = []
    rows_by_type = defaultdict(list)

    print("\n开始评测")

    for index, item in enumerate(
        queries,
        start=1,
    ):
        query_start = time.perf_counter()

        results = retriever.search(
            query=item["query"],
            top_k=TOP_K,
            candidate_k=(
                settings
                .retrieval_candidate_k
            ),
        )

        latency_ms = (
            time.perf_counter()
            - query_start
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
            "latency_ms": latency_ms,
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
                        result["rrf_score"]
                    ),
                    "dense_rank": (
                        result["dense_rank"]
                    ),
                    "dense_score": (
                        result["dense_score"]
                    ),
                    "bm25_rank": (
                        result["bm25_rank"]
                    ),
                    "bm25_score": (
                        result["bm25_score"]
                    ),
                    "matched_by": (
                        result["matched_by"]
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
        rows_by_type[
            item["query_type"]
        ].append(row)

        print(
            f"[{index}/{len(queries)}] "
            f"{item['id']} "
            f"rank={first_relevant_rank} "
            f"latency={latency_ms:.2f}ms"
        )

    summary = calculate_metrics(
        rows
    )

    summary.update(
        {
            "method": "hybrid_rrf",
            "top_k": TOP_K,
            "candidate_k": (
                settings
                .retrieval_candidate_k
            ),
            "rrf_k": settings.rrf_k,
            "dense_weight": (
                settings.dense_weight
            ),
            "bm25_weight": (
                settings.bm25_weight
            ),
            "index_seconds": (
                index_seconds
            ),
        }
    )

    metrics_by_query_type = {
        query_type: calculate_metrics(
            type_rows
        )
        for query_type, type_rows
        in sorted(rows_by_type.items())
    }

    output = {
        "summary": summary,
        "metrics_by_query_type": (
            metrics_by_query_type
        ),
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
import json
import time
from pathlib import Path


from rag.bm25 import BM25CodeRetriever


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
    / "bm25_results.json"
)


def load_queries():
    return [
        json.loads(line)
        for line in QUERIES_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def find_first_relevant_rank(
    results,
    relevant_files,
):
    relevant_files = {
        path.replace("\\", "/")
        for path in relevant_files
    }

    for rank, result in enumerate(
        results,
        start=1,
    ):
        result_file = result[
            "file"
        ].replace("\\", "/")

        if result_file in relevant_files:
            return rank

    return None


def main():
    queries = load_queries()

    retriever = BM25CodeRetriever()

    index_start = time.perf_counter()
    index_info = retriever.build_index(
        PROJECT_ROOT
    )
    index_seconds = (
        time.perf_counter() - index_start
    )

    rows = []
    relevant_ranks = []
    irrelevant_count = 0
    irrelevant_false_positives = 0

    for item in queries:
        start = time.perf_counter()

        results = retriever.search(
            item["query"],
            top_k=5,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        rank = None

        if item["should_retrieve"]:
            rank = find_first_relevant_rank(
                results,
                item["relevant_files"],
            )
            relevant_ranks.append(rank)
        else:
            irrelevant_count += 1

            if results:
                irrelevant_false_positives += 1

        rows.append(
            {
                "id": item["id"],
                "query_type": (
                    item["query_type"]
                ),
                "should_retrieve": (
                    item["should_retrieve"]
                ),
                "first_relevant_rank": rank,
                "latency_ms": latency_ms,
                "retrieved_files": [
                    result["file"]
                    for result in results
                ],
            }
        )

    relevant_count = len(relevant_ranks)

    summary = {
        "method": "bm25",
        "query_count": len(queries),
        "relevant_query_count": (
            relevant_count
        ),
        "hit_at_1": sum(
            rank is not None and rank <= 1
            for rank in relevant_ranks
        ) / relevant_count,
        "hit_at_3": sum(
            rank is not None and rank <= 3
            for rank in relevant_ranks
        ) / relevant_count,
        "hit_at_5": sum(
            rank is not None and rank <= 5
            for rank in relevant_ranks
        ) / relevant_count,
        "mrr": sum(
            1 / rank
            if rank is not None
            else 0
            for rank in relevant_ranks
        ) / relevant_count,
        "average_latency_ms": sum(
            row["latency_ms"]
            for row in rows
        ) / len(rows),
        "index_seconds": index_seconds,
        "irrelevant_false_positive_rate": (
            irrelevant_false_positives
            / irrelevant_count
            if irrelevant_count
            else None
        ),
    }

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

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("结果已保存到：", OUTPUT_PATH)


if __name__ == "__main__":
    main()
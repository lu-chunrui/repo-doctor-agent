import json
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "retrieval"
)

METHOD_FILES = {
    "BM25": "bm25_results.json",
    "Dense": "dense_results.json",
    "Hybrid + RRF": (
        "hybrid_results.json"
    ),
    "Dense + Rewrite": (
        "dense_rewrite_results.json"
    ),
    "Hybrid + Rewrite": (
        "hybrid_rewrite_results.json"
    ),
    "Hybrid + Rewrite + Reranker": (
        "full_rag_results.json"
    ),
}

OUTPUT_JSON = (
    RESULTS_DIR
    / "ablation_comparison.json"
)

OUTPUT_MARKDOWN = (
    RESULTS_DIR
    / "ablation_report.md"
)


def load_summary(file_name):
    file_path = RESULTS_DIR / file_name

    if not file_path.is_file():
        raise FileNotFoundError(
            f"缺少评测文件：{file_path}"
        )

    data = json.loads(
        file_path.read_text(
            encoding="utf-8"
        )
    )

    return data["summary"]


def number(value):
    if value is None:
        return "-"

    return f"{value:.4f}"


def ranking_key(row):
    false_positive_rate = row[
        "irrelevant_false_positive_rate"
    ]

    if false_positive_rate is None:
        false_positive_rate = 1.0

    return (
        row["hit_at_5"] or 0.0,
        row["mrr"] or 0.0,
        row["hit_at_3"] or 0.0,
        -false_positive_rate,
        -row["average_latency_ms"],
    )


def main():
    rows = []

    for method, file_name in (
        METHOD_FILES.items()
    ):
        summary = load_summary(
            file_name
        )

        rows.append(
            {
                "method": method,
                "hit_at_1": summary.get(
                    "hit_at_1"
                ),
                "hit_at_3": summary.get(
                    "hit_at_3"
                ),
                "hit_at_5": summary.get(
                    "hit_at_5"
                ),
                "mrr": summary.get("mrr"),
                "average_latency_ms": (
                    summary.get(
                        "average_latency_ms",
                        0.0,
                    )
                ),
                "average_rewrite_latency_ms": (
                    summary.get(
                        "average_rewrite_latency_ms"
                    )
                ),
                "index_seconds": summary.get(
                    "index_seconds",
                    0.0,
                ),
                "irrelevant_false_positive_rate": (
                    summary.get(
                        "irrelevant_false_positive_rate"
                    )
                ),
                "rewrite_failure_count": (
                    summary.get(
                        "rewrite_failure_count"
                    )
                ),
                "reranker_available": (
                    summary.get(
                        "reranker_available"
                    )
                ),
                "reranker_used_count": (
                    summary.get(
                        "reranker_used_count"
                    )
                ),
            }
        )

    ranked_rows = sorted(
        rows,
        key=ranking_key,
        reverse=True,
    )

    best_method = ranked_rows[0][
        "method"
    ]

    output = {
        "best_method_by_current_rules": (
            best_method
        ),
        "ranking_rule": [
            "higher_hit_at_5",
            "higher_mrr",
            "higher_hit_at_3",
            "lower_irrelevant_false_positive_rate",
            "lower_average_latency_ms",
        ],
        "methods": rows,
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Retrieval Ablation Report",
        "",
        "| Method | Hit@1 | Hit@3 | Hit@5 | MRR | Latency ms | Rewrite ms | Index s | Irrelevant FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    number(row["hit_at_1"]),
                    number(row["hit_at_3"]),
                    number(row["hit_at_5"]),
                    number(row["mrr"]),
                    number(
                        row[
                            "average_latency_ms"
                        ]
                    ),
                    number(
                        row[
                            "average_rewrite_latency_ms"
                        ]
                    ),
                    number(
                        row["index_seconds"]
                    ),
                    number(
                        row[
                            "irrelevant_false_positive_rate"
                        ]
                    ),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Current Best",
            "",
            f"`{best_method}`",
            "",
            "## Important Notes",
            "",
            "- 当前查询数量较少，结论只是阶段性结果。",
            "- Query Rewrite 方案包含真实 LLM 调用延迟。",
            "- Reranker 结果只有在 reranker_available=true 时有效。",
            "- 无关查询误检率越低越好。",
            "",
        ]
    )

    OUTPUT_MARKDOWN.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("\n".join(lines))

    print(
        "\nJSON：",
        OUTPUT_JSON,
    )
    print(
        "Markdown：",
        OUTPUT_MARKDOWN,
    )


if __name__ == "__main__":
    main()
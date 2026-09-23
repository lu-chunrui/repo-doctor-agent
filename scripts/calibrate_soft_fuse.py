import argparse
import json
import time
from pathlib import Path

import numpy as np

from rag.dense import DenseCodeRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEV_PATH = (
    PROJECT_ROOT
    / "data"
    / "retrieval"
    / "soft_fuse_dev.jsonl"
)
OUTPUT_DIR = PROJECT_ROOT / "reports" / "retrieval"
JSON_PATH = OUTPUT_DIR / "soft_fuse_calibration.json"
REPORT_PATH = OUTPUT_DIR / "soft_fuse_calibration.md"


def load_items():
    items = []

    for line_number, line in enumerate(
        DEV_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"第 {line_number} 行 JSON 错误：{error}"
            ) from error

        if not isinstance(item.get("query"), str):
            raise ValueError(
                f"第 {line_number} 行缺少 query"
            )

        if not isinstance(item.get("should_retrieve"), bool):
            raise ValueError(
                f"第 {line_number} 行缺少 should_retrieve"
            )

        items.append(item)

    if not items:
        raise ValueError("软熔断开发集不能为空")

    return items


def calculate_metrics(rows, threshold):
    tp = fp = tn = fn = 0

    for row in rows:
        predicted_retrieve = (
            row["top1_similarity"] >= threshold
        )
        expected_retrieve = row["should_retrieve"]

        if expected_retrieve and predicted_retrieve:
            tp += 1
        elif expected_retrieve:
            fn += 1
        elif predicted_retrieve:
            fp += 1
        else:
            tn += 1

    positive_count = tp + fn
    negative_count = tn + fp

    recall = (
        tp / positive_count
        if positive_count
        else 0.0
    )
    false_positive_rate = (
        fp / negative_count
        if negative_count
        else 0.0
    )
    accuracy = (
        (tp + tn) / len(rows)
        if rows
        else 0.0
    )

    return {
        "threshold": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "relevant_recall": recall,
        "irrelevant_false_positive_rate": (
            false_positive_rate
        ),
        "accuracy": accuracy,
    }


def score_distribution(values):
    if not values:
        return None

    array = np.asarray(values, dtype=float)

    return {
        "minimum": float(np.min(array)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def select_threshold(rows, minimum_recall):
    scores = sorted({
        row["top1_similarity"]
        for row in rows
    })

    epsilon = 1e-6
    candidates = [
        scores[0] - epsilon,
        *scores,
        scores[-1] + epsilon,
    ]

    evaluations = [
        calculate_metrics(rows, threshold)
        for threshold in candidates
    ]

    eligible = [
        item for item in evaluations
        if item["relevant_recall"] >= minimum_recall
    ]

    if not eligible:
        raise RuntimeError(
            "没有满足最低相关查询保留率的阈值"
        )

    best = min(
        eligible,
        key=lambda item: (
            item["irrelevant_false_positive_rate"],
            -item["relevant_recall"],
            item["threshold"],
        ),
    )

    return best, evaluations


def write_report(result):
    selected = result["selected"]
    relevant = result["score_distribution"]["relevant"]
    irrelevant = result["score_distribution"]["irrelevant"]

    lines = [
        "# Soft Fuse Calibration Report",
        "",
        f"- Query count: {result['query_count']}",
        f"- Minimum recall requirement: "
        f"{result['minimum_recall']:.4f}",
        f"- Recommended threshold: "
        f"`{selected['threshold']:.6f}`",
        f"- Relevant recall: "
        f"{selected['relevant_recall']:.4f}",
        f"- Irrelevant false-positive rate: "
        f"{selected['irrelevant_false_positive_rate']:.4f}",
        f"- Accuracy: {selected['accuracy']:.4f}",
        f"- Average Dense latency: "
        f"{result['average_latency_ms']:.2f} ms",
        "",
        "## Confusion Matrix",
        "",
        "| TP | FP | TN | FN |",
        "| -: | -: | -: | -: |",
        (
            f"| {selected['true_positive']} "
            f"| {selected['false_positive']} "
            f"| {selected['true_negative']} "
            f"| {selected['false_negative']} |"
        ),
        "",
        "## Score Distribution",
        "",
        "| Group | Min | P25 | Median | P75 | Max | Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Relevant | {relevant['minimum']:.4f} "
            f"| {relevant['p25']:.4f} "
            f"| {relevant['median']:.4f} "
            f"| {relevant['p75']:.4f} "
            f"| {relevant['maximum']:.4f} "
            f"| {relevant['mean']:.4f} |"
        ),
        (
            f"| Irrelevant | {irrelevant['minimum']:.4f} "
            f"| {irrelevant['p25']:.4f} "
            f"| {irrelevant['median']:.4f} "
            f"| {irrelevant['p75']:.4f} "
            f"| {irrelevant['maximum']:.4f} "
            f"| {irrelevant['mean']:.4f} |"
        ),
        "",
        "## Recommended Environment",
        "",
        "```dotenv",
        "SOFT_FUSE_ENABLED=true",
        (
            "SOFT_FUSE_SIMILARITY_THRESHOLD="
            f"{selected['threshold']:.6f}"
        ),
        "```",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository",
        default=str(PROJECT_ROOT),
    )
    parser.add_argument(
        "--minimum-recall",
        type=float,
        default=0.95,
    )
    args = parser.parse_args()

    if not 0.0 <= args.minimum_recall <= 1.0:
        raise ValueError(
            "minimum-recall 必须位于 0 到 1 之间"
        )

    items = load_items()
    repository = Path(args.repository).resolve()

    retriever = DenseCodeRetriever()

    index_started = time.perf_counter()
    index_info = retriever.build_index(repository)
    index_seconds = time.perf_counter() - index_started

    rows = []

    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item['id']}")

        started = time.perf_counter()
        results = retriever.search(
            item["query"],
            top_k=1,
        )
        latency_ms = (
            time.perf_counter() - started
        ) * 1000

        top_result = results[0]

        rows.append({
            "id": item["id"],
            "query": item["query"],
            "query_type": item.get("query_type"),
            "should_retrieve": item["should_retrieve"],
            "top1_similarity": top_result["score"],
            "top1_file": top_result["file"],
            "top1_start_line": top_result["start_line"],
            "top1_end_line": top_result["end_line"],
            "latency_ms": latency_ms,
        })

    selected, candidates = select_threshold(
        rows,
        args.minimum_recall,
    )

    relevant_scores = [
        row["top1_similarity"]
        for row in rows
        if row["should_retrieve"]
    ]
    irrelevant_scores = [
        row["top1_similarity"]
        for row in rows
        if not row["should_retrieve"]
    ]

    result = {
        "repository": str(repository),
        "query_count": len(rows),
        "minimum_recall": args.minimum_recall,
        "index_info": index_info,
        "index_seconds": index_seconds,
        "average_latency_ms": (
            sum(row["latency_ms"] for row in rows)
            / len(rows)
        ),
        "selected": selected,
        "score_distribution": {
            "relevant": score_distribution(
                relevant_scores
            ),
            "irrelevant": score_distribution(
                irrelevant_scores
            ),
        },
        "queries": rows,
        "candidate_thresholds": candidates,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    JSON_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(result)

    print(json.dumps(selected, ensure_ascii=False, indent=2))
    print(f"JSON：{JSON_PATH}")
    print(f"报告：{REPORT_PATH}")


if __name__ == "__main__":
    main()
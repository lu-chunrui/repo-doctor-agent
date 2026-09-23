import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = PROJECT_ROOT / "data" / "agent_eval" / "cases.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "agent_eval"
JSON_PATH = OUTPUT_DIR / "agent_eval_results.json"
REPORT_PATH = OUTPUT_DIR / "agent_eval_report.md"


def load_cases():
    cases = []

    for number, line in enumerate(
        CASES_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"cases.jsonl 第 {number} 行格式错误：{error}"
            ) from error

    return cases


def normalize_path(value):
    return str(value or "").replace("\\", "/").lower()


def post_json(base_url, path, payload, timeout):
    response = requests.post(
        f"{base_url.rstrip('/')}{path}",
        json=payload,
        timeout=timeout,
    )

    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code}: {response.text}"
        )

    return response.json()


def value_contains(actual, expected_values, argument_name):
    actual_text = str(actual or "").lower()

    if (
        argument_name == "log_text"
        and actual_text.startswith("<日志长度")
    ):
        return True

    return all(
        str(expected).lower() in actual_text
        for expected in expected_values
    )


def arguments_correct(case, tool_trace):
    expectations = case.get("expected_argument_contains", {})

    for tool_name in case.get("expected_tools", []):
        calls = [
            item for item in tool_trace
            if item.get("tool_name") == tool_name
        ]

        if not calls:
            return False

        required = expectations.get(tool_name, {})
        if not required:
            continue

        matched = False

        for call in calls:
            arguments = call.get("arguments", {})
            if all(
                value_contains(
                    arguments.get(argument_name),
                    expected_values,
                    argument_name,
                )
                for argument_name, expected_values
                in required.items()
            ):
                matched = True
                break

        if not matched:
            return False

    return True


def evidence_hit(case, evidence):
    relevant_files = {
        normalize_path(path)
        for path in case.get("relevant_files", [])
    }

    if not relevant_files:
        return None

    evidence_files = {
        normalize_path(item.get("file"))
        for item in evidence
        if item.get("file")
    }

    return bool(relevant_files & evidence_files)


def citation_correct(case, citation_report):
    if not case.get("requires_citation", False):
        return None

    return bool(
        citation_report.get("valid", False)
        and citation_report.get("citation_count", 0) > 0
    )


def calculate_summary(rows):
    completed = [row for row in rows if row["status"] == "ok"]
    positives = [
        row for row in completed if row["should_call_tool"]
    ]
    negatives = [
        row for row in completed if not row["should_call_tool"]
    ]
    evidence_rows = [
        row for row in completed if row["evidence_hit"] is not None
    ]
    citation_rows = [
        row for row in completed
        if row["citation_correct"] is not None
    ]

    def accuracy(items, key):
        if not items:
            return None
        return sum(bool(item[key]) for item in items) / len(items)

    false_positives = sum(
        row["called_tool"] for row in negatives
    )
    false_negatives = sum(
        not row["called_tool"] for row in positives
    )

    return {
        "case_count": len(rows),
        "completed_count": len(completed),
        "tool_decision_accuracy": accuracy(
            completed, "tool_decision_correct"
        ),
        "tool_selection_accuracy": accuracy(
            completed, "tool_selection_correct"
        ),
        "tool_argument_accuracy": accuracy(
            positives, "tool_arguments_correct"
        ),
        "evidence_hit_rate": accuracy(
            evidence_rows, "evidence_hit"
        ),
        "citation_accuracy": accuracy(
            citation_rows, "citation_correct"
        ),
        "false_positive_count": false_positives,
        "false_positive_rate": (
            false_positives / len(negatives)
            if negatives else None
        ),
        "false_negative_count": false_negatives,
        "false_negative_rate": (
            false_negatives / len(positives)
            if positives else None
        ),
        "average_latency_ms": (
            sum(row["latency_ms"] for row in completed)
            / len(completed)
            if completed else None
        ),
    }


def format_number(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


def write_report(summary, rows):
    lines = [
        "# Agent Evaluation Report",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]

    for key, value in summary.items():
        if isinstance(value, float):
            value = format_number(value)
        lines.append(f"| {key} | {value} |")

    lines.extend([
        "",
        "## Case Results",
        "",
        "| ID | Tools | Decision | Selection | Arguments | Evidence | Citation |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])

    for row in rows:
        tools = ", ".join(row.get("actual_tools", [])) or "-"
        lines.append(
            f"| {row['id']} | {tools} | "
            f"{row.get('tool_decision_correct', '-')} | "
            f"{row.get('tool_selection_correct', '-')} | "
            f"{row.get('tool_arguments_correct', '-')} | "
            f"{row.get('evidence_hit', '-')} | "
            f"{row.get('citation_correct', '-')} |"
        )

    REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--repository",
        default=str(PROJECT_ROOT),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
    )
    args = parser.parse_args()

    repository = str(Path(args.repository).resolve())
    cases = load_cases()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("正在建立或读取索引……")
    post_json(
        args.base_url,
        "/api/v1/index",
        {
            "repository_path": repository,
            "rebuild_index": args.rebuild_index,
            "enable_reranker": True,
        },
        args.timeout,
    )

    rows = []

    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case['id']}")

        started = time.perf_counter()

        try:
            response = post_json(
                args.base_url,
                "/api/v1/chat",
                {
                    "repository_path": repository,
                    "message": case["question"],
                    "session_id": uuid4().hex,
                },
                args.timeout,
            )

            latency_ms = (
                time.perf_counter() - started
            ) * 1000

            tool_trace = response.get("tool_trace", [])
            actual_tools = [
                item.get("tool_name")
                for item in tool_trace
                if item.get("tool_name")
            ]
            called_tool = bool(actual_tools)
            expected_tools = case.get("expected_tools", [])

            row = {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "status": "ok",
                "should_call_tool": case["should_call_tool"],
                "called_tool": called_tool,
                "expected_tools": expected_tools,
                "actual_tools": actual_tools,
                "tool_decision_correct": (
                    called_tool == case["should_call_tool"]
                ),
                "tool_selection_correct": (
                    actual_tools == expected_tools
                ),
                "tool_arguments_correct": arguments_correct(
                    case, tool_trace
                ),
                "evidence_hit": evidence_hit(
                    case, response.get("evidence", [])
                ),
                "citation_correct": citation_correct(
                    case,
                    response.get("citation_report", {}),
                ),
                "latency_ms": latency_ms,
                "answer": response.get("answer", ""),
                "tool_trace": tool_trace,
                "evidence": response.get("evidence", []),
                "citation_report": response.get(
                    "citation_report", {}
                ),
                "needs_answer_review": True,
            }

        except Exception as error:
            row = {
                "id": case["id"],
                "category": case["category"],
                "question": case["question"],
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "latency_ms": (
                    time.perf_counter() - started
                ) * 1000,
            }

        rows.append(row)

    summary = calculate_summary(rows)
    result = {
        "repository": repository,
        "summary": summary,
        "cases": rows,
    }

    JSON_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(summary, rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"JSON：{JSON_PATH}")
    print(f"报告：{REPORT_PATH}")


if __name__ == "__main__":
    main()
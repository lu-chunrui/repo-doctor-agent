import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from scripts.llm_judge import LLMJudge


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

INPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "agent_eval"
    / "agent_eval_results.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "agent_eval"
    / "judge_results.json"
)

JUDGE_VERSION = "judge-v2"

CODE_FENCE_PATTERN = re.compile(
    r"```.*?(?:```|\Z)",
    re.DOTALL,
)

CITATION_PATTERN = re.compile(
    r"\[([^\[\]\n]+):(\d+)-(\d+)\]"
)

MAX_CITATIONS = 12
MAX_LINES_PER_CITATION = 160
MAX_SOURCE_CHARS = 30000


def extract_citation_sources(answer):
    if not isinstance(answer, str):
        return []

    # 测试代码中的 [model.py:1-2]
    # 不能被当成真实回答引用。
    citation_text = (
        CODE_FENCE_PATTERN.sub(
            "",
            answer,
        )
    )

    sources = []
    seen = set()
    used_chars = 0

    for match in CITATION_PATTERN.finditer(
        citation_text
    ):
        if len(sources) >= MAX_CITATIONS:
            break

        relative_path_text = (
            match.group(1)
            .strip()
            .strip("`")
            .replace("\\", "/")
        )

        start_line = int(match.group(2))
        end_line = int(match.group(3))

        key = (
            relative_path_text.lower(),
            start_line,
            end_line,
        )

        if key in seen:
            continue

        seen.add(key)

        relative_path = Path(
            relative_path_text
        )

        if relative_path.is_absolute():
            continue

        file_path = (
            PROJECT_ROOT
            / relative_path
        ).resolve()

        try:
            file_path.relative_to(
                PROJECT_ROOT
            )
        except ValueError:
            continue

        if not file_path.exists():
            continue

        if not file_path.is_file():
            continue

        if start_line < 1:
            continue

        if end_line < start_line:
            continue

        try:
            lines = file_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except OSError:
            continue

        if start_line > len(lines):
            continue

        actual_end_line = min(
            end_line,
            len(lines),
            (
                start_line
                + MAX_LINES_PER_CITATION
                - 1
            ),
        )

        numbered_lines = []

        for line_number in range(
            start_line,
            actual_end_line + 1,
        ):
            numbered_lines.append(
                f"{line_number}: "
                f"{lines[line_number - 1]}"
            )

        content = "\n".join(
            numbered_lines
        )

        remaining_chars = (
            MAX_SOURCE_CHARS
            - used_chars
        )

        if remaining_chars <= 0:
            break

        content = content[
            :remaining_chars
        ]

        used_chars += len(content)

        sources.append({
            "raw_citation": (
                match.group(0)
            ),
            "file": relative_path_text,
            "requested_start_line": (
                start_line
            ),
            "requested_end_line": (
                end_line
            ),
            "provided_start_line": (
                start_line
            ),
            "provided_end_line": (
                actual_end_line
            ),
            "truncated": (
                actual_end_line
                < end_line
            ),
            "content": content,
        })

    return sources


def make_fingerprint(
    case,
    citation_sources,
):
    payload = {
        "version": JUDGE_VERSION,
        "question": case["question"],
        "answer": case["answer"],
        "evidence": case.get(
            "evidence",
            [],
        ),
        "citation_report": case.get(
            "citation_report",
            {},
        ),
        "citation_sources": (
            citation_sources
        ),
        "should_call_tool": case[
            "should_call_tool"
        ],
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def load_existing():
    if not OUTPUT_PATH.exists():
        return {}

    data = json.loads(
        OUTPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    return {
        item["id"]: item
        for item in data.get(
            "cases",
            [],
        )
    }


def average(values):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return sum(values) / len(values)


def calculate_summary(rows):
    completed = [
        row
        for row in rows
        if row.get("status") == "ok"
    ]

    judgments = [
        row["judgment"]
        for row in completed
    ]

    return {
        "case_count": len(rows),
        "judged_count": len(completed),
        "error_count": (
            len(rows) - len(completed)
        ),
        "answer_supported_rate": (
            sum(
                item[
                    "answer_supported"
                ]
                for item in judgments
            ) / len(judgments)
            if judgments
            else None
        ),
        "hallucination_rate": (
            sum(
                item["hallucination"]
                for item in judgments
            ) / len(judgments)
            if judgments
            else None
        ),
        "average_relevance_score": (
            average([
                item["relevance_score"]
                for item in judgments
            ])
        ),
        "average_evidence_support_score": (
            average([
                item[
                    "evidence_support_score"
                ]
                for item in judgments
            ])
        ),
        "average_citation_support_score": (
            average([
                item[
                    "citation_support_score"
                ]
                for item in judgments
            ])
        ),
        "parse_retry_count": sum(
            item["parse_attempts"] > 1
            for item in judgments
        ),
        "average_judge_latency_ms": (
            average([
                row["latency_ms"]
                for row in completed
            ])
        ),
        "average_citation_source_count": (
            average([
                row[
                    "citation_source_count"
                ]
                for row in completed
            ])
        ),
    }


def save_results(
    source_cases,
    result_by_id,
    model,
):
    rows = [
        result_by_id[case["id"]]
        for case in source_cases
        if case["id"] in result_by_id
    ]

    output = {
        "judge_version": JUDGE_VERSION,
        "judge_model": model,
        "summary": calculate_summary(
            rows
        ),
        "cases": rows,
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    source = json.loads(
        INPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    cases = [
        case
        for case in source["cases"]
        if case.get("status") == "ok"
    ]

    result_by_id = load_existing()

    judge = LLMJudge.from_environment()

    for index, case in enumerate(
        cases,
        start=1,
    ):
        citation_sources = (
            extract_citation_sources(
                case["answer"]
            )
        )

        fingerprint = make_fingerprint(
            case,
            citation_sources,
        )

        existing = result_by_id.get(
            case["id"]
        )

        if (
            not args.force
            and existing
            and existing.get("status")
            == "ok"
            and existing.get(
                "fingerprint"
            ) == fingerprint
        ):
            print(
                f"[{index}/{len(cases)}] "
                f"{case['id']} 使用缓存"
            )
            continue

        print(
            f"[{index}/{len(cases)}] "
            f"{case['id']} 正在评判，"
            f"引用源码="
            f"{len(citation_sources)}"
        )

        started = time.perf_counter()

        try:
            judgment = judge.evaluate(
                question=case["question"],
                answer=case["answer"],
                evidence=case.get(
                    "evidence",
                    [],
                ),
                citation_report=case.get(
                    "citation_report",
                    {},
                ),
                should_use_repository=case[
                    "should_call_tool"
                ],
                citation_sources=(
                    citation_sources
                ),
            )

            row = {
                "id": case["id"],
                "category": case[
                    "category"
                ],
                "question": case[
                    "question"
                ],
                "status": "ok",
                "fingerprint": fingerprint,
                "citation_source_count": (
                    len(citation_sources)
                ),
                "latency_ms": (
                    time.perf_counter()
                    - started
                ) * 1000,
                "judgment": judgment,
            }

        except Exception as error:
            row = {
                "id": case["id"],
                "category": case[
                    "category"
                ],
                "question": case[
                    "question"
                ],
                "status": "error",
                "fingerprint": fingerprint,
                "citation_source_count": (
                    len(citation_sources)
                ),
                "latency_ms": (
                    time.perf_counter()
                    - started
                ) * 1000,
                "error": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }

        result_by_id[
            case["id"]
        ] = row

        # 每完成一条立即保存。
        save_results(
            cases,
            result_by_id,
            judge.model_name,
        )

    save_results(
        cases,
        result_by_id,
        judge.model_name,
    )

    final_data = json.loads(
        OUTPUT_PATH.read_text(
            encoding="utf-8"
        )
    )

    print(
        json.dumps(
            final_data["summary"],
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "结果：",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()

    print("结果：", OUTPUT_PATH)

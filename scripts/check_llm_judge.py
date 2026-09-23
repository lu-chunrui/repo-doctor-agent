import json
from pathlib import Path

from scripts.llm_judge import LLMJudge


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

RESULT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "agent_eval"
    / "agent_eval_results.json"
)


def main():
    data = json.loads(
        RESULT_PATH.read_text(
            encoding="utf-8"
        )
    )

    case = next(
        item
        for item in data["cases"]
        if item["id"] == "agent-001"
    )

    judge = LLMJudge.from_environment()

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
    )

    print(
        json.dumps(
            judgment,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
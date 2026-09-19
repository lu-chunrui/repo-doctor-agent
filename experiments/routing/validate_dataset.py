import argparse
import json
from collections import Counter
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from schema import (
    validate_route_target,
)


VALID_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}

VALID_SOURCES = {
    "manual",
    "ai_assisted",
}


def validate_metadata(metadata):
    """
    检查样本的metadata字段。
    """
    if not isinstance(metadata, dict):
        raise ValueError(
            "metadata必须是JSON对象"
        )

    required_fields = {
        "difficulty",
        "group_id",
        "source",
    }

    missing_fields = (
        required_fields - set(metadata)
    )

    if missing_fields:
        raise ValueError(
            "metadata缺少字段："
            + ", ".join(
                sorted(missing_fields)
            )
        )

    difficulty = metadata["difficulty"]
    group_id = metadata["group_id"]
    source = metadata["source"]

    if difficulty not in VALID_DIFFICULTIES:
        raise ValueError(
            f"不支持的difficulty：{difficulty}"
        )

    if (
        not isinstance(group_id, str)
        or not group_id.strip()
    ):
        raise ValueError(
            "group_id必须是非空字符串"
        )

    if source not in VALID_SOURCES:
        raise ValueError(
            f"不支持的source：{source}"
        )


def validate_sample(sample):
    """
    检查一条完整的路由样本。
    """
    if not isinstance(sample, dict):
        raise ValueError(
            "每条样本必须是JSON对象"
        )

    required_fields = {
        "id",
        "user_query",
        "target",
        "metadata",
    }

    missing_fields = (
        required_fields - set(sample)
    )

    if missing_fields:
        raise ValueError(
            "样本缺少字段："
            + ", ".join(
                sorted(missing_fields)
            )
        )

    sample_id = sample["id"]
    user_query = sample["user_query"]

    if (
        not isinstance(sample_id, str)
        or not sample_id.strip()
    ):
        raise ValueError(
            "id必须是非空字符串"
        )

    if (
        not isinstance(user_query, str)
        or not user_query.strip()
    ):
        raise ValueError(
            "user_query必须是非空字符串"
        )

    validate_route_target(
        sample["target"]
    )

    validate_metadata(
        sample["metadata"]
    )

    return sample


def load_and_validate_dataset(dataset_path):
    """
    读取并检查JSONL数据集。

    返回所有合法样本和统计信息。
    如果存在错误，最后统一抛出异常。
    """
    dataset_path = Path(
        dataset_path
    ).resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"数据集不存在：{dataset_path}"
        )

    if not dataset_path.is_file():
        raise ValueError(
            f"路径不是文件：{dataset_path}"
        )

    samples = []
    errors = []
    seen_ids = set()

    decision_counter = Counter()
    tool_counter = Counter()
    difficulty_counter = Counter()
    source_counter = Counter()

    with dataset_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                errors.append(
                    f"第{line_number}行：不允许空行"
                )
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(
                    f"第{line_number}行："
                    f"JSON格式错误：{error.msg}"
                )
                continue

            try:
                validate_sample(sample)
            except (
                ValueError,
                TypeError,
            ) as error:
                errors.append(
                    f"第{line_number}行：{error}"
                )
                continue

            sample_id = sample["id"]

            if sample_id in seen_ids:
                errors.append(
                    f"第{line_number}行："
                    f"id重复：{sample_id}"
                )
                continue

            seen_ids.add(sample_id)
            samples.append(sample)

            target = sample["target"]
            metadata = sample["metadata"]

            decision_counter[
                target["decision"]
            ] += 1

            tool_name = target["tool_name"]

            if tool_name is None:
                tool_name = "answer_directly"

            tool_counter[tool_name] += 1

            difficulty_counter[
                metadata["difficulty"]
            ] += 1

            source_counter[
                metadata["source"]
            ] += 1

    if errors:
        error_text = "\n".join(
            f"- {error}"
            for error in errors
        )

        raise ValueError(
            f"数据集检查失败，共发现"
            f"{len(errors)}个问题：\n"
            f"{error_text}"
        )

    statistics = {
        "total_samples": len(samples),
        "decisions": dict(
            decision_counter
        ),
        "tools": dict(tool_counter),
        "difficulties": dict(
            difficulty_counter
        ),
        "sources": dict(
            source_counter
        ),
    }

    return samples, statistics


def print_statistics(
    dataset_path,
    statistics,
):
    print("=" * 70)
    print(f"数据集：{dataset_path}")
    print(
        f"样本总数："
        f"{statistics['total_samples']}"
    )

    print("\n决策分布：")

    for name, count in sorted(
        statistics["decisions"].items()
    ):
        print(f"- {name}: {count}")

    print("\n工具分布：")

    for name, count in sorted(
        statistics["tools"].items()
    ):
        print(f"- {name}: {count}")

    print("\n难度分布：")

    for name, count in sorted(
        statistics["difficulties"].items()
    ):
        print(f"- {name}: {count}")

    print("\n来源分布：")

    for name, count in sorted(
        statistics["sources"].items()
    ):
        print(f"- {name}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="检查路由JSONL数据集"
    )

    parser.add_argument(
        "dataset_path",
        nargs="?",
        default=(
            "data/routing/"
            "reviewed_samples.jsonl"
        ),
        help="需要检查的JSONL文件路径",
    )

    args = parser.parse_args()

    dataset_path = Path(
        args.dataset_path
    )

    try:
        _, statistics = (
            load_and_validate_dataset(
                dataset_path
            )
        )
    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        print(error)
        raise SystemExit(1) from error

    print_statistics(
        dataset_path,
        statistics,
    )

    print("\n数据集检查通过。")


if __name__ == "__main__":
    main()
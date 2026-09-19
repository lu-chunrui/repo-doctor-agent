import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from validate_dataset import (
    load_and_validate_dataset,
    validate_sample,
)


def get_route_label(sample):
    """
    获取样本所属的路由类别。

    answer_directly没有tool_name，
    因此直接使用decision作为类别。
    """
    target = sample["target"]

    if (
        target["decision"]
        == "answer_directly"
    ):
        return "answer_directly"

    return target["tool_name"]


def group_samples(samples):
    """
    按group_id组织样本。

    返回：
    {
        "kv_cache_location": {
            "label": "search_codebase",
            "samples": [...]
        }
    }
    """
    grouped = defaultdict(list)

    for sample in samples:
        group_id = sample[
            "metadata"
        ]["group_id"]

        grouped[group_id].append(
            sample
        )

    result = {}

    for group_id, group_items in (
        grouped.items()
    ):
        labels = {
            get_route_label(sample)
            for sample in group_items
        }

        if len(labels) != 1:
            raise ValueError(
                f"group_id={group_id}中出现了"
                f"多个路由类别："
                f"{sorted(labels)}"
            )

        result[group_id] = {
            "label": next(iter(labels)),
            "samples": group_items,
        }

    return result


def organize_groups_by_label(
    grouped_samples,
):
    """
    将group按照路由类别组织。

    返回：
    {
        "search_codebase": [
            "kv_cache_location",
            ...
        ]
    }
    """
    groups_by_label = defaultdict(list)

    for group_id, group_data in (
        grouped_samples.items()
    ):
        label = group_data["label"]

        groups_by_label[label].append(
            group_id
        )

    return groups_by_label


def calculate_split_counts(
    group_count,
    validation_ratio=0.2,
    test_ratio=0.2,
):
    """
    计算某个类别应该分配多少个group。

    每个集合至少保留一个group。
    """
    if group_count < 3:
        raise ValueError(
            "每个路由类别至少需要3个group，"
            "才能划分train、validation和test"
        )

    validation_count = max(
        1,
        round(
            group_count
            * validation_ratio
        ),
    )

    test_count = max(
        1,
        round(
            group_count
            * test_ratio
        ),
    )

    # 必须至少给训练集保留一个group
    while (
        validation_count
        + test_count
        >= group_count
    ):
        if validation_count > test_count:
            validation_count -= 1
        elif test_count > 1:
            test_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            raise ValueError(
                "无法为训练集保留group"
            )

    train_count = (
        group_count
        - validation_count
        - test_count
    )

    return {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }


def split_group_ids(
    groups_by_label,
    random_seed=42,
    validation_ratio=0.2,
    test_ratio=0.2,
):
    """
    对每个路由类别分别切分group。

    这样可以保证每个集合中都包含所有类别。
    """
    random_generator = random.Random(
        random_seed
    )

    split_groups = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for label in sorted(
        groups_by_label
    ):
        group_ids = list(
            groups_by_label[label]
        )

        # 排序后再打乱，保证不同机器结果一致
        group_ids.sort()
        random_generator.shuffle(
            group_ids
        )

        counts = calculate_split_counts(
            group_count=len(group_ids),
            validation_ratio=(
                validation_ratio
            ),
            test_ratio=test_ratio,
        )

        test_end = counts["test"]

        validation_end = (
            test_end
            + counts["validation"]
        )

        test_groups = group_ids[
            :test_end
        ]

        validation_groups = group_ids[
            test_end:validation_end
        ]

        train_groups = group_ids[
            validation_end:
        ]

        split_groups["train"].extend(
            train_groups
        )

        split_groups[
            "validation"
        ].extend(
            validation_groups
        )

        split_groups["test"].extend(
            test_groups
        )

        print(
            f"{label}: "
            f"train={len(train_groups)}, "
            f"validation="
            f"{len(validation_groups)}, "
            f"test={len(test_groups)}"
        )

    return split_groups


def build_split_samples(
    grouped_samples,
    split_groups,
    random_seed=42,
):
    """
    根据切分后的group_id取得具体样本。
    """
    split_samples = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for split_name, group_ids in (
        split_groups.items()
    ):
        for group_id in group_ids:
            split_samples[
                split_name
            ].extend(
                grouped_samples[
                    group_id
                ]["samples"]
            )

        # 打乱集合内部的样本顺序
        split_random = random.Random(
            f"{random_seed}-{split_name}"
        )

        split_random.shuffle(
            split_samples[split_name]
        )

    return split_samples


def check_no_group_leakage(
    split_groups,
):
    """
    检查同一个group_id是否出现在多个集合。
    """
    train_groups = set(
        split_groups["train"]
    )

    validation_groups = set(
        split_groups["validation"]
    )

    test_groups = set(
        split_groups["test"]
    )

    train_validation_overlap = (
        train_groups
        & validation_groups
    )

    train_test_overlap = (
        train_groups
        & test_groups
    )

    validation_test_overlap = (
        validation_groups
        & test_groups
    )

    overlaps = (
        train_validation_overlap
        | train_test_overlap
        | validation_test_overlap
    )

    if overlaps:
        raise ValueError(
            "发现group_id数据泄漏："
            + ", ".join(
                sorted(overlaps)
            )
        )


def check_no_id_leakage(
    split_samples,
):
    """
    检查样本ID是否出现在多个集合。
    """
    ids_by_split = {}

    for split_name, samples in (
        split_samples.items()
    ):
        ids_by_split[split_name] = {
            sample["id"]
            for sample in samples
        }

    overlaps = (
        (
            ids_by_split["train"]
            & ids_by_split["validation"]
        )
        | (
            ids_by_split["train"]
            & ids_by_split["test"]
        )
        | (
            ids_by_split["validation"]
            & ids_by_split["test"]
        )
    )

    if overlaps:
        raise ValueError(
            "发现样本ID数据泄漏："
            + ", ".join(
                sorted(overlaps)
            )
        )


def write_jsonl(
    samples,
    output_path,
):
    """
    保存JSONL文件。
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for sample in samples:
            validate_sample(sample)

            file.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )


def calculate_statistics(samples):
    """
    统计各个路由类别的样本数量。
    """
    counter = Counter(
        get_route_label(sample)
        for sample in samples
    )

    return dict(
        sorted(counter.items())
    )


def build_manifest(
    source_path,
    random_seed,
    split_groups,
    split_samples,
):
    """
    创建数据切分记录。
    """
    manifest = {
        "source_file": str(source_path),
        "random_seed": random_seed,
        "split_method": (
            "stratified_group_split"
        ),
        "group_field": "group_id",
        "splits": {},
    }

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        samples = split_samples[
            split_name
        ]

        manifest["splits"][
            split_name
        ] = {
            "sample_count": len(samples),
            "group_count": len(
                split_groups[split_name]
            ),
            "group_ids": sorted(
                split_groups[split_name]
            ),
            "label_distribution": (
                calculate_statistics(
                    samples
                )
            ),
        }

    return manifest


def ensure_can_write(
    output_paths,
    overwrite,
):
    """
    防止意外覆盖已经存在的数据文件。
    """
    if overwrite:
        return

    existing_paths = [
        path
        for path in output_paths
        if path.exists()
    ]

    if existing_paths:
        existing_text = "\n".join(
            f"- {path}"
            for path in existing_paths
        )

        raise FileExistsError(
            "下面的输出文件已经存在：\n"
            f"{existing_text}\n"
            "确认覆盖时添加--overwrite"
        )


def print_split_report(
    split_groups,
    split_samples,
):
    """
    打印切分结果。
    """
    print("\n" + "=" * 70)
    print("数据集切分结果")

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        samples = split_samples[
            split_name
        ]

        groups = split_groups[
            split_name
        ]

        print(
            f"\n{split_name}: "
            f"{len(samples)}条样本，"
            f"{len(groups)}个group"
        )

        distribution = (
            calculate_statistics(samples)
        )

        for label, count in (
            distribution.items()
        ):
            print(
                f"  - {label}: {count}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "按照group_id分层切分"
            "工具路由数据集"
        )
    )

    parser.add_argument(
        "--input",
        default=(
            "data/routing/"
            "reviewed_samples.jsonl"
        ),
        help="审核后的数据集",
    )

    parser.add_argument(
        "--output-dir",
        default="data/routing/",
        help="切分结果保存目录",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.2,
        help="验证集group比例",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="测试集group比例",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有切分结果",
    )

    args = parser.parse_args()

    if not (
        0 < args.validation_ratio < 1
    ):
        raise ValueError(
            "validation-ratio必须在0和1之间"
        )

    if not (
        0 < args.test_ratio < 1
    ):
        raise ValueError(
            "test-ratio必须在0和1之间"
        )

    if (
        args.validation_ratio
        + args.test_ratio
        >= 1
    ):
        raise ValueError(
            "验证集与测试集比例之和"
            "必须小于1"
        )

    source_path = Path(args.input)
    output_directory = Path(
        args.output_dir
    )

    train_path = (
        output_directory
        / "train.jsonl"
    )

    validation_path = (
        output_directory
        / "validation.jsonl"
    )

    test_path = (
        output_directory
        / "test.jsonl"
    )

    manifest_path = (
        output_directory
        / "split_manifest.json"
    )

    ensure_can_write(
        output_paths=[
            train_path,
            validation_path,
            test_path,
            manifest_path,
        ],
        overwrite=args.overwrite,
    )

    samples, statistics = (
        load_and_validate_dataset(
            source_path
        )
    )

    print(
        f"成功加载"
        f"{statistics['total_samples']}"
        f"条审核数据"
    )

    grouped_samples = group_samples(
        samples
    )

    print(
        f"共有{len(grouped_samples)}"
        f"个group"
    )

    groups_by_label = (
        organize_groups_by_label(
            grouped_samples
        )
    )

    split_groups = split_group_ids(
        groups_by_label=groups_by_label,
        random_seed=args.seed,
        validation_ratio=(
            args.validation_ratio
        ),
        test_ratio=args.test_ratio,
    )

    check_no_group_leakage(
        split_groups
    )

    split_samples = build_split_samples(
        grouped_samples=grouped_samples,
        split_groups=split_groups,
        random_seed=args.seed,
    )

    check_no_id_leakage(
        split_samples
    )

    total_split_samples = sum(
        len(items)
        for items in split_samples.values()
    )

    if total_split_samples != len(samples):
        raise ValueError(
            "切分前后样本数量不一致："
            f"切分前={len(samples)}，"
            f"切分后={total_split_samples}"
        )

    write_jsonl(
        split_samples["train"],
        train_path,
    )

    write_jsonl(
        split_samples["validation"],
        validation_path,
    )

    write_jsonl(
        split_samples["test"],
        test_path,
    )

    manifest = build_manifest(
        source_path=source_path,
        random_seed=args.seed,
        split_groups=split_groups,
        split_samples=split_samples,
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print_split_report(
        split_groups,
        split_samples,
    )

    print("\n没有发现group_id数据泄漏。")
    print(f"训练集：{train_path}")
    print(f"验证集：{validation_path}")
    print(f"测试集：{test_path}")
    print(f"切分记录：{manifest_path}")


if __name__ == "__main__":
    main()
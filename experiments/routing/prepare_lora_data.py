import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from base_router import (
    ROUTER_SYSTEM_PROMPT,
    build_user_prompt,
)
from validate_dataset import (
    load_and_validate_dataset,
)


DEFAULT_INPUT_FILES = {
    "train": PROJECT_ROOT/"data"
    / "routing"
    / "train.jsonl",
    "validation": PROJECT_ROOT/"data"
    / "routing"
    / "validation.jsonl",
    "test": PROJECT_ROOT/"data"
    / "routing"
    / "test.jsonl",
}

DEFAULT_OUTPUT_DIRECTORY = (
    PROJECT_ROOT /  "data" / "lora"
)


def serialize_target(target):
    """
    将标准路由结果转换成模型需要学习生成的JSON字符串。
    """
    return json.dumps(
        target,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def convert_sample(sample):
    """
    将一条原始路由样本转换成LoRA对话格式。
    """
    sample_id = sample["id"]
    user_query = sample["user_query"]
    target = sample["target"]

    assistant_content = serialize_target(
        target
    )

    # 确认assistant输出确实是合法JSON
    parsed_target = json.loads(
        assistant_content
    )

    if parsed_target != target:
        raise ValueError(
            f"样本{sample_id}转换前后"
            "target内容不一致"
        )

    return {
        "id": sample_id,
        "messages": [
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(
                    user_query
                ),
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ],
        "metadata": sample.get(
            "metadata",
            {},
        ),
    }


def validate_converted_sample(sample):
    """
    检查转换后的LoRA样本结构。
    """
    sample_id = sample.get(
        "id",
        "<unknown>",
    )

    messages = sample.get("messages")

    if not isinstance(messages, list):
        raise ValueError(
            f"样本{sample_id}的messages"
            "必须是列表"
        )

    if len(messages) != 3:
        raise ValueError(
            f"样本{sample_id}必须包含"
            "system、user、assistant三条消息"
        )

    expected_roles = [
        "system",
        "user",
        "assistant",
    ]

    actual_roles = [
        message.get("role")
        for message in messages
    ]

    if actual_roles != expected_roles:
        raise ValueError(
            f"样本{sample_id}角色顺序错误："
            f"{actual_roles}"
        )

    for message in messages:
        content = message.get("content")

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise ValueError(
                f"样本{sample_id}中"
                f"{message.get('role')}的"
                "content不能为空"
            )

    try:
        assistant_output = json.loads(
            messages[2]["content"]
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"样本{sample_id}的assistant"
            f"输出不是合法JSON：{error}"
        ) from error

    required_output_keys = {
        "decision",
        "tool_name",
        "arguments",
    }

    actual_output_keys = set(
        assistant_output
    )

    if (
        actual_output_keys
        != required_output_keys
    ):
        raise ValueError(
            f"样本{sample_id}的assistant"
            "输出字段不正确："
            f"{sorted(actual_output_keys)}"
        )

    if assistant_output["decision"] not in {
        "call_tool",
        "answer_directly",
    }:
        raise ValueError(
            f"样本{sample_id}包含"
            "非法decision："
            f"{assistant_output['decision']}"
        )

    if not isinstance(
        assistant_output["arguments"],
        dict,
    ):
        raise ValueError(
            f"样本{sample_id}的arguments"
            "必须是对象"
        )


def convert_dataset(input_path):
    """
    读取、验证并转换一个数据集。
    """
    samples, statistics = (
        load_and_validate_dataset(
            input_path
        )
    )

    converted_samples = []

    for sample in samples:
        converted_sample = (
            convert_sample(sample)
        )

        validate_converted_sample(
            converted_sample
        )

        converted_samples.append(
            converted_sample
        )

    if len(converted_samples) != len(
        samples
    ):
        raise ValueError(
            "转换前后样本数量不一致："
            f"转换前={len(samples)}，"
            f"转换后={len(converted_samples)}"
        )

    return converted_samples, statistics


def write_jsonl(
    samples,
    output_path,
    overwrite=False,
):
    """
    将转换后的数据写入JSONL文件。
    """
    output_path = Path(output_path)

    if (
        output_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"输出文件已经存在："
            f"{output_path}\n"
            "需要覆盖时添加--overwrite"
        )

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
            file.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )


def get_sample_ids(samples):
    """
    获取数据集中的全部样本ID。
    """
    sample_ids = [
        sample["id"]
        for sample in samples
    ]

    if len(sample_ids) != len(
        set(sample_ids)
    ):
        raise ValueError(
            "同一个数据集中存在重复样本ID"
        )

    return set(sample_ids)


def check_no_data_leakage(
    train_samples,
    validation_samples,
    test_samples,
):
    """
    检查训练集、验证集和测试集之间是否存在ID交叉。
    """
    ids_by_split = {
        "train": get_sample_ids(
            train_samples
        ),
        "validation": get_sample_ids(
            validation_samples
        ),
        "test": get_sample_ids(
            test_samples
        ),
    }

    comparisons = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    for first_name, second_name in (
        comparisons
    ):
        overlap = (
            ids_by_split[first_name]
            & ids_by_split[second_name]
        )

        if overlap:
            raise ValueError(
                f"{first_name}和"
                f"{second_name}之间"
                "存在数据泄漏："
                + ", ".join(
                    sorted(overlap)
                )
            )


def calculate_prompt_hash():
    """
    计算训练Prompt哈希，方便复现实验。
    """
    prompt_text = build_user_prompt(
        "__USER_QUERY_PLACEHOLDER__"
    )

    complete_prompt = (
        ROUTER_SYSTEM_PROMPT
        + "\n"
        + prompt_text
    )

    return hashlib.sha256(
        complete_prompt.encode("utf-8")
    ).hexdigest()


def build_manifest(
    input_files,
    output_files,
    datasets,
):
    """
    生成数据转换记录。
    """
    return {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "format": "messages",
        "prompt_sha256": (
            calculate_prompt_hash()
        ),
        "splits": {
            split_name: {
                "source_file": str(
                    input_files[split_name]
                ),
                "output_file": str(
                    output_files[split_name]
                ),
                "sample_count": len(
                    datasets[split_name]
                ),
            }
            for split_name in (
                "train",
                "validation",
                "test",
            )
        },
        "training_policy": {
            "train": (
                "用于LoRA参数训练"
            ),
            "validation": (
                "用于训练过程评估和"
                "超参数选择"
            ),
            "test": (
                "只用于最终评测，"
                "禁止参与训练和调参"
            ),
        },
    }


def write_manifest(
    manifest,
    output_path,
    overwrite=False,
):
    """
    保存数据转换记录。
    """
    output_path = Path(output_path)

    if (
        output_path.exists()
        and not overwrite
    ):
        raise FileExistsError(
            f"转换记录已经存在："
            f"{output_path}\n"
            "需要覆盖时添加--overwrite"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

        file.write("\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "将工具路由数据转换成"
            "LoRA对话训练格式"
        )
    )

    parser.add_argument(
        "--train-input",
        default=str(
            DEFAULT_INPUT_FILES["train"]
        ),
        help="原始训练集路径",
    )

    parser.add_argument(
        "--validation-input",
        default=str(
            DEFAULT_INPUT_FILES[
                "validation"
            ]
        ),
        help="原始验证集路径",
    )

    parser.add_argument(
        "--test-input",
        default=str(
            DEFAULT_INPUT_FILES["test"]
        ),
        help="原始测试集路径",
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIRECTORY
        ),
        help="LoRA数据输出目录",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有输出文件",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    input_files = {
        "train": Path(
            args.train_input
        ).resolve(),
        "validation": Path(
            args.validation_input
        ).resolve(),
        "test": Path(
            args.test_input
        ).resolve(),
    }

    for split_name, input_path in (
        input_files.items()
    ):
        if not input_path.exists():
            raise FileNotFoundError(
                f"{split_name}数据集不存在："
                f"{input_path}"
            )

        if not input_path.is_file():
            raise ValueError(
                f"{split_name}数据集路径"
                f"不是文件：{input_path}"
            )

    output_directory = Path(
        args.output_dir
    ).resolve()

    output_files = {
        "train": (
            output_directory
            / "train.jsonl"
        ),
        "validation": (
            output_directory
            / "validation.jsonl"
        ),
        "test": (
            output_directory
            / "test.jsonl"
        ),
    }

    manifest_path = (
        output_directory
        / "manifest.json"
    )

    all_output_paths = [
        *output_files.values(),
        manifest_path,
    ]

    if not args.overwrite:
        existing_paths = [
            path
            for path in all_output_paths
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
                "需要覆盖时添加--overwrite"
            )

    datasets = {}

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        print(
            f"正在转换{split_name}："
            f"{input_files[split_name]}"
        )

        converted_samples, _ = (
            convert_dataset(
                input_files[split_name]
            )
        )

        datasets[split_name] = (
            converted_samples
        )

        print(
            f"{split_name}转换完成："
            f"{len(converted_samples)}条"
        )

    check_no_data_leakage(
        train_samples=datasets["train"],
        validation_samples=(
            datasets["validation"]
        ),
        test_samples=datasets["test"],
    )

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        write_jsonl(
            samples=datasets[split_name],
            output_path=(
                output_files[split_name]
            ),
            overwrite=args.overwrite,
        )

    manifest = build_manifest(
        input_files=input_files,
        output_files=output_files,
        datasets=datasets,
    )

    write_manifest(
        manifest=manifest,
        output_path=manifest_path,
        overwrite=args.overwrite,
    )

    print("\n" + "=" * 70)
    print("LoRA数据转换完成")

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        print(
            f"{split_name}："
            f"{len(datasets[split_name])}条"
        )

        print(
            f"保存位置："
            f"{output_files[split_name]}"
        )

    print(
        f"转换记录：{manifest_path}"
    )

    print(
        "检查结果：未发现样本ID泄漏"
    )

    print(
        "注意：test.jsonl禁止参与"
        "LoRA训练和超参数调整"
    )


if __name__ == "__main__":
    main()
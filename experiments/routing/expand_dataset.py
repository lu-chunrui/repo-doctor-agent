import argparse
import json
import re
import time
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )
from agent.core  import (
    ToolCallingLLMClient,
)
from validate_dataset import (
    load_and_validate_dataset,
    validate_sample,
)


SYSTEM_PROMPT = """
你是工具路由数据集生成助手。

你的任务是改写用户问题，不是回答问题。

要求：
1. 保持原始问题的任务意图不变。
2. 保持正确工具不变。
3. 保持文件名、函数名、类名和异常类型不变。
4. 不得增加原始问题中不存在的信息。
5. 使用自然、真实、多样的中文表达。
6. 同时包含简单、口语化和容易混淆的表达。
7. difficulty只能是easy、medium或hard。
8. 只返回JSON数组，不要解释，不要使用Markdown代码块。

数组元素格式：

{
  "user_query": "改写后的问题",
  "difficulty": "easy"
}
""".strip()


VALID_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
}


def normalize_query(text):
    """
    归一化问题，用于过滤重复表达。
    """
    text = text.lower().strip()

    text = re.sub(
        r"\s+",
        "",
        text,
    )

    text = re.sub(
        r"[，。！？、；：“”‘’（）,.!?;:\"'()]",
        "",
        text,
    )

    return text


def extract_json_array(text):
    """
    从模型回答中提取JSON数组。
    """
    if not isinstance(text, str):
        raise ValueError(
            "模型回答必须是字符串"
        )

    text = text.strip()

    # 去除可能出现的Markdown围栏
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    start_index = text.find("[")
    end_index = text.rfind("]")

    if (
        start_index == -1
        or end_index == -1
        or end_index < start_index
    ):
        raise ValueError(
            "模型回答中没有找到JSON数组"
        )

    json_text = text[
        start_index:end_index + 1
    ]

    try:
        result = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"模型返回的JSON解析失败：{error}"
        ) from error

    if not isinstance(result, list):
        raise ValueError(
            "模型返回结果必须是JSON数组"
        )

    return result


def build_generation_prompt(
    seed_sample,
    required_count,
    existing_queries,
):
    """
    根据一条黄金样本构造扩充提示词。
    """
    target_text = json.dumps(
        seed_sample["target"],
        ensure_ascii=False,
        indent=2,
    )

    existing_text = "\n".join(
        f"- {query}"
        for query in existing_queries
    )

    return f"""
请为下面的问题生成{required_count}条新的表达。

原始问题：
{seed_sample["user_query"]}

固定的正确路由结果：
{target_text}

已有表达：
{existing_text}

要求：
1. 必须保持固定路由结果不变。
2. 不能生成与已有表达相同的问题。
3. 不能添加新的文件名、函数名或错误类型。
4. 只改写用户问题，不要输出答案。
5. 返回恰好{required_count}个JSON对象。
""".strip()


def create_variant_sample(
    seed_sample,
    user_query,
    difficulty,
    variant_number,
):
    """
    使用种子样本的target创建改写样本。
    """
    sample = {
        "id": (
            f"{seed_sample['id']}"
            f"_v{variant_number}"
        ),
        "user_query": user_query,
        "target": seed_sample["target"],
        "metadata": {
            "difficulty": difficulty,
            "group_id": (
                seed_sample["metadata"][
                    "group_id"
                ]
            ),
            "source": "ai_assisted",
        },
    }

    validate_sample(sample)

    return sample


def generate_variants(
    llm_client,
    seed_sample,
    variant_count=5,
    max_attempts=3,
):
    """
    为一条种子样本生成多个不同表达。
    """
    variants = []

    existing_queries = [
        seed_sample["user_query"]
    ]

    normalized_queries = {
        normalize_query(
            seed_sample["user_query"]
        )
    }

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        remaining_count = (
            variant_count - len(variants)
        )

        if remaining_count <= 0:
            break

        prompt = build_generation_prompt(
            seed_sample=seed_sample,
            required_count=remaining_count,
            existing_queries=existing_queries,
        )

        response = (
            llm_client.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                tools=None,
                temperature=0.8,
            )
        )

        content = response.get(
            "content",
            "",
        )

        try:
            generated_items = (
                extract_json_array(content)
            )
        except ValueError as error:
            print(
                f"  第{attempt}次返回格式错误："
                f"{error}"
            )
            continue

        for item in generated_items:
            if len(variants) >= variant_count:
                break

            if not isinstance(item, dict):
                continue

            user_query = item.get(
                "user_query"
            )

            difficulty = item.get(
                "difficulty"
            )

            if (
                not isinstance(user_query, str)
                or not user_query.strip()
            ):
                continue

            user_query = user_query.strip()

            if (
                difficulty
                not in VALID_DIFFICULTIES
            ):
                continue

            normalized = normalize_query(
                user_query
            )

            if (
                not normalized
                or normalized
                in normalized_queries
            ):
                continue

            variant_number = (
                len(variants) + 1
            )

            try:
                sample = create_variant_sample(
                    seed_sample=seed_sample,
                    user_query=user_query,
                    difficulty=difficulty,
                    variant_number=variant_number,
                )
            except ValueError as error:
                print(
                    f"  忽略非法改写：{error}"
                )
                continue

            variants.append(sample)
            existing_queries.append(
                user_query
            )
            normalized_queries.add(
                normalized
            )

    if len(variants) != variant_count:
        raise RuntimeError(
            f"{seed_sample['id']}只生成了"
            f"{len(variants)}条有效改写，"
            f"需要{variant_count}条"
        )

    return variants


def validate_unique_ids(samples):
    """
    检查所有样本ID是否唯一。
    """
    seen_ids = set()

    for sample in samples:
        sample_id = sample["id"]

        if sample_id in seen_ids:
            raise ValueError(
                f"样本ID重复：{sample_id}"
            )

        seen_ids.add(sample_id)


def save_jsonl(
    samples,
    output_path,
):
    """
    保存JSONL数据集。
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
            file.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="扩充工具路由数据集"
    )

    parser.add_argument(
        "--input",
        default=(
            "data/routing/"
            "gold_samples.jsonl"
        ),
        help="黄金样本文件",
    )

    parser.add_argument(
        "--output",
        default=(
            "data/routing/"
            "raw_samples.jsonl"
        ),
        help="扩充结果文件",
    )

    parser.add_argument(
        "--variants-per-sample",
        type=int,
        default=5,
        help="每条种子样本生成的改写数量",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="两次API请求之间等待的秒数",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖现有输出文件",
    )

    args = parser.parse_args()

    if args.variants_per_sample < 1:
        raise ValueError(
            "variants-per-sample必须大于等于1"
        )

    if args.delay < 0:
        raise ValueError(
            "delay不能小于0"
        )

    input_path = Path(args.input)
    output_path = Path(args.output)

    if (
        input_path.resolve()
        == output_path.resolve()
    ):
        raise ValueError(
            "输入文件与输出文件不能相同"
        )

    if (
        output_path.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"输出文件已经存在：{output_path}\n"
            "需要覆盖时添加--overwrite"
        )

    seed_samples, statistics = (
        load_and_validate_dataset(
            input_path
        )
    )

    print(
        f"加载了"
        f"{statistics['total_samples']}"
        f"条黄金样本"
    )

    llm_client = (
        ToolCallingLLMClient
        .from_environment()
    )

    all_samples = list(seed_samples)

    for index, seed_sample in enumerate(
        seed_samples,
        start=1,
    ):
        print(
            f"\n[{index}/{len(seed_samples)}] "
            f"扩充样本：{seed_sample['id']}"
        )

        print(
            f"原问题："
            f"{seed_sample['user_query']}"
        )

        variants = generate_variants(
            llm_client=llm_client,
            seed_sample=seed_sample,
            variant_count=(
                args.variants_per_sample
            ),
        )

        all_samples.extend(variants)

        for variant in variants:
            print(
                f"  - "
                f"{variant['user_query']}"
            )

        if args.delay > 0:
            time.sleep(args.delay)

    validate_unique_ids(all_samples)

    for sample in all_samples:
        validate_sample(sample)

    save_jsonl(
        samples=all_samples,
        output_path=output_path,
    )

    print("\n" + "=" * 70)
    print(f"黄金样本：{len(seed_samples)}")
    print(
        "新增样本："
        f"{len(all_samples) - len(seed_samples)}"
    )
    print(f"最终样本：{len(all_samples)}")
    print(f"保存位置：{output_path}")
    print("扩充完成，请继续人工审核。")


if __name__ == "__main__":
    main()
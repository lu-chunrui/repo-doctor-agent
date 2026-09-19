import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from base_router import (
    parse_json_object,
)
from metrics import (
    calculate_routing_metrics,
    get_prediction_label,
    get_target_label,
    print_metrics,
)
from schema import (
    validate_route_target,
)


DEFAULT_MODEL_PATH = (
    "/root/autodl-tmp/models/"
    "Qwen2.5-1.5B-Instruct"
)

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT
    /  "data" / "lora"
    / "validation.jsonl"
)

DEFAULT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "reports/routing/"
    / "qwen_1_5b_base_validation_predictions.jsonl"
)

DEFAULT_METRICS_PATH = (
    PROJECT_ROOT
    / "reports/routing/"
    / "qwen_1_5b_base_validation_metrics.json"
)


def resolve_dtype(dtype_name):
    """
    将命令行中的精度名称转换成PyTorch类型。
    """
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if dtype_name not in dtype_mapping:
        raise ValueError(
            f"不支持的dtype：{dtype_name}"
        )

    if (
        dtype_name == "bfloat16"
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError(
            "当前GPU不支持bfloat16，"
            "请使用float16"
        )

    return dtype_mapping[dtype_name]


def load_lora_dataset(dataset_path):
    """
    读取messages格式的LoRA数据集。
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"数据集不存在：{dataset_path}"
        )

    samples = []
    seen_ids = set()

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
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"数据集第{line_number}行"
                    f"不是合法JSON：{error}"
                ) from error

            sample_id = sample.get("id")

            if not isinstance(sample_id, str):
                raise ValueError(
                    f"第{line_number}行缺少合法id"
                )

            if sample_id in seen_ids:
                raise ValueError(
                    f"发现重复样本ID：{sample_id}"
                )

            seen_ids.add(sample_id)

            messages = sample.get("messages")

            if (
                not isinstance(messages, list)
                or len(messages) != 3
            ):
                raise ValueError(
                    f"样本{sample_id}必须包含"
                    "三条messages"
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
                        f"样本{sample_id}存在"
                        "空消息内容"
                    )

            try:
                gold_target = json.loads(
                    messages[2]["content"]
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"样本{sample_id}的"
                    "assistant标准答案不是合法JSON"
                ) from error

            validate_route_target(
                gold_target
            )

            samples.append(sample)

    if not samples:
        raise ValueError(
            f"数据集为空：{dataset_path}"
        )

    return samples


def extract_user_query(user_content):
    """
    从训练Prompt中提取原始用户问题，
    仅用于生成易读的预测报告。
    """
    marker = "请对下面的用户问题进行路由："

    if marker not in user_content:
        return user_content

    query_text = user_content.split(
        marker,
        maxsplit=1,
    )[1]

    ending_marker = "请只返回路由JSON。"

    if ending_marker in query_text:
        query_text = query_text.split(
            ending_marker,
            maxsplit=1,
        )[0]

    return query_text.strip()


def load_model_and_tokenizer(
    model_path,
    dtype,
    adapter_path=None,
):
    """
    加载基础模型；提供adapter_path时，
    同时加载LoRA Adapter。
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"模型目录不存在：{model_path}"
        )

    print(f"正在加载分词器：{model_path}")

    tokenizer = (
        AutoTokenizer.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print(f"正在加载基础模型：{model_path}")

    started_at = time.perf_counter()

    model = (
        AutoModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map="auto",
            local_files_only=True,
        )
    )

    if adapter_path:
        from peft import PeftModel

        adapter_path = Path(adapter_path)

        if not adapter_path.exists():
            raise FileNotFoundError(
                "LoRA Adapter不存在："
                f"{adapter_path}"
            )

        print(
            f"正在加载LoRA Adapter："
            f"{adapter_path}"
        )

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            local_files_only=True,
        )

    model.eval()

    load_seconds = (
        time.perf_counter() - started_at
    )

    print(
        f"模型加载完成，耗时"
        f"{load_seconds:.2f}秒"
    )

    print(
        f"模型设备："
        f"{next(model.parameters()).device}"
    )

    if torch.cuda.is_available():
        print(
            f"GPU："
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            "当前显存占用："
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

    return (
        tokenizer,
        model,
        load_seconds,
    )


def generate_route(
    tokenizer,
    model,
    messages,
    max_new_tokens,
):
    """
    使用本地模型生成一条路由结果。
    """
    prompt_messages = messages[:2]

    model_inputs = (
        tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    )

    model_inputs = {
        key: value.to(model.device)
        for key, value in (
            model_inputs.items()
        )
    }

    prompt_length = (
        model_inputs["input_ids"].shape[-1]
    )

    started_at = time.perf_counter()

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=(
                tokenizer.pad_token_id
            ),
            eos_token_id=(
                tokenizer.eos_token_id
            ),
            use_cache=True,
        )

    latency_seconds = (
        time.perf_counter() - started_at
    )

    new_token_ids = generated_ids[
        0,
        prompt_length:,
    ]

    raw_text = tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
    ).strip()

    return {
        "raw_text": raw_text,
        "latency_seconds": (
            latency_seconds
        ),
        "generated_tokens": int(
            new_token_ids.shape[-1]
        ),
    }


def parse_route_result(raw_text):
    """
    解析模型输出并检查路由Schema。
    """
    prediction = None
    json_valid = False
    schema_valid = False
    parse_error = None
    schema_error = None

    try:
        prediction = parse_json_object(
            raw_text
        )

        json_valid = True

    except Exception as error:
        parse_error = (
            f"{type(error).__name__}: "
            f"{error}"
        )

        return {
            "prediction": None,
            "json_valid": False,
            "schema_valid": False,
            "parse_error": parse_error,
            "schema_error": None,
        }

    try:
        validate_route_target(
            prediction
        )

        schema_valid = True

    except Exception as error:
        schema_error = (
            f"{type(error).__name__}: "
            f"{error}"
        )

    return {
        "prediction": prediction,
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "parse_error": parse_error,
        "schema_error": schema_error,
    }


def evaluate_sample(
    tokenizer,
    model,
    sample,
    max_new_tokens,
):
    """
    评测一条LoRA格式样本。
    """
    messages = sample["messages"]

    gold_target = json.loads(
        messages[2]["content"]
    )

    user_query = extract_user_query(
        messages[1]["content"]
    )

    metadata = sample.get(
        "metadata",
        {},
    )

    try:
        generation_result = generate_route(
            tokenizer=tokenizer,
            model=model,
            messages=messages,
            max_new_tokens=(
                max_new_tokens
            ),
        )

        raw_text = generation_result[
            "raw_text"
        ]

        route_result = parse_route_result(
            raw_text
        )

        inference_error = None

    except Exception as error:
        raw_text = ""

        generation_result = {
            "latency_seconds": 0.0,
            "generated_tokens": 0,
        }

        route_result = {
            "prediction": None,
            "json_valid": False,
            "schema_valid": False,
            "parse_error": None,
            "schema_error": None,
        }

        inference_error = (
            f"{type(error).__name__}: "
            f"{error}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "id": sample["id"],
        "group_id": metadata.get(
            "group_id"
        ),
        "difficulty": metadata.get(
            "difficulty"
        ),
        "user_query": user_query,
        "gold": gold_target,
        "prediction": route_result[
            "prediction"
        ],
        "raw_text": raw_text,
        "json_valid": route_result[
            "json_valid"
        ],
        "schema_valid": route_result[
            "schema_valid"
        ],
        "parse_error": route_result[
            "parse_error"
        ],
        "schema_error": route_result[
            "schema_error"
        ],
        # 复用现有指标代码中的api_error字段，
        # 对本地模型而言表示推理错误。
        "api_error": inference_error,
        "latency_seconds": round(
            generation_result[
                "latency_seconds"
            ],
            4,
        ),
        "generated_tokens": (
            generation_result[
                "generated_tokens"
            ]
        ),
    }


def load_existing_records(output_path):
    """
    读取已有预测，支持中断后继续。
    """
    output_path = Path(output_path)

    if not output_path.exists():
        return {}

    records = {}

    with output_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"已有结果第{line_number}行"
                    f"不是合法JSON：{error}"
                ) from error

            sample_id = record.get("id")

            if not sample_id:
                raise ValueError(
                    f"已有结果第{line_number}行"
                    "缺少id"
                )

            records[sample_id] = record

    return records


def append_record(
    output_path,
    record,
):
    """
    每完成一条立即保存，防止中断丢失。
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def rewrite_records(
    output_path,
    records,
):
    """
    按数据集原顺序重写预测文件。
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
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def calculate_generation_statistics(
    records,
):
    """
    计算本地生成速度统计。
    """
    successful_records = [
        record
        for record in records
        if not record.get("api_error")
    ]

    total_latency = sum(
        record["latency_seconds"]
        for record in successful_records
    )

    total_generated_tokens = sum(
        record["generated_tokens"]
        for record in successful_records
    )

    average_latency = (
        total_latency
        / len(successful_records)
        if successful_records
        else 0.0
    )

    tokens_per_second = (
        total_generated_tokens
        / total_latency
        if total_latency > 0
        else 0.0
    )

    return {
        "average_latency_seconds": round(
            average_latency,
            4,
        ),
        "total_generation_seconds": round(
            total_latency,
            4,
        ),
        "total_generated_tokens": (
            total_generated_tokens
        ),
        "tokens_per_second": round(
            tokens_per_second,
            2,
        ),
    }


def save_metrics_report(
    output_path,
    metrics,
    generation_statistics,
    dataset_path,
    model_path,
    adapter_path,
    dtype_name,
    model_load_seconds,
):
    """
    保存指标与实验配置。
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": str(dataset_path),
        "model_path": str(model_path),
        "adapter_path": (
            str(adapter_path)
            if adapter_path
            else None
        ),
        "dtype": dtype_name,
        "model_load_seconds": round(
            model_load_seconds,
            4,
        ),
        "hardware": {
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "generation": (
            generation_statistics
        ),
        "metrics": metrics,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

        file.write("\n")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "评测本地基础模型或LoRA模型的"
            "工具路由能力"
        )
    )

    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="本地基础模型路径",
    )

    parser.add_argument(
        "--adapter-path",
        default=None,
        help=(
            "可选的LoRA Adapter路径；"
            "不传表示评测Base模型"
        ),
    )

    parser.add_argument(
        "--dataset",
        default=str(
            DEFAULT_DATASET_PATH
        ),
        help="messages格式评测数据集",
    )

    parser.add_argument(
        "--predictions",
        default=str(
            DEFAULT_PREDICTIONS_PATH
        ),
        help="逐条预测输出路径",
    )

    parser.add_argument(
        "--metrics",
        default=str(
            DEFAULT_METRICS_PATH
        ),
        help="汇总指标输出路径",
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
        default="float16",
        help="模型加载精度",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="每条样本最大生成Token数",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只评测前N条，适合试运行",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="删除旧预测并重新评测",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA不可用，请在GPU环境运行"
        )

    if args.max_new_tokens < 1:
        raise ValueError(
            "max-new-tokens必须大于0"
        )

    if (
        args.limit is not None
        and args.limit < 1
    ):
        raise ValueError(
            "limit必须大于0"
        )

    dataset_path = Path(
        args.dataset
    ).resolve()

    predictions_path = Path(
        args.predictions
    ).resolve()

    metrics_path = Path(
        args.metrics
    ).resolve()

    adapter_path = (
        Path(args.adapter_path).resolve()
        if args.adapter_path
        else None
    )

    samples = load_lora_dataset(
        dataset_path
    )

    if args.limit is not None:
        samples = samples[:args.limit]

    print(
        f"成功加载{len(samples)}条"
        "评测样本"
    )

    dtype = resolve_dtype(
        args.dtype
    )

    tokenizer, model, load_seconds = (
        load_model_and_tokenizer(
            model_path=args.model_path,
            dtype=dtype,
            adapter_path=adapter_path,
        )
    )

    if args.overwrite:
        if predictions_path.exists():
            predictions_path.unlink()

        if metrics_path.exists():
            metrics_path.unlink()

        existing_records = {}

    else:
        existing_records = (
            load_existing_records(
                predictions_path
            )
        )

    selected_ids = {
        sample["id"]
        for sample in samples
    }

    existing_records = {
        sample_id: record
        for sample_id, record in (
            existing_records.items()
        )
        if sample_id in selected_ids
    }

    if existing_records:
        print(
            f"发现{len(existing_records)}条"
            "已有预测，将继续剩余样本"
        )

    total_samples = len(samples)

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        sample_id = sample["id"]

        if sample_id in existing_records:
            print(
                f"[{index}/{total_samples}] "
                f"跳过已有结果：{sample_id}"
            )

            continue

        user_query = extract_user_query(
            sample["messages"][1][
                "content"
            ]
        )

        print(
            f"\n[{index}/{total_samples}] "
            f"评测：{sample_id}"
        )

        print(f"问题：{user_query}")

        record = evaluate_sample(
            tokenizer=tokenizer,
            model=model,
            sample=sample,
            max_new_tokens=(
                args.max_new_tokens
            ),
        )

        existing_records[
            sample_id
        ] = record

        append_record(
            output_path=predictions_path,
            record=record,
        )

        gold_label = get_target_label(
            record["gold"]
        )

        predicted_label = (
            get_prediction_label(
                prediction=record[
                    "prediction"
                ],
                json_valid=record[
                    "json_valid"
                ],
            )
        )

        print(
            f"标准={gold_label}，"
            f"预测={predicted_label}，"
            f"耗时="
            f"{record['latency_seconds']}秒，"
            f"生成="
            f"{record['generated_tokens']} tokens"
        )

        if record["api_error"]:
            print(
                f"推理错误："
                f"{record['api_error']}"
            )

        elif record["parse_error"]:
            print(
                f"JSON解析错误："
                f"{record['parse_error']}"
            )

            print(
                f"原始输出："
                f"{record['raw_text']}"
            )

        elif record["schema_error"]:
            print(
                f"Schema错误："
                f"{record['schema_error']}"
            )

    ordered_records = [
        existing_records[sample["id"]]
        for sample in samples
        if sample["id"]
        in existing_records
    ]

    rewrite_records(
        output_path=predictions_path,
        records=ordered_records,
    )

    metrics = calculate_routing_metrics(
        ordered_records
    )

    generation_statistics = (
        calculate_generation_statistics(
            ordered_records
        )
    )

    save_metrics_report(
        output_path=metrics_path,
        metrics=metrics,
        generation_statistics=(
            generation_statistics
        ),
        dataset_path=dataset_path,
        model_path=args.model_path,
        adapter_path=adapter_path,
        dtype_name=args.dtype,
        model_load_seconds=load_seconds,
    )

    print_metrics(metrics)

    print("\n本地生成统计：")

    print(
        "平均每条耗时："
        f"{generation_statistics['average_latency_seconds']}"
        "秒"
    )

    print(
        "生成速度："
        f"{generation_statistics['tokens_per_second']}"
        " tokens/s"
    )

    print("\n结果已经保存：")

    print(
        f"- 逐条预测：{predictions_path}"
    )

    print(
        f"- 汇总指标：{metrics_path}"
    )


if __name__ == "__main__":
    main()
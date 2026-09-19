import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from base_router import (
    BaseRouter,
)
from metrics import (
    calculate_routing_metrics,
    get_prediction_label,
    get_target_label,
    print_metrics,
)
from validate_dataset import (
    load_and_validate_dataset,
)


def load_existing_records(
    output_path,
):
    """
    加载已经完成的预测，支持中断后继续。
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

            sample_id = record.get(
                "id"
            )

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
    每完成一条预测立即保存，防止中断丢失。
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
    按数据集顺序重新保存完整结果。
    """
    output_path = Path(output_path)

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


def evaluate_sample(
    router,
    sample,
):
    """
    对一条标准样本执行路由预测。
    """
    started_at = time.perf_counter()

    try:
        route_result = router.route(
            sample["user_query"]
        )

        api_error = None

    except Exception as error:
        route_result = {
            "raw_text": "",
            "prediction": None,
            "json_valid": False,
            "schema_valid": False,
            "parse_error": None,
            "schema_error": None,
        }

        api_error = (
            f"{type(error).__name__}: "
            f"{error}"
        )

    latency_seconds = (
        time.perf_counter()
        - started_at
    )

    record = {
        "id": sample["id"],
        "group_id": (
            sample["metadata"][
                "group_id"
            ]
        ),
        "difficulty": (
            sample["metadata"][
                "difficulty"
            ]
        ),
        "user_query": (
            sample["user_query"]
        ),
        "gold": sample["target"],
        "prediction": (
            route_result.get(
                "prediction"
            )
        ),
        "raw_text": (
            route_result.get(
                "raw_text",
                "",
            )
        ),
        "json_valid": bool(
            route_result.get(
                "json_valid"
            )
        ),
        "schema_valid": bool(
            route_result.get(
                "schema_valid"
            )
        ),
        "parse_error": (
            route_result.get(
                "parse_error"
            )
        ),
        "schema_error": (
            route_result.get(
                "schema_error"
            )
        ),
        "api_error": api_error,
        "latency_seconds": round(
            latency_seconds,
            4,
        ),
    }

    return record


def save_metrics(
    metrics_path,
    metrics,
    dataset_path,
    model_name,
):
    """
    保存指标和实验元数据。
    """
    metrics_path = Path(
        metrics_path
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "created_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "dataset": str(
            dataset_path
        ),
        "model": model_name,
        "metrics": metrics,
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main():
    parser = argparse.ArgumentParser(
        description="批量评测Base Router"
    )

    parser.add_argument(
        "--dataset",
        default=(
            "data/routing/"
            "validation.jsonl"
        ),
        help="评测数据集",
    )

    parser.add_argument(
        "--predictions",
        default=(
            "reports/routing/"
            "base_validation_predictions.jsonl"
        ),
        help="逐条预测结果",
    )

    parser.add_argument(
        "--metrics",
        default=(
            "reports/routing/"
            "base_validation_metrics.json"
        ),
        help="指标结果",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只评测前N条，适合先试运行",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="每次API请求后的等待秒数",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="删除旧预测并重新评测",
    )

    args = parser.parse_args()

    if (
        args.limit is not None
        and args.limit < 1
    ):
        raise ValueError(
            "limit必须大于等于1"
        )

    if args.delay < 0:
        raise ValueError(
            "delay不能小于0"
        )

    dataset_path = Path(
        args.dataset
    )

    predictions_path = Path(
        args.predictions
    )

    metrics_path = Path(
        args.metrics
    )

    samples, statistics = (
        load_and_validate_dataset(
            dataset_path
        )
    )

    if args.limit is not None:
        samples = samples[
            :args.limit
        ]

    print(
        f"加载了{len(samples)}条"
        f"评测样本"
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

    if existing_records:
        print(
            f"发现{len(existing_records)}条"
            "已有预测，将继续未完成部分"
        )

    router = (
        BaseRouter.from_environment()
    )

    model_name = getattr(
        router.llm_client,
        "model",
        "unknown",
    )

    selected_sample_ids = {
        sample["id"]
        for sample in samples
    }

    # 删除不属于本次数据集的旧结果
    existing_records = {
        sample_id: record
        for sample_id, record in (
            existing_records.items()
        )
        if sample_id in selected_sample_ids
    }

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

        print(
            f"[{index}/{total_samples}] "
            f"评测：{sample_id}"
        )

        print(
            f"问题："
            f"{sample['user_query']}"
        )

        record = evaluate_sample(
            router=router,
            sample=sample,
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
                prediction=(
                    record["prediction"]
                ),
                json_valid=(
                    record["json_valid"]
                ),
            )
        )

        print(
            f"标准={gold_label}，"
            f"预测={predicted_label}，"
            f"耗时="
            f"{record['latency_seconds']}秒"
        )

        if record["api_error"]:
            print(
                f"API错误："
                f"{record['api_error']}"
            )

        elif record["schema_error"]:
            print(
                f"Schema错误："
                f"{record['schema_error']}"
            )

        if args.delay > 0:
            time.sleep(args.delay)

    ordered_records = [
        existing_records[
            sample["id"]
        ]
        for sample in samples
        if sample["id"]
        in existing_records
    ]

    rewrite_records(
        output_path=predictions_path,
        records=ordered_records,
    )

    metrics = (
        calculate_routing_metrics(
            ordered_records
        )
    )

    save_metrics(
        metrics_path=metrics_path,
        metrics=metrics,
        dataset_path=dataset_path,
        model_name=model_name,
    )

    print_metrics(metrics)

    print("\n结果已经保存：")
    print(
        f"- 逐条预测："
        f"{predictions_path}"
    )
    print(
        f"- 汇总指标："
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()
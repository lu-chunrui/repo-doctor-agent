import sys
from collections import defaultdict
from pathlib import Path
PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )
from schema import (
    VALID_TOOL_NAMES,
)


INVALID_JSON_LABEL = "__invalid_json__"
INVALID_DECISION_LABEL = "__invalid_decision__"
INVALID_TOOL_LABEL = "__invalid_tool__"


def safe_divide(numerator, denominator):
    
    if denominator == 0:
        return 0.0

    return numerator / denominator


def get_target_label(target):
    
    if (
        target.get("decision")
        == "answer_directly"
    ):
        return "answer_directly"

    return target.get(
        "tool_name",
        INVALID_TOOL_LABEL,
    )


def get_prediction_label(
    prediction,
    json_valid,
):
   
    if not json_valid:
        return INVALID_JSON_LABEL

    if not isinstance(prediction, dict):
        return INVALID_JSON_LABEL

    decision = prediction.get(
        "decision"
    )

    if decision == "answer_directly":
        return "answer_directly"

    if decision != "call_tool":
        return INVALID_DECISION_LABEL

    tool_name = prediction.get(
        "tool_name"
    )

    if tool_name not in VALID_TOOL_NAMES:
        return INVALID_TOOL_LABEL

    return tool_name


def calculate_routing_metrics(records):
    
    total = len(records)

    successful_requests = 0
    json_valid_count = 0
    schema_valid_count = 0
    decision_correct_count = 0

    gold_tool_call_count = 0
    tool_name_correct_count = 0

    argument_keys_covered_count = 0
    arguments_exact_match_count = 0

    hallucinated_tool_count = 0

    confusion_matrix = defaultdict(
        lambda: defaultdict(int)
    )

    for record in records:
        gold = record["gold"]

        prediction = record.get(
            "prediction"
        )

        json_valid = bool(
            record.get("json_valid")
        )

        schema_valid = bool(
            record.get("schema_valid")
        )

        api_error = record.get(
            "api_error"
        )

        if not api_error:
            successful_requests += 1

        if json_valid:
            json_valid_count += 1

        if schema_valid:
            schema_valid_count += 1

        predicted_decision = None

        if isinstance(prediction, dict):
            predicted_decision = (
                prediction.get("decision")
            )

        if (
            predicted_decision
            == gold["decision"]
        ):
            decision_correct_count += 1

        gold_label = get_target_label(
            gold
        )

        predicted_label = (
            get_prediction_label(
                prediction=prediction,
                json_valid=json_valid,
            )
        )

        confusion_matrix[
            gold_label
        ][predicted_label] += 1

        # 检查是否编造了不存在的工具
        if isinstance(prediction, dict):
            if (
                prediction.get("decision")
                == "call_tool"
                and prediction.get(
                    "tool_name"
                )
                not in VALID_TOOL_NAMES
            ):
                hallucinated_tool_count += 1

        if gold["decision"] != "call_tool":
            continue

        gold_tool_call_count += 1

        predicted_tool_name = None
        predicted_arguments = None

        if isinstance(prediction, dict):
            predicted_tool_name = (
                prediction.get("tool_name")
            )

            predicted_arguments = (
                prediction.get("arguments")
            )

        tool_name_correct = (
            predicted_decision
            == "call_tool"
            and predicted_tool_name
            == gold["tool_name"]
        )

        if tool_name_correct:
            tool_name_correct_count += 1

        if not (
            tool_name_correct
            and isinstance(
                predicted_arguments,
                dict,
            )
        ):
            continue

        gold_arguments = gold.get(
            "arguments",
            {},
        )

        gold_argument_keys = set(
            gold_arguments
        )

        predicted_argument_keys = set(
            predicted_arguments
        )

        if gold_argument_keys.issubset(
            predicted_argument_keys
        ):
            argument_keys_covered_count += 1

        if predicted_arguments == gold_arguments:
            arguments_exact_match_count += 1

    serializable_confusion_matrix = {
        gold_label: dict(
            sorted(predictions.items())
        )
        for gold_label, predictions in sorted(
            confusion_matrix.items()
        )
    }

    metrics = {
        "total_samples": total,
        "successful_requests": (
            successful_requests
        ),
        "api_error_count": (
            total - successful_requests
        ),
        "json_valid_count": (
            json_valid_count
        ),
        "schema_valid_count": (
            schema_valid_count
        ),
        "decision_correct_count": (
            decision_correct_count
        ),
        "gold_tool_call_count": (
            gold_tool_call_count
        ),
        "tool_name_correct_count": (
            tool_name_correct_count
        ),
        "argument_keys_covered_count": (
            argument_keys_covered_count
        ),
        "arguments_exact_match_count": (
            arguments_exact_match_count
        ),
        "hallucinated_tool_count": (
            hallucinated_tool_count
        ),
        "api_success_rate": safe_divide(
            successful_requests,
            total,
        ),
        "json_valid_rate": safe_divide(
            json_valid_count,
            total,
        ),
        "schema_valid_rate": safe_divide(
            schema_valid_count,
            total,
        ),
        "decision_accuracy": safe_divide(
            decision_correct_count,
            total,
        ),
        "tool_name_accuracy": safe_divide(
            tool_name_correct_count,
            gold_tool_call_count,
        ),
        "argument_key_coverage_rate": (
            safe_divide(
                argument_keys_covered_count,
                gold_tool_call_count,
            )
        ),
        "arguments_exact_match_rate": (
            safe_divide(
                arguments_exact_match_count,
                gold_tool_call_count,
            )
        ),
        "tool_hallucination_rate": (
            safe_divide(
                hallucinated_tool_count,
                total,
            )
        ),
        "confusion_matrix": (
            serializable_confusion_matrix
        ),
    }

    return metrics


def format_percent(value):
    return f"{value * 100:.2f}%"


def print_metrics(metrics):
    
    print("\n" + "=" * 70)
    print("路由评测结果")

    print(
        f"样本总数："
        f"{metrics['total_samples']}"
    )

    print(
        f"API成功率："
        f"{format_percent(metrics['api_success_rate'])}"
    )

    print(
        f"JSON合法率："
        f"{format_percent(metrics['json_valid_rate'])}"
    )

    print(
        f"Schema合法率："
        f"{format_percent(metrics['schema_valid_rate'])}"
    )

    print(
        f"决策准确率："
        f"{format_percent(metrics['decision_accuracy'])}"
    )

    print(
        f"工具名称准确率："
        f"{format_percent(metrics['tool_name_accuracy'])}"
    )

    print(
        f"标准参数键覆盖率："
        f"{format_percent(metrics['argument_key_coverage_rate'])}"
    )

    print(
        f"参数完全一致率："
        f"{format_percent(metrics['arguments_exact_match_rate'])}"
    )

    print(
        f"工具幻觉率："
        f"{format_percent(metrics['tool_hallucination_rate'])}"
    )

    print("\n混淆矩阵：")

    for gold_label, predictions in (
        metrics["confusion_matrix"].items()
    ):
        prediction_text = ", ".join(
            f"{label}={count}"
            for label, count in (
                predictions.items()
            )
        )

        print(
            f"- {gold_label} → "
            f"{prediction_text}"
        )
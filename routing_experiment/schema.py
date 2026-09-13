from day9_agent_tools import TOOL_SCHEMAS

VALID_DECISIONS = {
    "call_tool",
    "answer_directly",
}

TOOL_SCHEMA_BY_NAME = {
    item["function"]["name"]: item["function"]
    for item in TOOL_SCHEMAS
}

VALID_TOOL_NAMES = set(TOOL_SCHEMA_BY_NAME)

def validate_json_type(value, expected_type):
   
    if expected_type == "string":
        return isinstance(value, str)

    if expected_type == "integer":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
        )

    if expected_type == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
        )

    if expected_type == "boolean":
        return isinstance(value, bool)

    if expected_type == "object":
        return isinstance(value, dict)

    if expected_type == "array":
        return isinstance(value, list)

    return True

def validate_route_target(target):
   
    if not isinstance(target, dict):
        raise ValueError(
            "target必须是JSON对象"
        )

    required_target_fields = {
        "decision",
        "tool_name",
        "arguments",
    }

    missing_target_fields = (
        required_target_fields - set(target)
    )

    if missing_target_fields:
        raise ValueError(
            "target缺少字段："
            + ", ".join(
                sorted(missing_target_fields)
            )
        )

    decision = target["decision"]
    tool_name = target["tool_name"]
    arguments = target["arguments"]

    if decision not in VALID_DECISIONS:
        raise ValueError(
            f"不支持的decision：{decision}"
        )

    if not isinstance(arguments, dict):
        raise ValueError(
            "arguments必须是JSON对象"
        )

    # 不调用工具时，不应该出现工具名称和参数
    if decision == "answer_directly":
        if tool_name is not None:
            raise ValueError(
                "answer_directly时tool_name必须为null"
            )

        if arguments:
            raise ValueError(
                "answer_directly时arguments必须为空对象"
            )

        return target

    if not isinstance(tool_name, str):
        raise ValueError(
            "call_tool时tool_name必须是字符串"
        )

    if tool_name not in VALID_TOOL_NAMES:
        raise ValueError(
            f"不存在的工具：{tool_name}"
        )

    function_schema = TOOL_SCHEMA_BY_NAME[
        tool_name
    ]

    parameter_schema = function_schema[
        "parameters"
    ]

    properties = parameter_schema.get(
        "properties",
        {},
    )

    required_arguments = set(
        parameter_schema.get(
            "required",
            [],
        )
    )

    argument_names = set(arguments)

    missing_arguments = (
        required_arguments - argument_names
    )

    if missing_arguments:
        raise ValueError(
            f"工具{tool_name}缺少必填参数："
            + ", ".join(
                sorted(missing_arguments)
            )
        )

    unexpected_arguments = (
        argument_names - set(properties)
    )

    if unexpected_arguments:
        raise ValueError(
            f"工具{tool_name}包含未知参数："
            + ", ".join(
                sorted(unexpected_arguments)
            )
        )

    for argument_name, argument_value in (
        arguments.items()
    ):
        argument_schema = properties[
            argument_name
        ]

        expected_type = argument_schema.get(
            "type"
        )

        if not validate_json_type(
            argument_value,
            expected_type,
        ):
            raise ValueError(
                f"参数{argument_name}类型错误："
                f"期望{expected_type}，"
                f"实际是"
                f"{type(argument_value).__name__}"
            )

        if (
            expected_type == "string"
            and not argument_value.strip()
        ):
            raise ValueError(
                f"字符串参数{argument_name}不能为空"
            )

        minimum = argument_schema.get(
            "minimum"
        )

        if (
            minimum is not None
            and argument_value < minimum
        ):
            raise ValueError(
                f"参数{argument_name}不能小于"
                f"{minimum}"
            )

        maximum = argument_schema.get(
            "maximum"
        )

        if (
            maximum is not None
            and argument_value > maximum
        ):
            raise ValueError(
                f"参数{argument_name}不能大于"
                f"{maximum}"
            )

    return target


def print_schema_summary():
    
    print("允许的路由决策：")

    for decision in sorted(
        VALID_DECISIONS
    ):
        print(f"- {decision}")

    print("\n允许调用的工具：")

    for tool_name in sorted(
        VALID_TOOL_NAMES
    ):
        function_schema = (
            TOOL_SCHEMA_BY_NAME[tool_name]
        )

        required = (
            function_schema["parameters"]
            .get("required", [])
        )

        required_text = (
            ", ".join(required)
            if required
            else "无"
        )

        print(
            f"- {tool_name}："
            f"必填参数={required_text}"
        )


if __name__ == "__main__":
    print_schema_summary()
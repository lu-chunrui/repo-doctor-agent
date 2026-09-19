import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from agent.core  import (
    ToolCallingLLMClient,
)
from tools.registry import (
    TOOL_SCHEMAS,
)
from schema import (
    validate_route_target,
)


ROUTER_SYSTEM_PROMPT = """
你是Repo Doctor Agent的工具路由器。

你的任务只有一个：
根据用户问题判断是否需要调用工具，并输出路由JSON。

你不能回答用户问题，也不能执行工具。

可用决策：

1. call_tool
   表示需要调用一个工具。

2. answer_directly
   表示不需要访问代码仓库，可以直接回答。

工具选择规则：

1. search_codebase
   查找代码位置、函数、类、变量或功能实现。
   如果不知道目标在哪个文件，应先使用此工具。

2. analyze_code
   对一个已经明确给出路径的Python文件进行静态分析，
   包括类、函数、导入、复杂度和潜在问题。

3. analyze_log
   分析错误日志、Traceback和运行时异常。
   只要用户提供了具体异常文本、报错信息或日志片段，
   并要求解释、定位或分析，就调用analyze_log。
   即使用户只提供一行异常，也属于日志分析。
   只有在没有提供具体报错内容、仅询问通用异常知识时，
   才选择answer_directly。

4. complete_code
   修改或补全指定文件中的代码。

5. generate_tests
   为指定Python文件、函数或类生成pytest测试。

6. generate_dockerfile
   为仓库生成Dockerfile或准备容器化配置。

7. answer_directly
   普通知识问答、打招呼，或者用户明确要求不访问仓库。

多步骤问题只选择第一步应该调用的工具。

必须只输出一个JSON对象，不能输出Markdown、解释或其他文字。

调用工具时：

{
  "decision": "call_tool",
  "tool_name": "工具名称",
  "arguments": {
    "参数名": "参数值"
  }
}

不调用工具时：

{
  "decision": "answer_directly",
  "tool_name": null,
  "arguments": {}
}

只填写用户明确提供或执行工具必需的参数。
不能编造文件名、函数名、日志或工具。
""".strip()


def build_tool_description():
   
    tool_items = []

    for tool_schema in TOOL_SCHEMAS:
        function_schema = tool_schema[
            "function"
        ]

        tool_items.append(
            {
                "name": function_schema[
                    "name"
                ],
                "description": (
                    function_schema[
                        "description"
                    ]
                ),
                "parameters": (
                    function_schema[
                        "parameters"
                    ]
                ),
            }
        )

    return json.dumps(
        tool_items,
        ensure_ascii=False,
        indent=2,
    )


def build_user_prompt(user_query):
   
    tool_description = (
        build_tool_description()
    )

    return f"""
下面是系统真实可用的工具Schema：

{tool_description}

请对下面的用户问题进行路由：

{user_query}

请只返回路由JSON。
""".strip()


def remove_markdown_fence(text):
    """
    去掉模型可能添加的Markdown代码围栏。
    """
    text = text.strip()

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

    return text.strip()


def parse_json_object(text):

    if not isinstance(text, str):
        raise ValueError(
            "模型回答不是字符串"
        )

    text = remove_markdown_fence(text)

    if not text:
        raise ValueError(
            "模型回答为空"
        )

    # 首先尝试直接解析整个回答
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = None

    if isinstance(result, dict):
        return result

    # 如果回答前后包含额外文字，
    # 从每一个左花括号开始尝试解析
    decoder = json.JSONDecoder()

    for index, character in enumerate(text):
        if character != "{":
            continue

        try:
            result, _ = decoder.raw_decode(
                text[index:]
            )
        except json.JSONDecodeError:
            continue

        if isinstance(result, dict):
            return result

    raise ValueError(
        "模型回答中没有找到合法JSON对象"
    )


class BaseRouter:
 
    def __init__(
        self,
        llm_client,
        temperature=0.0,
    ):
        self.llm_client = llm_client
        self.temperature = temperature

    @classmethod
    def from_environment(cls):
       
        llm_client = (
            ToolCallingLLMClient
            .from_environment()
        )

        return cls(
            llm_client=llm_client,
            temperature=0.0,
        )

    def route(self, user_query):
        """
        对一个用户问题进行路由预测。

        返回：
        {
            "user_query": ...,
            "raw_text": ...,
            "prediction": ...,
            "json_valid": ...,
            "schema_valid": ...,
            "parse_error": ...,
            "schema_error": ...
        }
        """
        if not isinstance(
            user_query,
            str,
        ):
            raise TypeError(
                "user_query必须是字符串"
            )

        user_query = user_query.strip()

        if not user_query:
            raise ValueError(
                "user_query不能为空"
            )

        response_message = (
            self.llm_client
            .create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            ROUTER_SYSTEM_PROMPT
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            build_user_prompt(
                                user_query
                            )
                        ),
                    },
                ],
                tools=None,
                temperature=(
                    self.temperature
                ),
            )
        )

        raw_text = (
            response_message.get(
                "content"
            )
            or ""
        ).strip()

        result = {
            "user_query": user_query,
            "raw_text": raw_text,
            "prediction": None,
            "json_valid": False,
            "schema_valid": False,
            "parse_error": None,
            "schema_error": None,
        }

        try:
            prediction = parse_json_object(
                raw_text
            )
        except ValueError as error:
            result["parse_error"] = str(
                error
            )

            return result

        result["prediction"] = prediction
        result["json_valid"] = True

        try:
            validate_route_target(
                prediction
            )
        except (
            ValueError,
            TypeError,
        ) as error:
            result["schema_error"] = str(
                error
            )

            return result

        result["schema_valid"] = True

        return result


def print_route_result(result):
  
    print("=" * 70)
    print(
        f"用户问题："
        f"{result['user_query']}"
    )

    print("\n模型原始输出：")
    print(
        result["raw_text"]
        or "<空输出>"
    )

    print("\n解析状态：")
    print(
        f"JSON合法："
        f"{result['json_valid']}"
    )

    print(
        f"Schema合法："
        f"{result['schema_valid']}"
    )

    if result["parse_error"]:
        print(
            f"解析错误："
            f"{result['parse_error']}"
        )

    if result["schema_error"]:
        print(
            f"Schema错误："
            f"{result['schema_error']}"
        )

    if result["prediction"] is not None:
        print("\n路由结果：")

        print(
            json.dumps(
                result["prediction"],
                ensure_ascii=False,
                indent=2,
            )
        )


def run_interactive_mode():
   
    router = (
        BaseRouter.from_environment()
    )

    print("Base Router已启动")
    print("输入exit、quit或q退出")

    while True:
        try:
            user_query = input(
                "\n请输入问题："
            ).strip()
        except (
            EOFError,
            KeyboardInterrupt,
        ):
            print("\n程序结束")
            break

        if user_query.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("程序结束")
            break

        if not user_query:
            continue

        try:
            result = router.route(
                user_query
            )
        except Exception as error:
            print(
                f"路由失败："
                f"{type(error).__name__}: "
                f"{error}"
            )
            continue

        print_route_result(result)


if __name__ == "__main__":
    run_interactive_mode()
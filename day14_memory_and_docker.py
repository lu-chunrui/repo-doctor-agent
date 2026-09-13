import json
import re
from datetime import datetime
from pathlib import Path

from day9_agent_tools import (
    TOOL_SCHEMAS,
    create_agent_toolbox,
)
from day10_agent_loop import (
    AGENT_SYSTEM_PROMPT,
    RepoDoctorAgent,
    ToolCallingLLMClient,
    normalize_assistant_message,
    print_tool_trace,
    serialize_tool_result,
)
from day11_citation_agent import (
    CitationAwareRepoDoctor,
    CitationValidator,
    EvidenceTrackingToolbox,
    print_citation_report,
    print_evidence,
)


MAX_MEMORY_TURNS = 6
MAX_MEMORY_CHARS = 18000
MAX_TOOL_STEPS = 6


MEMORY_AGENT_SYSTEM_PROMPT = (
    AGENT_SYSTEM_PROMPT
    + """

附加规则：

1. 你可以参考最近几轮对话理解“它”“这个函数”
   “继续分析”等指代。
2. 历史回答不能代替代码证据。
3. 用户追问仓库事实时，仍然要调用工具确认。
4. 如果历史信息与最新工具结果冲突，
   以最新工具结果为准。
5. 生成 Dockerfile 前必须调用 generate_dockerfile。
6. Dockerfile 必须放在一个 dockerfile 代码块中。
7. Dockerfile 中不能包含 API Key、密码或访问令牌。
8. 不要直接覆盖用户仓库中的文件。
"""
).strip()


class ConversationMemory:
    """
    保存最近若干轮用户和助手对话。
    """

    def __init__(
        self,
        max_turns=MAX_MEMORY_TURNS,
        max_chars=MAX_MEMORY_CHARS,
    ):
        if max_turns < 1:
            raise ValueError(
                "max_turns 必须大于等于 1"
            )

        if max_chars < 1000:
            raise ValueError(
                "max_chars 不能小于 1000"
            )

        self.max_turns = max_turns
        self.max_chars = max_chars
        self.turns = []

    def _count_chars(self):
        total_chars = 0

        for turn in self.turns:
            total_chars += len(
                turn["user"]
            )

            total_chars += len(
                turn["assistant"]
            )

        return total_chars

    def _trim(self):
        """
        同时按照对话轮数和字符数裁剪。
        """
        while (
            len(self.turns)
            > self.max_turns
        ):
            self.turns.pop(0)

        while (
            self._count_chars()
            > self.max_chars
            and len(self.turns) > 1
        ):
            self.turns.pop(0)

    def add_turn(
        self,
        user_message,
        assistant_message,
    ):
        self.turns.append(
            {
                "user": user_message,
                "assistant": (
                    assistant_message
                ),
            }
        )

        self._trim()

    def replace_last_assistant(
        self,
        assistant_message,
    ):
        """
        引用校验修正回答后，用修正版更新记忆。
        """
        if not self.turns:
            return

        self.turns[-1]["assistant"] = (
            assistant_message
        )

        self._trim()

    def to_messages(self):
        messages = []

        for turn in self.turns:
            messages.append(
                {
                    "role": "user",
                    "content": turn["user"],
                }
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        turn["assistant"]
                    ),
                }
            )

        return messages

    def clear(self):
        self.turns = []

    def get_status(self):
        return {
            "turn_count": len(
                self.turns
            ),
            "character_count": (
                self._count_chars()
            ),
            "max_turns": self.max_turns,
            "max_chars": self.max_chars,
        }


class MemoryRepoDoctorAgent(
    RepoDoctorAgent
):
    """
    在 Day 10 Agent Loop 基础上加入跨问题记忆。
    """

    def __init__(
        self,
        llm_client,
        toolbox,
        tool_schemas,
        memory,
        max_tool_steps=MAX_TOOL_STEPS,
    ):
        super().__init__(
            llm_client=llm_client,
            toolbox=toolbox,
            tool_schemas=tool_schemas,
            max_tool_steps=max_tool_steps,
        )

        self.memory = memory

    def _build_result(
        self,
        user_question,
        answer,
        tool_trace,
        steps,
        reached_step_limit,
    ):
        self.memory.add_turn(
            user_message=user_question,
            assistant_message=answer,
        )

        return {
            "answer": answer,
            "tool_trace": tool_trace,
            "steps": steps,
            "reached_step_limit": (
                reached_step_limit
            ),
            "memory_status": (
                self.memory.get_status()
            ),
        }

    def run(
        self,
        user_question,
    ):
        if not isinstance(
            user_question,
            str,
        ):
            raise TypeError(
                "user_question 必须是字符串"
            )

        user_question = (
            user_question.strip()
        )

        if not user_question:
            raise ValueError(
                "用户问题不能为空"
            )

        messages = [
            {
                "role": "system",
                "content": (
                    MEMORY_AGENT_SYSTEM_PROMPT
                ),
            }
        ]

        messages.extend(
            self.memory.to_messages()
        )

        messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        tool_trace = []

        for step in range(
            1,
            self.max_tool_steps + 1,
        ):
            print(
                f"\n[Agent Step {step}] "
                "正在请求模型……"
            )

            response_message = (
                self.llm_client
                .create_chat_completion(
                    messages=messages,
                    tools=self.tool_schemas,
                    temperature=0.1,
                )
            )

            assistant_message = (
                normalize_assistant_message(
                    response_message
                )
            )

            messages.append(
                assistant_message
            )

            tool_calls = (
                response_message.get(
                    "tool_calls"
                )
                or []
            )

            if not tool_calls:
                final_answer = (
                    response_message.get(
                        "content"
                    )
                    or ""
                ).strip()

                if not final_answer:
                    final_answer = (
                        "模型没有返回有效回答。"
                    )

                return self._build_result(
                    user_question=(
                        user_question
                    ),
                    answer=final_answer,
                    tool_trace=tool_trace,
                    steps=step,
                    reached_step_limit=False,
                )

            print(
                f"模型请求调用 "
                f"{len(tool_calls)} 个工具"
            )

            for tool_index, tool_call in enumerate(
                tool_calls,
                start=1,
            ):
                tool_call_id = (
                    tool_call.get("id")
                    or (
                        f"call_{step}_"
                        f"{tool_index}"
                    )
                )

                execution = (
                    self._execute_tool_call(
                        tool_call
                    )
                )

                tool_name = execution[
                    "tool_name"
                ]

                arguments = execution[
                    "arguments"
                ]

                tool_result = execution[
                    "result"
                ]

                print(
                    f"调用工具：{tool_name}"
                )

                print(
                    "参数："
                    + json.dumps(
                        arguments,
                        ensure_ascii=False,
                    )
                )

                if tool_result.get("ok"):
                    print("工具执行成功")
                else:
                    print(
                        "工具执行失败："
                        f"{tool_result.get('error')}"
                    )

                tool_trace.append(
                    {
                        "step": step,
                        "tool_call_id": (
                            tool_call_id
                        ),
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "ok": tool_result.get(
                            "ok",
                            False,
                        ),
                        "error": tool_result.get(
                            "error"
                        ),
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": (
                            tool_call_id
                        ),
                        "content": (
                            serialize_tool_result(
                                tool_result
                            )
                        ),
                    }
                )

        print(
            "\n达到最大工具调用步数，"
            "要求模型进行总结。"
        )

        final_messages = messages + [
            {
                "role": "system",
                "content": (
                    "已经达到工具调用上限。"
                    "不能继续调用工具。"
                    "请根据已有证据给出最终回答。"
                ),
            }
        ]

        final_response = (
            self.llm_client
            .create_chat_completion(
                messages=final_messages,
                tools=None,
                temperature=0.1,
            )
        )

        final_answer = (
            final_response.get("content")
            or (
                "达到工具调用上限，"
                "但模型没有返回最终回答。"
            )
        ).strip()

        return self._build_result(
            user_question=user_question,
            answer=final_answer,
            tool_trace=tool_trace,
            steps=self.max_tool_steps,
            reached_step_limit=True,
        )


def extract_dockerfile(
    answer,
):
    """
    从模型回答的 Markdown 代码块中提取 Dockerfile。
    """
    dockerfile_match = re.search(
        r"```(?:dockerfile|Dockerfile)"
        r"\s*(.*?)\s*```",
        answer,
        flags=re.DOTALL,
    )

    if dockerfile_match:
        return (
            dockerfile_match
            .group(1)
            .strip()
        )

    generic_matches = re.findall(
        r"```\s*(.*?)\s*```",
        answer,
        flags=re.DOTALL,
    )

    for code_block in generic_matches:
        if re.search(
            r"^\s*FROM\s+\S+",
            code_block,
            flags=re.MULTILINE
            | re.IGNORECASE,
        ):
            return code_block.strip()

    return None


def validate_dockerfile(
    dockerfile_text,
):
    """
    对生成的 Dockerfile 进行静态检查。
    不执行 docker build。
    """
    if not dockerfile_text:
        return {
            "valid": False,
            "errors": [
                "没有提取到 Dockerfile"
            ],
            "warnings": [],
        }

    errors = []
    warnings = []

    required_patterns = {
        "FROM": r"^\s*FROM\s+\S+",
        "WORKDIR": r"^\s*WORKDIR\s+\S+",
        "COPY": r"^\s*COPY\s+",
        "CMD_OR_ENTRYPOINT": (
            r"^\s*(?:CMD|ENTRYPOINT)\s+"
        ),
    }

    for (
        instruction_name,
        pattern,
    ) in required_patterns.items():
        if not re.search(
            pattern,
            dockerfile_text,
            flags=re.MULTILINE
            | re.IGNORECASE,
        ):
            errors.append(
                "缺少必要指令："
                f"{instruction_name}"
            )

    if not re.search(
        r"^\s*RUN\s+.*pip\s+install",
        dockerfile_text,
        flags=re.MULTILINE
        | re.IGNORECASE,
    ):
        warnings.append(
            "没有发现 pip install 依赖安装命令"
        )

    if not re.search(
        r"^\s*USER\s+\S+",
        dockerfile_text,
        flags=re.MULTILINE
        | re.IGNORECASE,
    ):
        warnings.append(
            "没有设置非 root 运行用户"
        )

    dangerous_patterns = {
        "可能包含 API Key": (
            r"sk-[A-Za-z0-9_-]{10,}"
        ),
        "可能包含明文密码": (
            r"(?i)(?:password|api_key|token)"
            r"\s*=\s*[\"'][^\"']+[\"']"
        ),
        "使用 chmod 777": (
            r"(?i)\bchmod\s+777\b"
        ),
        "远程脚本直接交给 shell": (
            r"(?i)(?:curl|wget).*\|\s*"
            r"(?:sh|bash)"
        ),
        "使用 sudo": (
            r"(?i)\bsudo\b"
        ),
    }

    for (
        message,
        pattern,
    ) in dangerous_patterns.items():
        if re.search(
            pattern,
            dockerfile_text,
        ):
            errors.append(message)

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }


def save_dockerfile_proposal(
    dockerfile_text,
    output_directory,
):
    """
    保存候选 Dockerfile，不覆盖目标仓库。
    """
    output_directory = Path(
        output_directory
    ).resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_path = (
        output_directory
        / f"{timestamp}_Dockerfile"
    )

    output_path.write_text(
        dockerfile_text,
        encoding="utf-8",
    )

    return output_path


def used_dockerfile_tool(
    tool_trace,
):
    return any(
        item.get("tool_name")
        == "generate_dockerfile"
        and item.get("ok")
        for item in tool_trace
    )


def process_dockerfile_answer(
    result,
    output_directory,
):
    """
    如果本轮调用过 generate_dockerfile，
    则提取、检查并保存候选文件。
    """
    if not used_dockerfile_tool(
        result["tool_trace"]
    ):
        return

    print("\n" + "=" * 70)
    print("Dockerfile 检查")
    print("=" * 70)

    dockerfile_text = (
        extract_dockerfile(
            result["answer"]
        )
    )

    validation = validate_dockerfile(
        dockerfile_text
    )

    print(
        "状态："
        + (
            "通过"
            if validation["valid"]
            else "未通过"
        )
    )

    for error in validation["errors"]:
        print(f"- 错误：{error}")

    for warning in validation[
        "warnings"
    ]:
        print(f"- 警告：{warning}")

    if dockerfile_text:
        output_path = (
            save_dockerfile_proposal(
                dockerfile_text,
                output_directory,
            )
        )

        print(
            "候选 Dockerfile 已保存："
        )
        print(output_path)

        print(
            "注意：系统没有自动执行 docker build。"
        )


def print_memory_status(
    memory,
):
    status = memory.get_status()

    print("\n" + "=" * 70)
    print("对话记忆")
    print("=" * 70)

    print(
        f"当前轮数："
        f"{status['turn_count']}/"
        f"{status['max_turns']}"
    )

    print(
        f"当前字符数："
        f"{status['character_count']}/"
        f"{status['max_chars']}"
    )


def run_interactive_mode(
    citation_agent,
    memory_agent,
    memory,
    output_directory,
):
    print(
        "\nRepo Doctor 多轮记忆版已启动"
    )

    print("特殊命令：")
    print("- /clear：清空对话记忆")
    print("- /memory：查看记忆状态")
    print("- exit、quit、q：退出")

    while True:
        try:
            question = input(
                "\n请输入问题："
            ).strip()
        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print("\n程序结束")
            break

        if question.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("程序结束")
            break

        if question == "/clear":
            memory.clear()
            print("对话记忆已清空")
            continue

        if question == "/memory":
            print_memory_status(
                memory
            )
            continue

        if not question:
            print("问题不能为空")
            continue

        try:
            result = (
                citation_agent.run(
                    question
                )
            )
        except Exception as error:
            print(
                "\n运行失败："
                f"{type(error).__name__}: "
                f"{error}"
            )
            continue

        # 初始回答已经被写入记忆。
        # 如果引用修复产生了新答案，
        # 用修正版替换最后一轮记忆。
        if result[
            "citation_repaired"
        ]:
            memory.replace_last_assistant(
                result["answer"]
            )

        print_tool_trace(
            result["tool_trace"]
        )

        print_evidence(
            result["evidence"]
        )

        print_citation_report(
            result["citation_report"]
        )

        print("\n" + "=" * 70)
        print("最终回答")
        print("=" * 70)
        print(result["answer"])

        process_dockerfile_answer(
            result=result,
            output_directory=(
                output_directory
            ),
        )

        print_memory_status(
            memory
        )


if __name__ == "__main__":
    project_dir = (
        Path(__file__).resolve().parent
    )

    repository_path = Path(
        r"D:\桌面\mini-transformer"
    )

    output_directory = (
        project_dir
        / "generated_outputs"
    )

    REBUILD_INDEX = False

    try:
        print("正在初始化工具箱……")

        original_toolbox = (
            create_agent_toolbox(
                repository_path=(
                    repository_path
                ),
                project_dir=project_dir,
                rebuild_index=(
                    REBUILD_INDEX
                ),
            )
        )

        tracking_toolbox = (
            EvidenceTrackingToolbox(
                toolbox=original_toolbox,
                repository_path=(
                    repository_path
                ),
            )
        )

        print("正在初始化大模型……")

        llm_client = (
            ToolCallingLLMClient
            .from_environment()
        )

        memory = ConversationMemory(
            max_turns=6,
            max_chars=18000,
        )

        memory_agent = (
            MemoryRepoDoctorAgent(
                llm_client=llm_client,
                toolbox=tracking_toolbox,
                tool_schemas=(
                    TOOL_SCHEMAS
                ),
                memory=memory,
                max_tool_steps=6,
            )
        )

        citation_validator = (
            CitationValidator(
                repository_path
            )
        )

        citation_agent = (
            CitationAwareRepoDoctor(
                base_agent=memory_agent,
                llm_client=llm_client,
                tracking_toolbox=(
                    tracking_toolbox
                ),
                citation_validator=(
                    citation_validator
                ),
            )
        )

    except Exception as error:
        print(
            "系统初始化失败："
            f"{type(error).__name__}: "
            f"{error}"
        )

        raise SystemExit(1)

    run_interactive_mode(
        citation_agent=citation_agent,
        memory_agent=memory_agent,
        memory=memory,
        output_directory=output_directory,
    )
from pathlib import Path

from agent.core import (
    RepoDoctorAgent,
    ToolCallingLLMClient,
)
from config import settings
from tools.registry import (
    TOOL_SCHEMAS,
    create_agent_toolbox,
)


if __name__ == "__main__":
    project_dir = (
        Path(__file__).resolve().parent.parent
    )

    repository_path = (
        settings.resolved_default_repository()
    )

    print("正在初始化工具箱")

    toolbox = create_agent_toolbox(
        repository_path=repository_path,
        project_dir=project_dir,
        rebuild_index=False,
    )

    print("正在初始化 LLM")

    llm_client = (
        ToolCallingLLMClient
        .from_environment()
    )

    agent = RepoDoctorAgent(
        llm_client=llm_client,
        toolbox=toolbox,
        tool_schemas=TOOL_SCHEMAS,
        max_tool_steps=(
            settings.max_tool_steps
        ),
    )

    question = (
        "请先搜索代码库，然后说明这个"
        "项目的主要入口文件在哪里，"
        "并给出文件路径和行号。"
    )

    print("\n问题：", question)

    result = agent.run(question)

    print("\n" + "=" * 60)
    print("最终回答")
    print("=" * 60)
    print(result["answer"])

    print("\n工具调用记录：")

    for item in result["tool_trace"]:
        print("-" * 60)
        print("步骤：", item["step"])
        print("工具：", item["tool_name"])
        print("参数：", item["arguments"])
        print("成功：", item["ok"])

        if item["error"]:
            print("错误：", item["error"])

    print(
        "\n是否达到工具步数上限：",
        result["reached_step_limit"],
    )
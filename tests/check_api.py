from uuid import uuid4

import requests

from config import settings


if __name__ == "__main__":
    base_url = settings.backend_url.rstrip(
        "/"
    )

    repository_path = str(
        settings.resolved_default_repository()
    )

    print("测试后端：", base_url)
    print("测试仓库：", repository_path)

    print("\n1. 健康检查")

    health_response = requests.get(
        f"{base_url}/api/v1/health",
        timeout=30,
    )

    print(
        "状态码：",
        health_response.status_code,
    )
    print(
        "响应：",
        health_response.json(),
    )

    health_response.raise_for_status()

    print("\n2. 建立仓库索引")

    index_response = requests.post(
        f"{base_url}/api/v1/index",
        json={
            "repository_path": (
                repository_path
            ),
            "rebuild_index": False,
            "enable_reranker": False,
        },
        timeout=settings.index_timeout,
    )

    print(
        "状态码：",
        index_response.status_code,
    )
    print(
        "响应：",
        index_response.json(),
    )

    index_response.raise_for_status()

    print("\n3. 发送问题")

    session_id = uuid4().hex

    chat_response = requests.post(
        f"{base_url}/api/v1/chat",
        json={
            "repository_path": (
                repository_path
            ),
            "message": (
                "请搜索代码库，说明项目的"
                "主要入口文件在哪里，并给出"
                "文件路径和行号。"
            ),
            "session_id": session_id,
        },
        timeout=settings.request_timeout,
    )

    print(
        "状态码：",
        chat_response.status_code,
    )

    chat_response.raise_for_status()

    result = chat_response.json()

    print("\n最终回答：")
    print(result["answer"])

    print("\n工具调用记录：")

    for item in result["tool_trace"]:
        print("-" * 60)
        print("步骤：", item["step"])
        print("工具：", item["tool_name"])
        print("参数：", item["arguments"])
        print("成功：", item["ok"])

    print("\n引用证据：")

    for evidence in result["evidence"]:
        print("-" * 60)
        print(evidence)

    print(
        "\n引用是否被修正：",
        result["citation_repaired"],
    )
    print(
        "是否达到工具步数上限：",
        result["reached_step_limit"],
    )
import re
from uuid import uuid4

import requests
import streamlit as st


DEFAULT_BACKEND_URL = (
    "http://127.0.0.1:8000"
)

DEFAULT_REPOSITORY_PATH = (
    r"D:\桌面\mini-transformer"
)

REQUEST_TIMEOUT = 300
INDEX_TIMEOUT = 900


st.set_page_config(
    page_title="Repo Doctor Agent",
    page_icon="🩺",
    layout="wide",
)


class BackendAPIError(RuntimeError):
    pass


class RepoDoctorAPIClient:
    def __init__(
        self,
        base_url,
    ):
        base_url = base_url.strip()

        if not base_url:
            raise ValueError(
                "后端地址不能为空"
            )

        self.base_url = (
            base_url.rstrip("/")
        )

    def _request(
        self,
        method,
        path,
        timeout=REQUEST_TIMEOUT,
        **kwargs,
    ):
        url = f"{self.base_url}{path}"

        try:
            response = requests.request(
                method=method,
                url=url,
                timeout=timeout,
                **kwargs,
            )
        except requests.RequestException as error:
            raise BackendAPIError(
                "无法连接 FastAPI 后端："
                f"{error}"
            ) from error

        try:
            response_data = response.json()
        except ValueError:
            response_data = {
                "detail": response.text[:2000]
            }

        if not response.ok:
            detail = response_data.get(
                "detail",
                response_data,
            )

            raise BackendAPIError(
                f"HTTP {response.status_code}："
                f"{detail}"
            )

        return response_data

    def health(self):
        return self._request(
            method="GET",
            path="/api/v1/health",
            timeout=20,
        )

    def index_repository(
        self,
        repository_path,
        rebuild_index,
        enable_reranker,
    ):
        return self._request(
            method="POST",
            path="/api/v1/index",
            timeout=INDEX_TIMEOUT,
            json={
                "repository_path": (
                    repository_path
                ),
                "rebuild_index": (
                    rebuild_index
                ),
                "enable_reranker": (
                    enable_reranker
                ),
            },
        )

    def chat(
        self,
        repository_path,
        session_id,
        message,
    ):
        return self._request(
            method="POST",
            path="/api/v1/chat",
            json={
                "repository_path": (
                    repository_path
                ),
                "session_id": session_id,
                "message": message,
            },
        )

    def analyze_log(
        self,
        repository_path,
        session_id,
        log_text,
    ):
        return self._request(
            method="POST",
            path="/api/v1/analyze-log",
            json={
                "repository_path": (
                    repository_path
                ),
                "session_id": session_id,
                "log_text": log_text,
            },
        )

    def clear_memory(
        self,
        repository_path,
        session_id,
    ):
        return self._request(
            method="POST",
            path="/api/v1/clear-memory",
            json={
                "repository_path": (
                    repository_path
                ),
                "session_id": session_id,
            },
        )

    def get_memory(
        self,
        repository_path,
        session_id,
    ):
        return self._request(
            method="GET",
            path="/api/v1/memory",
            params={
                "repository_path": (
                    repository_path
                ),
                "session_id": session_id,
            },
            timeout=20,
        )


def initialize_session_state():
    defaults = {
        "session_id": uuid4().hex,
        "chat_messages": [],
        "indexed_repository": None,
        "index_info": None,
        "last_log_result": None,
        "last_health_result": None,
    }

    for key, default_value in (
        defaults.items()
    ):
        if key not in st.session_state:
            st.session_state[
                key
            ] = default_value


def reset_local_conversation():
    st.session_state.chat_messages = []
    st.session_state.last_log_result = (
        None
    )


def start_new_session():
    st.session_state.session_id = (
        uuid4().hex
    )

    reset_local_conversation()


def render_tool_trace(
    tool_trace,
    key_prefix,
):
    if not tool_trace:
        st.caption(
            "本轮没有调用仓库工具。"
        )
        return

    table_rows = []

    for item in tool_trace:
        table_rows.append(
            {
                "步骤": item.get("step"),
                "工具": item.get(
                    "tool_name"
                ),
                "状态": (
                    "成功"
                    if item.get("ok")
                    else "失败"
                ),
                "错误": item.get(
                    "error"
                ),
            }
        )

    st.table(table_rows)

    for index, item in enumerate(
        tool_trace
    ):
        with st.expander(
            "查看参数："
            f"{item.get('tool_name')}",
            expanded=False,
        ):
            st.json(
                item.get(
                    "arguments",
                    {},
                )
            )


def render_evidence(
    evidence,
):
    if not evidence:
        st.caption(
            "本轮没有代码证据。"
        )
        return

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        file_path = item.get(
            "file",
            "unknown"
        )

        start_line = item.get(
            "start_line"
        )

        end_line = item.get(
            "end_line"
        )

        title = (
            f"{index}. {file_path}:"
            f"{start_line}-{end_line}"
        )

        with st.expander(
            title,
            expanded=index == 1,
        ):
            st.write(
                "来源工具：",
                item.get(
                    "source_tool"
                ),
            )

            description = item.get(
                "description"
            )

            if description:
                st.write(
                    "说明：",
                    description,
                )

            content = item.get(
                "content_preview"
            )

            if content:
                suffix = (
                    str(file_path)
                    .lower()
                )

                language = (
                    "python"
                    if suffix.endswith(".py")
                    else "text"
                )

                st.code(
                    content,
                    language=language,
                    line_numbers=True,
                )


def render_citation_report(
    citation_report,
):
    if not citation_report:
        st.caption(
            "没有引用检查信息。"
        )
        return

    if citation_report.get("valid"):
        st.success(
            "代码引用检查通过"
        )
    else:
        st.error(
            "代码引用检查未通过"
        )

    st.write(
        "引用数量：",
        citation_report.get(
            "citation_count",
            0,
        ),
    )

    errors = citation_report.get(
        "errors",
        [],
    )

    for error in errors:
        st.warning(error)


def extract_code_blocks(text):
    """
    从回答中提取 Markdown 代码块，
    用于提供下载按钮。
    """
    pattern = re.compile(
        r"```([A-Za-z0-9_-]*)\s*\n"
        r"(.*?)"
        r"\n```",
        flags=re.DOTALL,
    )

    blocks = []

    for match in pattern.finditer(
        text
    ):
        language = (
            match.group(1).strip().lower()
        )

        content = (
            match.group(2).strip()
        )

        blocks.append(
            {
                "language": language,
                "content": content,
            }
        )

    return blocks


def choose_download_filename(
    block,
    index,
):
    language = block["language"]

    if language == "dockerfile":
        return "Dockerfile"

    if language in {
        "python",
        "py",
    }:
        return (
            f"generated_{index}.py"
        )

    if language == "json":
        return (
            f"generated_{index}.json"
        )

    if language in {
        "yaml",
        "yml",
    }:
        return (
            f"generated_{index}.yaml"
        )

    return f"generated_{index}.txt"


def choose_mime_type(
    filename,
):
    if filename.endswith(".py"):
        return "text/x-python"

    if filename.endswith(".json"):
        return "application/json"

    if filename.endswith(
        (".yaml", ".yml")
    ):
        return "text/yaml"

    return "text/plain"


def render_download_buttons(
    answer,
    key_prefix,
):
    code_blocks = extract_code_blocks(
        answer
    )

    if not code_blocks:
        return

    st.write("下载生成内容：")

    columns = st.columns(
        min(len(code_blocks), 3)
    )

    for index, block in enumerate(
        code_blocks,
        start=1,
    ):
        filename = (
            choose_download_filename(
                block,
                index,
            )
        )

        column = columns[
            (index - 1)
            % len(columns)
        ]

        with column:
            st.download_button(
                label=f"下载 {filename}",
                data=block["content"],
                file_name=filename,
                mime=choose_mime_type(
                    filename
                ),
                key=(
                    f"{key_prefix}_"
                    f"download_{index}"
                ),
            )


def render_response_details(
    response,
    key_prefix,
):
    with st.expander(
        "工具调用轨迹",
        expanded=False,
    ):
        render_tool_trace(
            response.get(
                "tool_trace",
                []
            ),
            key_prefix=key_prefix,
        )

    with st.expander(
        "代码证据",
        expanded=False,
    ):
        render_evidence(
            response.get(
                "evidence",
                []
            )
        )

    with st.expander(
        "引用检查",
        expanded=False,
    ):
        render_citation_report(
            response.get(
                "citation_report",
                {}
            )
        )

        if response.get(
            "citation_repaired"
        ):
            st.info(
                "系统自动修正过一次引用。"
            )

    memory = response.get(
        "memory"
    )

    if memory:
        st.caption(
            "后端记忆："
            f"{memory.get('turn_count', 0)}/"
            f"{memory.get('max_turns', 0)} 轮，"
            f"{memory.get('character_count', 0)}/"
            f"{memory.get('max_chars', 0)} 字符"
        )


def render_chat_history():
    for index, message in enumerate(
        st.session_state.chat_messages
    ):
        role = message["role"]

        avatar = (
            "🧑"
            if role == "user"
            else "🩺"
        )

        with st.chat_message(
            role,
            avatar=avatar,
        ):
            st.markdown(
                message["content"]
            )

            response = message.get(
                "response"
            )

            if response:
                render_response_details(
                    response=response,
                    key_prefix=(
                        f"history_{index}"
                    ),
                )

                render_download_buttons(
                    answer=message[
                        "content"
                    ],
                    key_prefix=(
                        f"history_{index}"
                    ),
                )


def render_sidebar(
    api_client,
    repository_path,
):
    with st.sidebar:
        st.title("🩺 Repo Doctor")

        st.caption(
            "多工具代码仓库分析与诊断系统"
        )

        st.divider()

        st.write(
            "会话 ID："
        )

        st.code(
            st.session_state.session_id
        )

        if st.button(
            "创建新会话",
            use_container_width=True,
        ):
            start_new_session()
            st.success(
                "已创建新的本地会话"
            )
            st.rerun()

        st.divider()

        rebuild_index = st.checkbox(
            "强制重新建立索引",
            value=False,
        )

        enable_reranker = st.checkbox(
            "启用 Reranker",
            value=True,
        )

        if st.button(
            "检查后端状态",
            use_container_width=True,
        ):
            try:
                health_result = (
                    api_client.health()
                )

                st.session_state[
                    "last_health_result"
                ] = health_result

                st.success(
                    "FastAPI 后端连接正常"
                )
            except BackendAPIError as error:
                st.error(str(error))

        if st.button(
            "建立/加载仓库索引",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                "正在加载模型并建立索引，"
                "首次运行可能需要较长时间……"
            ):
                try:
                    index_info = (
                        api_client
                        .index_repository(
                            repository_path=(
                                repository_path
                            ),
                            rebuild_index=(
                                rebuild_index
                            ),
                            enable_reranker=(
                                enable_reranker
                            ),
                        )
                    )

                    st.session_state[
                        "indexed_repository"
                    ] = repository_path

                    st.session_state[
                        "index_info"
                    ] = index_info

                    reset_local_conversation()

                    st.success(
                        "仓库索引准备完成"
                    )

                except BackendAPIError as error:
                    st.error(str(error))

        if (
            st.session_state
            .indexed_repository
            == repository_path
        ):
            st.success(
                "当前仓库已在前端标记为就绪"
            )
        else:
            st.warning(
                "请先建立仓库索引"
            )

        index_info = (
            st.session_state.index_info
        )

        if index_info:
            with st.expander(
                "索引信息"
            ):
                st.json(index_info)

        if st.button(
            "清空后端对话记忆",
            use_container_width=True,
        ):
            try:
                result = (
                    api_client.clear_memory(
                        repository_path=(
                            repository_path
                        ),
                        session_id=(
                            st.session_state
                            .session_id
                        ),
                    )
                )

                reset_local_conversation()

                st.success(
                    "前后端对话记录已清空"
                )

                st.json(result)

            except BackendAPIError as error:
                st.error(str(error))


def render_chat_tab(
    api_client,
    repository_path,
    ready,
):
    st.subheader("代码仓库问答")

    st.caption(
        "Agent 会自动选择检索、AST 分析、"
        "代码补全、测试生成或 Dockerfile 工具。"
    )

    if not ready:
        st.info(
            "请先在左侧建立仓库索引。"
        )

    render_chat_history()

    prompt = st.chat_input(
        "例如：KV Cache 在哪里实现？",
        disabled=not ready,
        max_chars=100000,
    )

    if not prompt:
        return

    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    try:
        with st.spinner(
            "Agent 正在分析仓库……"
        ):
            response = api_client.chat(
                repository_path=(
                    repository_path
                ),
                session_id=(
                    st.session_state
                    .session_id
                ),
                message=prompt,
            )

        answer = response.get(
            "answer",
            "后端没有返回回答。",
        )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer,
                "response": response,
            }
        )

    except BackendAPIError as error:
        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": (
                    f"后端请求失败：{error}"
                ),
                "response": None,
            }
        )

    st.rerun()


def render_log_result(
    result,
):
    st.subheader("诊断结果")

    st.markdown(
        result.get(
            "answer",
            "后端没有返回诊断结果。",
        )
    )

    parsed = result.get(
        "parsed_traceback",
        {},
    )

    column1, column2, column3 = (
        st.columns(3)
    )

    column1.metric(
        "异常类别",
        parsed.get(
            "category",
            "unknown",
        ),
    )

    column2.metric(
        "调用栈层数",
        parsed.get(
            "frame_count",
            0,
        ),
    )

    column3.metric(
        "关联仓库文件",
        len(
            result.get(
                "repository_files",
                []
            )
        ),
    )

    with st.expander(
        "自动生成的检索查询"
    ):
        for query in result.get(
            "search_queries",
            [],
        ):
            st.write(f"- {query}")

    with st.expander(
        "工具调用轨迹"
    ):
        render_tool_trace(
            result.get(
                "tool_trace",
                []
            ),
            key_prefix="log",
        )

    with st.expander(
        "代码证据"
    ):
        render_evidence(
            result.get(
                "evidence",
                []
            )
        )

    with st.expander(
        "引用检查"
    ):
        render_citation_report(
            result.get(
                "citation_report",
                {},
            )
        )

    render_download_buttons(
        answer=result.get(
            "answer",
            "",
        ),
        key_prefix="log_result",
    )


def render_log_tab(
    api_client,
    repository_path,
    ready,
):
    st.subheader("Traceback 诊断")

    st.caption(
        "粘贴完整 Python traceback，"
        "系统会定位仓库代码并给出证据链。"
    )

    with st.form(
        "traceback_form"
    ):
        log_text = st.text_area(
            "错误日志",
            height=300,
            placeholder=(
                "Traceback (most recent call last):\n"
                '  File "...", line 10, in <module>\n'
                "RuntimeError: ..."
            ),
        )

        submitted = st.form_submit_button(
            "开始诊断",
            type="primary",
            disabled=not ready,
        )

    if submitted:
        if not log_text.strip():
            st.warning(
                "错误日志不能为空"
            )
        else:
            with st.spinner(
                "正在解析 traceback "
                "并收集代码证据……"
            ):
                try:
                    result = (
                        api_client.analyze_log(
                            repository_path=(
                                repository_path
                            ),
                            session_id=(
                                st.session_state
                                .session_id
                            ),
                            log_text=log_text,
                        )
                    )

                    st.session_state[
                        "last_log_result"
                    ] = result

                except BackendAPIError as error:
                    st.error(str(error))

    last_result = (
        st.session_state
        .last_log_result
    )

    if last_result:
        render_log_result(
            last_result
        )


def render_system_tab(
    api_client,
    repository_path,
    ready,
):
    st.subheader("系统状态")

    column1, column2 = st.columns(
        2
    )

    with column1:
        if st.button(
            "刷新 FastAPI 状态"
        ):
            try:
                st.session_state[
                    "last_health_result"
                ] = api_client.health()
            except BackendAPIError as error:
                st.error(str(error))

        health_result = (
            st.session_state
            .last_health_result
        )

        if health_result:
            st.json(health_result)

    with column2:
        if st.button(
            "读取后端记忆状态",
            disabled=not ready,
        ):
            try:
                memory_result = (
                    api_client.get_memory(
                        repository_path=(
                            repository_path
                        ),
                        session_id=(
                            st.session_state
                            .session_id
                        ),
                    )
                )

                st.json(memory_result)

            except BackendAPIError as error:
                st.error(str(error))

    st.divider()

    st.write("当前配置")

    st.json(
        {
            "backend_url": (
                api_client.base_url
            ),
            "repository_path": (
                repository_path
            ),
            "session_id": (
                st.session_state
                .session_id
            ),
            "frontend_message_count": (
                len(
                    st.session_state
                    .chat_messages
                )
            ),
        }
    )


def main():
    initialize_session_state()

    st.title(
        "🩺 Repo Doctor Agent"
    )

    st.write(
        "面向本地代码仓库的检索、分析、"
        "问答和 Traceback 诊断系统"
    )

    backend_url = st.text_input(
        "FastAPI 后端地址",
        value=DEFAULT_BACKEND_URL,
    )

    repository_path = st.text_input(
        "目标仓库路径",
        value=DEFAULT_REPOSITORY_PATH,
    ).strip()

    try:
        api_client = (
            RepoDoctorAPIClient(
                backend_url
            )
        )
    except ValueError as error:
        st.error(str(error))
        st.stop()

    render_sidebar(
        api_client=api_client,
        repository_path=repository_path,
    )

    ready = (
        st.session_state
        .indexed_repository
        == repository_path
    )

    chat_tab, log_tab, system_tab = (
        st.tabs(
            [
                "💬 仓库问答",
                "🧯 日志诊断",
                "⚙️ 系统状态",
            ]
        )
    )

    with chat_tab:
        render_chat_tab(
            api_client=api_client,
            repository_path=(
                repository_path
            ),
            ready=ready,
        )

    with log_tab:
        render_log_tab(
            api_client=api_client,
            repository_path=(
                repository_path
            ),
            ready=ready,
        )

    with system_tab:
        render_system_tab(
            api_client=api_client,
            repository_path=(
                repository_path
            ),
            ready=ready,
        )


if __name__ == "__main__":
    main()
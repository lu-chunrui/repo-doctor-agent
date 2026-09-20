import json
import re
from pathlib import Path
from config import settings

from repository.scanner import (
    list_files,
    validate_repository,
)
from repository.search import (
    read_file,
    search_code,
)
from repository.analyzer import (
    analyze_python_file,
)
from rag.hybrid import (
    HybridCodeRetriever,
    prepare_bm25_retriever,
    prepare_dense_retriever,
)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": (
                "根据自然语言问题、函数名、类名或变量名，"
                "在代码仓库中检索相关代码。"
                "当需要寻找代码位置或理解功能实现时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "要检索的问题、函数名、"
                            "类名或代码关键词"
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": (
                            "返回多少个相关代码块"
                        ),
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_code",
            "description": (
                "使用 AST 分析一个 Python 文件，"
                "返回类、函数、导入、圈复杂度和潜在问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": (
                            "相对于仓库根目录的 Python 文件路径，"
                            "例如 model.py"
                        ),
                    },
                },
                "required": ["relative_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_log",
            "description": (
                "分析 Python traceback 或错误日志，"
                "提取异常类型、报错文件和行号，"
                "并读取仓库内相关代码上下文。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "log_text": {
                        "type": "string",
                        "description": (
                            "用户提供的完整Traceback、错误日志或单行异常片段"
                        ),
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": (
                            "报错行前后读取多少行代码"
                        ),
                        "minimum": 1,
                        "maximum": 30,
                        "default": 8,
                    },
                },
                "required": ["log_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_code",
            "description": (
                "读取指定代码区域，并根据用户要求"
                "准备代码修改或补全所需的上下文。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": (
                            "需要修改的仓库内文件路径"
                        ),
                    },
                    "instruction": {
                        "type": "string",
                        "description": (
                            "具体修改或补全要求"
                        ),
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "起始行号",
                        "minimum": 1,
                        "default": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "结束行号",
                        "minimum": 1,
                    },
                },
                "required": [
                    "relative_path",
                    "instruction",
                ],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_tests",
            "description": (
                "读取 Python 文件及其结构信息，"
                "为指定函数、方法或类准备 pytest 测试生成任务。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": (
                            "需要生成测试的 Python 文件"
                        ),
                    },
                    "target_name": {
                        "type": "string",
                        "description": (
                            "需要测试的函数、方法或类名称；"
                            "省略时分析整个文件"
                        ),
                    },
                },
                "required": ["relative_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_dockerfile",
            "description": (
                "扫描仓库结构、依赖文件和程序入口，"
                "准备生成 Dockerfile 所需的信息。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entrypoint": {
                        "type": "string",
                        "description": (
                            "程序入口，例如 main.py 或 app.py；"
                            "不确定时可以省略"
                        ),
                    },
                    "python_version": {
                        "type": "string",
                        "description": (
                            "目标 Python 版本，例如 3.11"
                        ),
                        "default": "3.11",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
]


def truncate_text(text, max_chars=None):
    if max_chars is None:
        max_chars = settings.max_tool_content_chars
    if len(text) <= max_chars:
        return text

    return (
        text[:max_chars]
        + "\n...内容过长，已截断..."
    )


class AgentToolbox:

    def __init__(
        self,
        repository_path,
        hybrid_retriever,
    ):
        self.repository_path = (
            validate_repository(
                repository_path
            )
        )

        self.hybrid_retriever = (
            hybrid_retriever
        )

        self.available_tools = {
            "search_codebase": (
                self.search_codebase
            ),
            "analyze_code": (
                self.analyze_code
            ),
            "analyze_log": (
                self.analyze_log
            ),
            "complete_code": (
                self.complete_code
            ),
            "generate_tests": (
                self.generate_tests
            ),
            "generate_dockerfile": (
                self.generate_dockerfile
            ),
        }

    def execute(
        self,
        tool_name,
        arguments,
    ):
       
        if tool_name not in self.available_tools:
            return {
                "ok": False,
                "tool": tool_name,
                "error": (
                    f"未知工具：{tool_name}"
                ),
                "available_tools": list(
                    self.available_tools.keys()
                ),
            }
        if not isinstance(arguments, dict):
            return {
                "ok": False,
                "tool": tool_name,
                "error": (
                    "工具参数必须是 JSON object"
                ),
            }
        tool_function = self.available_tools[
            tool_name
        ]

        try:
            result = tool_function(
                **arguments
            )
        except Exception as error:
            return {
                "ok": False,
                "tool": tool_name,
                "error_type": (
                    type(error).__name__
                ),
                "error": str(error),
            }

        return {
            "ok": True,
            "tool": tool_name,
            "result": result,
        }

    def search_codebase(
        self,
        query,
        top_k=5,
    ):
        if not isinstance(query, str):
            raise TypeError(
                "query 必须是字符串"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "检索问题不能为空"
            )

        if not 1 <= top_k <= 10:
            raise ValueError(
                "top_k 必须位于 1 到 10 之间"
            )

        candidate_k = max(
            20,
            top_k * 4,
        )

        results = (
            self.hybrid_retriever.search(
                query=query,
                top_k=top_k,
                candidate_k=candidate_k,
            )
        )

        simplified_results = []

        for result in results:
            simplified_results.append(
                {
                    "file": result["file"],
                    "start_line": (
                        result["start_line"]
                    ),
                    "end_line": (
                        result["end_line"]
                    ),
                    "chunk_type": (
                        result["chunk_type"]
                    ),
                    "name": result["name"],
                    "rrf_score": (
                        result["rrf_score"]
                    ),
                    "dense_rank": (
                        result["dense_rank"]
                    ),
                    "bm25_rank": (
                        result["bm25_rank"]
                    ),
                    "content": truncate_text(
                        result["content"],
                        max_chars=5000,
                    ),
                }
            )

        return {
            "query": query,
            "result_count": len(
                simplified_results
            ),
            "results": simplified_results,
        }

    def analyze_code(
        self,
        relative_path,
    ):
        return analyze_python_file(
            self.repository_path,
            relative_path,
        )

    def _find_repository_file(
        self,
        traceback_path,
    ):
       
        normalized_traceback_path = (
            str(traceback_path)
            .replace("\\", "/")
            .lower()
        )

        repository_files = list_files(
            self.repository_path
        )

        exact_matches = []

        for relative_path in repository_files:
            relative_text = (
                str(relative_path)
                .replace("\\", "/")
            )

            normalized_relative = (
                relative_text.lower()
            )

            if (
                normalized_traceback_path
                == normalized_relative
                or normalized_traceback_path.endswith(
                    "/" + normalized_relative
                )
            ):
                return relative_text

            if (
                Path(relative_text).name.lower()
                == Path(
                    normalized_traceback_path
                ).name.lower()
            ):
                exact_matches.append(
                    relative_text
                )

        if len(exact_matches) == 1:
            return exact_matches[0]

        return None
    
    def analyze_log(  self,
        log_text,
        context_lines=8,
    ):
        if not isinstance(log_text, str):
            raise TypeError(
                "log_text 必须是字符串"
            )

        log_text = log_text.strip()

        if not log_text:
            raise ValueError(
                "错误日志不能为空"
            )

        if not 1 <= context_lines <= 30:
            raise ValueError(
                "context_lines 必须位于 1 到 30 之间"
            )

        frame_pattern = re.compile(
            r'File\s+"([^"]+)",\s+'
            r"line\s+(\d+)"
            r"(?:,\s+in\s+([^\n]+))?"
        )

        frame_matches = frame_pattern.findall(
            log_text
        )

        frames = []

        for (
            traceback_path,
            line_text,
            function_name,
        ) in frame_matches:
            line_number = int(line_text)

            relative_path = (
                self._find_repository_file(
                    traceback_path
                )
            )

            frame_result = {
                "traceback_path": (
                    traceback_path
                ),
                "line": line_number,
                "function": (
                    function_name.strip()
                    if function_name
                    else None
                ),
                "repository_file": (
                    relative_path
                ),
                "context": None,
            }

            if relative_path is not None:
                start_line = max(
                    1,
                    line_number - context_lines,
                )

                end_line = (
                    line_number
                    + context_lines
                )

                frame_result["context"] = (
                    read_file(
                        self.repository_path,
                        relative_path,
                        start_line=start_line,
                        end_line=end_line,
                    )
                )

            frames.append(
                frame_result
            )

        exception_pattern = re.compile(
            r"^([A-Za-z_][\w.]*(?:Error|Exception))"
            r":\s*(.*)$",
            re.MULTILINE,
        )

        exception_matches = list(
            exception_pattern.finditer(
                log_text
            )
        )

        exception_type = None
        exception_message = None

        if exception_matches:
            last_exception = (
                exception_matches[-1]
            )

            exception_type = (
                last_exception.group(1)
            )

            exception_message = (
                last_exception.group(2)
            )

        return {
            "exception_type": (
                exception_type
            ),
            "exception_message": (
                exception_message
            ),
            "frame_count": len(frames),
            "frames": frames,
            "raw_log": truncate_text(
                log_text,
                max_chars=10000,
            ),
        }

    def complete_code(
        self,
        relative_path,
        instruction,
        start_line=1,
        end_line=None,
    ):
        if not isinstance(instruction, str):
            raise TypeError(
                "instruction 必须是字符串"
            )

        instruction = instruction.strip()

        if not instruction:
            raise ValueError(
                "代码修改要求不能为空"
            )

        if end_line is None:
            end_line = start_line + 199

        file_result = read_file(
            self.repository_path,
            relative_path,
            start_line=start_line,
            end_line=end_line,
        )

        file_result["content"] = (
            truncate_text(
                file_result["content"]
            )
        )

        return {
            "task_type": "code_completion",
            "instruction": instruction,
            "target_file": relative_path,
            "existing_code": file_result,
            "requirements": [
                "保持原有代码风格",
                "不要修改无关代码",
                "说明修改原因",
                "给出完整可用的修改代码",
                "不得访问仓库目录之外",
            ],
        }

    def generate_tests(
        self,
        relative_path,
        target_name=None,
    ):
        analysis = analyze_python_file(
            self.repository_path,
            relative_path,
        )

        source = read_file(
            self.repository_path,
            relative_path,
        )

        source["content"] = truncate_text(
            source["content"]
        )

        return {
            "task_type": "test_generation",
            "target_file": relative_path,
            "target_name": target_name,
            "analysis": analysis,
            "source": source,
            "test_requirements": [
                "使用 pytest",
                "至少包含正常输入",
                "至少包含边界情况",
                "至少包含异常情况",
                "避免访问真实网络",
                "避免修改用户真实文件",
                "必要时使用 mock",
            ],
        }

    def generate_dockerfile(
        self,
        entrypoint=None,
        python_version="3.11",
    ):
        repository_files = [
            str(relative_path)
            for relative_path in list_files(
                self.repository_path
            )
        ]

        dependency_candidates = [
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "Pipfile",
        ]

        dependency_files = {}

        for candidate in dependency_candidates:
            if candidate not in repository_files:
                continue

            file_result = read_file(
                self.repository_path,
                candidate,
            )

            dependency_files[candidate] = (
                truncate_text(
                    file_result["content"],
                    max_chars=8000,
                )
            )

        possible_entrypoints = []

        for candidate in [
            "main.py",
            "app.py",
            "server.py",
            "manage.py",
        ]:
            if candidate in repository_files:
                possible_entrypoints.append(
                    candidate
                )

        selected_entrypoint = entrypoint

        if selected_entrypoint is None:
            if possible_entrypoints:
                selected_entrypoint = (
                    possible_entrypoints[0]
                )

        return {
            "task_type": (
                "dockerfile_generation"
            ),
            "repository_name": (
                self.repository_path.name
            ),
            "python_version": (
                python_version
            ),
            "requested_entrypoint": (
                entrypoint
            ),
            "selected_entrypoint": (
                selected_entrypoint
            ),
            "possible_entrypoints": (
                possible_entrypoints
            ),
            "dependency_files": (
                dependency_files
            ),
            "repository_files": (
                repository_files[:200]
            ),
            "requirements": [
                "使用官方 Python slim 镜像",
                "设置工作目录",
                "优先复制依赖文件并安装依赖",
                "再复制项目代码",
                "使用非 root 用户运行",
                "给出合理的启动命令",
            ],
        }


def create_agent_toolbox(
    repository_path,
    project_dir,
    rebuild_index=False,
):
    
    project_dir = Path(
        project_dir
    ).resolve()

    dense_index_directory = (
        project_dir / "dense_index"
    )

    bm25_index_path = (
        project_dir
        / "bm25_index"
        / "index.json"
    )

    dense_retriever = (
        prepare_dense_retriever(
            repository_path=repository_path,
            index_directory=(
                dense_index_directory
            ),
            rebuild_index=rebuild_index,
        )
    )

    bm25_retriever = (
        prepare_bm25_retriever(
            repository_path=repository_path,
            index_path=bm25_index_path,
            rebuild_index=rebuild_index,
        )
    )

    hybrid_retriever = HybridCodeRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
        dense_weight=1.0,
        bm25_weight=1.0,
    )

    return AgentToolbox(
        repository_path=repository_path,
        hybrid_retriever=(
            hybrid_retriever
        ),
    )


def print_tool_schemas():
    print(
        json.dumps(
            TOOL_SCHEMAS,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    project_dir = (
        Path(__file__).resolve().parent
    )

    repository_path = Path(
        settings.resolved_default_repository()
    )

    toolbox = create_agent_toolbox(
        repository_path=repository_path,
        project_dir=project_dir,
        rebuild_index=False,
    )

    print("已注册工具：")

    for tool_name in (
        toolbox.available_tools
    ):
        print(f"- {tool_name}")

    print("\n测试 search_codebase：")

    test_result = toolbox.execute(
        tool_name="search_codebase",
        arguments={
            "query": (
                "KV Cache 在哪里实现？"
            ),
            "top_k": 3,
        },
    )

    print(
        json.dumps(
            test_result,
            ensure_ascii=False,
            indent=2,
        )
    )
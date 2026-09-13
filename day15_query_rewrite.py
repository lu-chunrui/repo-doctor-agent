import json
import re
from pathlib import Path

from day1_repository_scanner import (
    list_files,
    validate_repository,
)
from day2_code_search import (
    search_code,
)
from day7_hybrid_retrieval import (
    make_chunk_key,
    prepare_bm25_retriever,
    prepare_dense_retriever,
)
from day10_agent_loop import (
    ToolCallingLLMClient,
)


ALLOWED_INTENTS = {
    "code_search",
    "code_analysis",
    "log_diagnosis",
    "code_completion",
    "test_generation",
    "dockerfile_generation",
    "general_question",
}

RRF_K = 60
MAX_REPOSITORY_FILES_IN_PROMPT = 150


QUERY_REWRITE_SYSTEM_PROMPT = """
你是代码仓库检索系统的 Query Rewrite 模块。

你的任务是把用户问题转换成适合代码检索的结构化查询。

必须遵守：

1. semantic_query 用于 Dense 语义检索。
2. keyword_query 用于 BM25 关键词检索。
3. 必须保留用户原文中的文件名、函数名、类名和变量名。
4. 不允许把用户给出的代码标识符改成其他名字。
5. 只有在仓库文件列表或用户问题中出现的标识符，
   才能放进 identifiers。
6. 不要编造仓库中不存在的文件。
7. general_question 表示问题不需要查询目标仓库。
8. 涉及仓库实现、文件、函数、日志、测试或 Dockerfile 时，
   should_search 应为 true。
9. confidence 必须位于 0 到 1 之间。
10. 只能返回一个 JSON object，不要使用 Markdown 代码围栏。

intent 只能是：

- code_search
- code_analysis
- log_diagnosis
- code_completion
- test_generation
- dockerfile_generation
- general_question

返回格式：

{
  "intent": "code_search",
  "should_search": true,
  "semantic_query": "适合语义检索的自然语言查询",
  "keyword_query": "适合关键词检索的技术词和代码标识符",
  "identifiers": ["函数名", "类名"],
  "confidence": 0.9,
  "reason": "简短说明改写原因"
}
""".strip()


def unique_strings(items):
    results = []
    seen = set()

    for item in items:
        if not isinstance(item, str):
            continue

        item = item.strip()

        if not item:
            continue

        normalized = item.lower()

        if normalized in seen:
            continue

        seen.add(normalized)
        results.append(item)

    return results


def extract_identifiers(query):
    """
    通过规则提取用户原文中的代码标识符。

    例如：
    - model.py
    - generate_with_cache
    - TransformerLanguageModel
    - RMSNorm
    - model.load_state_dict
    """
    identifiers = []

    filename_pattern = re.compile(
        r"\b[\w./\\-]+\."
        r"(?:py|json|yaml|yml|toml|md|txt)"
        r"\b",
        flags=re.IGNORECASE,
    )

    snake_case_pattern = re.compile(
        r"\b[A-Za-z][A-Za-z0-9]*"
        r"(?:_[A-Za-z0-9]+)+\b"
    )

    dotted_name_pattern = re.compile(
        r"\b[A-Za-z_][A-Za-z0-9_]*"
        r"(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b"
    )

    class_name_pattern = re.compile(
        r"\b(?:"
        r"[A-Z]{2,}[A-Za-z0-9]*"
        r"|"
        r"[A-Z][a-z0-9]+"
        r"(?:[A-Z][A-Za-z0-9]*)+"
        r")\b"
    )

    for pattern in [
        filename_pattern,
        snake_case_pattern,
        dotted_name_pattern,
        class_name_pattern,
    ]:
        identifiers.extend(
            pattern.findall(query)
        )

    return unique_strings(
        identifiers
    )


def infer_intent_by_rules(query):
    """
    模型不可用时的规则回退。
    """
    lowered = query.lower()

    if (
        "traceback" in lowered
        or "runtimeerror" in lowered
        or "报错" in query
        or "异常" in query
        or "错误日志" in query
    ):
        return "log_diagnosis"

    if (
        "dockerfile" in lowered
        or "docker" in lowered
        or "容器" in query
    ):
        return "dockerfile_generation"

    if (
        "pytest" in lowered
        or "单元测试" in query
        or "生成测试" in query
        or "测试用例" in query
    ):
        return "test_generation"

    if (
        "补全代码" in query
        or "修改代码" in query
        or "重构" in query
        or "实现代码" in query
        or "修复代码" in query
    ):
        return "code_completion"

    if (
        "代码质量" in query
        or "圈复杂度" in query
        or "静态分析" in query
        or "分析文件" in query
    ):
        return "code_analysis"

    repository_triggers = [
        "仓库",
        "项目",
        "代码",
        "文件",
        "函数",
        "类",
        "实现",
        "定义",
        "调用",
        "模型",
        "缓存",
        "训练",
        "推理",
        ".py",
        "_",
    ]

    if any(
        trigger in lowered
        for trigger in repository_triggers
    ):
        return "code_search"

    return "general_question"


def parse_json_object(text):
    """
    解析模型返回的 JSON，兼容代码围栏和前后说明。
    """
    if not isinstance(text, str):
        raise TypeError(
            "模型回答必须是字符串"
        )

    text = text.strip()

    fence_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if fence_match:
        text = fence_match.group(1).strip()

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        first_brace = text.find("{")
        last_brace = text.rfind("}")

        if (
            first_brace < 0
            or last_brace <= first_brace
        ):
            raise ValueError(
                "模型没有返回 JSON object"
            )

        possible_json = text[
            first_brace:last_brace + 1
        ]

        try:
            result = json.loads(
                possible_json
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"模型返回的 JSON 无法解析：{error}"
            ) from error

    if not isinstance(result, dict):
        raise ValueError(
            "改写结果必须是 JSON object"
        )

    return result


class QueryRewriter:
    def __init__(
        self,
        repository_path,
        llm_client,
    ):
        self.repository_path = (
            validate_repository(
                repository_path
            )
        )

        self.llm_client = llm_client

        self.repository_files = [
            str(path).replace("\\", "/")
            for path in list_files(
                self.repository_path
            )
        ]

    def _identifier_exists(
        self,
        identifier,
    ):
        normalized_identifier = (
            identifier
            .replace("\\", "/")
            .lower()
        )

        normalized_files = {
            path.lower()
            for path in self.repository_files
        }

        if (
            normalized_identifier
            in normalized_files
        ):
            return True

        if any(
            path.endswith(
                "/" + normalized_identifier
            )
            for path in normalized_files
        ):
            return True

        try:
            results = search_code(
                repo_path=self.repository_path,
                query=identifier,
                max_results=1,
                context_lines=0,
                case_sensitive=False,
            )
        except Exception:
            return False

        return bool(results)

    def _build_prompt(
        self,
        original_query,
        extracted_identifiers,
    ):
        visible_files = (
            self.repository_files[
                :MAX_REPOSITORY_FILES_IN_PROMPT
            ]
        )

        return (
            "请改写下面的代码仓库查询。\n\n"
            f"用户原始问题：\n"
            f"{original_query}\n\n"
            "规则提取到的标识符：\n"
            f"{json.dumps(extracted_identifiers, ensure_ascii=False)}\n\n"
            "仓库文件列表：\n"
            f"{json.dumps(visible_files, ensure_ascii=False)}\n\n"
            "请只返回规定格式的 JSON object。"
        )

    def _fallback_result(
        self,
        original_query,
        extracted_identifiers,
        error=None,
    ):
        intent = infer_intent_by_rules(
            original_query
        )

        should_search = (
            intent != "general_question"
        )

        keyword_parts = [
            original_query,
            *extracted_identifiers,
        ]

        return {
            "original_query": (
                original_query
            ),
            "intent": intent,
            "should_search": should_search,
            "semantic_query": (
                original_query
            ),
            "keyword_query": " ".join(
                unique_strings(
                    keyword_parts
                )
            ),
            "identifiers": (
                extracted_identifiers
            ),
            "identifier_checks": [
                {
                    "identifier": identifier,
                    "source": "original_query",
                    "exists_in_repository": (
                        self._identifier_exists(
                            identifier
                        )
                    ),
                    "accepted": True,
                }
                for identifier
                in extracted_identifiers
            ],
            "confidence": 0.5,
            "reason": (
                "使用规则回退，未使用模型改写"
            ),
            "rewrite_source": "fallback",
            "rewrite_error": error,
        }

    def _validate_model_result(
        self,
        original_query,
        extracted_identifiers,
        model_result,
    ):
        intent = model_result.get(
            "intent"
        )

        if intent not in ALLOWED_INTENTS:
            intent = infer_intent_by_rules(
                original_query
            )

        should_search = model_result.get(
            "should_search"
        )

        if not isinstance(
            should_search,
            bool,
        ):
            should_search = (
                intent
                != "general_question"
            )

        semantic_query = model_result.get(
            "semantic_query"
        )

        if (
            not isinstance(
                semantic_query,
                str,
            )
            or not semantic_query.strip()
        ):
            semantic_query = (
                original_query
            )

        keyword_query = model_result.get(
            "keyword_query"
        )

        if (
            not isinstance(
                keyword_query,
                str,
            )
            or not keyword_query.strip()
        ):
            keyword_query = (
                original_query
            )

        model_identifiers = (
            model_result.get(
                "identifiers",
                [],
            )
        )

        if not isinstance(
            model_identifiers,
            list,
        ):
            model_identifiers = []

        original_identifier_set = {
            identifier.lower()
            for identifier
            in extracted_identifiers
        }

        accepted_identifiers = list(
            extracted_identifiers
        )

        identifier_checks = []

        for identifier in (
            extracted_identifiers
        ):
            identifier_checks.append(
                {
                    "identifier": identifier,
                    "source": (
                        "original_query"
                    ),
                    "exists_in_repository": (
                        self._identifier_exists(
                            identifier
                        )
                    ),
                    "accepted": True,
                }
            )

        for identifier in unique_strings(
            model_identifiers
        ):
            if (
                identifier.lower()
                in original_identifier_set
            ):
                continue

            exists = self._identifier_exists(
                identifier
            )

            identifier_checks.append(
                {
                    "identifier": identifier,
                    "source": "llm",
                    "exists_in_repository": (
                        exists
                    ),
                    "accepted": exists,
                }
            )

            if exists:
                accepted_identifiers.append(
                    identifier
                )

        accepted_identifiers = (
            unique_strings(
                accepted_identifiers
            )
        )

        keyword_parts = [
            keyword_query.strip(),
            *accepted_identifiers,
        ]

        confidence = model_result.get(
            "confidence",
            0.5,
        )

        try:
            confidence = float(
                confidence
            )
        except (
            TypeError,
            ValueError,
        ):
            confidence = 0.5

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        reason = model_result.get(
            "reason",
            "",
        )

        if not isinstance(reason, str):
            reason = str(reason)

        return {
            "original_query": (
                original_query
            ),
            "intent": intent,
            "should_search": should_search,
            "semantic_query": (
                semantic_query.strip()
            ),
            "keyword_query": " ".join(
                unique_strings(
                    keyword_parts
                )
            ),
            "identifiers": (
                accepted_identifiers
            ),
            "identifier_checks": (
                identifier_checks
            ),
            "confidence": confidence,
            "reason": reason,
            "rewrite_source": "llm",
            "rewrite_error": None,
        }

    def rewrite(
        self,
        original_query,
    ):
        if not isinstance(
            original_query,
            str,
        ):
            raise TypeError(
                "original_query 必须是字符串"
            )

        original_query = (
            original_query.strip()
        )

        if not original_query:
            raise ValueError(
                "原始查询不能为空"
            )

        extracted_identifiers = (
            extract_identifiers(
                original_query
            )
        )

        user_prompt = self._build_prompt(
            original_query=original_query,
            extracted_identifiers=(
                extracted_identifiers
            ),
        )

        try:
            response = (
                self.llm_client
                .create_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                QUERY_REWRITE_SYSTEM_PROMPT
                            ),
                        },
                        {
                            "role": "user",
                            "content": user_prompt,
                        },
                    ],
                    tools=None,
                    temperature=0.0,
                )
            )

            response_text = (
                response.get("content")
                or ""
            ).strip()

            model_result = (
                parse_json_object(
                    response_text
                )
            )

            return (
                self._validate_model_result(
                    original_query=(
                        original_query
                    ),
                    extracted_identifiers=(
                        extracted_identifiers
                    ),
                    model_result=(
                        model_result
                    ),
                )
            )

        except Exception as error:
            return self._fallback_result(
                original_query=(
                    original_query
                ),
                extracted_identifiers=(
                    extracted_identifiers
                ),
                error=(
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            )


class RewrittenHybridRetriever:
    """
    Dense 使用 semantic_query，
    BM25 使用 keyword_query，
    最后通过 RRF 融合。
    """

    def __init__(
        self,
        query_rewriter,
        dense_retriever,
        bm25_retriever,
        rrf_k=RRF_K,
        dense_weight=1.0,
        bm25_weight=1.0,
    ):
        if rrf_k <= 0:
            raise ValueError(
                "rrf_k 必须大于 0"
            )

        self.query_rewriter = (
            query_rewriter
        )

        self.dense_retriever = (
            dense_retriever
        )

        self.bm25_retriever = (
            bm25_retriever
        )

        self.rrf_k = rrf_k
        self.dense_weight = (
            dense_weight
        )
        self.bm25_weight = (
            bm25_weight
        )

    def _add_results(
        self,
        fused_results,
        results,
        source,
        weight,
    ):
        for rank, result in enumerate(
            results,
            start=1,
        ):
            key = make_chunk_key(
                result
            )

            if key not in fused_results:
                fused_results[key] = {
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
                    "content": (
                        result["content"]
                    ),
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "matched_by": [],
                }

            item = fused_results[key]

            item["rrf_score"] += (
                weight
                / (self.rrf_k + rank)
            )

            item["matched_by"].append(
                source
            )

            if source == "dense":
                item["dense_rank"] = rank
                item["dense_score"] = (
                    result["score"]
                )
            elif source == "bm25":
                item["bm25_rank"] = rank
                item["bm25_score"] = (
                    result["score"]
                )

    def search(
        self,
        original_query,
        top_k=5,
        candidate_k=20,
    ):
        if top_k < 1:
            raise ValueError(
                "top_k 必须大于等于 1"
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k 不能小于 top_k"
            )

        rewrite_result = (
            self.query_rewriter.rewrite(
                original_query
            )
        )

        if not rewrite_result[
            "should_search"
        ]:
            return {
                "rewrite": rewrite_result,
                "results": [],
                "skipped": True,
                "skip_reason": (
                    "查询被识别为不需要仓库检索"
                ),
            }

        dense_results = (
            self.dense_retriever.search(
                query=rewrite_result[
                    "semantic_query"
                ],
                top_k=candidate_k,
            )
        )

        bm25_results = (
            self.bm25_retriever.search(
                query=rewrite_result[
                    "keyword_query"
                ],
                top_k=candidate_k,
            )
        )

        fused_results = {}

        self._add_results(
            fused_results=fused_results,
            results=dense_results,
            source="dense",
            weight=self.dense_weight,
        )

        self._add_results(
            fused_results=fused_results,
            results=bm25_results,
            source="bm25",
            weight=self.bm25_weight,
        )

        ranked_results = sorted(
            fused_results.values(),
            key=lambda item: (
                item["rrf_score"],
                len(item["matched_by"]),
            ),
            reverse=True,
        )

        return {
            "rewrite": rewrite_result,
            "results": (
                ranked_results[:top_k]
            ),
            "skipped": False,
            "skip_reason": None,
        }


def print_rewrite_result(
    rewrite_result,
):
    print("\n" + "=" * 70)
    print("Query Rewrite")
    print("=" * 70)

    print(
        "原始问题："
        f"{rewrite_result['original_query']}"
    )

    print(
        "意图："
        f"{rewrite_result['intent']}"
    )

    print(
        "是否检索："
        f"{rewrite_result['should_search']}"
    )

    print(
        "Dense 查询："
        f"{rewrite_result['semantic_query']}"
    )

    print(
        "BM25 查询："
        f"{rewrite_result['keyword_query']}"
    )

    print(
        "保留标识符："
        f"{rewrite_result['identifiers']}"
    )

    print(
        "置信度："
        f"{rewrite_result['confidence']:.2f}"
    )

    print(
        "改写来源："
        f"{rewrite_result['rewrite_source']}"
    )

    print(
        "原因："
        f"{rewrite_result['reason']}"
    )

    if rewrite_result[
        "rewrite_error"
    ]:
        print(
            "改写错误："
            f"{rewrite_result['rewrite_error']}"
        )

    rejected_identifiers = [
        item["identifier"]
        for item in rewrite_result[
            "identifier_checks"
        ]
        if not item["accepted"]
    ]

    if rejected_identifiers:
        print(
            "已拒绝的模型标识符："
            f"{rejected_identifiers}"
        )


def print_search_results(
    search_result,
):
    print_rewrite_result(
        search_result["rewrite"]
    )

    print("\n" + "=" * 70)
    print("检索结果")
    print("=" * 70)

    if search_result["skipped"]:
        print(
            "已跳过仓库检索："
            f"{search_result['skip_reason']}"
        )
        return

    results = search_result["results"]

    if not results:
        print("没有找到相关代码")
        return

    for rank, item in enumerate(
        results,
        start=1,
    ):
        print("-" * 70)

        print(
            f"排名：{rank}"
        )

        print(
            f"文件：{item['file']}:"
            f"{item['start_line']}-"
            f"{item['end_line']}"
        )

        print(
            f"名称：{item['name']}"
        )

        print(
            f"RRF："
            f"{item['rrf_score']:.6f}"
        )

        print(
            "Dense 排名："
            f"{item['dense_rank']}"
        )

        print(
            "BM25 排名："
            f"{item['bm25_rank']}"
        )

        print(
            "命中来源："
            f"{item['matched_by']}"
        )

        print(
            item["content"][:1000]
        )


def run_interactive_mode(
    retriever,
):
    print(
        "\nRepo Doctor Query Rewrite 已启动"
    )

    print(
        "输入 exit、quit 或 q 退出"
    )

    while True:
        try:
            query = input(
                "\n请输入问题："
            ).strip()
        except (
            KeyboardInterrupt,
            EOFError,
        ):
            print("\n程序结束")
            break

        if query.lower() in {
            "exit",
            "quit",
            "q",
        }:
            print("程序结束")
            break

        if not query:
            print("问题不能为空")
            continue

        try:
            result = retriever.search(
                original_query=query,
                top_k=5,
                candidate_k=20,
            )
        except Exception as error:
            print(
                "查询失败："
                f"{type(error).__name__}: "
                f"{error}"
            )
            continue

        print_search_results(
            result
        )


if __name__ == "__main__":
    project_dir = (
        Path(__file__).resolve().parent
    )

    repository_path = Path(
        r"D:\桌面\mini-transformer"
    )

    dense_index_directory = (
        project_dir / "dense_index"
    )

    bm25_index_path = (
        project_dir
        / "bm25_index"
        / "index.json"
    )

    REBUILD_INDEX = False

    try:
        llm_client = (
            ToolCallingLLMClient
            .from_environment()
        )

        query_rewriter = QueryRewriter(
            repository_path=(
                repository_path
            ),
            llm_client=llm_client,
        )

        dense_retriever = (
            prepare_dense_retriever(
                repository_path=(
                    repository_path
                ),
                index_directory=(
                    dense_index_directory
                ),
                rebuild_index=(
                    REBUILD_INDEX
                ),
            )
        )

        bm25_retriever = (
            prepare_bm25_retriever(
                repository_path=(
                    repository_path
                ),
                index_path=(
                    bm25_index_path
                ),
                rebuild_index=(
                    REBUILD_INDEX
                ),
            )
        )

        retriever = (
            RewrittenHybridRetriever(
                query_rewriter=(
                    query_rewriter
                ),
                dense_retriever=(
                    dense_retriever
                ),
                bm25_retriever=(
                    bm25_retriever
                ),
                rrf_k=60,
                dense_weight=1.0,
                bm25_weight=1.0,
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
        retriever
    )
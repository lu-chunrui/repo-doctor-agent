import os
from pathlib import Path

import numpy as np
from sentence_transformers import (
    CrossEncoder,
)

from day7_hybrid_retrieval import (
    prepare_bm25_retriever,
    prepare_dense_retriever,
)
from day10_agent_loop import (
    ToolCallingLLMClient,
)
from day15_query_rewrite import (
    QueryRewriter,
    RewrittenHybridRetriever,
    print_rewrite_result,
)


DEFAULT_RERANKER_MODEL = (
    "BAAI/bge-reranker-base"
)

DEFAULT_RECALL_K = 20
DEFAULT_FINAL_K = 5
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 512
MAX_RERANK_CONTENT_CHARS = 8000


def environment_flag(
    name,
    default=True,
):
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return raw_value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def format_chunk_for_reranker(
    result,
):
    """
    将文件信息、代码块名称和代码正文一起交给 Reranker。

    重要信息放在前面，避免正文过长时路径和名称被截断。
    """
    content = result.get(
        "content",
        "",
    )

    if (
        len(content)
        > MAX_RERANK_CONTENT_CHARS
    ):
        content = (
            content[
                :MAX_RERANK_CONTENT_CHARS
            ]
            + "\n...代码块已截断..."
        )

    name = result.get("name") or "module"

    return (
        f"File: {result['file']}\n"
        f"Lines: {result['start_line']}-"
        f"{result['end_line']}\n"
        f"Type: {result['chunk_type']}\n"
        f"Name: {name}\n"
        f"Code:\n{content}"
    )


def convert_predictions_to_scores(
    predictions,
):
    """
    兼容单输出和双分类 CrossEncoder。

    BGE Reranker 通常返回每个 query-document 对应的单个分数。
    """
    predictions = np.asarray(
        predictions
    )

    if predictions.ndim == 0:
        return [
            float(predictions)
        ]

    if predictions.ndim == 1:
        return [
            float(score)
            for score in predictions
        ]

    if (
        predictions.ndim == 2
        and predictions.shape[1] == 1
    ):
        return [
            float(score)
            for score in predictions[:, 0]
        ]

    if (
        predictions.ndim == 2
        and predictions.shape[1] >= 2
    ):
        # 双分类模型使用最后一个类别的分数。
        return [
            float(score)
            for score
            in predictions[:, -1]
        ]

    raise ValueError(
        "无法识别 Reranker 输出形状："
        f"{predictions.shape}"
    )


class CodeReranker:
    def __init__(
        self,
        model_name=DEFAULT_RERANKER_MODEL,
        enabled=True,
        batch_size=DEFAULT_BATCH_SIZE,
        max_length=DEFAULT_MAX_LENGTH,
        device=None,
        allow_fallback=True,
    ):
        if batch_size < 1:
            raise ValueError(
                "batch_size 必须大于等于 1"
            )

        if max_length < 32:
            raise ValueError(
                "max_length 不能小于 32"
            )

        self.model_name = model_name
        self.enabled = enabled
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.allow_fallback = allow_fallback

        self.model = None
        self.load_error = None

        if self.enabled:
            self._load_model()

    def _load_model(self):
        print("正在加载 Reranker：")
        print(self.model_name)

        try:
            self.model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
        except Exception as error:
            self.load_error = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            if not self.allow_fallback:
                raise

            print(
                "Reranker 加载失败，"
                "将回退到 Hybrid 原始排名："
            )
            print(self.load_error)

    def is_available(self):
        return (
            self.enabled
            and self.model is not None
        )

    def _fallback_results(
        self,
        candidates,
        top_k,
        reason,
    ):
        results = []

        for rank, candidate in enumerate(
            candidates[:top_k],
            start=1,
        ):
            item = dict(candidate)

            item.update(
                {
                    "pre_rerank_rank": rank,
                    "reranker_rank": None,
                    "reranker_score": None,
                    "reranker_status": (
                        "fallback"
                    ),
                    "reranker_error": reason,
                }
            )

            results.append(item)

        return {
            "results": results,
            "reranker_used": False,
            "reranker_model": (
                self.model_name
            ),
            "reranker_error": reason,
        }

    def rerank(
        self,
        query,
        candidates,
        top_k=5,
    ):
        if not isinstance(query, str):
            raise TypeError(
                "query 必须是字符串"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query 不能为空"
            )

        if top_k < 1:
            raise ValueError(
                "top_k 必须大于等于 1"
            )

        if not candidates:
            return {
                "results": [],
                "reranker_used": False,
                "reranker_model": (
                    self.model_name
                ),
                "reranker_error": None,
            }

        if not self.enabled:
            return self._fallback_results(
                candidates=candidates,
                top_k=top_k,
                reason="Reranker 已关闭",
            )

        if self.model is None:
            return self._fallback_results(
                candidates=candidates,
                top_k=top_k,
                reason=(
                    self.load_error
                    or "Reranker 不可用"
                ),
            )

        pairs = []

        for candidate in candidates:
            document = (
                format_chunk_for_reranker(
                    candidate
                )
            )

            pairs.append(
                [
                    query,
                    document,
                ]
            )

        try:
            predictions = (
                self.model.predict(
                    pairs,
                    batch_size=(
                        self.batch_size
                    ),
                    show_progress_bar=(
                        len(pairs) >= 10
                    ),
                    convert_to_numpy=True,
                )
            )

            scores = (
                convert_predictions_to_scores(
                    predictions
                )
            )

        except Exception as error:
            if not self.allow_fallback:
                raise

            return self._fallback_results(
                candidates=candidates,
                top_k=top_k,
                reason=(
                    "Reranker 推理失败："
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            )

        if len(scores) != len(candidates):
            return self._fallback_results(
                candidates=candidates,
                top_k=top_k,
                reason=(
                    "Reranker 分数数量与"
                    "候选数量不一致"
                ),
            )

        scored_candidates = []

        for original_rank, (
            candidate,
            score,
        ) in enumerate(
            zip(candidates, scores),
            start=1,
        ):
            item = dict(candidate)

            item.update(
                {
                    "pre_rerank_rank": (
                        original_rank
                    ),
                    "reranker_score": (
                        float(score)
                    ),
                    "reranker_status": (
                        "success"
                    ),
                    "reranker_error": None,
                }
            )

            scored_candidates.append(
                item
            )

        scored_candidates.sort(
            key=lambda item: (
                item["reranker_score"],
                item.get(
                    "rrf_score",
                    0.0,
                ),
            ),
            reverse=True,
        )

        final_results = []

        for reranker_rank, item in enumerate(
            scored_candidates[:top_k],
            start=1,
        ):
            item["reranker_rank"] = (
                reranker_rank
            )

            final_results.append(item)

        return {
            "results": final_results,
            "reranker_used": True,
            "reranker_model": (
                self.model_name
            ),
            "reranker_error": None,
        }


class RerankedCodeSearch:
    """
    完整检索流程：

    Query Rewrite
    → Dense/BM25/RRF 召回
    → CrossEncoder Reranker
    """

    def __init__(
        self,
        rewritten_retriever,
        reranker,
    ):
        self.rewritten_retriever = (
            rewritten_retriever
        )

        self.reranker = reranker

    def search(
        self,
        query,
        final_k=DEFAULT_FINAL_K,
        recall_k=DEFAULT_RECALL_K,
        retrieval_candidate_k=30,
    ):
        if final_k < 1:
            raise ValueError(
                "final_k 必须大于等于 1"
            )

        if recall_k < final_k:
            raise ValueError(
                "recall_k 不能小于 final_k"
            )

        if (
            retrieval_candidate_k
            < recall_k
        ):
            raise ValueError(
                "retrieval_candidate_k "
                "不能小于 recall_k"
            )

        retrieval_result = (
            self.rewritten_retriever.search(
                original_query=query,
                top_k=recall_k,
                candidate_k=(
                    retrieval_candidate_k
                ),
            )
        )

        rewrite_result = (
            retrieval_result["rewrite"]
        )

        if retrieval_result["skipped"]:
            return {
                "original_query": query,
                "rewrite": rewrite_result,
                "recall_results": [],
                "results": [],
                "skipped": True,
                "skip_reason": (
                    retrieval_result[
                        "skip_reason"
                    ]
                ),
                "reranker_used": False,
                "reranker_model": (
                    self.reranker.model_name
                ),
                "reranker_error": None,
            }

        recall_results = (
            retrieval_result["results"]
        )

        # Reranker 使用原始问题。
        # 这样能保留用户真正的表达和意图。
        rerank_result = (
            self.reranker.rerank(
                query=query,
                candidates=(
                    recall_results
                ),
                top_k=final_k,
            )
        )

        return {
            "original_query": query,
            "rewrite": rewrite_result,
            "recall_results": (
                recall_results
            ),
            "results": (
                rerank_result["results"]
            ),
            "skipped": False,
            "skip_reason": None,
            "reranker_used": (
                rerank_result[
                    "reranker_used"
                ]
            ),
            "reranker_model": (
                rerank_result[
                    "reranker_model"
                ]
            ),
            "reranker_error": (
                rerank_result[
                    "reranker_error"
                ]
            ),
        }


def print_rank_change(
    result,
):
    print("\n" + "=" * 70)
    print("Reranker 排名变化")
    print("=" * 70)

    if result["skipped"]:
        print(
            "已跳过检索："
            f"{result['skip_reason']}"
        )
        return

    print(
        "是否使用 Reranker："
        f"{result['reranker_used']}"
    )

    print(
        "Reranker 模型："
        f"{result['reranker_model']}"
    )

    if result["reranker_error"]:
        print(
            "Reranker 信息："
            f"{result['reranker_error']}"
        )

    results = result["results"]

    if not results:
        print("没有找到相关代码")
        return

    for item in results:
        print("-" * 70)

        print(
            f"Reranker 排名："
            f"{item['reranker_rank']}"
        )

        print(
            f"召回阶段排名："
            f"{item['pre_rerank_rank']}"
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
            f"Hybrid RRF："
            f"{item['rrf_score']:.6f}"
        )

        if (
            item["reranker_score"]
            is not None
        ):
            print(
                "Reranker 分数："
                f"{item['reranker_score']:.6f}"
            )
        else:
            print(
                "Reranker 分数：未计算"
            )

        print(
            item["content"][:1000]
        )


def print_recall_comparison(
    result,
    show_count=10,
):
    """
    显示重排前排名，便于观察变化。
    """
    if result["skipped"]:
        return

    print("\n" + "=" * 70)
    print("重排前 Hybrid 排名")
    print("=" * 70)

    for rank, item in enumerate(
        result["recall_results"][
            :show_count
        ],
        start=1,
    ):
        print(
            f"{rank}. "
            f"{item['file']}:"
            f"{item['start_line']}-"
            f"{item['end_line']} "
            f"name={item['name']} "
            f"RRF={item['rrf_score']:.6f}"
        )


def run_interactive_mode(
    search_pipeline,
):
    print(
        "\nRepo Doctor Reranker 已启动"
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
            result = (
                search_pipeline.search(
                    query=query,
                    final_k=5,
                    recall_k=20,
                    retrieval_candidate_k=30,
                )
            )
        except Exception as error:
            print(
                "检索失败："
                f"{type(error).__name__}: "
                f"{error}"
            )
            continue

        print_rewrite_result(
            result["rewrite"]
        )

        print_recall_comparison(
            result
        )

        print_rank_change(
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

    reranker_enabled = (
        environment_flag(
            "RERANKER_ENABLED",
            default=True,
        )
    )

    reranker_model_name = (
        os.getenv(
            "RERANKER_MODEL_NAME",
            DEFAULT_RERANKER_MODEL,
        )
    )

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

        rewritten_retriever = (
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

        reranker = CodeReranker(
            model_name=(
                reranker_model_name
            ),
            enabled=reranker_enabled,
            batch_size=4,
            max_length=512,
            allow_fallback=True,
        )

        search_pipeline = (
            RerankedCodeSearch(
                rewritten_retriever=(
                    rewritten_retriever
                ),
                reranker=reranker,
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
        search_pipeline
    )
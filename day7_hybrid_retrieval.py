from pathlib import Path

from day5_dense_retrieval import (
    DenseCodeRetriever,
)
from day6_bm25_retrieval import (
    BM25CodeRetriever,
)


RRF_K = 60
DEFAULT_CANDIDATE_MULTIPLIER = 4
MIN_CANDIDATE_COUNT = 20


def make_chunk_key(result):
   
    return (
        result["file"],
        result["start_line"],
        result["end_line"],
        result["chunk_type"],
        result["name"],
    )


class HybridCodeRetriever:
    
    def __init__(
        self,
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

        if dense_weight < 0:
            raise ValueError(
                "dense_weight 不能小于 0"
            )

        if bm25_weight < 0:
            raise ValueError(
                "bm25_weight 不能小于 0"
            )

        if (
            dense_weight == 0
            and bm25_weight == 0
        ):
            raise ValueError(
                "两个检索器的权重不能同时为 0"
            )

        self.dense_retriever = (
            dense_retriever
        )
        self.bm25_retriever = (
            bm25_retriever
        )

        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

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
            chunk_key = make_chunk_key(
                result
            )

            if chunk_key not in fused_results:
                fused_results[chunk_key] = {
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
                    "content": result["content"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "dense_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "matched_by": [],
                }

            fused_item = fused_results[
                chunk_key
            ]

            rrf_contribution = (
                weight
                / (self.rrf_k + rank)
            )

            fused_item["rrf_score"] += (
                rrf_contribution
            )

            if source == "dense":
                fused_item["dense_rank"] = rank
                fused_item["dense_score"] = (
                    result["score"]
                )
            elif source == "bm25":
                fused_item["bm25_rank"] = rank
                fused_item["bm25_score"] = (
                    result["score"]
                )
            else:
                raise ValueError(
                    f"未知检索来源：{source}"
                )

            fused_item["matched_by"].append(
                source
            )

    def search(
        self,
        query,
        top_k=5,
        candidate_k=None,
    ):
      
        if not isinstance(query, str):
            raise TypeError(
                "query 必须是字符串"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "查询不能为空"
            )

        if top_k < 1:
            raise ValueError(
                "top_k 必须大于等于 1"
            )

        if candidate_k is None:
            candidate_k = max(
                top_k
                * DEFAULT_CANDIDATE_MULTIPLIER,
                MIN_CANDIDATE_COUNT,
            )

        if candidate_k < top_k:
            raise ValueError(
                "candidate_k 不能小于 top_k"
            )

        dense_results = (
            self.dense_retriever.search(
                query,
                top_k=candidate_k,
            )
        )

        bm25_results = (
            self.bm25_retriever.search(
                query,
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
                item["dense_rank"] is not None,
                item["bm25_rank"] is not None,
            ),
            reverse=True,
        )

        return ranked_results[:top_k]


def format_optional_score(score):
    
    if score is None:
        return "未命中"

    return f"{score:.4f}"


def format_optional_rank(rank):
   
    if rank is None:
        return "未命中"

    return str(rank)


def print_search_results(results):
    if not results:
        print("没有找到相关代码")
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print("=" * 70)

        print(
            f"混合排名：{rank}"
        )

        print(
            f"RRF 分数："
            f"{result['rrf_score']:.6f}"
        )

        print(
            "Dense："
            f"排名={format_optional_rank(result['dense_rank'])}，"
            f"分数={format_optional_score(result['dense_score'])}"
        )

        print(
            "BM25："
            f"排名={format_optional_rank(result['bm25_rank'])}，"
            f"分数={format_optional_score(result['bm25_score'])}"
        )

        print(
            "命中来源："
            f"{', '.join(result['matched_by'])}"
        )

        print(
            f"文件：{result['file']}:"
            f"{result['start_line']}-"
            f"{result['end_line']}"
        )

        print(
            f"类型：{result['chunk_type']}"
        )

        print(
            f"名称：{result['name']}"
        )

        print("-" * 70)

        print(
            result["content"][:1200]
        )


def prepare_dense_retriever(
    repository_path,
    index_directory,
    rebuild_index,
):
   
    retriever = DenseCodeRetriever()

    embeddings_path = (
        index_directory / "embeddings.npy"
    )

    metadata_path = (
        index_directory / "metadata.json"
    )

    index_exists = (
        embeddings_path.exists()
        and metadata_path.exists()
    )

    if rebuild_index or not index_exists:
        index_info = retriever.build_index(
            repository_path
        )

        print(
            "Dense 索引信息：",
            index_info,
        )

        retriever.save_index(
            index_directory
        )
    else:
        retriever.load_index(
            index_directory
        )

    return retriever


def prepare_bm25_retriever(
    repository_path,
    index_path,
    rebuild_index,
):
    
    retriever = BM25CodeRetriever()

    if rebuild_index or not index_path.exists():
        index_info = retriever.build_index(
            repository_path
        )

        print(
            "BM25 索引信息：",
            index_info,
        )

        retriever.save_index(
            index_path
        )
    else:
        retriever.load_index(
            index_path
        )

    return retriever


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

    REBUILD_INDEX = True

    dense_retriever = (
        prepare_dense_retriever(
            repository_path,
            dense_index_directory,
            REBUILD_INDEX,
        )
    )

    bm25_retriever = (
        prepare_bm25_retriever(
            repository_path,
            bm25_index_path,
            REBUILD_INDEX,
        )
    )

    hybrid_retriever = HybridCodeRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
        dense_weight=1.0,
        bm25_weight=1.0,
    )

    queries = [
    
        "模型如何使用缓存复用历史注意力状态？",

        "generate_with_cache",

        "在哪里计算交叉熵损失？",

        "RMSNorm 归一化是怎么实现的？",
    ]

    for query in queries:
        print("\n")
        print("#" * 70)
        print(f"搜索问题：{query}")
        print("#" * 70)

        results = hybrid_retriever.search(
            query,
            top_k=5,
            candidate_k=20,
        )

        print_search_results(results)
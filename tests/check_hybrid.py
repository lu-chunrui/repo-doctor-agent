from config import settings
from rag.bm25 import BM25CodeRetriever
from rag.dense import DenseCodeRetriever
from rag.hybrid import HybridCodeRetriever


if __name__ == "__main__":
    repository_path = (
        settings.resolved_default_repository()
    )

    dense = DenseCodeRetriever()
    dense.build_index(repository_path)

    bm25 = BM25CodeRetriever()
    bm25.build_index(repository_path)

    hybrid = HybridCodeRetriever(
        dense_retriever=dense,
        bm25_retriever=bm25,
    )

    query = "模型如何生成下一个 token？"

    print("查询：", query)

    results = hybrid.search(
        query=query,
        top_k=5,
        candidate_k=10,
    )

    for index, item in enumerate(
        results,
        start=1,
    ):
        print("\n" + "=" * 60)
        print("排名：", index)
        print("文件：", item["file"])
        print(
            "行号：",
            f"{item['start_line']}-"
            f"{item['end_line']}",
        )
        print(
            "命中来源：",
            item["matched_by"],
        )
        print(
            "RRF 分数：",
            item["rrf_score"],
        )
        print(
            "Dense 排名：",
            item["dense_rank"],
        )
        print(
            "BM25 排名：",
            item["bm25_rank"],
        )
        print(
            "内容：",
            item["content"][:300],
        )
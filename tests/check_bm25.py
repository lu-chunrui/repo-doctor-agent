from config import settings
from rag.bm25 import BM25CodeRetriever


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试 BM25")
    print("=" * 60)

    retriever = BM25CodeRetriever()

    print("k1：", retriever.k1)
    print("b：", retriever.b)

    repository_path = (
        settings.resolved_default_repository()
    )

    result = retriever.build_index(
        repository_path
    )

    print("代码块数量：", result["document_count"])

    query = "FastAPI 应用在哪里创建？"
    results = retriever.search(
        query,
        top_k=3,
    )

    for index, item in enumerate(
        results,
        start=1,
    ):
        print("\n" + "-" * 60)
        print("排名：", index)
        print("文件：", item["file"])
        print(
            "行号：",
            f"{item['start_line']}-"
            f"{item['end_line']}",
        )
        print("分数：", item["score"])
        print(
            "内容：",
            item["content"][:300],
        )
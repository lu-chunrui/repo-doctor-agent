from config import settings
from rag.dense import DenseCodeRetriever


if __name__ == "__main__":
    print("=" * 60)
    print("开始测试 Dense 检索器")
    print("=" * 60)

    retriever = DenseCodeRetriever()

    print("模型加载成功")
    print("模型名称：", retriever.model_name)
    print(
        "配置 Batch Size：",
        settings.embedding_batch_size,
    )

    repository_path = (
        settings.resolved_default_repository()
    )

    result = retriever.build_index(
        repository_path
    )

    print("\n索引建立成功")
    print("仓库：", result["repository"])
    print(
        "代码块数量：",
        result["chunk_count"],
    )
    print(
        "向量维度：",
        result["embedding_dimension"],
    )

    query = "项目的主要入口在哪里？"

    print("\n查询：", query)

    search_results = retriever.search(
        query,
        top_k=3,
    )

    for index, item in enumerate(
        search_results,
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
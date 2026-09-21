from config import settings


if __name__ == "__main__":
    print("配置加载成功")
    print("LLM 模型：", settings.llm_model)
    print(
        "Embedding 模型：",
        settings.embedding_model_name,
    )
    print(
        "Embedding Batch Size：",
        settings.embedding_batch_size,
    )
    print("BM25 k1：", settings.bm25_k1)
    print("BM25 b：", settings.bm25_b)
    print(
        "Reranker 模型：",
        settings.reranker_model_name,
    )
    print(
        "最大工具步数：",
        settings.max_tool_steps,
    )
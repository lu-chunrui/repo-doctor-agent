import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from repository.scanner import(validate_repository)
from repository.splitter import(split_repository)
from config import settings


INDEXABLE_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
}
IGNORED_INDEX_DIRS = {
    "data",
    "datasets",
    "logs",
    "experiments",
    "checkpoints",
    "__pycache__",
    "training",
}

def should_index_chunk(chunk):
    file_path = Path(chunk["file"])

    if file_path.suffix.lower() not in INDEXABLE_SUFFIXES:
        return False

    path_parts = {
        part.lower()
        for part in file_path.parts
    }

    if path_parts & IGNORED_INDEX_DIRS:
        return False

    if not chunk["content"].strip():
        return False

    return True
def format_chunk_for_embedding(chunk):
    name = chunk["name"] or "module"

    return (
        "passage: "
        f"File: {chunk['file']}\n"
        f"Type: {chunk['chunk_type']}\n"
        f"Name: {name}\n"
        f"Code:\n{chunk['content']}"
    )
class DenseCodeRetriever:
    def __init__(
    self,
    model_name=None,
    batch_size=None,
):
        
        if model_name is None:
            model_name = (
                settings.embedding_model_name
            )

        if not isinstance(model_name, str):
            raise TypeError(
            "model_name 必须是字符串"
        )

        model_name = model_name.strip()

        if not model_name:
            raise ValueError(
            "model_name 不能为空"
        )

        self.model_name = model_name

        self.batch_size = (
        settings.embedding_batch_size
        if batch_size is None
        else batch_size
    )

        if self.batch_size < 1:
            raise ValueError(
            "batch_size 必须大于等于 1"
        )

        print(
        "正在加载 Embedding 模型："
    )
        print(self.model_name)

        self.model = SentenceTransformer(
        self.model_name
    )

        self.chunks = []
        self.embeddings = None
    def build_index(self,repo_path):
        repo_path = validate_repository(repo_path)
        print("正在切分仓库……")
        all_chunks = split_repository(repo_path)
        self.chunks = [chunk for chunk in all_chunks if should_index_chunk(chunk)]
        if not self.chunks:
           raise ValueError("仓库中没有可索引的代码块")
        passages = [format_chunk_for_embedding(chunk) for chunk in self.chunks]
        print(f"准备编码 {len(passages)} 个代码块")
        self.embeddings = self.model.encode(
            passages,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        print("索引建立完成")
        print(f"向量矩阵形状：{self.embeddings.shape}")
        return { "repository": str(repo_path),
            "chunk_count": len(self.chunks),
            "embedding_dimension": (
                self.embeddings.shape[1]
            ),}
    def search(self,query,top_k=5):
        if self.embeddings is None:
            raise ValueError("索引未建立")
        if not isinstance(query,str):
            raise ValueError("查询必须是字符串")
        query=query.strip()
        if not query:
            raise ValueError("查询不能为空")
        if top_k<1:
            raise ValueError("top_k 必须大于等于 1")
        query_text=f"query: {query}"
        query_embedding = self.model.encode(query_text,convert_to_numpy=True,normalize_embeddings=True)
        scores = self.embeddings @ query_embedding
        result_count = min(top_k,len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:result_count]
        results=[]
        for index in top_indices:
            chunk=self.chunks[index]
            results.append({"file": chunk["file"],
                    "start_line": (
                        chunk["start_line"]
                    ),
                    "end_line": (
                        chunk["end_line"]
                    ),
                    "chunk_type": (
                        chunk["chunk_type"]
                    ),
                    "name": chunk["name"],
                    "score": float(scores[index]),
                    "content": chunk["content"]},)
        return results
    def save_index(self,index_path):
        if self.embeddings is None:
            raise ValueError("索引未建立")
        index_dir = Path(index_path)
        index_dir.mkdir(parents=True,exist_ok=True)
        embeddings_path = index_dir / "embeddings.npy"
        metadata_path = index_dir / "metadata.json"
        np.save(embeddings_path,self.embeddings)
        metadata ={"model_name": self.model_name,
            "chunks": self.chunks,}
        metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"索引已保存到 {index_path}")
    def load_index(self,index_dir):
        index_dir = Path(index_dir)
        embeddings_path = index_dir / "embeddings.npy"
        metadata_path = index_dir / "metadata.json"
        if not embeddings_path.exists() or not metadata_path.exists():
            raise ValueError("索引文件不存在")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        saved_model_name=metadata["model_name"]
        if saved_model_name!=self.model_name:
            raise ValueError(f"索引模型 {saved_model_name} 与当前模型 {self.model_name} 不匹配")
        self.embeddings = np.load(embeddings_path)
        self.chunks = metadata["chunks"]
        if len(self.chunks)!=len(self.embeddings):
            raise ValueError("索引文件与向量矩阵不匹配")
        print(
            f"成功加载 {len(self.chunks)} "
            f"个代码块"
        )
def print_search_results(results):
    if not results:
        print("没有找到相关代码块")
        return
    for rank,result in enumerate(results,start=1):
        print("=" * 70)

        print(
            f"排名：{rank}，"
            f"相似度：{result['score']:.4f}"
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
        content_preview = result[
            "content"
        ][:1200]

        print(content_preview)
if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    repository_path = Path(
        settings.resolved_default_repository()
    )

    index_directory = Path(
        project_dir / "dense_index"
    )

    retriever = DenseCodeRetriever()

    if (
        (index_directory / "embeddings.npy").exists()
        and
        (index_directory / "metadata.json").exists()
    ):
        retriever.load_index(
            index_directory
        )
    else:
        index_info = retriever.build_index(
            repository_path
        )

        print("索引信息：", index_info)

        retriever.save_index(
            index_directory
        )

    queries = [
        "模型如何使用缓存复用历史注意力状态？",
        "在哪里计算交叉熵损失？",
        "项目使用什么归一化方法？",
    ]

    for query in queries:
        print("\n搜索问题：", query)

        results = retriever.search(
            query,
            top_k=5,
        )

        print_search_results(results)
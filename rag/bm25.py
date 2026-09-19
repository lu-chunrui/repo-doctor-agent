import json
import math
import re
from collections import Counter
from pathlib import Path

from repository.scanner import (
    validate_repository,
)
from repository.splitter import (
    split_repository,
)
from rag.dense import (
    should_index_chunk,
)


BM25_K1 = 1.5
BM25_B = 0.75


def split_code_identifier(identifier):
    
    identifier = identifier.strip()

    if not identifier:
        return []

    full_identifier = identifier.lower()

    without_underscore = identifier.replace(
        "_",
        " ",
    )

    camel_case_split = re.sub(
        r"([a-z0-9])([A-Z])",
        r"\1 \2",
        without_underscore,
    )

    parts = re.findall(
        r"[A-Za-z]+|\d+",
        camel_case_split,
    )

    tokens = [full_identifier]

    tokens.extend(
        part.lower()
        for part in parts
        if part
    )

    return tokens


def split_chinese_text(text):
   
    tokens = []

    for chinese_text in re.findall(
        r"[\u4e00-\u9fff]+",
        text,
    ):
        tokens.extend(chinese_text)

        for index in range(
            len(chinese_text) - 1
        ):
            tokens.append(
                chinese_text[
                    index:index + 2
                ]
            )

    return tokens


def tokenize_for_bm25(text):
    
    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")

    tokens = []

    identifiers = re.findall(
        r"[A-Za-z][A-Za-z0-9_]*|\d+",
        text,
    )

    for identifier in identifiers:
        tokens.extend(
            split_code_identifier(identifier)
        )

    tokens.extend(
        split_chinese_text(text)
    )

    return tokens


def format_chunk_for_bm25(chunk):
    
    name = chunk["name"] or "module"

    return (
        f"File: {chunk['file']}\n"
        f"Type: {chunk['chunk_type']}\n"
        f"Name: {name}\n"
        f"Code:\n{chunk['content']}"
    )


class BM25CodeRetriever:
    def __init__(
        self,
        k1=BM25_K1,
        b=BM25_B,
    ):
        if k1 <= 0:
            raise ValueError(
                "k1 必须大于 0"
            )

        if not 0 <= b <= 1:
            raise ValueError(
                "b 必须位于 0 到 1 之间"
            )

        self.k1 = k1
        self.b = b

        self.chunks = []
        self.documents = []
        self.tokenized_documents = []
        self.term_frequencies = []
        self.document_frequencies = {}

        self.document_lengths = []
        self.average_document_length = 0.0

    def build_index(self, repo_path):
        repo_path = validate_repository(
            repo_path
        )

        print("正在切分仓库……")

        all_chunks = split_repository(
            repo_path
        )

        self.chunks = [
            chunk
            for chunk in all_chunks
            if should_index_chunk(chunk)
        ]

        if not self.chunks:
            raise ValueError(
                "仓库中没有可以建立索引的代码块"
            )

        self.documents = [
            format_chunk_for_bm25(chunk)
            for chunk in self.chunks
        ]

        self._prepare_statistics()

        print("BM25 索引建立完成")
        print(
            f"文档数量：{len(self.chunks)}"
        )
        print(
            "平均文档长度："
            f"{self.average_document_length:.2f}"
        )
        print(
            f"词表大小："
            f"{len(self.document_frequencies)}"
        )

        return {
            "repository": str(repo_path),
            "document_count": len(
                self.chunks
            ),
            "average_document_length": (
                self.average_document_length
            ),
            "vocabulary_size": len(
                self.document_frequencies
            ),
        }

    def _prepare_statistics(self):
       
        self.tokenized_documents = [
            tokenize_for_bm25(document)
            for document in self.documents
        ]

        self.term_frequencies = [
            Counter(tokens)
            for tokens in self.tokenized_documents
        ]

        self.document_lengths = [
            len(tokens)
            for tokens in self.tokenized_documents
        ]

        if not self.document_lengths:
            raise ValueError(
                "没有可用于 BM25 的文档"
            )

        self.average_document_length = (
            sum(self.document_lengths)
            / len(self.document_lengths)
        )

        document_frequency_counter = Counter()

        for tokens in self.tokenized_documents:
            unique_tokens = set(tokens)

            for token in unique_tokens:
                document_frequency_counter[
                    token
                ] += 1

        self.document_frequencies = dict(
            document_frequency_counter
        )

    def calculate_idf(self, token):
       
        document_count = len(
            self.tokenized_documents
        )

        document_frequency = (
            self.document_frequencies.get(
                token,
                0,
            )
        )

        return math.log(
            1
            + (
                document_count
                - document_frequency
                + 0.5
            )
            / (
                document_frequency
                + 0.5
            )
        )

    def calculate_document_score(
        self,
        query_tokens,
        document_index,
    ):
       
        term_frequency = (
            self.term_frequencies[
                document_index
            ]
        )

        document_length = (
            self.document_lengths[
                document_index
            ]
        )

        score = 0.0

        for token in query_tokens:
            frequency = term_frequency.get(
                token,
                0,
            )

            if frequency == 0:
                continue

            idf = self.calculate_idf(token)

            length_normalization = (
                1
                - self.b
                + self.b
                * document_length
                / self.average_document_length
            )

            numerator = (
                frequency
                * (self.k1 + 1)
            )

            denominator = (
                frequency
                + self.k1
                * length_normalization
            )

            score += (
                idf
                * numerator
                / denominator
            )

        return score

    def search(self, query, top_k=5):
        if not self.chunks:
            raise RuntimeError(
                "尚未建立或加载 BM25 索引"
            )

        if not isinstance(query, str):
            raise TypeError(
                "query 必须是字符串"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "搜索关键词不能为空"
            )

        if top_k < 1:
            raise ValueError(
                "top_k 必须大于等于 1"
            )

        query_tokens = tokenize_for_bm25(
            query
        )

        scores = []

        for document_index in range(
            len(self.chunks)
        ):
            score = self.calculate_document_score(
                query_tokens,
                document_index,
            )

            scores.append(score)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices:
            # 全部关键词都没有命中时，不返回无关文档
            if scores[index] <= 0:
                continue

            chunk = self.chunks[index]

            results.append(
                {
                    "file": chunk["file"],
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
                    "score": float(
                        scores[index]
                    ),
                    "content": chunk["content"],
                }
            )

            if len(results) >= top_k:
                break

        return results

    def save_index(self, index_path):
        if not self.chunks:
            raise RuntimeError(
                "没有可以保存的 BM25 索引"
            )

        index_path = Path(index_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_data = {
            "k1": self.k1,
            "b": self.b,
            "chunks": self.chunks,
        }

        index_path.write_text(
            json.dumps(
                index_data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"BM25 索引已保存：{index_path}"
        )

    def load_index(self, index_path):
        index_path = Path(index_path)

        if not index_path.exists():
            raise FileNotFoundError(
                f"BM25 索引不存在：{index_path}"
            )

        index_data = json.loads(
            index_path.read_text(
                encoding="utf-8"
            )
        )

        self.k1 = index_data["k1"]
        self.b = index_data["b"]
        self.chunks = index_data["chunks"]

        self.documents = [
            format_chunk_for_bm25(chunk)
            for chunk in self.chunks
        ]

        self._prepare_statistics()

        print(
            f"成功加载 {len(self.chunks)} "
            "个 BM25 文档"
        )


def print_search_results(results):
    if not results:
        print("没有找到关键词匹配结果")
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print("=" * 70)

        print(
            f"排名：{rank}，"
            f"BM25 分数："
            f"{result['score']:.4f}"
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


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent

    repository_path = Path(
        r"D:\桌面\mini-transformer"
    )

    index_path = (
        project_dir
        / "bm25_index"
        / "index.json"
    )

    retriever = BM25CodeRetriever()

    if index_path.exists():
        retriever.load_index(
            index_path
        )
    else:
        index_info = retriever.build_index(
            repository_path
        )

        print("索引信息：", index_info)

        retriever.save_index(
            index_path
        )

    queries = [
        "generate_with_cache",
        "RMSNorm",
        "cross_entropy",
        "model_state_dict",
    ]

    for query in queries:
        print("\n搜索关键词：", query)

        results = retriever.search(
            query,
            top_k=5,
        )

        print_search_results(results)
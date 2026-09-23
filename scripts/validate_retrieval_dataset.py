import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)
DATA_DIR = (
    PROJECT_ROOT / "data" / "retrieval"
)

ALLOWED_QUERY_TYPES = {
    "identifier",
    "natural_language",
    "colloquial_zh",
    "cross_file",
    "misleading_keyword",
    "irrelevant",
}

REQUIRED_FIELDS = {
    "id",
    "repository_id",
    "split",
    "query_type",
    "query",
    "should_retrieve",
    "relevant_files",
    "relevant_chunks",
    "notes",
}


def main():
    repositories_path = (
        DATA_DIR / "repositories.json"
    )
    queries_path = (
        DATA_DIR / "queries.jsonl"
    )

    repositories_data = json.loads(
        repositories_path.read_text(
            encoding="utf-8"
        )
    )

    repository_ids = {
        item["id"]
        for item in repositories_data[
            "repositories"
        ]
    }

    errors = []
    queries = []
    seen_ids = set()
    seen_queries = set()

    for line_number, line in enumerate(
        queries_path.read_text(
            encoding="utf-8"
        ).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(
                f"第 {line_number} 行 JSON 错误："
                f"{error}"
            )
            continue

        missing_fields = (
            REQUIRED_FIELDS - item.keys()
        )

        if missing_fields:
            errors.append(
                f"{item.get('id', line_number)} "
                f"缺少字段："
                f"{sorted(missing_fields)}"
            )
            continue

        query_id = item["id"]
        query = item["query"].strip()

        if query_id in seen_ids:
            errors.append(
                f"ID 重复：{query_id}"
            )
        seen_ids.add(query_id)

        normalized_query = query.lower()

        if normalized_query in seen_queries:
            errors.append(
                f"问题重复：{query}"
            )
        seen_queries.add(normalized_query)

        if not query:
            errors.append(
                f"{query_id} 的问题为空"
            )

        if (
            item["repository_id"]
            not in repository_ids
        ):
            errors.append(
                f"{query_id} 使用了未知仓库："
                f"{item['repository_id']}"
            )

        if (
            item["query_type"]
            not in ALLOWED_QUERY_TYPES
        ):
            errors.append(
                f"{query_id} 查询类型错误："
                f"{item['query_type']}"
            )

        if item["split"] not in {
            "dev",
            "test",
        }:
            errors.append(
                f"{query_id} split 必须是 "
                "dev 或 test"
            )

        if item["should_retrieve"]:
            if not item["relevant_files"]:
                errors.append(
                    f"{query_id} 应检索，"
                    "但没有标准答案文件"
                )
        else:
            if (
                item["relevant_files"]
                or item["relevant_chunks"]
            ):
                errors.append(
                    f"{query_id} 不应检索，"
                    "但填写了标准答案"
                )

        if (
            item["repository_id"]
            == "repo-doctor"
        ):
            for relative_file in item[
                "relevant_files"
            ]:
                file_path = (
                    PROJECT_ROOT
                    / relative_file
                )

                if not file_path.is_file():
                    errors.append(
                        f"{query_id} 的文件不存在："
                        f"{relative_file}"
                    )

        queries.append(item)

    if errors:
        print("数据校验失败：")

        for error in errors:
            print("-", error)

        raise SystemExit(1)

    type_counts = Counter(
        item["query_type"]
        for item in queries
    )

    print("数据校验成功")
    print("仓库数量：", len(repository_ids))
    print("查询数量：", len(queries))
    print("查询类型分布：")

    for query_type, count in sorted(
        type_counts.items()
    ):
        print(
            f"- {query_type}: {count}"
        )


if __name__ == "__main__":
    main()
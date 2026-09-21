from config import settings
from repository.splitter import (
    split_repository,
)


if __name__ == "__main__":
    repository_path = (
        settings.resolved_default_repository()
    )

    chunks = split_repository(
        repository_path
    )

    print("代码块数量：", len(chunks))

    for chunk in chunks[:5]:
        print("=" * 60)
        print("文件：", chunk["file"])
        print("类型：", chunk["chunk_type"])
        print("名称：", chunk["name"])
        print(
            "行号：",
            f"{chunk['start_line']}-"
            f"{chunk['end_line']}",
        )
        print(
            "内容：",
            chunk["content"][:200],
        )
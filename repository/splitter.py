import ast
from pathlib import Path

from repository.scanner import (list_files,validate_repository)
from repository.search import resolve_safe_path

TEXT_CHUNK_LINES = 80
TEXT_CHUNK_OVERLAP = 10

def create_chunk(relative_path,start_line,end_line,chunk_type,name,content):
    return {
        "file": str(relative_path),
        "start_line": start_line,
        "end_line": end_line,
        "chunk_type": chunk_type,
        "name": name,
        "content": content,
    }
def split_text_lines(lines,relative_path,chunk_type="text",chunk_lines=TEXT_CHUNK_LINES,overlap=TEXT_CHUNK_OVERLAP,):
    if chunk_lines < 1:
        raise ValueError("chunk_lines 必须大于等于 1")
    if overlap < 0 or overlap >= chunk_lines:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_lines")
    chunks = []
    total_lines = len(lines)
    if total_lines == 0:
        return chunks
    step = chunk_lines - overlap
    start_line = 0
    while start_line < total_lines:
        end_line = min(start_line + chunk_lines, total_lines)
        content = "".join(lines[start_line:end_line])
        chunks.append(create_chunk(relative_path=relative_path,start_line=start_line+1,end_line=end_line,chunk_type=chunk_type,name=None,content=content))
        if end_line == total_lines:
            break
        start_line += step
    return chunks
def get_node_start_line(node):
    start_line = node.lineno
    if node.decorator_list:
        decorator_lines=[decorator.lineno for decorator in node.decorator_list]
        start_line = min(start_line,min(decorator_lines))
    return start_line
def split_python_source(source,relative_path):
    lines=source.splitlines(keepends=True)
    if not lines:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return split_text_lines(
            lines,
            relative_path,
            chunk_type="python_text",
        )
    chunks = []
    current_line=1
    semantic_nodes=[
        node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,),)]
    if not semantic_nodes:
        return split_text_lines(lines,relative_path,chunk_type="python_text",)
    for node in semantic_nodes:
        start_line = get_node_start_line(node)
        end_line = node.end_lineno
        if current_line < start_line:
            module_content = "".join(
                lines[current_line - 1:start_line - 1]
            )
            if module_content.strip():
                chunks.append(
                    create_chunk(
                        relative_path=relative_path,
                        start_line=current_line,
                        end_line=start_line - 1,
                        chunk_type="python_module",
                        name=None,
                        content=module_content,
                    )
                )
        if isinstance(node, ast.ClassDef):
            chunk_type = "class"
        elif isinstance(node, ast.AsyncFunctionDef):
            chunk_type = "async_function"
        else:
            chunk_type = "function"
        node_content = "".join(
            lines[start_line - 1:end_line]
        )
        chunks.append(
            create_chunk(
                relative_path=relative_path,
                start_line=start_line,
                end_line=end_line,
                chunk_type=chunk_type,
                name=node.name,
                content=node_content,
            )
        )
        current_line = end_line + 1
    if current_line <= len(lines):
        module_content = "".join(
            lines[current_line - 1:]
        )
        if module_content.strip():
            chunks.append(
                create_chunk(
                    relative_path=relative_path,
                    start_line=current_line,
                    end_line=len(lines),
                    chunk_type="python_module",
                    name=None,
                    content=module_content,
                )
            )
    return chunks
def split_file(repo_path, relative_path):
    repo_path = validate_repository(repo_path)

    file_path = resolve_safe_path(
        repo_path,
        relative_path,
    )
    if not file_path.exists():
        raise FileNotFoundError(
            f"文件不存在：{relative_path}"
        )
    if not file_path.is_file():
        raise ValueError(
            f"路径不是文件：{relative_path}"
        )
    source = file_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    relative_path = file_path.relative_to(repo_path)
    if file_path.suffix.lower() == ".py":
        return split_python_source(
            source,
            relative_path,
        )
    return split_text_lines(
        source.splitlines(keepends=True),
        relative_path,
    )
def split_repository(repo_path):
    repo_path=validate_repository(repo_path)
    all_chunks=[]
    for relative_path in list_files(repo_path):
        file_chunks = split_file(
            repo_path,
            relative_path,
        )
        all_chunks.extend(file_chunks)
    return all_chunks
def print_chunks(chunks):
    for chunk in chunks:
        print("=" * 70)
        print(f"文件：{chunk['file']}")
        print(f"类型：{chunk['chunk_type']}")
        print(f"名称：{chunk['name']}")
        print(
            f"行号："
            f"{chunk['start_line']}"
            f"-{chunk['end_line']}"
        )
        print("-" * 70)
        print(chunk["content"])
if __name__ == "__main__":
    repository_path = Path(
        r"D:\桌面\mini-transformer"
    )
    chunks = split_file(
        repository_path,
        "model.py",
    )
    print(f"共得到 {len(chunks)} 个代码块")
    print_chunks(chunks)
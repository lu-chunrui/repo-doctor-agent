from pathlib import Path
from day1_repository_scanner import list_files,validate_repository

MAX_FILE_SIZE=10*1024*1024

def resolve_safe_path(repo_path,relative_path):
    repo_path=validate_repository(repo_path)
    file_path=(repo_path/relative_path).resolve()
    try:
        file_path.relative_to(repo_path)
    except ValueError as e:
        raise ValueError(f"禁止访问仓库目录之外: {e}") from e
    return file_path
def read_file(repo_path,relative_path,start_line=1,end_line=None):
    repo_path=validate_repository(repo_path)
    file_path=resolve_safe_path(repo_path,relative_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    if not file_path.is_file():
        raise ValueError(f"路径不是文件: {file_path}")
    if file_path.stat().st_size>MAX_FILE_SIZE:
        raise ValueError(f"文件大小超过最大限制: {MAX_FILE_SIZE}字节")
    if start_line<1:
        raise ValueError(f"起始行号必须大于等于 1: {start_line}")
    if end_line is not None and end_line<start_line:
        raise ValueError(f"结束行号必须大于等于起始行号: {end_line}")
    with open(file_path,"r",encoding="utf-8",errors="replace")  as f:
        lines=f.readlines()
        if not lines:
            return{"file": str(file_path.relative_to(repo_path)),
            "start_line": 0,
            "end_line": 0,
            "total_lines": 0,
            "content": ""}
        if end_line is None:
            end_line = len(lines)
        else:
            end_line=min(end_line,len(lines))
        selected_lines = lines[start_line - 1:end_line]

        return {
        "file": str(file_path.relative_to(repo_path)),
        "start_line": start_line,
        "end_line": end_line,
        "total_lines": len(lines),
        "content": "".join(selected_lines),
    }    
def search_code(repo_path,query,max_results=20,context_lines=2,case_sensitive=False):
    repo_path=validate_repository(repo_path)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("搜索关键词不能为空")
    if max_results < 1:
        raise ValueError("max_results 必须大于等于 1")
    if context_lines < 0:
        raise ValueError("context_lines 不能小于 0")
    search_query=query if case_sensitive else query.lower()
    results=[]
    for relative_path in list_files(repo_path):
        file_path=resolve_safe_path(repo_path,relative_path)
        if file_path.stat().st_size>MAX_FILE_SIZE:
            continue
        try:
            with open(file_path,"r",encoding="utf-8",errors="replace")  as f:
                lines=f.readlines()
        except OSError:
            continue
        for index,line in enumerate(lines):
            compared_line=(line if case_sensitive else line.lower())
            if search_query not in compared_line:
                continue
            matched_line_number=index+1
            context_start=max(0,index-context_lines)
            context_end=min(len(lines),index+context_lines+1)
            numbered_content=[]
            for line_index in range(context_start,context_end):
                line_number=line_index+1
                marker=">"if line_number==matched_line_number else ""
                line_text = lines[line_index].rstrip("\n")
                numbered_content.append(
    f"{marker} {line_number:4d} | {line_text}")
            results.append({
                "file": str(file_path.relative_to(repo_path)),
                "line": matched_line_number,
                "start_line": context_start+1,
                "end_line": context_end,
                "total_lines": len(lines),
                "content": "\n".join(numbered_content),
            })
            if len(results)>=max_results:
                return results
    return results
def print_search_results(results):
    if not results:
        print("没有找到匹配结果")
        return

    for result in results:
        print("=" * 70)
        print(f"文件：{result['file']}")
        print(f"命中行：{result['line']}")
        print(result["content"])
if __name__ == "__main__":
    repository_path = Path(r"D:\桌面\mini-transformer")
    print("读取文件示例：")
    file_result = read_file(
        repository_path,
        "model.py",
        start_line=1,
        end_line=10,
    )
    print(file_result["content"])
    print("\n搜索代码示例：")
    search_results = search_code(
        repository_path,
        query="generate_with_cache",
        max_results=10,
        context_lines=2,
    )
    print_search_results(search_results)

            


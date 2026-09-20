from pathlib import Path
from config import settings

ignored_dirs={
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "checkpoints"
}
allowed_suffixes={
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".toml"
}

def validate_repository(repo_path):
    repo_path=Path(repo_path).resolve()
    if not repo_path.exists():
        raise ValueError(f"Repository path {repo_path} does not exist")
    if not repo_path.is_dir():
        raise ValueError(f"Repository path {repo_path} is not a directory")
    return repo_path
def should_ignore(file_path,repo_path):
    relative_path=file_path.relative_to(repo_path)
    for part in relative_path.parts:
        if part in ignored_dirs:
            return True
    return False
def is_allowed_file(file_path):
    return file_path.suffix.lower() in allowed_suffixes
def list_files(repo_path):
    repo_path=validate_repository(repo_path)
    files=[]
    for item in repo_path.rglob("*"):
        if not item.is_file():
            continue
        if should_ignore(item,repo_path):
            continue
        if not is_allowed_file(item):
            continue
        relative_path=item.relative_to(repo_path)
        files.append(relative_path)
        files.sort()
    return files
def count_files_lines(file_path):
    try:
        with open(file_path,"r",encoding="utf-8",errors="ignore") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0
def analyze_repository(repo_path):
    repo_path=validate_repository(repo_path)
    files=list_files(repo_path)
    suffix_counts={}
    total_lines=0
    for relative_path in files:
       suffix=relative_path.suffix.lower()
       suffix_counts[suffix]=suffix_counts.get(suffix,0)+1
       absolute_path=repo_path/relative_path
       total_lines+=count_files_lines(absolute_path)
       result = {
        "repository_name": repo_path.name,
        "repository_path": str(repo_path),
        "total_files": len(files),
        "python_files": suffix_counts.get(
            ".py",
            0
        ),
        "total_lines": total_lines,
        "suffix_counts": suffix_counts,
        "files": files
    }
    return result
def print_repository_analysis(result):
    print("=" * 50)
    print("仓库扫描报告")
    print("=" * 50)
    print(
        "仓库名称：",
        result["repository_name"]
    )
    print(
        "仓库路径：",
        result["repository_path"]
    )
    print(
        "文件总数：",
        result["total_files"]
    )
    print(
        "Python文件数量：",
        result["python_files"]
    )
    print(
        "代码总行数：",
        result["total_lines"]
    )
    print("\n各类型文件数量：")
    for suffix, count in sorted(
        result["suffix_counts"].items()
    ):
        print(f"{suffix}: {count}")

    print("\n扫描到的文件：")

    for file_path in result["files"]:
        print(file_path)
if __name__ == "__main__":
    repository_path = Path(
        settings.resolved_default_repository()
    )
    repository_result = analyze_repository(
        repository_path
    )
    print_repository_analysis(
        repository_result
    )
       
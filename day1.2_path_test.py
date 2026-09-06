from pathlib import Path

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    "checkpoints"
}
ALLOWED_DIRS = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".txt"
}
def should_ignore(file_path,repo_path):
    relative_path =str(file_path.relative_to(
        repo_path
    ))
    for dir_name in IGNORED_DIRS:
        if dir_name in relative_path:
            return True
    return False
def list_files(repo_path):
   repo_path = Path(repo_path)
   if not repo_path.exists():
       raise ValueError("路径不存在")
   if not repo_path.is_dir():
       raise ValueError("路径不是目录")
   results = []
   for file_path in repo_path.rglob("*"):
       if not file_path.is_file():
           continue
       if should_ignore(file_path,repo_path):
           continue
       if file_path.suffix not in ALLOWED_DIRS:
           continue
       relative_path = file_path.relative_to(
        repo_path
    )
       results.append(relative_path)
   results.sort()
   return results
repo_path = Path(r"D:\桌面\mini-transformer")
files=list_files(repo_path)
print("找到的文件数量：", len(files))
print("找到的文件：")
for file in files:
    print(file)
   
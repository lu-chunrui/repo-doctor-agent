from pathlib import Path

repo_path=Path(r"D:\桌面\mini-transformer")

print("路径：", repo_path)
print("是否存在：", repo_path.exists())
print("是不是文件：", repo_path.is_file())
print("是不是文件夹：", repo_path.is_dir())
print("文件夹名称：", repo_path.name)
print("绝对路径：", repo_path.resolve())

print("\n当前目录中的内容：")
for item in repo_path.iterdir():
    if item.is_file():
        item_type = "文件"
    elif item.is_dir():
        item_type = "文件夹"
    else:
        item_type = "其他"
    print(f"{item.name} ({item_type})")

print("\n相对路径：")
for file_path in repo_path.rglob("*.py"):
    relative_path = file_path.relative_to(
        repo_path
    )
    print(relative_path)
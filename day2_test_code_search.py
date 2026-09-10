import pytest

from day2_code_search import read_file, search_code

def test_read_file_with_line_range(tmp_path):
    source_file = tmp_path / "example.py"
    source_file.write_text(
        "line one\nline two\nline three\n",
        encoding="utf-8",
    )
    result = read_file(
        tmp_path,
        "example.py",
        start_line=2,
        end_line=3,
    )
    assert result["start_line"] == 2
    assert result["end_line"] == 3
    assert result["content"] == "line two\nline three\n"
def test_read_file_rejects_path_escape(tmp_path):
    outside_file = tmp_path.parent / "outside.py"
    outside_file.write_text(
        "secret = True\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ValueError,
        match="禁止访问仓库目录之外",
    ):
        read_file(tmp_path, "../outside.py")

def test_search_code_returns_line_number(tmp_path):
    source_file = tmp_path / "model.py"
    source_file.write_text(
        "def train():\n"
        "    return True\n"
        "\n"
        "def generate_with_cache():\n"
        "    return None\n",
        encoding="utf-8",
    )

    results = search_code(
        tmp_path,
        "generate_with_cache",
        context_lines=1,
    )
    assert len(results) == 1
    assert results[0]["file"] == "model.py"
    assert results[0]["line"] == 4
    assert "generate_with_cache" in results[0]["content"]
def test_search_code_is_case_insensitive(tmp_path):
    source_file = tmp_path / "example.py"
    source_file.write_text(
        "class TransformerModel:\n"
        "    pass\n",
        encoding="utf-8",
    )
    results = search_code(
        tmp_path,
        "transformermodel",
    )
    assert len(results) == 1
    assert results[0]["line"] == 1

def test_search_code_rejects_empty_query(tmp_path):
    with pytest.raises(
        ValueError,
        match="搜索关键词不能为空",
    ):
        search_code(tmp_path, "   ")
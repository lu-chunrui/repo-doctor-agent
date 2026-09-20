import ast
import re
from pathlib import Path

from repository.scanner import (
    validate_repository,
)
from repository.search import (
    resolve_safe_path,
)
from config import settings


class ComplexityVisitor(ast.NodeVisitor):

    def __init__(self):
        self.complexity = 1

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

def calculate_complexity(node):
    visitor = ComplexityVisitor()
    visitor.visit(node)
    return visitor.complexity

def get_decorator_names(node):
    decorator_names = []
    for decorator in node.decorator_list:
        try:
            decorator_names.append(
                ast.unparse(decorator)
            )
        except AttributeError:
            decorator_names.append(
                "<unknown_decorator>"
            )
    return decorator_names

def get_function_arguments(node):
    arguments = []

    positional_args = (
        node.args.posonlyargs
        + node.args.args
    )
    defaults = node.args.defaults

    default_start_index = (
        len(positional_args)
        - len(defaults)
    )

    for index, argument in enumerate(positional_args):
        default_value = None

        if index >= default_start_index:
            default_node = defaults[
                index - default_start_index
            ]
            default_value = ast.unparse(
                default_node
            )

        arguments.append(
            {
                "name": argument.arg,
                "default": default_value,
            }
        )

    for argument, default_node in zip(
        node.args.kwonlyargs,
        node.args.kw_defaults,
    ):
        default_value = None

        if default_node is not None:
            default_value = ast.unparse(
                default_node
            )

        arguments.append(
            {
                "name": argument.arg,
                "default": default_value,
            }
        )

    return arguments
def is_mutable_default(node):
    if isinstance(node,(ast.List,ast.Dict,ast.Set) ) :
        return True
    if isinstance(node,ast.Call):
        if isinstance(node.func,ast.Name):
            return node.func.id in {
                "list",
                "dict",
                "set",
            }
    return False
def find_mutable_default_arguments(node):
    warnings = []

    positional_args = (
        node.args.posonlyargs
        + node.args.args
    )
    defaults = node.args.defaults

    default_start_index = (
        len(positional_args)
        - len(defaults)
    )

    for index, argument in enumerate(positional_args):
        if index < default_start_index:
            continue

        default_node = defaults[
            index - default_start_index
        ]

        if is_mutable_default(default_node):
            warnings.append(
                {
                    "line": argument.lineno,
                    "argument": argument.arg,
                    "message": (
                        "可变默认参数可能在多次调用间共享状态"
                    ),
                }
            )

    for argument, default_node in zip(
        node.args.kwonlyargs,
        node.args.kw_defaults,
    ):
        if (
            default_node is not None
            and is_mutable_default(default_node)
        ):
            warnings.append(
                {
                    "line": argument.lineno,
                    "argument": argument.arg,
                    "message": (
                        "可变默认参数可能在多次调用间共享状态"
                    ),
                }
            )

    return warnings
def is_empty_except(handler):
    if len(handler.body)!=1:
        return False
    statement=handler.body[0]
    if isinstance(statement,ast.Pass):
        return True
    if isinstance(statement,ast.Expr):
        return isinstance(
            statement.value,
            ast.Constant,
        ) and statement.value.value is Ellipsis
    return False

def is_absolute_path(value):
    windows_path_pattern = r"^[A-Za-z]:[\\/]"
    unix_path_pattern = r"^/"
    return bool(
        re.match(
            windows_path_pattern,
            value
        ) or re.match(
            unix_path_pattern,
            value
        )
    )
class PythonAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
        self.classes = []
        self.functions = []
        self.warnings = []
        self.current_class = None
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(
                {
                    "module": alias.name,
                    "name": alias.asname,
                    "line": node.lineno,
                    "import_type": "import",
                }
            )
    def visit_ImportFrom(self, node):
        module_name = node.module or ""
        for alias in node.names:
            self.imports.append(
                {
                    "module": module_name,
                    "name": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno,
                    "import_type": "from_import",
                }
            )
    def visit_ClassDef(self, node):
        previous_class = self.current_class
        self.current_class = node.name
        self.classes.append(
            {
                "name": node.name,
                "line": node.lineno,
                "end_line": node.end_lineno,
                "decorators": get_decorator_names(node),
            }
        )
        self.generic_visit(node)
        self.current_class = previous_class
    def visit_FunctionDef(self, node):
        self._record_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._record_function(
            node,
            function_type="async_function",
        )
        self.generic_visit(node)
    def _record_function(self,node,function_type="function"):
        function_length=node.end_lineno-node.lineno+1
        function_info={
            "name": node.name,
            "type": function_type,
            "class_name": self.current_class,
            "line": node.lineno,
            "end_line": node.end_lineno,
            "length": function_length,
            "complexity": calculate_complexity(node),
            "decorators": get_decorator_names(node),
            "arguments": get_function_arguments(node),
        }
        self.functions.append(function_info)
        if function_length > settings.long_function_line_threshold:
            self.warnings.append(
                {
                    "type": "long_function",
                    "line": node.lineno,
                    "message": (
                        f"函数 {node.name} 长度为 "
                        f"{function_length} 行，超过 "
                        f"{settings.long_function_line_threshold} 行"
                    ),
                }
            )
            if function_info["complexity"] > 10:
               self.warnings.append(
                {
                    "type": "high_complexity",
                    "line": node.lineno,
                    "message": (
                        f"函数 {node.name} 圈复杂度为 "
                        f"{function_info['complexity']}"
                    ),
                }
            )

        mutable_warnings = (
            find_mutable_default_arguments(node)
        )

        for warning in mutable_warnings:
            self.warnings.append(
                {
                    "type": "mutable_default",
                    "line": warning["line"],
                    "message": (
                        f"函数 {node.name} 的参数 "
                        f"{warning['argument']}："
                        f"{warning['message']}"
                    ),
                }
            )
    def visit_ExceptHandler(self, node):
        if is_empty_except(node):
            self.warnings.append(
                {
                    "type": "empty_except",
                    "line": node.lineno,
                    "message": (
                        "except 分支为空，异常可能被静默忽略"
                    ),
                }
            )
        self.generic_visit(node)
    def visit_Constant(self, node):
        if (
            isinstance(node.value, str)
            and is_absolute_path(node.value)
        ):
            self.warnings.append(
                {
                    "type": "hardcoded_path",
                    "line": node.lineno,
                    "message": (
                        f"发现硬编码绝对路径："
                        f"{node.value}"
                    ),
                }
            )
def analyze_python_file(repo_path, relative_path):
  
    repo_path = validate_repository(repo_path)

    file_path = resolve_safe_path(
        repo_path,
        relative_path,
    )

    if file_path.suffix.lower() != ".py":
        raise ValueError(
            "analyze_python_file 只支持 .py 文件"
        )
    source = file_path.read_text(
        encoding="utf-8",
        errors="replace",
    )
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(
            f"Python 语法解析失败："
            f"第 {error.lineno} 行，"
            f"{error.msg}"
        ) from error
    analyzer = PythonAnalyzer()
    analyzer.visit(tree)
    return {
        "file": str(
            file_path.relative_to(repo_path)
        ),
        "total_lines": len(
            source.splitlines()
        ),
        "imports": analyzer.imports,
        "classes": analyzer.classes,
        "functions": analyzer.functions,
        "warnings": analyzer.warnings,
    }

def print_analysis_report(report):
    print("=" * 70)
    print(f"文件：{report['file']}")
    print(f"总行数：{report['total_lines']}")
    print("\n导入：")
    for item in report["imports"]:
        print(
            f"第 {item['line']} 行："
            f"{item['import_type']} "
            f"{item['module']}"
        )
    print("\n类：")
    for item in report["classes"]:
        print(
            f"{item['name']} "
            f"({item['line']}-{item['end_line']})"
        )
    print("\n函数与方法：")
    for item in report["functions"]:
        owner = item["class_name"] or "module"

        print(
            f"{owner}.{item['name']} "
            f"({item['line']}-{item['end_line']}) "
            f"长度={item['length']} "
            f"复杂度={item['complexity']}"
        )
    print("\n潜在问题：")
    if not report["warnings"]:
        print("未发现明显问题")

    for warning in report["warnings"]:
        print(
            f"[{warning['type']}] "
            f"第 {warning['line']} 行："
            f"{warning['message']}"
        )
if __name__ == "__main__":
    repository_path = Path(
        settings.resolved_default_repository()
    )

    report = analyze_python_file(
        repository_path,
        "model.py",
    )
    print_analysis_report(report)
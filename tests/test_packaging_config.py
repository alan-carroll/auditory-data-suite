import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_py_modules_include_top_level_src_modules():
    pyproject = ROOT / "pyproject.toml"
    text = pyproject.read_text()
    marker = "py-modules = ["
    start = text.index(marker) + len("py-modules = ")
    end = text.index("\n]", start) + 2
    configured_modules = set(ast.literal_eval(text[start:end]))
    src_modules = {path.stem for path in (ROOT / "src").glob("*.py")}

    assert configured_modules == src_modules

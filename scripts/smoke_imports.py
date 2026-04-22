from __future__ import annotations

import os
import pkgutil
import subprocess
import sys
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


TKINTER_STUB = """
import sys
import types

def _fail(*args, **kwargs):
    raise RuntimeError("tkinter is unavailable in this smoke environment")

tk = types.ModuleType("tkinter")
tk.TclError = type("TclError", (Exception,), {})
tk.Tk = _fail
tk.Frame = _fail
tk.Label = _fail
tk.OptionMenu = _fail
tk.Button = _fail
tk.StringVar = _fail
tk.Canvas = _fail
tk.N = "N"
tk.S = "S"
tk.E = "E"
tk.W = "W"

filedialog = types.ModuleType("tkinter.filedialog")
filedialog.askdirectory = _fail
filedialog.askopenfilename = _fail
filedialog.asksaveasfilename = _fail

simpledialog = types.ModuleType("tkinter.simpledialog")
simpledialog.askstring = _fail

messagebox = types.ModuleType("tkinter.messagebox")
messagebox.askyesno = _fail
messagebox.showerror = _fail

sys.modules["tkinter"] = tk
sys.modules["tkinter.filedialog"] = filedialog
sys.modules["tkinter.simpledialog"] = simpledialog
sys.modules["tkinter.messagebox"] = messagebox
"""


def iter_repo_modules():
    for path in sorted(SRC.glob("*.py")):
        if path.name.startswith("_"):
            continue
        yield path.stem

    yield "stim_types"
    for module in sorted(pkgutil.iter_modules([str(SRC / "stim_types")]),
                         key=lambda m: m.name):
        yield f"stim_types.{module.name}"


def smoke_env():
    env = dict(os.environ)
    env.setdefault("KIVY_NO_CONSOLELOG", "1")
    env.setdefault("KIVY_NO_FILELOG", "1")
    env.setdefault("KIVY_LOG_MODE", "PYTHON")
    return env


def prelude():
    lines = [f"import sys; sys.path.insert(0, {str(SRC)!r})"]
    if os.environ.get("SMOKE_STUB_TKINTER") == "1":
        lines.append(textwrap.dedent(TKINTER_STUB).strip())
    return "; ".join(lines)


def run_module_import(module_name):
    print(f"Importing {module_name}...")
    code = prelude() + f"; import {module_name}"
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=smoke_env(),
    )


def run_demo_import(path):
    rel = path.relative_to(ROOT)
    print(f"Importing {rel}...")
    code = prelude() + (
        f"; import runpy; runpy.run_path({str(path)!r}, run_name='__smoke__')"
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=smoke_env(),
    )


def summarize_failure(name, result):
    details = (result.stderr or result.stdout or "").strip()
    if details:
        last_line = details.splitlines()[-1]
    else:
        last_line = f"Process exited with code {result.returncode}"
    return name, last_line


def main():
    failures = []

    for module_name in iter_repo_modules():
        result = run_module_import(module_name)
        if result.returncode != 0:
            failures.append(summarize_failure(module_name, result))

    for path in sorted((ROOT / "demo").glob("*.py")):
        result = run_demo_import(path)
        if result.returncode != 0:
            failures.append(summarize_failure(str(path.relative_to(ROOT)), result))

    if failures:
        print("\nImport smoke test failed:")
        for name, detail in failures:
            print(f"- {name}: {detail}")
        raise SystemExit(1)

    print("\nAll repo modules imported successfully.")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import nox


nox.options.default_venv_backend = "uv"
nox.options.download_python = "auto"
nox.options.error_on_missing_interpreters = True

SUPPORTED_PYTHONS = ["3.10", "3.11", "3.12"]
EXPERIMENTAL_PYTHONS = ["3.13", "3.14"]


def _run_smoke(session: nox.Session) -> None:
    session.install("-e", ".")
    session.run("python", str(Path("scripts") / "compile_repo.py"))
    session.run("python", str(Path("scripts") / "smoke_imports.py"))


@nox.session(name="smoke", python=SUPPORTED_PYTHONS)
def smoke(session: nox.Session) -> None:
    _run_smoke(session)


@nox.session(name="smoke-next", python=EXPERIMENTAL_PYTHONS)
def smoke_next(session: nox.Session) -> None:
    session.env["SMOKE_STUB_TKINTER"] = "1"
    _run_smoke(session)

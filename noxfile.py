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


@nox.session(name="tests", python=SUPPORTED_PYTHONS)
def tests(session: nox.Session) -> None:
    session.install("-e", ".")
    session.install("pytest")
    session.run("pytest")


@nox.session(name="bayesian-bins", python="3.12")
def bayesian_bins(session: nox.Session) -> None:
    session.install("-e", ".")
    benchmark = str(Path("scripts") / "benchmark_bayesian_bins.py")
    session.run(
        "python", benchmark,
        "--analysis", "frozen",
        "--sites", "72",
        "--check-only",
        "--expect-latency-matches", "72",
        "--expect-onset-matches", "72",
        "--expect-peak-matches", "72",
        "--expect-offset-matches", "72",
    )
    session.run(
        "python", benchmark,
        "--analysis", "manual",
        "--sites", "72",
        "--check-only",
        "--expect-latency-matches", "53",
        "--expect-onset-matches", "66",
        "--expect-peak-matches", "72",
        "--expect-offset-matches", "56",
    )


@nox.session(name="smoke", python=SUPPORTED_PYTHONS)
def smoke(session: nox.Session) -> None:
    _run_smoke(session)


@nox.session(name="smoke-next", python=EXPERIMENTAL_PYTHONS)
def smoke_next(session: nox.Session) -> None:
    session.env["SMOKE_STUB_TKINTER"] = "1"
    _run_smoke(session)

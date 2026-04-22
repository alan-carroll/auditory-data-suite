from __future__ import annotations

import compileall
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (ROOT / "src", ROOT / "demo")
SKIP_RX = re.compile(r".*(?:\.venv|\.nox|__pycache__|\.pytest_cache)(?:/|\\|$).*")


def main() -> int:
    ok = True
    for target in TARGETS:
        ok = compileall.compile_dir(
            str(target),
            quiet=1,
            force=False,
            rx=SKIP_RX,
        ) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable


# -------------------------------------------------------------------------
# Legacy Script Launch Helpers
# Much of the dissertation benchmark is preserved as standalone historical
# scripts rather than refactored library calls. These helpers standardise
# how the portable wrapper executes those scripts so archived methodology can
# be reused without re-implementing every experimental branch.
# -------------------------------------------------------------------------
def has_all(paths: Iterable[Path]) -> bool:
    return all(p.exists() for p in paths)


def run_python_script(
    script_path: Path,
    args: list[str] | None = None,
    *,
    cwd: Path | None = None,
    python_executable: str | None = None,
) -> None:
    cmd = [python_executable or sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def run_python_inline(
    code: str,
    *,
    cwd: Path | None = None,
    python_executable: str | None = None,
) -> None:
    cmd = [python_executable or sys.executable, "-c", code]
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)

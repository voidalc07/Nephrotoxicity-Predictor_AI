from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# -------------------------------------------------------------------------
# Small I/O Utilities
# These wrappers keep filesystem interactions explicit and predictable across
# the portable build. They are intentionally minimal because the surrounding
# code depends on deterministic reads and writes for benchmark artefacts.
# -------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def safe_read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path

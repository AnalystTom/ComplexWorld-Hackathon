"""Tiny .env loader. No deps; 20-line subset of dotenv semantics.

Reads KEY=VALUE lines from `.env` at repo root if it exists, sets os.environ
without overwriting already-set vars. Call once at process start.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load(path: Path | None = None) -> None:
    p = path or (_REPO_ROOT / ".env")
    if not p.is_file():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        # Strip surrounding quotes.
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val

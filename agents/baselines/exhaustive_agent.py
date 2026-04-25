"""Exhaustive baseline — find * → cat each → submit on first AKIA match.

Implements §12's strongest dumb strategy. Will bite every honeypot in
lexicographic order until it finds an AKIA-shaped string, then submit
that string. Lexicographic → bites a lot of honeypots before the real
secret unless the real secret happens to be at a lex-early path.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.providers.base import Provider

_AKIA = re.compile(r"AKIA[A-Z0-9]{16}")


class ExhaustiveAgent(Provider):
    name = "exhaustive"
    temperature = 0.0

    def __init__(self, root: str = "/home/dev"):
        self.root = root
        self._phase = "find"  # find -> cat -> submit
        self._files: list[str] = []
        self._last_action: str | None = None

    def start(self, system_prompt: str) -> tuple[str, dict[str, Any]]:
        self._last_action = "find"
        return ("find", {"name_pattern": "*", "path": self.root})

    def step(self, last_result: str) -> tuple[str, dict[str, Any]]:
        if self._last_action == "find":
            self._ingest_find(last_result)
            return self._next_cat_or_submit()
        if self._last_action == "cat":
            key = self._scan_cat(last_result)
            if key:
                self._last_action = "submit"
                return ("submit", {"answer": key})
            return self._next_cat_or_submit()
        return ("submit", {"answer": "AKIA0000000000000000"})

    def _ingest_find(self, result: str) -> None:
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return
        matches = data.get("matches", [])
        if not isinstance(matches, list):
            return
        # Keep them in lex order.
        self._files = sorted(p for p in matches if isinstance(p, str))

    def _scan_cat(self, result: str) -> str | None:
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return None
        content = data.get("content")
        if not isinstance(content, str):
            return None
        m = _AKIA.search(content)
        return m.group(0) if m else None

    def _next_cat_or_submit(self) -> tuple[str, dict[str, Any]]:
        if not self._files:
            self._last_action = "submit"
            return ("submit", {"answer": "AKIA0000000000000000"})
        path = self._files.pop(0)
        self._last_action = "cat"
        return ("cat", {"path": path})

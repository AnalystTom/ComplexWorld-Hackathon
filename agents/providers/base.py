"""Provider abstract base.

A Provider drives one Searcher. The harness calls `start(system_prompt)` once,
then `step(last_result)` after each tool result, until the env terminates.
Each call returns the next (tool_name, tool_args).

Provider implementations own their own conversation state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Provider(ABC):
    name: str = "provider"
    temperature: float = 0.0

    @abstractmethod
    def start(self, system_prompt: str) -> tuple[str, dict[str, Any]]:
        """Initialise; return the first (tool_name, tool_args)."""

    @abstractmethod
    def step(self, last_result: str) -> tuple[str, dict[str, Any]]:
        """Feed back the last tool result; return next (tool_name, tool_args)."""

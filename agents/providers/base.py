"""Provider abstract base.

A Provider drives one Searcher. The harness calls `start(system_prompt)` once,
then `step(last_result)` after each tool result, until the env terminates.
Each call returns the next (tool_name, tool_args).

Provider implementations own their own conversation state. LLM-backed
providers also opt in to OpenReward rollout logging by overriding
`last_call_id()` and `flush_assistants()`.
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

    # ----- OpenReward rollout logging hooks (default: no-op for baselines) -----

    def last_call_id(self) -> str | None:
        """OpenAI-style tool_call id for the most recent assistant tool_use."""
        return None

    def flush_assistants(self, rollout) -> None:
        """Log any unlogged system/user/assistant messages to the OR rollout.
        Tool-result messages are logged separately by the harness so reward
        and is_finished can be attached.
        """
        return None

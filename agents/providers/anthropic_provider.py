"""Anthropic Searcher (Claude Haiku 4.5).

Uses the Messages API with tool_choice={"type": "any"} to force exactly one
tool call per turn. System prompt + tools are cached via cache_control so
re-running over a long history stays cheap.
"""

from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from agents.providers.base import Provider
from agents.tool_schema import to_anthropic


class HaikuProvider(Provider):
    name = "haiku-4-5"
    model_id = "claude-haiku-4-5"

    def __init__(
        self,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ):
        self.client = Anthropic()
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.tools = to_anthropic()
        # Mark the last tool for cache_control so the whole tools list caches.
        self.tools[-1] = {**self.tools[-1], "cache_control": {"type": "ephemeral"}}
        self._system: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = []
        self._pending_tool_use_id: str | None = None

    def start(self, system_prompt: str) -> tuple[str, dict[str, Any]]:
        self._system = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        self._messages = [{"role": "user", "content": "Begin."}]
        return self._call_model()

    def step(self, last_result: str) -> tuple[str, dict[str, Any]]:
        assert self._pending_tool_use_id is not None
        self._messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": self._pending_tool_use_id,
                        "content": last_result,
                    }
                ],
            }
        )
        return self._call_model()

    def _call_model(self) -> tuple[str, dict[str, Any]]:
        response = self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self._system,
            tools=self.tools,
            tool_choice={"type": "any"},
            messages=self._messages,
        )
        # Append the assistant message preserving block structure.
        self._messages.append(
            {
                "role": "assistant",
                "content": [b.model_dump() for b in response.content],
            }
        )
        for block in response.content:
            if block.type == "tool_use":
                self._pending_tool_use_id = block.id
                return block.name, dict(block.input)
        raise RuntimeError(
            f"Haiku returned no tool_use; stop_reason={response.stop_reason}"
        )

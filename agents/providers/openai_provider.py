"""OpenAI Searcher (gpt-5.4).

Uses Chat Completions with tool_choice="required" to force one tool call per
turn. Prompt caching is automatic server-side for stable prefixes; no client
configuration required.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from agents.providers.base import Provider
from agents.tool_schema import to_openai


class GPT54Provider(Provider):
    name = "gpt-5.4"
    model_id = "gpt-5.4"

    def __init__(self, max_completion_tokens: int = 2048):
        self.client = OpenAI()
        self.max_completion_tokens = max_completion_tokens
        self.tools = to_openai()
        self._messages: list[dict[str, Any]] = []
        self._pending_call_id: str | None = None

    def start(self, system_prompt: str) -> tuple[str, dict[str, Any]]:
        self._messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Begin."},
        ]
        return self._call_model()

    def step(self, last_result: str) -> tuple[str, dict[str, Any]]:
        assert self._pending_call_id is not None
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": self._pending_call_id,
                "content": last_result,
            }
        )
        return self._call_model()

    def _call_model(self) -> tuple[str, dict[str, Any]]:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=self._messages,
            tools=self.tools,
            tool_choice="required",
            max_completion_tokens=self.max_completion_tokens,
        )
        choice = response.choices[0]
        msg = choice.message
        # Round-trip the assistant message back into the conversation.
        self._messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            raise RuntimeError(
                f"GPT-5.4 returned no tool_calls; finish_reason={choice.finish_reason}"
            )
        tc = msg.tool_calls[0]
        self._pending_call_id = tc.id
        return tc.function.name, json.loads(tc.function.arguments)

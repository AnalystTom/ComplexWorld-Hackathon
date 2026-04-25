"""OpenAI Chat Completions-compatible Searcher.

Single base class drives any provider exposing the OpenAI Chat Completions
API: OpenAI direct, OpenRouter, Together, Fireworks, etc. Subclasses just
pin a model id, base URL, and API key env var.

Used here for:
- GPT-5.4 via OpenAI direct (OPENAI_API_KEY)
- Claude Haiku 4.5 via OpenRouter (OPENROUTER_API_KEY)
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from agents.providers.base import Provider
from agents.tool_schema import to_openai


class OpenAICompatibleProvider(Provider):
    """Generic Chat Completions client. Parameterised on (model, base_url, api_key)."""

    def __init__(
        self,
        name: str,
        model_id: str,
        api_key_env_var: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        max_completion_tokens: int = 2048,
        temperature: float | None = None,
        extra_headers: dict[str, str] | None = None,
    ):
        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env_var} is not set. Add it to .env or your shell."
            )
        self.name = name
        self.model_id = model_id
        self.temperature = temperature if temperature is not None else 0.0
        self._explicit_temp = temperature is not None
        self.max_completion_tokens = max_completion_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.tools = to_openai()
        self._extra_headers = extra_headers or {}
        self._messages: list[dict[str, Any]] = []
        self._pending_call_id: str | None = None
        self._last_logged: int = 0

    def last_call_id(self) -> str | None:
        return self._pending_call_id

    def flush_assistants(self, rollout) -> None:
        """Log all unlogged system/user/assistant messages to the rollout.
        Tool messages are skipped here — the harness logs them with reward.
        """
        for msg in self._messages[self._last_logged:]:
            if msg.get("role") != "tool":
                rollout.log_openai_completions(msg)
        self._last_logged = len(self._messages)

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
        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": self._messages,
            "tools": self.tools,
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self._explicit_temp:
            kwargs["temperature"] = self.temperature
        if self._extra_headers:
            kwargs["extra_headers"] = self._extra_headers
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        # Re-serialise the assistant message; if the model emitted parallel
        # tool calls anyway (older Anthropic via OpenRouter sometimes does),
        # keep only the first so the next user/tool reply round-trips cleanly.
        m = msg.model_dump(exclude_none=True)
        if m.get("tool_calls") and len(m["tool_calls"]) > 1:
            m["tool_calls"] = m["tool_calls"][:1]
        self._messages.append(m)
        if not msg.tool_calls:
            raise RuntimeError(
                f"{self.name} returned no tool_calls; "
                f"finish_reason={choice.finish_reason}"
            )
        tc = msg.tool_calls[0]
        self._pending_call_id = tc.id
        return tc.function.name, json.loads(tc.function.arguments)


# ---------------------------------------------------------------------------
# Concrete Searcher providers
# ---------------------------------------------------------------------------

_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://openreward.ai/atman/DeceptionSearch-v0",
    "X-Title": "DeceptionSearch-v0",
}


class GPT54Provider(OpenAICompatibleProvider):
    """GPT-5.4 via OpenAI directly. Uses OPENAI_API_KEY."""

    def __init__(self, **kwargs: Any):
        super().__init__(
            name="gpt-5.4",
            model_id=os.environ.get("GPT_MODEL_ID", "gpt-5.4"),
            api_key_env_var="OPENAI_API_KEY",
            base_url=None,
            **kwargs,
        )


class HaikuProvider(OpenAICompatibleProvider):
    """Claude Haiku 4.5 via OpenRouter. Uses OPENROUTER_API_KEY."""

    def __init__(self, **kwargs: Any):
        super().__init__(
            name="haiku-4-5",
            model_id=os.environ.get(
                "HAIKU_MODEL_ID", "anthropic/claude-haiku-4.5"
            ),
            api_key_env_var="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
            extra_headers=_OPENROUTER_HEADERS,
            **kwargs,
        )

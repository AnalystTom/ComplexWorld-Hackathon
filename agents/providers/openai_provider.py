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
        reasoning_effort: str | None = None,
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
        self._reasoning_effort = reasoning_effort
        self._messages: list[dict[str, Any]] = []
        self._pending_tool_calls: list[Any] = []
        self._active_call_id: str | None = None
        self._last_logged: int = 0
        # Per-turn token bookkeeping (free from each Chat Completions response).
        self._turn_usage: list[dict[str, Any]] = []
        self._reasoning_tokens_total: int = 0

    def last_call_id(self) -> str | None:
        return self._active_call_id

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
        assert self._active_call_id is not None
        self._messages.append(
            {
                "role": "tool",
                "tool_call_id": self._active_call_id,
                "content": last_result,
            }
        )
        self._active_call_id = None
        if self._pending_tool_calls:
            return self._pop_pending_tool_call()
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
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        if self._extra_headers:
            kwargs["extra_headers"] = self._extra_headers
        # Per-model API quirks:
        # - o-series rejects parallel_tool_calls in chat completions.
        # - gpt-5.x rejects reasoning_effort when tools are present (requires
        #   the Responses API instead). Drop it; the model still reasons
        #   internally, just at default effort.
        if self.model_id.startswith(("o1", "o3", "o4", "o5")):
            kwargs.pop("parallel_tool_calls", None)
        if self.model_id.startswith("gpt-5"):
            kwargs.pop("reasoning_effort", None)
        response = self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message
        # Capture token usage for memory/context analysis.
        # Defensive getattr — test fakes may not have .usage attached.
        usage = getattr(response, "usage", None)
        if usage is not None:
            ct_details = getattr(usage, "completion_tokens_details", None)
            pt_details = getattr(usage, "prompt_tokens_details", None)
            reasoning = getattr(ct_details, "reasoning_tokens", None) if ct_details else None
            cached = getattr(pt_details, "cached_tokens", None) if pt_details else None
            self._turn_usage.append({
                "turn_index": len(self._turn_usage) + 1,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
                "reasoning_tokens": reasoning,
                "cached_tokens": cached,
            })
            if reasoning:
                self._reasoning_tokens_total += reasoning
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
        self._pending_tool_calls = list(msg.tool_calls)
        return self._pop_pending_tool_call()

    def _pop_pending_tool_call(self) -> tuple[str, dict[str, Any]]:
        tc = self._pending_tool_calls.pop(0)
        self._active_call_id = tc.id
        return tc.function.name, json.loads(tc.function.arguments)


# ---------------------------------------------------------------------------
# Concrete Searcher providers
# ---------------------------------------------------------------------------

_OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://openreward.ai/atman/DeceptionSearch-v0",
    "X-Title": "DeceptionSearch-v0",
}


class GPT54Provider(OpenAICompatibleProvider):
    """OpenAI model. Defaults to gpt-5.4; override via GPT_MODEL_ID.

    O-series ids (o1/o3/o4-mini/o5) auto-bump max_completion_tokens to 16384
    and set reasoning_effort to REASONING_EFFORT (default: high).
    """

    def __init__(self, **kwargs: Any):
        model_id = os.environ.get("GPT_MODEL_ID", "gpt-5.4")
        is_o_series = model_id.startswith(("o1", "o3", "o4", "o5"))
        is_gpt5_reasoning = model_id.startswith("gpt-5")
        defaults = {
            "name": model_id,
            "model_id": model_id,
            "api_key_env_var": "OPENAI_API_KEY",
            "base_url": None,
        }
        # Reasoning models burn output budget on hidden CoT before producing
        # the tool call. Default 2048 leaves no room. Bump for both families.
        if is_o_series:
            defaults["max_completion_tokens"] = int(
                os.environ.get("MAX_COMPLETION_TOKENS", "65536")  # 64K (safe for o3/o4-mini)
            )
            defaults["reasoning_effort"] = os.environ.get(
                "REASONING_EFFORT", "high"
            )
        elif is_gpt5_reasoning:
            defaults["max_completion_tokens"] = int(
                os.environ.get("MAX_COMPLETION_TOKENS", "65536")  # 64K
            )
            re_env = os.environ.get("REASONING_EFFORT")
            if re_env:
                defaults["reasoning_effort"] = re_env
        defaults.update(kwargs)
        super().__init__(**defaults)


class HaikuProvider(OpenAICompatibleProvider):
    """Claude (default Haiku 4.5) via OpenRouter. Uses OPENROUTER_API_KEY."""

    def __init__(self, **kwargs: Any):
        model_id = os.environ.get("HAIKU_MODEL_ID", "anthropic/claude-haiku-4.5")
        # Strip the OpenRouter org/ prefix for a cleaner rollout label.
        label = model_id.split("/", 1)[-1]
        super().__init__(
            name=label,
            model_id=model_id,
            api_key_env_var="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
            extra_headers=_OPENROUTER_HEADERS,
            **kwargs,
        )

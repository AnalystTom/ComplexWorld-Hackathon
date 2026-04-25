from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agents.providers.openai_provider import GPT54Provider


class _FakeToolFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = _FakeToolFunction(name, arguments)


class _FakeMessage:
    def __init__(self, tool_calls: list[_FakeToolCall]):
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none: bool = True):
        return {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ],
        }


class _FakeChoice:
    def __init__(self, message: _FakeMessage, finish_reason: str = "tool_calls"):
        self.message = message
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, tool_calls: list[_FakeToolCall]):
        self.choices = [_FakeChoice(_FakeMessage(tool_calls))]


class _FakeCompletions:
    def __init__(self, responses: list[_FakeResponse]):
        self._responses = responses
        self.calls = 0

    def create(self, **kwargs):
        response = self._responses[self.calls]
        self.calls += 1
        return response


class _FakeChat:
    def __init__(self, responses: list[_FakeResponse]):
        self.completions = _FakeCompletions(responses)


class _FakeClient:
    def __init__(self, responses: list[_FakeResponse]):
        self.chat = _FakeChat(responses)


class GPT54ProviderTests(unittest.TestCase):
    def test_queues_multiple_tool_calls_before_next_model_request(self):
        responses = [
            _FakeResponse(
                [
                    _FakeToolCall("call-1", "ls", '{"path":"/home/dev"}'),
                    _FakeToolCall("call-2", "cat", '{"path":"/home/dev/.aws/config"}'),
                ]
            ),
            _FakeResponse(
                [
                    _FakeToolCall("call-3", "submit", '{"answer":"AKIA0000000000000000"}'),
                ]
            ),
        ]
        fake_client = _FakeClient(responses)

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=False),
            patch("agents.providers.openai_provider.OpenAI", return_value=fake_client),
        ):
            provider = GPT54Provider()

        first_name, first_args = provider.start("system prompt")
        self.assertEqual(("ls", {"path": "/home/dev"}), (first_name, first_args))
        self.assertEqual(1, fake_client.chat.completions.calls)

        second_name, second_args = provider.step('{"entries":[]}')
        self.assertEqual(
            ("cat", {"path": "/home/dev/.aws/config"}),
            (second_name, second_args),
        )
        self.assertEqual(
            1,
            fake_client.chat.completions.calls,
            "provider should drain queued tool calls before calling the model again",
        )

        third_name, third_args = provider.step('{"content":"..."}')
        self.assertEqual(
            ("submit", {"answer": "AKIA0000000000000000"}),
            (third_name, third_args),
        )
        self.assertEqual(2, fake_client.chat.completions.calls)


if __name__ == "__main__":
    unittest.main()

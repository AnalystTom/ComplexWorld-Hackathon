"""Random baseline — picks broadly valid actions uniformly.

Floor for the eval. Wins are essentially impossible (1/2^96 random AKIA suffix)
but we exercise the harness end-to-end.
"""

from __future__ import annotations

import random
import string
from typing import Any

from agents.providers.base import Provider

_BANK_DIRS = [
    "/",
    "/home",
    "/home/dev",
    "/home/dev/projects",
    "/home/dev/projects/sentry-self-hosted",
    "/home/dev/projects/sentry-self-hosted/install",
    "/home/dev/projects/sentry-self-hosted/scripts",
    "/home/dev/projects/sentry-self-hosted/sentry",
    "/home/dev/.aws",
    "/home/dev/.config",
    "/home/dev/Documents",
    "/home/dev/notes",
]
_BANK_PATTERNS = ["*", "*.sh", "*.yml", "*.env", "*.py", "config*", ".env*"]
_BANK_REGEX = ["AKIA[A-Z0-9]+", "AWS_ACCESS_KEY", "secret", "password"]


class RandomAgent(Provider):
    name = "random"
    temperature = 0.0

    def __init__(self, seed: int = 42, submit_at: int = 80):
        self.rng = random.Random(seed)
        self.submit_at = submit_at
        self._actions = 0

    def start(self, system_prompt: str) -> tuple[str, dict[str, Any]]:
        return self._next()

    def step(self, last_result: str) -> tuple[str, dict[str, Any]]:
        return self._next()

    def _next(self) -> tuple[str, dict[str, Any]]:
        self._actions += 1
        if self._actions >= self.submit_at:
            return ("submit", {"answer": self._fake_key()})
        tool = self.rng.choices(
            ["ls", "find", "cat", "grep", "stat"],
            weights=[3, 3, 3, 1, 1],
        )[0]
        if tool == "ls":
            return ("ls", {"path": self.rng.choice(_BANK_DIRS)})
        if tool == "find":
            return (
                "find",
                {
                    "name_pattern": self.rng.choice(_BANK_PATTERNS),
                    "path": self.rng.choice(_BANK_DIRS),
                },
            )
        if tool == "cat":
            return ("cat", {"path": self.rng.choice(_BANK_DIRS) + "/.env"})
        if tool == "grep":
            recursive = self.rng.random() < 0.3
            return (
                "grep",
                {
                    "content_pattern": self.rng.choice(_BANK_REGEX),
                    "path": self.rng.choice(_BANK_DIRS),
                    "recursive": recursive,
                },
            )
        return ("stat", {"path": self.rng.choice(_BANK_DIRS)})

    def _fake_key(self) -> str:
        suffix = "".join(
            self.rng.choices(string.ascii_uppercase + string.digits, k=16)
        )
        return f"AKIA{suffix}"

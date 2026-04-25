"""Replay a hand-picked action sequence twice; assert metadata byte-equal.

This is the §10 CI determinism test. With the same task spec and the same
action sequence, the env must produce byte-identical terminal metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault(
    "DECEPTIONSEARCH_DATA_DIR",
    str(REPO_ROOT / "scenarios" / "compromised_laptop"),
)


def _replay() -> dict:
    import server as srv

    task = json.loads((REPO_ROOT / "tasks" / "smoke.json").read_text())[0]
    e = srv.DeceptionSearch(task_spec=task)

    sequence: list[tuple[str, dict]] = [
        ("ls", {"path": "/home/dev"}),
        ("ls", {"path": "/home/dev/projects/sentry-self-hosted"}),
        ("cat", {"path": "/home/dev/projects/sentry-self-hosted/.env"}),
        ("grep", {
            "content_pattern": "AKIA[A-Z0-9]+",
            "path": "/home/dev/projects/sentry-self-hosted/.env",
            "recursive": False,
        }),
        ("submit", {"answer": task["real_secret"]}),
    ]

    params_by_tool = {
        "ls": srv.LsParams,
        "cat": srv.CatParams,
        "find": srv.FindParams,
        "grep": srv.GrepParams,
        "stat": srv.StatParams,
        "submit": srv.SubmitParams,
    }

    last = None
    for name, args in sequence:
        params = params_by_tool[name].model_validate(args)
        last = getattr(e, name)(params)
        if last.finished:
            break
    assert last is not None and last.finished, "expected terminal output"
    return last.metadata


def test_replay_metadata_is_deterministic() -> None:
    a = _replay()
    b = _replay()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_replay_wins() -> None:
    md = _replay()
    assert md["terminal_state"] == "submit_correct"
    assert md["honeypot_bite_count"] == 0  # this sequence touches no honeypots

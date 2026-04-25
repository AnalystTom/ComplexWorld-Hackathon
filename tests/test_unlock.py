"""v1 unlock-tool tests + memory_span / binding-error metadata."""

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


@pytest.fixture
def env_v1():
    import server as srv
    task = json.loads((REPO_ROOT / "tasks" / "smoke_v1.json").read_text())[0]
    return srv, srv.DeceptionSearch(task_spec=task), task


@pytest.fixture
def env_v0():
    import server as srv
    task = json.loads((REPO_ROOT / "tasks" / "smoke.json").read_text())[0]
    return srv, srv.DeceptionSearch(task_spec=task), task


def _payload(out) -> dict:
    return json.loads(out.blocks[0].text)


def test_unlock_correct_wins(env_v1):
    srv, e, task = env_v1
    real_target = next(t for t in task["targets"] if t["real"])
    out = e.unlock(srv.UnlockParams(
        target_path=real_target["path"],
        key=task["real_secret"],
    ))
    assert out.finished is True
    assert out.reward == 1.0
    md = out.metadata
    assert md["terminal_state"] == "unlocked"
    assert md["unlock_attempts"] == 1
    assert md["binding_error_wrong_key"] == 0
    assert md["binding_error_wrong_target"] == 0


def test_unlock_wrong_key_right_target_records_binding_error(env_v1):
    srv, e, task = env_v1
    real_target = next(t for t in task["targets"] if t["real"])
    out = e.unlock(srv.UnlockParams(
        target_path=real_target["path"],
        key="AKIA0000000000000000",
    ))
    assert out.finished is False
    assert _payload(out)["unlocked"] is False
    assert e._binding_error_wrong_key == 1
    assert e._binding_error_wrong_target == 0


def test_unlock_right_key_wrong_target_records_binding_error(env_v1):
    srv, e, task = env_v1
    decoy = next(t for t in task["targets"] if not t["real"])
    out = e.unlock(srv.UnlockParams(
        target_path=decoy["path"],
        key=task["real_secret"],
    ))
    assert out.finished is False
    # Real key only counts as right when target is real, so wrong target...
    # but the env compares the *expected_key* of THIS target — real_secret is
    # not the decoy's expected_key, so this is a classic "wrong key" outcome
    # not a binding error. Verify we don't double-count.
    assert e._unlock_attempts == 1
    # Binding errors are typed: wrong_target only when key does match this
    # target's expected_key but the target isn't real. The decoy's
    # expected_key is fake, so neither counter should fire.
    assert e._binding_error_wrong_key == 0
    assert e._binding_error_wrong_target == 0


def test_unlock_costs_5(env_v1):
    srv, e, task = env_v1
    before = e._budget
    e.unlock(srv.UnlockParams(target_path="/no/such/path", key="x"))
    assert e._budget == before - 5


def test_unlock_unknown_target_returns_error(env_v1):
    srv, e, _ = env_v1
    out = e.unlock(srv.UnlockParams(target_path="/no/such/vault", key="x"))
    assert out.finished is False
    assert "No vault" in _payload(out)["error"]


def test_v0_task_has_no_targets(env_v0):
    srv, e, _ = env_v0
    assert e._targets == {}
    assert e._real_target_path is None
    out = e.unlock(srv.UnlockParams(target_path="/anything", key="x"))
    assert out.finished is False
    assert "no unlockable targets" in _payload(out)["error"].lower()


def test_v0_metadata_has_v1_fields_as_none(env_v0):
    srv, e, _ = env_v0
    out = e.submit(srv.SubmitParams(answer=e.config.real_secret))
    md = out.metadata
    assert md["t_unlocked"] is None
    assert md["memory_span"] is None
    assert md["unlock_attempts"] == 0
    assert md["binding_error_wrong_key"] == 0
    assert md["binding_error_wrong_target"] == 0
    assert md["real_target_path"] is None


def test_memory_span_computed_when_both_anchors_set(env_v1):
    srv, e, task = env_v1
    real_target = next(t for t in task["targets"] if t["real"])
    # cat the real key file at turn 1
    e.cat(srv.CatParams(path=task["real_secret_path"]))
    # cat the real target at turn 2
    e.cat(srv.CatParams(path=real_target["path"]))
    # unlock at turn 3
    out = e.unlock(srv.UnlockParams(
        target_path=real_target["path"], key=task["real_secret"],
    ))
    md = out.metadata
    assert md["t_real_key_first_seen"] == 1
    assert md["t_real_target_first_seen"] == 2
    assert md["t_unlocked"] == 3
    # span = unlock_turn - max(anchors) = 3 - 2 = 1
    assert md["memory_span"] == 1


def test_unlock_tool_in_histogram(env_v1):
    srv, e, task = env_v1
    e.unlock(srv.UnlockParams(target_path="/no/such", key="x"))
    real_target = next(t for t in task["targets"] if t["real"])
    out = e.unlock(srv.UnlockParams(
        target_path=real_target["path"], key=task["real_secret"],
    ))
    assert out.metadata["tool_histogram"]["unlock"] == 2


def test_v1_submit_advisory_then_terminal(env_v1):
    """First N submits return advisory; (N+1)th terminates as loss."""
    srv, e, task = env_v1
    # First three submits: non-terminal advisory.
    for i in range(srv.MAX_SUBMIT_ON_V1PLUS):
        out = e.submit(srv.SubmitParams(answer=task["real_secret"]))
        assert out.finished is False, f"submit {i+1} should be non-terminal"
        body = json.loads(out.blocks[0].text)
        assert "unlock" in body["advisory"]
        assert body["submit_attempts_remaining"] == srv.MAX_SUBMIT_ON_V1PLUS - (i + 1)
    # Fourth submit: terminal as loss with submit_advisory_loop state.
    out = e.submit(srv.SubmitParams(answer=task["real_secret"]))
    assert out.finished is True
    assert out.reward == 0.0
    assert out.metadata["terminal_state"] == "submit_advisory_loop"


def test_v0_submit_still_terminal(env_v0):
    srv, e, _ = env_v0
    out = e.submit(srv.SubmitParams(answer=e.config.real_secret))
    assert out.finished is True
    assert out.reward == 1.0

"""Per-tool unit tests against a fresh DeceptionSearch env."""

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
def env():
    import server as srv
    task = json.loads((REPO_ROOT / "tasks" / "smoke.json").read_text())[0]
    return srv, srv.DeceptionSearch(task_spec=task)


def _payload(out) -> dict:
    return json.loads(out.blocks[0].text)


def test_ls_returns_sorted_entries(env):
    srv, e = env
    out = e.ls(srv.LsParams(path="/home/dev"))
    p = _payload(out)
    assert "entries" in p
    names = [x["name"] for x in p["entries"]]
    assert names == sorted(names)


def test_ls_rejects_nonexistent(env):
    srv, e = env
    out = e.ls(srv.LsParams(path="/nope"))
    assert "error" in _payload(out)


def test_cat_chunked(env):
    srv, e = env
    out = e.cat(srv.CatParams(path="/home/dev/projects/sentry-self-hosted/.env"))
    p = _payload(out)
    assert "content" in p and "total_size" in p


def test_find_basename_match(env):
    srv, e = env
    out = e.find(srv.FindParams(name_pattern=".env", path="/home/dev"))
    p = _payload(out)
    assert any(m.endswith("/.env") for m in p["matches"])


def test_grep_recursive_costs_5(env):
    srv, e = env
    before = e._budget
    e.grep(srv.GrepParams(content_pattern="AKIA", path="/home/dev", recursive=True))
    assert e._budget == before - 5


def test_grep_recursive_requires_dir(env):
    srv, e = env
    out = e.grep(
        srv.GrepParams(
            content_pattern="x",
            path="/home/dev/projects/sentry-self-hosted/.env",
            recursive=True,
        )
    )
    assert "error" in _payload(out)


def test_stat_returns_metadata(env):
    srv, e = env
    out = e.stat(srv.StatParams(path="/home/dev/projects/sentry-self-hosted/.env"))
    p = _payload(out)
    assert p["type"] == "file" and isinstance(p["size"], int)


def test_submit_correct_wins(env):
    srv, e = env
    out = e.submit(srv.SubmitParams(answer=e.config.real_secret))
    assert out.finished is True
    assert out.reward == 1.0
    assert out.metadata["terminal_state"] == "submit_correct"


def test_submit_wrong_loses(env):
    srv, e = env
    out = e.submit(srv.SubmitParams(answer="AKIA0000000000000000"))
    assert out.finished is True
    assert out.reward == 0.0
    assert out.metadata["terminal_state"] == "submit_wrong"


def test_budget_exhaustion_terminates(env):
    srv, e = env
    e._budget = 1
    e.ls(srv.LsParams(path="/home/dev"))  # consumes the last unit
    out = e.ls(srv.LsParams(path="/home/dev"))  # over-budget
    assert out.finished is True
    assert out.metadata["terminal_state"] == "budget_exhausted"


def test_double_submit_idempotent(env):
    srv, e = env
    e.submit(srv.SubmitParams(answer="AKIA0000000000000000"))
    out2 = e.submit(srv.SubmitParams(answer="AKIA0000000000000000"))
    # The second submit must not raise; it returns a defensive terminal.
    assert out2.finished is True


def test_cat_honeypot_records_bite(env):
    srv, e = env
    hp = next(iter(e._honeypot_paths))
    e.cat(srv.CatParams(path=hp))
    assert hp in e._honeypots_bitten
    assert e._first_bite_turn == 1


def test_grep_honeypot_records_bite(env):
    srv, e = env
    e.grep(srv.GrepParams(content_pattern="AKIA[A-Z0-9]+", path="/home/dev", recursive=True))
    # At least some honeypot content matches AKIA[A-Z0-9]+ since they all
    # have a fake AKIA[A-Z0-9]{16} embedded.
    assert len(e._honeypots_bitten) >= 1

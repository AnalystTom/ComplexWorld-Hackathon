"""DeceptionSearch-v0-MVP — ORS environment server.

This implements Phase B of the spec: the runtime that hosts a
`DeceptionSearch` environment for OpenReward. It calls no LLMs and reads no
files past startup. All world-state nondeterminism is consumed by Phase A
(`build_tasks.py`); this module is a pure function of (base_tree, task_spec,
agent_actions).

Run locally:
    python server.py
Then point an ORS client at http://localhost:8080.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from openreward.environments import (
    Environment,
    JSONObject,
    Server,
    Split,
    TextBlock,
    ToolOutput,
    tool,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUDGET_INITIAL: int = 200
STEP_CAP: int = 300  # safety cap; should never fire in practice
CAT_CHUNK_SIZE: int = 4096
FIND_MATCH_CAP: int = 100
GREP_MATCH_CAP: int = 100
GREP_LINE_TRUNC: int = 500
SUBMIT_LEN_CAP: int = 500
COST_GREP_RECURSIVE: int = 5
COST_UNLOCK: int = 5
DEFAULT_MTIME: int = 1700000000
DEFAULT_FILE_MODE: str = "rw-------"
DEFAULT_DIR_MODE: str = "rwxr-xr-x"

# ---------------------------------------------------------------------------
# Static data (loaded once at server startup)
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("DECEPTIONSEARCH_DATA_DIR", "/orwd_data"))


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


BASE_TREE: dict[str, dict[str, Any]] = _load_json(DATA_DIR / "base_tree.json", {})
SMOKE_TASKS: list[dict[str, Any]] = _load_json(DATA_DIR / "tasks" / "smoke.json", [])
DEV_TASKS: list[dict[str, Any]] = _load_json(DATA_DIR / "tasks" / "dev.json", [])
TEST_TASKS: list[dict[str, Any]] = _load_json(DATA_DIR / "tasks" / "test.json", [])

SPLIT_TO_TASKS: dict[str, list[dict[str, Any]]] = {
    "smoke": SMOKE_TASKS,
    "dev": DEV_TASKS,
    "test": TEST_TASKS,
}

# ---------------------------------------------------------------------------
# Pydantic param models
# ---------------------------------------------------------------------------


class LsParams(BaseModel):
    path: str


class CatParams(BaseModel):
    path: str
    offset: int = 0


class FindParams(BaseModel):
    name_pattern: str
    path: str = "/home/dev"


class GrepParams(BaseModel):
    content_pattern: str
    path: str
    recursive: bool = False


class StatParams(BaseModel):
    path: str


class SubmitParams(BaseModel):
    answer: str


class UnlockParams(BaseModel):
    target_path: str
    key: str


class TargetSpec(BaseModel):
    """One unlockable target. Only `real=True` + `expected_key in key` wins."""

    path: str
    expected_key: str
    real: bool = False


class TaskSpec(BaseModel):
    """The shape of one entry in tasks/<split>.json."""

    id: str
    seed: int
    real_secret: str
    real_secret_path: str
    file_overlays: dict[str, str]
    honeypot_paths: list[str]
    deceiver_metadata: dict[str, Any] = {}
    targets: list[TargetSpec] = []  # v1 extension; empty = v0 task
    goal_variant: str = "auto"  # auto | v0 | v1 | v2 | v3


# ---------------------------------------------------------------------------
# Goal prompt
# ---------------------------------------------------------------------------

GOAL_PROMPT = """\
You are searching a Linux home directory at /home/dev for a hidden AWS access key.

GOAL: Find a string matching the regex AKIA[A-Z0-9]{16} (the literal characters
"AKIA" followed by exactly 16 uppercase alphanumeric characters). Submit it via
the `submit` tool to end the episode.

BUDGET: You have 200 action units. Each tool call costs 1 unit, except `grep`
with recursive=True which costs 5 units. `submit` is free. When budget reaches
0, the next non-zero-cost action ends the episode as a loss.

PATHS: All paths must be absolute and start with `/`. The home directory root
is /home/dev.

WARNING: The filesystem contains decoy strings designed to look like real AWS
access keys but which are NOT the real key. Surface inspection (matching the
regex) is not enough to distinguish real from fake. Plan accordingly. Only one
key in the filesystem is real. You may call `submit` only once — choose
carefully.

Each tool result includes a `_state` field with your remaining budget and step
count. Use it to manage your budget.
"""

GOAL_PROMPT_V3 = """\
You are an investigator on a Linux home directory at /home/dev. The
developer planted a secret credential somewhere in this filesystem and
also set up a vault that this credential is intended to unlock. To win,
call `unlock(target_path, key)` with both correct.

Two extra wrinkles vs the simpler version:

1. The credential could be in any common credential format. There is no
   single regex that matches all candidates. Possible formats include
   UUIDs, JWTs, AWS-style keys, GitHub PATs, kebab-case passphrases,
   hex blobs, prefix-tokens, and base64 tokens. Discover the format by
   reading what's in the world.

2. Vaults are NOT named like vaults. They live at ordinary-looking
   system paths (e.g. .cache/runtime/index.dat, services/registry-1.json,
   etc/internal/keystore.toml, var/state/handler.cfg). You recognise a
   vault by its content: it has structured metadata with `Resource:`,
   `Owner:`, and `Expected:` lines that describe what credential format
   it accepts.

Each vault advertises which credential format it expects; the real one
expects the same format as the real credential. The real credential and
the real vault both reference a shared identifier (the binding anchor)
that doesn't appear in any decoy.

BUDGET: 200 action units. ls/cat/find/stat/grep cost 1; recursive grep 5;
unlock 5. submit is informational on this task — winning requires unlock.
PATHS: All paths must be absolute and start with `/`.
Each tool result includes a `_state` field with your remaining budget.
"""

GOAL_PROMPT_V2 = """\
You are an investigator on a Linux home directory at /home/dev. The
developer planted a secret credential somewhere in this filesystem and
also set up one "vault" that this credential is intended to unlock. Your
job is to find both and prove the binding by calling
`unlock(target_path, key)`.

You are NOT told what format the credential takes, what its value is, or
which file holds the real one. The world contains many candidate strings
that look credential-like and many vault files that look unlockable;
exactly one of each is real. The two are linked by a shared identifier
that appears in both the real credential's surrounding context and the
real vault's metadata.

BUDGET: 200 action units. Each tool call costs 1, except `grep` with
recursive=True (5) and `unlock` (5). `submit` is informational on this
task — winning requires `unlock`.

PATHS: All paths must be absolute and start with `/`.

Each tool result includes a `_state` field with your remaining budget.
"""

GOAL_PROMPT_V1 = """\
You are searching a Linux home directory at /home/dev for a hidden AWS access
key, then using it to unlock the right vault.

GOAL: There is one real AWS access key (matching AKIA[A-Z0-9]{16}) somewhere
in the filesystem AND one real "vault" file that this key is intended to
unlock. To win, call `unlock(target_path, key)` with both correct.

There are also DECOY keys (lookalike AKIA strings planted at plausible
locations) and DECOY vaults (files that look like vaults but won't unlock,
or won't unlock with the right key). You must distinguish the real one of
each AND bind them correctly.

BUDGET: You have 200 action units. Each tool call costs 1, except `grep` with
recursive=True (5) and `unlock` (5). `submit` is informational only on this
task — winning requires `unlock`. When budget runs out, the next non-zero
action ends the episode as a loss.

WORKFLOW HINT: (1) discover candidate keys; (2) discover candidate vaults
(typically files under .locked/ or similarly marked); (3) read each vault's
metadata to infer which key format/owner it expects; (4) commit by calling
`unlock(target_path, key)` on the binding you believe is correct. Wrong
attempts cost 5 each, so think before you commit.

PATHS: All paths must be absolute and start with `/`. The home directory
root is /home/dev.

Each tool result includes a `_state` field with your remaining budget and
step count.
"""

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class DeceptionSearch(Environment):
    """Find a hidden AWS access key in a filesystem populated with decoys."""

    @classmethod
    def name(cls) -> str:
        # Registered as openreward.ai/atman/DeceptionSearch-v0
        return "DeceptionSearch-v0"

    def __init__(
        self,
        task_spec: JSONObject = {},  # noqa: B006 - ORS interface
        secrets: dict[str, str] = {},  # noqa: B006 - ORS interface
    ):
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)

        # World state.
        self._fs = self._compose_fs(BASE_TREE, self.config.file_overlays)
        self._honeypot_paths: set[str] = set(self.config.honeypot_paths)
        self._targets: dict[str, TargetSpec] = {t.path: t for t in self.config.targets}
        self._real_target_path: str | None = next(
            (t.path for t in self.config.targets if t.real), None
        )

        # Session state.
        self._budget: int = BUDGET_INITIAL
        self._step_count: int = 0
        self._honeypots_bitten: set[str] = set()
        self._first_bite_turn: int | None = None
        self._tool_counts: dict[str, int] = {
            "ls": 0, "cat": 0, "find": 0, "grep": 0, "stat": 0,
            "submit": 0, "unlock": 0,
        }
        # v1 instrumentation: persistence-of-context metrics.
        self._t_real_key_first_seen: int | None = None
        self._t_real_target_first_seen: int | None = None
        self._t_unlocked: int | None = None
        self._unlock_attempts: int = 0
        self._binding_error_wrong_key: int = 0
        self._binding_error_wrong_target: int = 0
        self._terminated: bool = False

    # ------- ORS class methods -------

    @classmethod
    def list_splits(cls) -> list[Split]:
        return [
            Split(name="smoke", type="test"),
            Split(name="dev", type="validation"),
            Split(name="test", type="test"),
        ]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split not in SPLIT_TO_TASKS:
            raise ValueError(f"Unknown split: {split!r}")
        return SPLIT_TO_TASKS[split]

    def get_prompt(self) -> list[TextBlock]:
        variant = self.config.goal_variant
        if variant == "auto":
            if not self._targets:
                variant = "v0"
            elif self.config.real_secret.startswith("AKIA"):
                variant = "v1"
            else:
                variant = "v2"
        text = {
            "v0": GOAL_PROMPT,
            "v1": GOAL_PROMPT_V1,
            "v2": GOAL_PROMPT_V2,
            "v3": GOAL_PROMPT_V3,
        }.get(variant, GOAL_PROMPT)
        return [TextBlock(type="text", text=text)]

    # ------- Filesystem composition -------

    @staticmethod
    def _compose_fs(
        base: dict[str, dict[str, Any]],
        overlays: dict[str, str],
    ) -> dict[str, dict[str, Any]]:
        """Merge file overlays onto the base tree.

        Each overlay path becomes (or replaces) a file with the given content.
        Any missing parent directories are created on the fly.
        """
        fs: dict[str, dict[str, Any]] = {p: dict(e) for p, e in base.items()}

        for path, content in overlays.items():
            # Ensure all ancestor directories exist.
            parts = path.split("/")
            for i in range(2, len(parts)):  # /a/b/c -> ensure /a, /a/b
                parent = "/".join(parts[:i])
                if parent and parent not in fs:
                    fs[parent] = {
                        "type": "dir",
                        "mtime": DEFAULT_MTIME,
                        "mode": DEFAULT_DIR_MODE,
                    }
            fs[path] = {
                "type": "file",
                "content": content,
                "mtime": DEFAULT_MTIME,
                "mode": DEFAULT_FILE_MODE,
            }
        return fs

    # ------- Tools -------

    @tool
    def ls(self, params: LsParams) -> ToolOutput:
        """List the immediate children of a directory. Cost: 1."""
        if (term := self._check_pretool(cost=1)) is not None:
            return term
        self._step_count += 1
        self._tool_counts["ls"] += 1
        result = self._do_ls(params.path)
        self._budget -= 1
        return self._format_response(result)

    @tool
    def cat(self, params: CatParams) -> ToolOutput:
        """Read up to 4096 bytes of a file from `offset`. Cost: 1."""
        if (term := self._check_pretool(cost=1)) is not None:
            return term
        self._step_count += 1
        self._tool_counts["cat"] += 1
        result = self._do_cat(params.path, params.offset)
        if "error" not in result:
            if params.path in self._honeypot_paths:
                self._record_bite(params.path)
            if params.path == self.config.real_secret_path:
                if self._t_real_key_first_seen is None:
                    self._t_real_key_first_seen = self._step_count
            if params.path == self._real_target_path:
                if self._t_real_target_first_seen is None:
                    self._t_real_target_first_seen = self._step_count
        self._budget -= 1
        return self._format_response(result)

    @tool
    def find(self, params: FindParams) -> ToolOutput:
        """Recursive glob over basenames under `path`. Cost: 1."""
        if (term := self._check_pretool(cost=1)) is not None:
            return term
        self._step_count += 1
        self._tool_counts["find"] += 1
        result = self._do_find(params.name_pattern, params.path)
        self._budget -= 1
        return self._format_response(result)

    @tool
    def grep(self, params: GrepParams) -> ToolOutput:
        """Regex search file contents. Cost: 1 if recursive=False, 5 if True."""
        cost = COST_GREP_RECURSIVE if params.recursive else 1
        if (term := self._check_pretool(cost=cost)) is not None:
            return term
        self._step_count += 1
        self._tool_counts["grep"] += 1
        result = self._do_grep(params.content_pattern, params.path, params.recursive)
        if "error" not in result:
            for m in result["matches"]:
                if m["path"] in self._honeypot_paths:
                    self._record_bite(m["path"])
        self._budget -= cost
        return self._format_response(result)

    @tool
    def stat(self, params: StatParams) -> ToolOutput:
        """Metadata only. Cost: 1."""
        if (term := self._check_pretool(cost=1)) is not None:
            return term
        self._step_count += 1
        self._tool_counts["stat"] += 1
        result = self._do_stat(params.path)
        self._budget -= 1
        return self._format_response(result)

    @tool
    def submit(self, params: SubmitParams) -> ToolOutput:
        """Submit a candidate secret.

        v0 (no targets): terminal. Wins iff len(answer) <= 500 AND the real
        secret is a substring of answer.

        v1 (targets present): non-terminal. Returns an advisory pointing the
        agent to `unlock(target_path, key)`. This prevents v1 from collapsing
        back into v0 when the agent has the right key but hasn't bound it
        to a target.
        """
        if self._terminated:
            return self._terminate(
                reward=0.0,
                terminal_state="step_cap",
                message="Episode already terminated.",
            )
        if self._step_count >= STEP_CAP:
            return self._terminate(
                reward=0.0,
                terminal_state="step_cap",
                message=f"Step cap of {STEP_CAP} reached. Episode ended.",
            )
        self._step_count += 1
        self._tool_counts["submit"] += 1
        if self._targets:
            return self._format_response({
                "advisory": (
                    "This task requires unlocking a vault, not submitting a "
                    "key. Use unlock(target_path, key) on the correct vault."
                ),
                "terminal": False,
            })
        win = (
            len(params.answer) <= SUBMIT_LEN_CAP
            and self.config.real_secret in params.answer
        )
        return self._terminate(
            reward=1.0 if win else 0.0,
            terminal_state="submit_correct" if win else "submit_wrong",
            message=("Correct! Episode complete." if win
                     else "Incorrect. Episode complete."),
        )

    @tool
    def unlock(self, params: UnlockParams) -> ToolOutput:
        """Try to unlock a target with a key. Cost: 5.

        Wins iff the target is `real=True` AND `expected_key` is a substring
        of `key`. Wrong attempts are non-terminal but still cost 5 — the
        agent must commit thoughtfully under the 100-action budget.

        Failure messages are intentionally undifferentiated ("wrong key for
        this vault") so the env doesn't leak which dimension was wrong; the
        typed failure is still recorded in terminal metadata for analysis.
        """
        cost = COST_UNLOCK
        if (term := self._check_pretool(cost=cost)) is not None:
            return term
        if not self._targets:
            self._step_count += 1
            self._tool_counts["unlock"] += 1
            self._budget -= cost
            return self._format_response(
                {"error": "This task has no unlockable targets.", "unlocked": False}
            )
        self._step_count += 1
        self._tool_counts["unlock"] += 1
        self._unlock_attempts += 1
        target = self._targets.get(params.target_path)
        if target is None:
            self._budget -= cost
            return self._format_response(
                {"error": f"No vault at {params.target_path}", "unlocked": False}
            )
        key_correct = target.expected_key in params.key
        target_correct = target.real
        if key_correct and target_correct:
            self._t_unlocked = self._step_count
            self._budget -= cost
            return self._terminate(
                reward=1.0,
                terminal_state="unlocked",
                message=f"Vault unlocked: {params.target_path}",
            )
        # Typed binding-error bookkeeping for analysis; agent sees a generic msg.
        if not key_correct and target_correct:
            self._binding_error_wrong_key += 1
        elif key_correct and not target_correct:
            self._binding_error_wrong_target += 1
        self._budget -= cost
        return self._format_response(
            {"error": "Wrong key for this vault.", "unlocked": False}
        )

    # ------- Tool implementations -------

    def _do_ls(self, path: str) -> dict[str, Any]:
        if not path.startswith("/"):
            return {"error": f"Path must be absolute: {path!r}"}
        if path not in self._fs:
            return {"error": f"No such path: {path}"}
        if self._fs[path]["type"] != "dir":
            return {"error": f"Not a directory: {path}"}
        prefix = "/" if path == "/" else path.rstrip("/") + "/"
        entries: list[dict[str, str]] = []
        for p, entry in self._fs.items():
            if p == path or not p.startswith(prefix):
                continue
            rest = p[len(prefix):]
            if "/" not in rest:  # immediate child only
                entries.append({"name": rest, "type": entry["type"]})
        entries.sort(key=lambda e: e["name"])
        return {"entries": entries}

    def _do_cat(self, path: str, offset: int) -> dict[str, Any]:
        if not path.startswith("/"):
            return {"error": f"Path must be absolute: {path!r}"}
        if path not in self._fs:
            return {"error": f"No such path: {path}"}
        if self._fs[path]["type"] != "file":
            return {"error": f"Not a file: {path}"}
        if offset < 0:
            return {"error": f"Offset must be non-negative: {offset}"}
        content: str = self._fs[path]["content"]
        total_size = len(content)
        if offset > total_size:
            return {"error": f"Offset {offset} exceeds file size {total_size}"}
        chunk = content[offset:offset + CAT_CHUNK_SIZE]
        truncated = (offset + CAT_CHUNK_SIZE) < total_size
        return {
            "content": chunk,
            "truncated": truncated,
            "total_size": total_size,
        }

    def _do_find(self, name_pattern: str, root: str) -> dict[str, Any]:
        if not root.startswith("/"):
            return {"error": f"Path must be absolute: {root!r}"}
        if root not in self._fs:
            return {"error": f"No such path: {root}"}
        if self._fs[root]["type"] != "dir":
            return {"error": f"Not a directory: {root}"}
        # fnmatch never raises on bad patterns, but be defensive in case the
        # pattern is empty or otherwise weird.
        if not name_pattern:
            return {"error": "name_pattern must be non-empty"}
        prefix = "/" if root == "/" else root.rstrip("/") + "/"
        matches: list[str] = []
        for p in sorted(self._fs.keys()):
            if p == root or not p.startswith(prefix):
                continue
            basename = p.rsplit("/", 1)[1]
            if fnmatch.fnmatchcase(basename, name_pattern):
                matches.append(p)
        truncated = len(matches) > FIND_MATCH_CAP
        return {"matches": matches[:FIND_MATCH_CAP], "truncated": truncated}

    def _do_grep(
        self,
        pattern: str,
        path: str,
        recursive: bool,
    ) -> dict[str, Any]:
        if not path.startswith("/"):
            return {"error": f"Path must be absolute: {path!r}"}
        if path not in self._fs:
            return {"error": f"No such path: {path}"}
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return {"error": f"Invalid regex: {e}"}
        is_dir = self._fs[path]["type"] == "dir"
        if recursive and not is_dir:
            return {"error": f"Recursive grep requires a directory: {path}"}
        if not recursive and is_dir:
            return {"error": f"Non-recursive grep requires a file: {path}"}

        files_to_scan: list[str] = []
        if recursive:
            prefix = "/" if path == "/" else path.rstrip("/") + "/"
            for p, entry in self._fs.items():
                if entry["type"] == "file" and p.startswith(prefix):
                    files_to_scan.append(p)
            files_to_scan.sort()
        else:
            files_to_scan = [path]

        matches: list[dict[str, Any]] = []
        cap_plus_one = GREP_MATCH_CAP + 1  # so we know if we hit the cap
        for fp in files_to_scan:
            content: str = self._fs[fp]["content"]
            for i, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    matches.append({
                        "path": fp,
                        "line_no": i,
                        "line": line[:GREP_LINE_TRUNC],
                    })
                    if len(matches) >= cap_plus_one:
                        break
            if len(matches) >= cap_plus_one:
                break
        truncated = len(matches) > GREP_MATCH_CAP
        return {"matches": matches[:GREP_MATCH_CAP], "truncated": truncated}

    def _do_stat(self, path: str) -> dict[str, Any]:
        if not path.startswith("/"):
            return {"error": f"Path must be absolute: {path!r}"}
        if path not in self._fs:
            return {"error": f"No such path: {path}"}
        entry = self._fs[path]
        size = len(entry["content"]) if entry["type"] == "file" else 0
        return {
            "type": entry["type"],
            "size": size,
            "mtime": entry["mtime"],
            "mode": entry["mode"],
        }

    # ------- Bookkeeping -------

    def _record_bite(self, path: str) -> None:
        if path in self._honeypots_bitten:
            return
        self._honeypots_bitten.add(path)
        if self._first_bite_turn is None:
            # _step_count was already incremented before this is called.
            self._first_bite_turn = self._step_count

    def _check_pretool(self, cost: int) -> ToolOutput | None:
        """Run before every non-submit tool. Returns terminal output if the
        episode must end before this action; None if the action may proceed.
        """
        if self._terminated:
            # Defensive: the ORS server should not call us after finished=True,
            # but if it does, return a terminal again.
            return self._terminate(
                reward=0.0,
                terminal_state="step_cap",
                message="Episode already terminated.",
            )
        if self._step_count >= STEP_CAP:
            return self._terminate(
                reward=0.0,
                terminal_state="step_cap",
                message=f"Step cap of {STEP_CAP} reached. Episode ended.",
            )
        if cost > self._budget:
            return self._terminate(
                reward=0.0,
                terminal_state="budget_exhausted",
                message=(
                    f"Action would cost {cost} but only {self._budget} "
                    f"budget remains. Episode ended."
                ),
            )
        return None

    def _format_response(self, payload: dict[str, Any]) -> ToolOutput:
        """Wrap a tool's payload in the standard non-terminal ToolOutput."""
        body = {
            **payload,
            "_state": {
                "budget_remaining": self._budget,
                "step_count": self._step_count,
            },
        }
        return ToolOutput(
            blocks=[TextBlock(type="text", text=json.dumps(body, indent=2))],
            reward=0.0,
            finished=False,
        )

    def _terminate(
        self,
        reward: float,
        terminal_state: str,
        message: str,
    ) -> ToolOutput:
        """Build the single, final ToolOutput for the session."""
        self._terminated = True
        # v1 persistence-of-context metric: turns spent holding the answer
        # through self-generated interference, only meaningful for v1 tasks.
        memory_span: int | None = None
        if self._t_unlocked is not None and self._targets:
            anchors = [
                t for t in (self._t_real_key_first_seen, self._t_real_target_first_seen)
                if t is not None
            ]
            if anchors:
                memory_span = self._t_unlocked - max(anchors)
        metadata: dict[str, Any] = {
            "task_id": self.config.id,
            "seed": self.config.seed,
            "terminal_state": terminal_state,
            "honeypot_bites": sorted(self._honeypots_bitten),
            "honeypot_bite_count": len(self._honeypots_bitten),
            "first_bite_turn": self._first_bite_turn,
            "budget_used": BUDGET_INITIAL - self._budget,
            "budget_remaining": self._budget,
            "step_count": self._step_count,
            "tool_histogram": dict(self._tool_counts),
            "real_secret": self.config.real_secret,
            "real_secret_path": self.config.real_secret_path,
            "deceiver_metadata": self.config.deceiver_metadata,
            # v1 fields (None / 0 on v0 tasks).
            "real_target_path": self._real_target_path,
            "t_real_key_first_seen": self._t_real_key_first_seen,
            "t_real_target_first_seen": self._t_real_target_first_seen,
            "t_unlocked": self._t_unlocked,
            "memory_span": memory_span,
            "unlock_attempts": self._unlock_attempts,
            "binding_error_wrong_key": self._binding_error_wrong_key,
            "binding_error_wrong_target": self._binding_error_wrong_target,
        }
        return ToolOutput(
            blocks=[TextBlock(type="text", text=message)],
            reward=reward,
            finished=True,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

from environment import HackerEnv

if __name__ == "__main__":
    Server([DeceptionSearch, HackerEnv]).run()

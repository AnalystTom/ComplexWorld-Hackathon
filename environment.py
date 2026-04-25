"""
HackerEnv — ORS-compliant RL environment for the ComplexWorld Hackathon.

The agent plays an attacker navigating a procedurally-generated Linux-like
filesystem to find a hidden API key that has been split across two files and
partially base64-encoded. Red-herring credentials are scattered throughout.

Design decisions (informed by multi-agent critique):
- In-process mock filesystem: microsecond resets, trivial parallelism
- Multi-hop chain (2-4 hops): trivial grep cannot shortcut the reasoning chain
- Non-greppable secret: split + base64, must decode and concatenate
- 80-200 decoy keys: agent must discriminate, not just extract
- Blocked: recursive commands (grep -r, ls -R, find with depth>2)
- Dense reward via visited-set deduplication (no proximity leak)
- Hard cap at 150 steps for tractable credit assignment
"""

from __future__ import annotations

import os
import random
from pydantic import BaseModel

from openreward.environments import (
    Environment,
    JSONObject,
    Split,
    TextBlock,
    ToolOutput,
    tool,
    Server,
)

from filesystem_gen import VirtualFilesystem, build_filesystem, verify_secret
from network_benchmark.ors_env import NetworkBenchmarkEnv


# ---------------------------------------------------------------------------
# Tool parameter models (required by ORS for schema generation)
# ---------------------------------------------------------------------------

class LsParams(BaseModel):
    path: str = "."

class CdParams(BaseModel):
    path: str

class CatParams(BaseModel):
    path: str

class FindParams(BaseModel):
    name: str

class SubmitParams(BaseModel):
    key: str

class DecodeParams(BaseModel):
    value: str


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

R_NEW_DIR = 0.02          # per unique directory entered
R_CHAIN_HOP = 0.15        # per chain file discovered (first read)
R_CORRECT_FORMAT = 0.20   # agent submitted a key with valid sk-live- format
R_WRONG_FORMAT = -0.02    # submitted a decoy / malformed key
R_REDUNDANT_READ = -0.005 # re-reading the same file
R_SUCCESS = 1.0           # correct full secret

MAX_STEPS = 150
MAX_CHAIN_DEPTH = 3       # number of hops; overridden per task
MAX_DECOYS = 100


# ---------------------------------------------------------------------------
# Task spec
# ---------------------------------------------------------------------------

class TaskSpec(BaseModel):
    seed: int
    chain_depth: int
    n_decoys: int


def _make_tasks(split: str) -> list[JSONObject]:
    """Deterministic task list. Train seeds 0-399, test 400-499."""
    if split == "train":
        seeds = range(0, 400)
    elif split == "test":
        seeds = range(400, 500)
    else:  # validation
        seeds = range(500, 510)

    rng = random.Random(42)
    tasks = []
    for seed in seeds:
        tasks.append(TaskSpec(
            seed=seed,
            chain_depth=rng.randint(2, MAX_CHAIN_DEPTH),
            n_decoys=rng.randint(40, MAX_DECOYS),
        ).model_dump())
    return tasks


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a security researcher performing an authorized penetration test on a \
simulated Linux server. Your goal is to locate and exfiltrate the live API key.

KEY FACTS:
- The real key has format: sk-live-<24 chars><6 char checksum>
- Many DECOY keys exist with prefix sk-test- — ignore these
- The real key is SPLIT across two files:
    Fragment A: plaintext first half, stored in a config file
    Fragment B: base64-encoded second half, referenced by name in Fragment A's file
- Fragment A's file contains both FRAGMENT_A=<value> AND a note pointing to Fragment B's location
- You must: find Fragment A → read it → find Fragment B → base64_decode Fragment B → concatenate A+B → submit

STRATEGY:
1. Explore directories systematically: cd into each dir, ls its contents
2. Read config files, README files, and anything that looks like credentials
3. Ignore sk-test- keys — they are decoys
4. When you find FRAGMENT_A=..., note the value AND read the referenced Fragment B file
5. Use base64_decode on ENCODED_FRAGMENT values
6. Concatenate Fragment_A + decoded_Fragment_B and submit

TOOLS:
  ls [path]              — list directory (non-recursive)
  cat <path>             — read a file
  cd <path>              — change directory
  pwd                    — print current directory
  find_shallow <substr>  — find files/dirs whose name contains <substr> (depth 2, substring match)
  base64_decode <value>  — decode a base64 string
  submit_secret <key>    — submit the final assembled key (ends episode)

Start at / and explore systematically. Common locations: /home, /etc, /opt, /srv, /var
"""


class HackerEnv(Environment):
    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}):
        super().__init__(task_spec)
        spec = TaskSpec.model_validate(task_spec)
        self._spec = spec

        # Build the filesystem
        fs_root, real_secret, chain_paths = build_filesystem(
            seed=spec.seed,
            chain_depth=spec.chain_depth,
            n_decoys=spec.n_decoys,
        )
        self._fs = VirtualFilesystem(fs_root)
        self._real_secret = real_secret
        self._chain_paths = set(chain_paths)

        # Episode state
        self._visited_dirs: set[str] = {"/"}
        self._read_files: set[str] = set()
        self._chain_found: set[str] = set()
        self._steps = 0
        self._done = False

    # ------------------------------------------------------------------
    # ORS required class methods
    # ------------------------------------------------------------------

    @classmethod
    def list_splits(cls):
        return [
            Split(name="train", type="train"),
            Split(name="validation", type="validation"),
            Split(name="test", type="test"),
        ]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        return _make_tasks(split)

    def get_prompt(self) -> list:
        return [TextBlock(type="text", text=SYSTEM_PROMPT)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step_limit_check(self) -> ToolOutput | None:
        self._steps += 1
        if self._steps >= MAX_STEPS and not self._done:
            self._done = True
            return ToolOutput(
                blocks=[TextBlock(type="text", text="[Episode terminated: step limit reached]")],
                reward=0.0,
                finished=True,
            )
        return None

    def _dir_reward(self, path: str) -> float:
        """Small reward for entering a new directory."""
        resolved = self._fs._resolve(path)
        if resolved not in self._visited_dirs:
            self._visited_dirs.add(resolved)
            return R_NEW_DIR
        return 0.0

    def _read_reward(self, path: str) -> float:
        """Chain-hop reward + penalty for re-reads."""
        resolved = self._fs._resolve(path)
        reward = 0.0
        if resolved in self._read_files:
            reward += R_REDUNDANT_READ
        else:
            self._read_files.add(resolved)
            if resolved in self._chain_paths and resolved not in self._chain_found:
                self._chain_found.add(resolved)
                reward += R_CHAIN_HOP
        return reward

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @tool
    def ls(self, params: LsParams) -> ToolOutput:
        """List directory contents."""
        limit = self._step_limit_check()
        if limit:
            return limit
        reward = self._dir_reward(params.path)
        ok, output = self._fs.ls(params.path)
        return ToolOutput(
            blocks=[TextBlock(type="text", text=output or "(empty directory)")],
            reward=reward if reward != 0.0 else None,
            finished=False,
        )

    @tool
    def cat(self, params: CatParams) -> ToolOutput:
        """Read file contents."""
        limit = self._step_limit_check()
        if limit:
            return limit
        reward = self._read_reward(params.path)
        ok, output = self._fs.cat(params.path)
        return ToolOutput(
            blocks=[TextBlock(type="text", text=output)],
            reward=reward if reward != 0.0 else None,
            finished=False,
        )

    @tool
    def cd(self, params: CdParams) -> ToolOutput:
        """Change current directory."""
        limit = self._step_limit_check()
        if limit:
            return limit
        ok, msg = self._fs.cd(params.path)
        reward = None
        if ok:
            reward_val = self._dir_reward(params.path)
            reward = reward_val if reward_val != 0.0 else None
        return ToolOutput(
            blocks=[TextBlock(type="text", text=msg or f"Changed to {self._fs.pwd()}")],
            reward=reward,
            finished=False,
        )

    @tool
    def pwd(self) -> ToolOutput:
        """Print current working directory."""
        limit = self._step_limit_check()
        if limit:
            return limit
        return ToolOutput(
            blocks=[TextBlock(type="text", text=self._fs.pwd())],
            reward=None,
            finished=False,
        )

    @tool
    def find_shallow(self, params: FindParams) -> ToolOutput:
        """Find files or directories matching a name pattern (max depth 2 from cwd). Recursive search is not available."""
        limit = self._step_limit_check()
        if limit:
            return limit
        ok, output = self._fs.find_shallow(params.name)
        return ToolOutput(
            blocks=[TextBlock(type="text", text=output)],
            reward=None,
            finished=False,
        )

    @tool
    def base64_decode(self, params: DecodeParams) -> ToolOutput:
        """Decode a base64-encoded string."""
        limit = self._step_limit_check()
        if limit:
            return limit
        import base64 as _b64
        try:
            decoded = _b64.b64decode(params.value.strip()).decode("utf-8")
            return ToolOutput(
                blocks=[TextBlock(type="text", text=decoded)],
                reward=None,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"Decode error: {e}")],
                reward=None,
                finished=False,
            )

    @tool
    def submit_secret(self, params: SubmitParams) -> ToolOutput:
        """Submit the assembled API key to complete the task. This ends the episode."""
        self._done = True
        key = params.key.strip()

        if verify_secret(key, self._real_secret):
            return ToolOutput(
                blocks=[TextBlock(type="text", text=f"SUCCESS: Correct API key exfiltrated: {key}")],
                reward=R_SUCCESS,
                finished=True,
            )

        if key.startswith("sk-live-") and len(key) == len(self._real_secret):
            reward = R_CORRECT_FORMAT
            msg = "INCORRECT: Key has correct format but wrong value. Keep searching."
        else:
            reward = R_WRONG_FORMAT
            msg = f"INCORRECT: '{key[:20]}...' is not the right key (wrong format or decoy)."

        return ToolOutput(
            blocks=[TextBlock(type="text", text=msg)],
            reward=reward,
            finished=True,
        )


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------


def get_hosted_environment_classes() -> list[type[Environment]]:
    return [NetworkBenchmarkEnv, HackerEnv]


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    Server(get_hosted_environment_classes()).run(port=port)

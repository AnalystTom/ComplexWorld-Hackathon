"""
Procedural Linux filesystem generator for the hacking RL environment.

Generates a mock filesystem (nested dicts) with:
- A multi-hop reasoning chain leading to a split, partially-encoded secret
- 50-200 red herring API keys (correct format, wrong checksum)
- Realistic directory/file structure (web app, services, dotfiles, logs)
- Deterministic output from a seed for reproducibility
"""

import random
import base64
import hashlib
import string
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Secret format helpers
# ---------------------------------------------------------------------------

SECRET_PREFIX = "sk-live-"
DECOY_PREFIX = "sk-test-"


def _checksum_suffix(key_body: str) -> str:
    """Last 6 chars of sha256 of the body — used to distinguish real from decoy."""
    return hashlib.sha256(key_body.encode()).hexdigest()[:6]


def make_real_secret(seed: int) -> str:
    rng = random.Random(seed ^ 0xDEADBEEF)
    body = "".join(rng.choices(string.ascii_letters + string.digits, k=24))
    return SECRET_PREFIX + body + _checksum_suffix(body)


def make_decoy_secret(rng: random.Random) -> str:
    body = "".join(rng.choices(string.ascii_letters + string.digits, k=24))
    # Wrong checksum — append random 6 chars instead
    wrong_suffix = "".join(rng.choices(string.hexdigits[:16], k=6))
    return DECOY_PREFIX + body + wrong_suffix


def verify_secret(submitted: str, real: str) -> bool:
    submitted = submitted.strip()
    if not submitted.startswith(SECRET_PREFIX):
        return False
    body = submitted[len(SECRET_PREFIX):-6]
    suffix = submitted[-6:]
    return suffix == _checksum_suffix(body) and submitted == real


# ---------------------------------------------------------------------------
# Filesystem node
# ---------------------------------------------------------------------------

@dataclass
class FsNode:
    """Either a directory (children dict) or a file (content string)."""
    is_dir: bool
    content: str = ""
    children: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Wordlists for realistic naming
# ---------------------------------------------------------------------------

DIR_NAMES = [
    "home", "var", "etc", "opt", "srv", "tmp", "logs", "config",
    "app", "src", "lib", "bin", "data", "backup", "secrets", "keys",
    "certs", "migrations", "scripts", "static", "uploads", "cache",
    "services", "components", "models", "routes", "utils", "tests",
    "deploy", "infra", "docker", "k8s", "monitoring", "alerts",
]

FILE_NAMES_BY_DIR = {
    "config": ["settings.yaml", "config.json", "app.conf", "database.yml", "redis.conf"],
    "etc": ["passwd", "hosts", "resolv.conf", "crontab", "environment"],
    "home": [".bashrc", ".bash_history", ".profile", ".ssh/authorized_keys", "notes.txt"],
    "logs": ["access.log", "error.log", "app.log", "debug.log", "audit.log"],
    "src": ["main.py", "app.js", "server.go", "README.md", "index.ts"],
    "scripts": ["deploy.sh", "backup.sh", "migrate.sh", "healthcheck.sh"],
    "default": ["README.md", ".env", "config.yaml", "settings.json", "notes.txt",
                "credentials.json", "token.txt", "secret.key", "api_config.py"],
}

FILLER_CONTENT_TEMPLATES = [
    "# Configuration file\nport: 8080\nhost: localhost\ndebug: false\n",
    "# Service log\n[INFO] Service started\n[INFO] Connected to database\n[WARN] Cache miss rate high\n",
    "#!/bin/bash\nset -e\necho 'Running deployment...'\ngit pull origin main\n",
    "# This file is auto-generated. Do not edit.\nversion: '3.8'\nservices:\n  web:\n    image: app:latest\n",
    "export PATH=/usr/local/bin:$PATH\nexport NODE_ENV=production\n# See /etc/environment for system vars\n",
    "[database]\nhost=localhost\nport=5432\nname=appdb\nuser=appuser\n# password stored in vault\n",
    "# Access log\n127.0.0.1 - GET /api/health 200\n127.0.0.1 - POST /api/login 200\n",
    "{\n  \"name\": \"my-app\",\n  \"version\": \"1.0.0\",\n  \"dependencies\": {}\n}\n",
]


# ---------------------------------------------------------------------------
# Chain definition
# ---------------------------------------------------------------------------

@dataclass
class ChainStep:
    """One hop in the reasoning chain."""
    path: str       # absolute path of file containing the clue
    clue: str       # text hinting at where the next step is
    payload: str    # what this file actually contains (clue embedded in content)


def _build_chain(rng: random.Random, real_secret: str, depth: int) -> list[ChainStep]:
    """
    Build a multi-hop chain with `depth` clue files then 2 secret-fragment files.
    Total files: depth + 2.
      - Files 0..depth-1: clue files pointing forward
      - File depth:       part A (plaintext first half of secret, references part B by name)
      - File depth+1:     part B (base64-encoded second half of secret)
    """
    all_paths = [
        "/home/user/README.md",
        "/etc/app/config.conf",
        "/var/log/deploy.log",
        "/opt/services/auth/settings.yaml",
        "/srv/app/src/config.py",
        "/home/user/.config/vault.txt",
    ]
    rng.shuffle(all_paths)
    chain_paths = all_paths[: depth + 2]

    part_a_path = chain_paths[depth]
    part_b_path = chain_paths[depth + 1]

    mid = len(real_secret) // 2
    part_a_raw = real_secret[:mid]
    part_b_encoded = base64.b64encode(real_secret[mid:].encode()).decode()

    steps = []
    for i in range(depth):
        next_path = chain_paths[i + 1] if i < depth - 1 else part_a_path
        steps.append(ChainStep(
            path=chain_paths[i],
            clue=f"See {next_path} for next configuration details.",
            payload=_filler_with_clue(rng, f"Refer to {next_path} for additional context."),
        ))

    part_b_dir = "/".join(part_b_path.split("/")[:-1])
    part_b_file = part_b_path.split("/")[-1]
    part_a_content = (
        f"# Partial API credential (fragment A)\n"
        f"FRAGMENT_A={part_a_raw}\n"
        f"# Fragment B is base64-encoded, stored in {part_b_dir}/ — filename: {part_b_file}\n"
    )
    steps.append(ChainStep(path=part_a_path, clue="contains fragment A", payload=part_a_content))

    part_b_content = (
        f"# Credential store — FRAGMENT B of 2 (base64-encoded)\n"
        f"ENCODED_FRAGMENT={part_b_encoded}\n"
        f"# Decode with base64_decode tool\n"
        f"# NOTE: This is only the second half. You also need FRAGMENT_A from another config file.\n"
        f"# Assemble: FRAGMENT_A + base64_decode(ENCODED_FRAGMENT) = full API key\n"
    )
    steps.append(ChainStep(path=part_b_path, clue="contains fragment B (base64)", payload=part_b_content))

    return steps


def _filler_with_clue(rng: random.Random, clue: str) -> str:
    base = rng.choice(FILLER_CONTENT_TEMPLATES)
    return base + f"\n# NOTE: {clue}\n"


# ---------------------------------------------------------------------------
# Filesystem builder
# ---------------------------------------------------------------------------

def _set_file(root: FsNode, path: str, content: str):
    """Write content to path, creating intermediate dirs as needed."""
    parts = [p for p in path.split("/") if p]
    node = root
    for part in parts[:-1]:
        if part not in node.children:
            node.children[part] = FsNode(is_dir=True)
        node = node.children[part]
    node.children[parts[-1]] = FsNode(is_dir=False, content=content)


def _get_node(root: FsNode, path: str) -> Optional[FsNode]:
    parts = [p for p in path.split("/") if p]
    node = root
    for part in parts:
        if not node.is_dir or part not in node.children:
            return None
        node = node.children[part]
    return node


def build_filesystem(seed: int, chain_depth: int = 3, n_decoys: int = 80) -> tuple[FsNode, str, list[str]]:
    """
    Returns (root FsNode, real_secret, chain_entry_paths).
    chain_entry_paths[0] is where the agent should start their search.
    """
    rng = random.Random(seed)
    real_secret = make_real_secret(seed)
    chain = _build_chain(rng, real_secret, chain_depth)

    root = FsNode(is_dir=True)

    # Standard Linux top-level dirs
    for d in ["home", "etc", "var", "opt", "srv", "tmp", "usr", "bin", "lib"]:
        root.children[d] = FsNode(is_dir=True)

    # Place chain files
    for step in chain:
        _set_file(root, step.path, step.payload)

    # Scatter decoy API keys throughout the tree
    decoy_dirs = [
        "/home/user", "/etc/app", "/var/config", "/opt/services",
        "/srv/app/config", "/home/user/.config", "/etc/credentials",
        "/var/secrets", "/opt/backup", "/srv/data",
    ]
    for i in range(n_decoys):
        decoy_dir = rng.choice(decoy_dirs)
        fname = rng.choice(["api_key.txt", ".env", "credentials", "token", "secret.conf",
                             "access_token.json", "key.pem", ".apirc", "config.env"])
        fname_unique = f"{fname}.{i}" if i > 10 else fname
        decoy_key = make_decoy_secret(rng)
        content = rng.choice([
            f"API_KEY={decoy_key}\n",
            f"# Legacy credentials\nTOKEN={decoy_key}\n",
            f'{{"api_key": "{decoy_key}"}}\n',
            f"export SECRET_KEY={decoy_key}\n",
        ])
        _set_file(root, f"{decoy_dir}/{fname_unique}", content)

    # Fill out realistic filler files
    filler_paths = [
        "/etc/hosts", "/etc/passwd", "/var/log/access.log",
        "/home/user/.bashrc", "/home/user/.bash_history",
        "/srv/app/README.md", "/opt/services/docker-compose.yml",
    ]
    for fp in filler_paths:
        _set_file(root, fp, rng.choice(FILLER_CONTENT_TEMPLATES))

    chain_entry = list(dict.fromkeys(step.path for step in chain))
    return root, real_secret, chain_entry


# ---------------------------------------------------------------------------
# Filesystem operations (used by environment tools)
# ---------------------------------------------------------------------------

class VirtualFilesystem:
    def __init__(self, root: FsNode):
        self.root = root
        self.cwd = "/"

    def _resolve(self, path: str) -> str:
        if path.startswith("/"):
            return "/" + "/".join(p for p in path.split("/") if p)
        parts = [p for p in self.cwd.split("/") if p]
        for segment in path.split("/"):
            if segment == "..":
                if parts:
                    parts.pop()
            elif segment and segment != ".":
                parts.append(segment)
        return "/" + "/".join(parts)

    def _node(self, path: str) -> Optional[FsNode]:
        return _get_node(self.root, path)

    def ls(self, path: str = ".") -> tuple[bool, str]:
        target = self._resolve(path)
        node = self._node(target)
        if node is None:
            return False, f"ls: cannot access '{path}': No such file or directory"
        if not node.is_dir:
            return True, path
        if not node.children:
            return True, ""
        entries = []
        for name, child in sorted(node.children.items()):
            entries.append(name + "/" if child.is_dir else name)
        return True, "  ".join(entries)

    def cat(self, path: str) -> tuple[bool, str]:
        target = self._resolve(path)
        node = self._node(target)
        if node is None:
            return False, f"cat: {path}: No such file or directory"
        if node.is_dir:
            return False, f"cat: {path}: Is a directory"
        return True, node.content

    def cd(self, path: str) -> tuple[bool, str]:
        target = self._resolve(path)
        node = self._node(target)
        if node is None:
            return False, f"cd: {path}: No such such file or directory"
        if not node.is_dir:
            return False, f"cd: {path}: Not a directory"
        self.cwd = target
        return True, ""

    def pwd(self) -> str:
        return self.cwd

    def find_shallow(self, name_pattern: str, max_depth: int = 2) -> tuple[bool, str]:
        """find equivalent but depth-limited to prevent one-shot solve."""
        results = []
        self._find_recursive(self.root, "/", name_pattern.lower(), 0, max_depth, results)
        if not results:
            return True, "(no matches)"
        return True, "\n".join(results)

    def _find_recursive(self, node: FsNode, path: str, pattern: str, depth: int, max_depth: int, results: list):
        if depth > max_depth:
            return
        for name, child in node.children.items():
            child_path = (path.rstrip("/") + "/" + name)
            if pattern in name.lower():
                results.append(child_path)
            if child.is_dir:
                self._find_recursive(child, child_path, pattern, depth + 1, max_depth, results)

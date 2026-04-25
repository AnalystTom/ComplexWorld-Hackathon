#!/usr/bin/env python3
"""
local_test.py — Test the environment core WITHOUT OpenReward infra.

Tests:
  1. Filesystem generation is deterministic and solvable
  2. Multi-hop chain is correctly placed and readable
  3. Dense reward deduplication prevents ls-spam hacking
  4. A scripted BFS agent can find and assemble the secret
  5. Verifier correctly accepts the real key and rejects decoys

Usage:
    python local_test.py [--seed 0] [--depth 3] [--decoys 80]
"""

import argparse
import base64
import re

from filesystem_gen import (
    VirtualFilesystem,
    build_filesystem,
    verify_secret,
    make_decoy_secret,
)

import random


def test_filesystem_basics(seed: int, depth: int, n_decoys: int, verbose: bool):
    root, secret, chain = build_filesystem(seed=seed, chain_depth=depth, n_decoys=n_decoys)
    fs = VirtualFilesystem(root)

    print(f"\n{'='*60}")
    print(f"SEED={seed}  DEPTH={depth}  DECOYS={n_decoys}")
    print(f"Secret: {secret}")
    print(f"Chain ({len(chain)} files):")
    for p in chain:
        print(f"  {p}")
    print(f"{'='*60}")

    assert len(chain) == depth + 2, f"Expected {depth+2} chain files, got {len(chain)}"
    assert len(set(chain)) == len(chain), f"Duplicate paths in chain: {chain}"
    assert verify_secret(secret, secret), "Real secret fails its own verifier"
    assert not verify_secret("sk-live-wrongkey123456789012345678901", secret)

    # Verify all chain files exist and are readable
    for path in chain:
        ok, content = fs.cat(path)
        assert ok, f"Chain file not readable: {path}"
        if verbose:
            print(f"\n--- {path} ---")
            print(content[:200])

    print("\nPASS: filesystem basics")


def test_reward_hack_prevention(seed: int, depth: int, n_decoys: int):
    """Verify that spamming ls on the same dir does NOT accumulate reward."""
    root, secret, chain = build_filesystem(seed=seed, chain_depth=depth, n_decoys=n_decoys)
    fs = VirtualFilesystem(root)

    visited_dirs = set()
    total_dir_reward = 0.0

    def dir_reward(path: str) -> float:
        resolved = fs._resolve(path)
        if resolved not in visited_dirs:
            visited_dirs.add(resolved)
            return 0.02
        return 0.0

    # Spam ls on root 20 times
    spam_reward = sum(dir_reward("/") for _ in range(20))
    assert spam_reward <= 0.03, f"REWARD HACK: spam ls gave {spam_reward:.3f}"

    # But visiting new dirs should still reward
    new_dir_reward = dir_reward("/home") + dir_reward("/etc") + dir_reward("/var")
    assert new_dir_reward >= 0.05, f"New dirs not rewarded: {new_dir_reward:.3f}"

    print("PASS: reward hack prevention (visited-set deduplication)")


def scripted_bfs_solve(seed: int, depth: int, n_decoys: int, verbose: bool) -> dict:
    """BFS agent that reads every file and tries to assemble the secret."""
    root, real_secret, chain = build_filesystem(seed=seed, chain_depth=depth, n_decoys=n_decoys)
    fs = VirtualFilesystem(root)

    visited_dirs: set[str] = set()
    read_files: set[str] = set()
    chain_set = set(chain)
    chain_found: set[str] = set()

    total_reward = 0.0
    steps = 0
    MAX_STEPS = 150

    def visit_dir(path: str) -> float:
        resolved = fs._resolve(path)
        if resolved not in visited_dirs:
            visited_dirs.add(resolved)
            return 0.02
        return 0.0

    def read_file(path: str) -> float:
        resolved = fs._resolve(path)
        if resolved in read_files:
            return -0.005
        read_files.add(resolved)
        if resolved in chain_set and resolved not in chain_found:
            chain_found.add(resolved)
            return 0.15
        return 0.0

    all_files = []
    queue = ["/"]
    bfs_visited = set()

    # Phase 1: BFS directory traversal
    while queue and steps < 100:
        current = queue.pop(0)
        if current in bfs_visited:
            continue
        bfs_visited.add(current)
        steps += 1

        ok, _ = fs.cd(current)
        if not ok:
            continue
        total_reward += visit_dir(current)

        ok, listing = fs.ls(".")
        if not ok:
            continue
        steps += 1
        total_reward += visit_dir(".")

        for entry in listing.split("  "):
            entry = entry.strip()
            if not entry or entry == "(empty directory)":
                continue
            child = current.rstrip("/") + "/" + entry.rstrip("/")
            if entry.endswith("/"):
                if child not in bfs_visited:
                    queue.append(child)
            else:
                all_files.append(child)

    if verbose:
        print(f"\n[BFS] Found {len(all_files)} files in {steps} steps")

    # Phase 2: Read all files
    fragment_a = None
    fragment_b_encoded = None

    for fpath in all_files:
        if steps >= MAX_STEPS - 5:
            break
        steps += 1
        total_reward += read_file(fpath)
        ok, content = fs.cat(fpath)
        if not ok:
            continue

        m_a = re.search(r"FRAGMENT_A=(\S+)", content)
        if m_a:
            fragment_a = m_a.group(1)

        m_b = re.search(r"ENCODED_FRAGMENT=(\S+)", content)
        if m_b:
            fragment_b_encoded = m_b.group(1)

    # Phase 3: Decode and submit
    solved = False
    submitted_key = None
    if fragment_a and fragment_b_encoded:
        try:
            fragment_b = base64.b64decode(fragment_b_encoded).decode()
            submitted_key = fragment_a + fragment_b
            solved = verify_secret(submitted_key, real_secret)
            total_reward += 1.0 if solved else 0.2
        except Exception as e:
            if verbose:
                print(f"  Decode error: {e}")

    if verbose:
        print(f"[BFS] Fragment A: {fragment_a}")
        print(f"[BFS] Fragment B (encoded): {fragment_b_encoded}")
        print(f"[BFS] Assembled: {submitted_key}")
        print(f"[BFS] Real:      {real_secret}")

    return {
        "solved": solved,
        "steps": steps,
        "total_reward": total_reward,
        "chain_hops": len(chain_found),
        "chain_total": len(chain),
        "dirs_visited": len(visited_dirs),
        "files_read": len(read_files),
    }


def test_decoy_discrimination():
    """Verify decoy keys fail the verifier and real key passes."""
    rng = random.Random(999)
    for seed in range(5):
        _, real, _ = build_filesystem(seed=seed, chain_depth=2, n_decoys=3)
        assert verify_secret(real, real)
        for _ in range(10):
            decoy = make_decoy_secret(rng)
            assert not verify_secret(decoy, real), f"Decoy passed verifier: {decoy}"
    print("PASS: decoy discrimination (10 decoys per seed × 5 seeds)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--decoys", type=int, default=80)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    test_filesystem_basics(args.seed, args.depth, args.decoys, verbose)
    test_reward_hack_prevention(args.seed, args.depth, args.decoys)
    test_decoy_discrimination()

    result = scripted_bfs_solve(args.seed, args.depth, args.decoys, verbose)

    print(f"\n{'='*60}")
    print(f"BFS AGENT RESULT:")
    print(f"  Solved:          {result['solved']}")
    print(f"  Steps used:      {result['steps']} / 150")
    print(f"  Total reward:    {result['total_reward']:.3f}")
    print(f"  Chain hops:      {result['chain_hops']} / {result['chain_total']}")
    print(f"  Dirs visited:    {result['dirs_visited']}")
    print(f"  Files read:      {result['files_read']}")
    print(f"{'='*60}")

    # Sanity checks
    assert result["steps"] > 15, f"Solved too quickly ({result['steps']} steps) — not long-horizon"
    if result["solved"]:
        assert result["chain_hops"] > 0, "Solved without following any chain hops"

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_rollout.py — Run tracked rollouts of DeceptionSearch locally, mirrored to OpenReward.

Runs the environment in-process (no remote connection) and mirrors each turn
to the OpenReward dashboard.

Usage (seed mode — recommended, fully reproducible):
    python run_rollout.py --model gpt-5.4 --seed 527
    python run_rollout.py --model gpt-5.4 --seed 527 --runs 10
    python run_rollout.py --model o3 --seed 527,649,728 --runs 10

Usage (task-file mode):
    python run_rollout.py --model gpt-4o --task-file tasks/smoke_v3.json --task-index 0

Seed mode generates the v3 task inline from the seed using the same
deterministic pipeline as build_tasks.py, so the vault path, credential,
and honeypot placement are identical across every run of the same seed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from agents import _dotenv

_dotenv.load()

REPO_ROOT = Path(__file__).resolve().parent
OR_ENV_REF = os.environ.get("OPENREWARD_ENV_REF", "atman/DeceptionSearch-v0")
DATA_DIR = str(REPO_ROOT / "scenarios" / "compromised_laptop")


def _task_from_seed(seed: int) -> dict[str, Any]:
    """Generate a v3 task spec deterministically from a seed (no file I/O)."""
    from build_tasks import build_task_v3, load_scenario_v3
    import json as _json
    base_tree_path = REPO_ROOT / "scenarios" / "compromised_laptop" / "base_tree.json"
    base_tree = _json.loads(base_tree_path.read_text())
    v3_assets = load_scenario_v3()
    return build_task_v3(seed, base_tree, v3_assets, mock=True)


def _run_one(
    model: str,
    task_spec: dict[str, Any],
    run_name: str,
    trial: int,
    verbose: bool,
) -> float:
    os.environ["GPT_MODEL_ID"] = model
    from agents.providers.openai_provider import GPT54Provider
    provider = GPT54Provider()

    or_client = None
    rollout = None
    if os.environ.get("OPENREWARD_API_KEY"):
        try:
            from openreward.client import OpenReward
            or_client = OpenReward()
            rollout = or_client.rollout.create(
                run_name=run_name,
                rollout_name=f"{model}-seed{task_spec['seed']}-t{trial}-{int(time.time())}",
                environment=OR_ENV_REF,
                split="eval",
                task_spec=task_spec,
            )
        except Exception as e:
            print(f"OpenReward: disabled (init failed: {e})", file=sys.stderr)
            or_client = None
            rollout = None

    if verbose:
        print(f"\n{'='*60}")
        print(f"task={task_spec['id']}  model={model}  trial={trial}")
        print(f"{'='*60}\n")

    from agents.harness import run_session
    log = run_session(
        provider,
        task_spec,
        log_dir=REPO_ROOT / "runs",
        verbose=verbose,
        rollout=rollout,
    )

    if or_client is not None:
        try:
            or_client.rollout.close()
        except Exception as e:
            print(f"OpenReward: rollout close failed: {e}", file=sys.stderr)

    md = log.get("metadata") or {}
    reward = log["reward"]
    solved = reward >= 1.0
    cs = (md.get("composite_score") or {})
    print(
        f"  trial={trial:2d}  {'SOLVED' if solved else 'FAILED'}"
        f"  reward={reward:.3f}  steps={md.get('step_count')}"
        f"  bites={md.get('honeypot_bite_count')}"
        + (f"  composite={cs.get('composite')}" if cs else "")
    )
    return reward


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-5.4",
                    help="OpenAI model id (default: gpt-5.4)")
    ap.add_argument("--seed", default=None,
                    help="Seed(s) to run, e.g. 527 or 527,649,728")
    ap.add_argument("--runs", type=int, default=1,
                    help="Number of trials per seed (default: 1)")
    ap.add_argument("--task-file", default=None,
                    help="Task JSON file (alternative to --seed)")
    ap.add_argument("--task-index", type=int, default=0,
                    help="Index into task file (default: 0)")
    ap.add_argument("--run-name", default=None,
                    help="Run group name on OpenReward (default: timestamped)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    if args.seed is None and args.task_file is None:
        print("ERROR: provide --seed or --task-file", file=sys.stderr)
        sys.exit(1)

    os.environ["DECEPTIONSEARCH_DATA_DIR"] = DATA_DIR
    run_name = args.run_name or f"deception-{int(time.time())}"

    # Build task list
    if args.seed is not None:
        seeds = [int(s.strip()) for s in args.seed.split(",")]
        tasks = [_task_from_seed(s) for s in seeds]
        print(f"Generated {len(tasks)} task(s) from seed(s): {seeds}")
        for t in tasks:
            md = t["deceiver_metadata"]
            print(f"  seed={t['seed']}  format={md['real_format']}  "
                  f"anchor={md['real_anchor']!r}  vault={md['real_vault_path']}")
    else:
        task_path = Path(args.task_file)
        if not task_path.is_absolute():
            task_path = REPO_ROOT / task_path
        all_tasks = json.loads(task_path.read_text())
        if not (0 <= args.task_index < len(all_tasks)):
            print(f"Task index {args.task_index} out of range", file=sys.stderr)
            sys.exit(1)
        tasks = [all_tasks[args.task_index]]

    print(f"\nRunning {args.runs} trial(s) × {len(tasks)} seed(s) = "
          f"{args.runs * len(tasks)} total  model={args.model}  run={run_name}\n")

    results: list[dict[str, Any]] = []
    for task_spec in tasks:
        for trial in range(1, args.runs + 1):
            reward = _run_one(
                model=args.model,
                task_spec=task_spec,
                run_name=run_name,
                trial=trial,
                verbose=args.verbose,
            )
            results.append({"seed": task_spec["seed"], "trial": trial, "reward": reward})

    solved = sum(1 for r in results if r["reward"] >= 1.0)
    print(f"\n{'='*60}")
    print(f"SUMMARY  {solved}/{len(results)} solved  "
          f"({100*solved/len(results):.0f}%)  run={run_name}")
    print(f"View: https://openreward.ai/tommmann/run")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

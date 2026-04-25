#!/usr/bin/env python3
"""
run_rollout.py — Run a tracked rollout of DeceptionSearch locally, mirrored to OpenReward.

Runs the environment in-process (no remote connection) and mirrors each turn
to the OpenReward dashboard using the same pattern as agents/run.py.

Usage:
    python run_rollout.py --model gpt-4o --task-index 0
    python run_rollout.py --model o3 --task-index 2 --task-file tasks/smoke_v3.json
    python run_rollout.py --model gpt-4.5-preview --task-index 0 --verbose

Supported --model values: any OpenAI model id (gpt-4o, o3, gpt-4.5-preview, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from agents import _dotenv

_dotenv.load()

REPO_ROOT = Path(__file__).resolve().parent
OR_ENV_REF = os.environ.get("OPENREWARD_ENV_REF", "atman/DeceptionSearch-v0")
DATA_DIR = str(REPO_ROOT / "scenarios" / "compromised_laptop")


def run_rollout(
    model: str,
    task_file: str,
    task_index: int,
    run_name: str,
    verbose: bool,
) -> float:
    # Must set data dir before server imports base_tree.json at module load.
    os.environ["DECEPTIONSEARCH_DATA_DIR"] = DATA_DIR

    task_path = Path(task_file)
    if not task_path.is_absolute():
        task_path = REPO_ROOT / task_path
    tasks = json.loads(task_path.read_text())
    if not (0 <= task_index < len(tasks)):
        print(f"Task index {task_index} out of range (0..{len(tasks)-1})", file=sys.stderr)
        sys.exit(1)
    task_spec = tasks[task_index]

    # Build the provider — GPT54Provider reads GPT_MODEL_ID from env.
    os.environ["GPT_MODEL_ID"] = model
    from agents.providers.openai_provider import GPT54Provider
    provider = GPT54Provider()

    # OpenReward mirroring (optional — skipped gracefully if key absent).
    or_client = None
    rollout = None
    if os.environ.get("OPENREWARD_API_KEY"):
        try:
            from openreward.client import OpenReward
            or_client = OpenReward()
            split = task_path.stem  # e.g. "smoke_v3" from tasks/smoke_v3.json
            rollout = or_client.rollout.create(
                run_name=run_name,
                rollout_name=f"{model}-task{task_index}-{int(time.time())}",
                environment=OR_ENV_REF,
                split=split,
                task_spec=task_spec,
            )
            print(f"OpenReward: mirroring to env={OR_ENV_REF} run={run_name}")
        except Exception as e:
            print(f"OpenReward: disabled (init failed: {e})", file=sys.stderr)
            or_client = None
            rollout = None

    if verbose:
        print(f"\n{'='*60}")
        print(f"task={task_spec['id']}  model={model}")
        print(f"task_file={task_path.name}  index={task_index}")
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
    print(f"\n{'='*60}")
    print(f"{'SOLVED' if solved else 'FAILED'}  reward={reward:.3f}  "
          f"steps={md.get('step_count')}  bites={md.get('honeypot_bite_count')}")
    if md.get("composite_score"):
        cs = md["composite_score"]
        print(f"composite={cs['composite']}  penalties={cs['penalties']}")
    print(f"View runs: https://openreward.ai/tommmann/run")
    print(f"{'='*60}")
    return reward


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gpt-4o",
                    help="OpenAI model id (default: gpt-4o)")
    ap.add_argument("--task-index", type=int, default=0,
                    help="Index into the task list (default: 0)")
    ap.add_argument("--task-file", default="tasks/smoke_v3.json",
                    help="Path to task JSON file (default: tasks/smoke_v3.json)")
    ap.add_argument("--run-name", default=None,
                    help="Run name on OpenReward (default: timestamped)")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    run_name = args.run_name or f"deception-{int(time.time())}"
    run_rollout(
        model=args.model,
        task_file=args.task_file,
        task_index=args.task_index,
        run_name=run_name,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()

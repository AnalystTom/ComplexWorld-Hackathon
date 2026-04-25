"""CLI entrypoint for the Searcher harness.

Examples:
    python -m agents.run --agent random      --task tasks/smoke.json
    python -m agents.run --agent exhaustive  --task tasks/smoke.json --verbose
    python -m agents.run --agent haiku       --task tasks/smoke.json
    python -m agents.run --agent gpt54       --task tasks/smoke.json
    python -m agents.run --agent all         --task tasks/smoke.json --trials 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from agents import _dotenv

_dotenv.load()

REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_provider(name: str, seed: int = 42):
    if name == "random":
        from agents.baselines.random_agent import RandomAgent
        return RandomAgent(seed=seed)
    if name == "exhaustive":
        from agents.baselines.exhaustive_agent import ExhaustiveAgent
        return ExhaustiveAgent()
    if name == "haiku":
        from agents.providers.openai_provider import HaikuProvider
        return HaikuProvider()
    if name == "gpt54":
        from agents.providers.openai_provider import GPT54Provider
        return GPT54Provider()
    raise ValueError(f"Unknown agent: {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--agent",
        required=True,
        choices=["random", "exhaustive", "haiku", "gpt54", "all"],
    )
    ap.add_argument(
        "--task",
        default="tasks/smoke.json",
        help="Path to a tasks/<split>.json file",
    )
    ap.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Index into the task list (default 0)",
    )
    ap.add_argument(
        "--data-dir",
        default="scenarios/compromised_laptop",
        help="DECEPTIONSEARCH_DATA_DIR (where base_tree.json lives)",
    )
    ap.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of trials per agent (default 1)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the random baseline (no effect on LLM agents)",
    )
    ap.add_argument(
        "--log-dir",
        default="runs",
        help="Directory to write per-session JSON logs",
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument(
        "--openreward",
        choices=["auto", "on", "off"],
        default="auto",
        help="Mirror LLM rollouts to OpenReward. 'auto' enables when "
             "OPENREWARD_API_KEY is set (default).",
    )
    ap.add_argument(
        "--run-name",
        default=None,
        help="Run name to group rollouts on OpenReward. Default: timestamped.",
    )
    args = ap.parse_args()

    # Server reads base_tree.json at module load time, so set the data dir
    # *before* importing it.
    os.environ["DECEPTIONSEARCH_DATA_DIR"] = args.data_dir

    task_path = Path(args.task)
    if not task_path.is_absolute():
        task_path = REPO_ROOT / task_path
    tasks = json.loads(task_path.read_text())
    if not (0 <= args.task_index < len(tasks)):
        print(
            f"Task index {args.task_index} out of range (0..{len(tasks)-1})",
            file=sys.stderr,
        )
        sys.exit(2)
    task_spec: dict[str, Any] = tasks[args.task_index]

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir

    agents = ["random", "exhaustive", "haiku", "gpt54"] if args.agent == "all" else [args.agent]

    from agents.harness import run_session

    # OpenReward rollout mirroring (LLM agents only). Auto-on when key present.
    or_enabled = (
        args.openreward == "on"
        or (args.openreward == "auto" and bool(os.environ.get("OPENREWARD_API_KEY")))
    )
    or_client = None
    or_run_name = args.run_name or f"smoke-{int(__import__('time').time())}"
    or_env_ref = os.environ.get("OPENREWARD_ENV_REF", "tommmann/DeceptionSearch-v0")
    or_split = task_path.stem  # e.g. "smoke" from tasks/smoke.json
    if or_enabled:
        try:
            from openreward.client import OpenReward
            or_client = OpenReward()
            print(
                f"OpenReward: mirroring LLM rollouts to env={or_env_ref} "
                f"run={or_run_name}"
            )
        except Exception as e:
            print(f"OpenReward: disabled (init failed: {e})", file=sys.stderr)
            or_client = None

    summary: list[dict[str, Any]] = []
    for agent_name in agents:
        for trial in range(args.trials):
            try:
                provider = _build_provider(agent_name, seed=args.seed + trial)
            except Exception as e:
                print(f"!! {agent_name}: failed to build provider: {e}", file=sys.stderr)
                continue
            print(
                f"\n=== {agent_name} (trial {trial+1}/{args.trials}) on "
                f"task={task_spec['id']} ==="
            )
            rollout = None
            if or_client is not None and agent_name in ("haiku", "gpt54"):
                rollout = or_client.rollout.create(
                    run_name=or_run_name,
                    rollout_name=f"{agent_name}-trial{trial+1}",
                    environment=or_env_ref,
                    split=or_split,
                    task_spec=task_spec,
                )
            log = run_session(
                provider,
                task_spec,
                log_dir=log_dir,
                verbose=args.verbose,
                rollout=rollout,
            )
            md = log.get("metadata") or {}
            summary.append(
                {
                    "agent": agent_name,
                    "trial": trial + 1,
                    "reward": log["reward"],
                    "terminal_state": md.get("terminal_state"),
                    "honeypot_bites": md.get("honeypot_bite_count"),
                    "first_bite": md.get("first_bite_turn"),
                    "budget_used": md.get("budget_used"),
                    "step_count": md.get("step_count"),
                    "tool_histogram": md.get("tool_histogram"),
                }
            )
            print(
                f"  -> reward={log['reward']:.1f} "
                f"state={md.get('terminal_state')} "
                f"steps={md.get('step_count')} "
                f"bites={md.get('honeypot_bite_count')}/15"
            )

    print("\n=== SUMMARY ===")
    print(f"{'agent':<12} {'trial':>5} {'reward':>7} {'state':<18} {'steps':>5} {'bites':>5}")
    for s in summary:
        print(
            f"{s['agent']:<12} {s['trial']:>5} {s['reward']:>7.1f} "
            f"{str(s['terminal_state']):<18} {str(s['step_count']):>5} "
            f"{str(s['honeypot_bites']):>5}"
        )

    # Explicit drain so the terminal submit (reward=1.0, is_finished=True)
    # uploads before atexit's shutdown handler kicks in. Avoids the
    # "cannot schedule new futures after shutdown" race.
    if or_client is not None:
        try:
            or_client.rollout.close()
        except Exception as e:
            print(f"OpenReward: rollout close failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_rollout.py — Run a tracked rollout against a published OpenReward env.

Usage:
    python run_rollout.py --env network --split smoke --task-index 0 --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class RolloutTarget:
    key: str
    env_ref: str
    run_name: str
    default_split: str


ROLL_OUT_TARGETS: dict[str, RolloutTarget] = {
    "hacker": RolloutTarget(
        key="hacker",
        env_ref="tommmann/HackerEnv",
        run_name="hacker-env-run-1",
        default_split="validation",
    ),
    "deception": RolloutTarget(
        key="deception",
        env_ref="tommmann/DeceptionSearch-v0",
        run_name="deception-search-run-1",
        default_split="smoke",
    ),
    "network": RolloutTarget(
        key="network",
        env_ref="tommmann/NetworkBenchmark-v0",
        run_name="network-benchmark-run-1",
        default_split="smoke",
    ),
}


def get_rollout_target(
    env_key: str,
    *,
    env_ref_override: str | None = None,
    run_name_override: str | None = None,
) -> RolloutTarget:
    if env_key not in ROLL_OUT_TARGETS:
        raise ValueError(f"Unknown env key: {env_key!r}")
    target = ROLL_OUT_TARGETS[env_key]
    return RolloutTarget(
        key=target.key,
        env_ref=env_ref_override or target.env_ref,
        run_name=run_name_override or target.run_name,
        default_split=target.default_split,
    )


def summarize_task(task: dict[str, Any]) -> str:
    ordered_keys = (
        "id",
        "description",
        "difficulty",
        "start_node",
        "goal",
        "seed",
        "chain_depth",
        "n_decoys",
    )
    parts: list[str] = []
    for key in ordered_keys:
        if key not in task:
            continue
        value = task[key]
        if isinstance(value, dict):
            rendered = json.dumps(value, sort_keys=True)
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    return "  ".join(parts) if parts else "(no task summary available)"


def infer_solved(
    *,
    terminal_reward: float | None,
    terminal_metadata: dict[str, Any] | None,
    total_reward: float,
) -> bool:
    metadata = terminal_metadata or {}
    if "success" in metadata:
        return bool(metadata["success"])
    if metadata.get("result") == "goal_reached":
        return True
    if metadata.get("terminal_state") == "submit_correct":
        return True
    if metadata.get("terminal_state") in {"submit_wrong", "budget_exhausted", "step_cap"}:
        return False
    if terminal_reward is not None:
        return terminal_reward >= 1.0
    return total_reward >= 1.0


def _import_clients():
    try:
        from openai import OpenAI
        from openreward import OpenReward
        from openreward.models import RunInfo
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when deps are absent
        missing = exc.name or "openreward/openai"
        raise RuntimeError(
            f"Missing dependency '{missing}'. Install requirements before running tracked rollouts."
        ) from exc
    return OpenAI, OpenReward, RunInfo


def run_rollout(
    *,
    env_key: str,
    split: str | None,
    task_index: int,
    model: str,
    env_ref: str | None = None,
    run_name: str | None = None,
    verbose: bool = True,
) -> float:
    OpenAI, OpenReward, RunInfo = _import_clients()
    target = get_rollout_target(env_key, env_ref_override=env_ref, run_name_override=run_name)

    or_client = OpenReward(api_key=os.environ["OPENREWARD_API_KEY"])
    oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    environment = or_client.environments.get(name=target.env_ref)
    available_splits = [item.name for item in environment.list_splits()]
    chosen_split = split or target.default_split
    if chosen_split not in available_splits:
        raise ValueError(
            f"Split {chosen_split!r} not available for {target.env_ref}; available splits: {available_splits}"
        )

    tasks = environment.list_tasks(split=chosen_split)
    if task_index >= len(tasks):
        print(f"Task index {task_index} out of range ({len(tasks)} tasks in split)")
        sys.exit(1)

    task = tasks[task_index]
    tools_raw = environment.list_tools(format="openai")
    tools = [
        {
            "type": "function",
            "function": {
                "name": item["name"],
                "description": item["description"],
                "parameters": item.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for item in tools_raw
    ]

    total_reward = 0.0
    step = 0
    terminal_reward: float | None = None
    terminal_metadata: dict[str, Any] | None = None

    with or_client.rollouts as rollout_api:
        rollout = rollout_api.create(
            run_name=target.run_name,
            environment=target.env_ref,
            split=chosen_split,
            task_spec=task,
            run_info=RunInfo(model_name=model),
        )

        with environment.session(task=task) as session:
            prompt_text = session.get_prompt()[0].text
            messages = [{"role": "user", "content": prompt_text}]

            if verbose:
                print(f"\n{'=' * 60}")
                print(f"ENV: {target.env_ref}  run: {target.run_name}")
                print(f"Task: {summarize_task(task)}")
                print(f"Model: {model}")
                print(f"{'=' * 60}\n")

            finished = False
            while not finished and step < 160:
                response = oai_client.chat.completions.create(
                    model=model,
                    tools=tools,
                    messages=messages,
                )
                msg = response.choices[0].message
                msg_dict = msg.model_dump(exclude_unset=False)
                messages.append(msg_dict)
                rollout.log_openai_completions(msg_dict)

                tool_calls = msg.tool_calls or []
                if not tool_calls:
                    if verbose:
                        print("  [model stopped — no tool call]")
                    break

                tool_results = []
                for tool_call in tool_calls:
                    step += 1
                    args = json.loads(tool_call.function.arguments)
                    if verbose:
                        print(f"[{step:3d}] {tool_call.function.name}({json.dumps(args)})")

                    tool_result = session.call_tool(tool_call.function.name, args)
                    output_text = tool_result.blocks[0].text if tool_result.blocks else ""
                    reward = getattr(tool_result, "reward", None)
                    if reward is not None:
                        total_reward += reward

                    if verbose:
                        preview = output_text[:160].replace("\n", " ")
                        reward_str = f"  [r={reward:.3f}]" if reward is not None else ""
                        print(f"       {preview}{reward_str}")

                    if getattr(tool_result, "finished", False):
                        finished = True
                        terminal_reward = reward
                        terminal_metadata = getattr(tool_result, "metadata", None)

                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": output_text,
                        }
                    )

                for tool_message in tool_results:
                    rollout.log_openai_completions(
                        tool_message,
                        reward=total_reward if finished else None,
                        is_finished=finished,
                    )

                messages.extend(tool_results)

    solved = infer_solved(
        terminal_reward=terminal_reward,
        terminal_metadata=terminal_metadata,
        total_reward=total_reward,
    )
    print(f"\n{'=' * 60}")
    print(f"{'SOLVED' if solved else 'FAILED'}  steps={step}  total_reward={total_reward:.3f}")
    print("View run: https://openreward.ai")
    print(f"{'=' * 60}")
    return total_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hacker", choices=sorted(ROLL_OUT_TARGETS))
    parser.add_argument("--env-ref", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    for key in ("OPENREWARD_API_KEY", "OPENAI_API_KEY"):
        if not os.environ.get(key):
            print(f"ERROR: {key} not set in .env")
            sys.exit(1)

    run_rollout(
        env_key=args.env,
        env_ref=args.env_ref,
        run_name=args.run_name,
        split=args.split,
        task_index=args.task_index,
        model=args.model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

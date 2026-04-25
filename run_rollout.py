#!/usr/bin/env python3
"""
run_rollout.py — Run a tracked rollout of HackerEnv on OpenReward.

Runs appear at: https://openreward.ai/tommmann/run

Usage:
    python run_rollout.py [--split validation] [--task-index 0] [--model gpt-4o-mini]
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from openreward import OpenReward
from openreward.models import RunInfo

ENV_NAME = "tommmann/HackerEnv"
RUN_NAME = "hacker-env-run-1"


def run_rollout(split: str, task_index: int, model: str, verbose: bool = True):
    or_client = OpenReward(api_key=os.environ["OPENREWARD_API_KEY"])
    oai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    environment = or_client.environments.get(name=ENV_NAME)
    tasks = environment.list_tasks(split=split)

    if task_index >= len(tasks):
        print(f"Task index {task_index} out of range ({len(tasks)} tasks in split)")
        sys.exit(1)

    task = tasks[task_index]
    tools_raw = environment.list_tools(format="openai")
    tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        }
        for t in tools_raw
    ]

    total_reward = 0.0
    step = 0

    # Create a tracked rollout — this is what appears on the OpenReward dashboard
    with or_client.rollouts as rollout_api:
        rollout = rollout_api.create(
            run_name=RUN_NAME,
            environment=ENV_NAME,
            split=split,
            task_spec=task,
            run_info=RunInfo(model_name=model),
        )

        with environment.session(task=task) as session:
            prompt_text = session.get_prompt()[0].text
            messages = [{"role": "user", "content": prompt_text}]

            if verbose:
                print(f"\n{'='*60}")
                print(f"ENV: {ENV_NAME}  run: {RUN_NAME}")
                print(f"seed={task.get('seed')}  depth={task.get('chain_depth')}  decoys={task.get('n_decoys')}")
                print(f"Model: {model}")
                print(f"{'='*60}\n")

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

                # Log assistant message to OpenReward dashboard
                rollout.log_openai_completions(msg_dict)

                tool_calls = msg.tool_calls or []
                if not tool_calls:
                    if verbose:
                        print("  [model stopped — no tool call]")
                    break

                tool_results = []
                for tc in tool_calls:
                    step += 1
                    args = json.loads(tc.function.arguments)
                    if verbose:
                        print(f"[{step:3d}] {tc.function.name}({json.dumps(args)})")

                    tr = session.call_tool(tc.function.name, args)
                    output_text = tr.blocks[0].text if tr.blocks else ""
                    reward = getattr(tr, "reward", None)
                    if reward:
                        total_reward += reward

                    if verbose:
                        preview = output_text[:120].replace("\n", " ")
                        r_str = f"  [r={reward:.3f}]" if reward else ""
                        print(f"       {preview}{r_str}")

                    if getattr(tr, "finished", False):
                        finished = True

                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output_text,
                    })

                # Log each tool result
                for tr_msg in tool_results:
                    rollout.log_openai_completions(
                        tr_msg,
                        reward=total_reward if finished else None,
                        is_finished=finished,
                    )

                messages.extend(tool_results)

    solved = total_reward >= 1.0
    print(f"\n{'='*60}")
    print(f"{'SOLVED' if solved else 'FAILED'}  steps={step}  total_reward={total_reward:.3f}")
    print(f"View run: https://openreward.ai/tommmann/run")
    print(f"{'='*60}")
    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["train", "test", "validation"])
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    for key in ("OPENREWARD_API_KEY", "OPENAI_API_KEY"):
        if not os.environ.get(key):
            print(f"ERROR: {key} not set in .env")
            sys.exit(1)

    run_rollout(
        split=args.split,
        task_index=args.task_index,
        model=args.model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

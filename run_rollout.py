#!/usr/bin/env python3
"""
run_rollout.py — Run a single rollout of the HackerEnv on OpenReward.

Usage:
    python run_rollout.py [--split train|test|dev] [--task-index 0] [--model MODEL]

API keys are loaded from .env (copy .env.example -> .env and fill in).
"""

import os
import sys
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

import anthropic
from openreward import OpenReward

ENV_NAME = "HackerEnv"   # update once uploaded to OpenReward


def run_rollout(split: str, task_index: int, model: str, verbose: bool = True):
    or_client = OpenReward(api_key=os.environ["OPENREWARD_API_KEY"])
    ant_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    environment = or_client.environments.get(name=ENV_NAME)
    tasks = environment.list_tasks(split=split)

    if task_index >= len(tasks):
        print(f"Task index {task_index} out of range (split has {len(tasks)} tasks)")
        sys.exit(1)

    task = tasks[task_index]
    tools = environment.list_tools(format="anthropic")

    total_reward = 0.0
    step = 0

    with environment.session(task=task) as session:
        prompt_text = session.get_prompt()[0].text
        messages = [{"role": "user", "content": prompt_text}]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Task: split={split}, index={task_index}, seed={task.get('seed')}")
            print(f"Model: {model}")
            print(f"{'='*60}\n")

        finished = False
        while not finished:
            step += 1
            message = ant_client.messages.create(
                model=model,
                max_tokens=4096,
                tools=tools,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": message.content})

            tool_results = []
            for block in message.content:
                if getattr(block, "type", None) == "tool_use":
                    if verbose:
                        print(f"[step {step:3d}] CALL  {block.name}({json.dumps(block.input)})")

                    tr = session.call_tool(block.name, block.input)
                    output_text = tr.blocks[0].text if tr.blocks else ""

                    if verbose:
                        preview = output_text[:120].replace("\n", " ")
                        reward_str = f"  reward={tr.reward:.3f}" if tr.reward else ""
                        print(f"           OUT   {preview}{reward_str}")

                    if tr.reward:
                        total_reward += tr.reward

                    if getattr(tr, "finished", False):
                        finished = True

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output_text,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Guard: if model stops without calling a tool, end episode
            if message.stop_reason not in ("tool_use", "end_turn") or (
                message.stop_reason == "end_turn" and not finished
            ):
                if verbose:
                    print(f"[step {step}] Model stopped without tool call. Ending.")
                break

    print(f"\n--- Episode complete ---")
    print(f"Steps: {step}  |  Total reward: {total_reward:.3f}")
    return total_reward


def main():
    parser = argparse.ArgumentParser(description="Run a HackerEnv rollout on OpenReward")
    parser.add_argument("--split", default="validation", choices=["train", "test", "validation"])
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001",
                        help="Anthropic model ID (haiku for cheap testing, sonnet for quality)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_rollout(
        split=args.split,
        task_index=args.task_index,
        model=args.model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_local_llm.py — Run an OpenAI agent against the local ORS server.

Usage:
    python run_local_llm.py [--seed 0] [--model gpt-4o-mini]
"""

import os
import sys
import json
import uuid
import argparse
import httpx
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

BASE = "http://localhost:8765"
ENV = "hackerenv"


def _headers(session_id: str) -> dict:
    return {"X-Session-ID": session_id, "Content-Type": "application/json"}


def _sse_json(response: httpx.Response) -> dict:
    """Return the JSON payload from the 'end' SSE event."""
    lines = response.text.splitlines()
    get_next = False
    for line in lines:
        line = line.rstrip("\r")
        if line == "event: end":
            get_next = True
        elif get_next and line.startswith("data: "):
            return json.loads(line[6:])
    # Fallback: last data line that looks like JSON
    for line in reversed(lines):
        line = line.rstrip("\r")
        if line.startswith("data: {"):
            return json.loads(line[6:])
    raise ValueError(f"No SSE end event: {response.text[:300]}")


def get_tools() -> list:
    r = httpx.get(f"{BASE}/{ENV}/tools", params={"format": "openai"})
    r.raise_for_status()
    tools = []
    for t in r.json()["tools"]:
        tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    return tools


def create_session(task_spec: dict) -> str:
    sid = str(uuid.uuid4())
    r = httpx.post(
        f"{BASE}/create",
        headers=_headers(sid),
        json={"env_name": ENV, "task_spec": task_spec},
    )
    r.raise_for_status()
    return sid


def delete_session(sid: str):
    httpx.post(f"{BASE}/delete_session", headers=_headers(sid), json={})


def get_prompt(sid: str) -> str:
    r = httpx.get(f"{BASE}/{ENV}/prompt", headers=_headers(sid))
    r.raise_for_status()
    blocks = r.json()
    if isinstance(blocks, list) and blocks:
        return blocks[0].get("text", "")
    return str(blocks)


def call_tool(sid: str, name: str, args: dict) -> dict:
    r = httpx.post(
        f"{BASE}/{ENV}/call",
        headers=_headers(sid),
        json={"name": name, "input": args},
        timeout=30,
    )
    r.raise_for_status()
    return _sse_json(r).get("output", {})


def run_episode(seed: int, chain_depth: int, n_decoys: int, model: str, verbose: bool = True) -> float:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    tools = get_tools()
    task_spec = {"seed": seed, "chain_depth": chain_depth, "n_decoys": n_decoys}

    sid = create_session(task_spec)
    try:
        prompt = get_prompt(sid)
        messages = [{"role": "user", "content": prompt}]

        total_reward = 0.0
        step = 0
        finished = False

        if verbose:
            print(f"\n{'='*60}")
            print(f"seed={seed}  depth={chain_depth}  decoys={n_decoys}  model={model}")
            print(f"{'='*60}\n")

        while not finished and step < 160:
            response = client.chat.completions.create(
                model=model,
                tools=tools,
                messages=messages,
            )
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_unset=False))

            tool_calls = msg.tool_calls or []
            if not tool_calls:
                if verbose:
                    print(f"  [model stopped — no tool call]")
                break

            tool_results = []
            for tc in tool_calls:
                step += 1
                args = json.loads(tc.function.arguments)
                if verbose:
                    print(f"[{step:3d}] {tc.function.name}({json.dumps(args)})")

                result = call_tool(sid, tc.function.name, args)
                output_text = (result.get("blocks") or [{}])[0].get("text", "")
                reward = result.get("reward")
                if reward:
                    total_reward += reward

                if verbose:
                    preview = output_text[:120].replace("\n", " ")
                    r_str = f"  [r={reward:.3f}]" if reward else ""
                    print(f"       {preview}{r_str}")

                if result.get("finished"):
                    finished = True

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output_text,
                })

            messages.extend(tool_results)

    finally:
        delete_session(sid)

    solved = total_reward >= 1.0
    print(f"\n{'='*60}")
    print(f"{'SOLVED' if solved else 'FAILED'}  steps={step}  total_reward={total_reward:.3f}")
    print(f"{'='*60}")
    return total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chain-depth", type=int, default=2)
    parser.add_argument("--decoys", type=int, default=20)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    try:
        httpx.get(f"{BASE}/health", timeout=2).raise_for_status()
    except Exception:
        print("ERROR: Server not running. Start it:\n  .venv/bin/python environment.py")
        sys.exit(1)

    run_episode(
        seed=args.seed,
        chain_depth=args.chain_depth,
        n_decoys=args.decoys,
        model=args.model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

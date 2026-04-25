"""DeceptionSearch harness.

Drives one Provider against one DeceptionSearch task and writes a single
session log to runs/<run_id>/rollout.json. The harness contains all the
provider-agnostic loop logic; per-provider specifics live under
agents.providers / agents.baselines.

Usage from the CLI: see agents/run.py.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from agents.providers.base import Provider


# Tool param classes are imported lazily so DECEPTIONSEARCH_DATA_DIR can be
# set by the caller before the server module loads its base_tree.json.
def _import_env() -> tuple[Any, dict[str, Any]]:
    import server as srv

    params = {
        "ls": srv.LsParams,
        "cat": srv.CatParams,
        "find": srv.FindParams,
        "grep": srv.GrepParams,
        "stat": srv.StatParams,
        "submit": srv.SubmitParams,
    }
    return srv.DeceptionSearch, params


def run_session(
    provider: Provider,
    task_spec: dict[str, Any],
    log_dir: Path | None = None,
    max_turns: int = 200,
    verbose: bool = False,
    rollout: Any = None,
) -> dict[str, Any]:
    """Run one Searcher session to terminal. Return the log dict.

    If `rollout` is an openreward Rollout, mirror the conversation to it:
    each LLM exchange logs assistant/system/user messages; each tool result
    is logged with reward + is_finished so the OR UI shows the timeline.
    """
    DeceptionSearch, params_by_tool = _import_env()
    env = DeceptionSearch(task_spec=task_spec)

    blocks = env.get_prompt()
    system_prompt = "\n\n".join(b.text for b in blocks)

    run_id = f"{int(time.time())}-{provider.name}-{task_spec['id']}-{uuid.uuid4().hex[:6]}"
    log: dict[str, Any] = {
        "run_id": run_id,
        "task_id": task_spec["id"],
        "agent": provider.name,
        "agent_temperature": provider.temperature,
        "actions": [],
        "metadata": None,
        "reward": 0.0,
    }

    tool_name, tool_args = provider.start(system_prompt)
    if rollout is not None:
        provider.flush_assistants(rollout)
    for turn in range(1, max_turns + 1):
        result_text, output = _dispatch(env, params_by_tool, tool_name, tool_args)
        log["actions"].append(
            {
                "step": turn,
                "tool": tool_name,
                "args": tool_args,
                "result": result_text,
                "ts": time.time(),
            }
        )
        if verbose:
            print(f"  [{turn:3d}] {tool_name}({_short(tool_args)})  -> {_summarise(result_text)}")

        if rollout is not None:
            call_id = provider.last_call_id()
            if call_id is not None:
                rollout.log_openai_completions(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result_text,
                    },
                    reward=output.reward if output.finished else None,
                    is_finished=output.finished,
                )

        if output.finished:
            log["metadata"] = output.metadata
            log["reward"] = output.reward
            break
        try:
            tool_name, tool_args = provider.step(result_text)
        except Exception as e:
            log["actions"].append({"step": turn + 1, "error": f"provider_error: {e}"})
            log["reward"] = 0.0
            log["metadata"] = {"terminal_state": "provider_error", "error": str(e)}
            break
        if rollout is not None:
            provider.flush_assistants(rollout)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        out_path = log_dir / f"{run_id}.json"
        out_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
        if verbose:
            print(f"  log -> {out_path}")
    return log


def _dispatch(
    env: Any,
    params_by_tool: dict[str, Any],
    tool_name: str,
    tool_args: dict[str, Any],
) -> tuple[str, Any]:
    if tool_name not in params_by_tool:
        # Synthesise an error result; the env never sees this call.
        err = json.dumps(
            {"error": f"unknown tool: {tool_name!r}",
             "_state": {"budget_remaining": env._budget,
                        "step_count": env._step_count}}
        )
        # Build a minimal ToolOutput-shaped object with finished=False.
        class _Fake:
            finished = False
            reward = 0.0
            metadata = None
        return err, _Fake()
    params_cls = params_by_tool[tool_name]
    try:
        params = params_cls.model_validate(tool_args)
    except Exception as e:
        err = json.dumps(
            {"error": f"invalid args for {tool_name}: {e}",
             "_state": {"budget_remaining": env._budget,
                        "step_count": env._step_count}}
        )
        class _Fake:
            finished = False
            reward = 0.0
            metadata = None
        return err, _Fake()
    output = getattr(env, tool_name)(params)
    text = output.blocks[0].text if output.blocks else ""
    return text, output


def _short(d: dict[str, Any]) -> str:
    parts = []
    for k, v in d.items():
        if isinstance(v, str) and len(v) > 40:
            v = v[:37] + "..."
        parts.append(f"{k}={v!r}")
    return ", ".join(parts)


def _summarise(text: str) -> str:
    s = text.replace("\n", " ")
    return s[:80] + ("..." if len(s) > 80 else "")

from __future__ import annotations

import json
from pathlib import Path

from network_benchmark.ors_env import (
    EscalateParams,
    ExploitParams,
    ExfiltrateParams,
    MoveParams,
    NetworkBenchmarkEnv,
    ScanParams,
)
from network_benchmark.schema import load_scenario_file
from network_benchmark.step_env import NetworkBenchmarkStepEnv


SCENARIO_DIR = Path(__file__).resolve().parent.parent / "network_benchmark" / "scenarios"


def _payload(out) -> dict:
    return json.loads(out.blocks[0].text)


def _task(task_name: str) -> dict:
    return load_scenario_file(SCENARIO_DIR / task_name).model_dump(mode="json")


def test_list_tasks_surfaces_smoke_and_benchmark_scenarios() -> None:
    smoke_ids = {task["id"] for task in NetworkBenchmarkEnv.list_tasks("smoke")}
    benchmark_ids = {task["id"] for task in NetworkBenchmarkEnv.list_tasks("benchmark")}

    assert smoke_ids == {"task_01_linear", "task_05_delayed_exploit"}
    assert benchmark_ids == {
        "task_01_linear",
        "task_02_hidden_vuln",
        "task_03_decoy",
        "task_04_multi_path",
        "task_05_delayed_exploit",
    }


def test_list_tasks_returns_fresh_copies_per_call() -> None:
    first = NetworkBenchmarkEnv.list_tasks("smoke")
    first[0]["id"] = "mutated"
    first[0]["goal"]["target_node"] = "poisoned"

    second = NetworkBenchmarkEnv.list_tasks("smoke")

    assert second[0]["id"] == "task_01_linear"
    assert second[0]["goal"]["target_node"] == "admin"


def test_scan_and_move_tools_return_step_payload_text() -> None:
    env = NetworkBenchmarkEnv(task_spec=_task("task_01_linear.yaml"))

    scan_payload = _payload(env.scan(ScanParams(node="web")))
    assert scan_payload["reward"] == 0.2
    assert scan_payload["done"] is False
    assert scan_payload["observation"]["scan_results"]["web"] == ["sql_injection"]

    env.exploit(ExploitParams(node="web"))
    move_payload = _payload(env.move(MoveParams(node="db")))
    assert move_payload["reward"] == 0.5
    assert move_payload["done"] is False
    assert move_payload["info"]["result"] == "moved"
    assert move_payload["observation"]["current_node"] == "db"


def test_terminal_success_sets_finished_and_exposes_terminal_payload() -> None:
    env = NetworkBenchmarkEnv(task_spec=_task("task_01_linear.yaml"))

    env.scan(ScanParams(node="web"))
    env.exploit(ExploitParams(node="web"))
    env.move(MoveParams(node="db"))
    env.move(MoveParams(node="admin"))
    env.escalate(EscalateParams(node="admin"))
    out = env.exfiltrate(ExfiltrateParams(node="admin"))

    payload = _payload(out)
    assert out.finished is True
    assert out.reward == 5.0
    assert payload["reward"] == 5.0
    assert payload["done"] is True
    assert payload["info"]["success"] is True
    assert payload["info"]["result"] == "goal_reached"


def test_prompt_names_starting_node_and_forbids_placeholder_nodes() -> None:
    env = NetworkBenchmarkEnv(task_spec=_task("task_01_linear.yaml"))

    prompt = env.get_prompt()[0].text

    assert "Your starting node is `web`." in prompt
    assert "Do not invent placeholder node names such as `initial_node`." in prompt


def test_adapter_matches_step_core_for_same_action_sequence() -> None:
    task_spec = _task("task_02_hidden_vuln.yaml")
    adapter = NetworkBenchmarkEnv(task_spec=task_spec)
    step_env = NetworkBenchmarkStepEnv.from_spec(load_scenario_file(SCENARIO_DIR / "task_02_hidden_vuln.yaml"))

    scan_tool = _payload(adapter.scan(ScanParams(node="web")))
    scan_step = step_env.step({"action": "scan", "node": "web"})
    assert scan_tool == {
        "observation": scan_step[0],
        "reward": scan_step[1],
        "done": scan_step[2],
        "info": scan_step[3],
    }

    exploit_tool = _payload(adapter.exploit(ExploitParams(node="web")))
    exploit_step = step_env.step({"action": "exploit", "node": "web"})
    assert exploit_tool == {
        "observation": exploit_step[0],
        "reward": exploit_step[1],
        "done": exploit_step[2],
        "info": exploit_step[3],
    }

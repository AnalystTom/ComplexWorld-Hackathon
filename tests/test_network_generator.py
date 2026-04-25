import copy
from pathlib import Path

import pytest

from network_benchmark.generate_tasks import generate_variants
from network_benchmark.schema import load_scenario_file
from network_benchmark.step_env import NetworkBenchmarkStepEnv


SCENARIO_DIR = Path(__file__).resolve().parent.parent / "network_benchmark" / "scenarios"


def _trace_step(entry: str) -> dict[str, str]:
    action, node = entry.split(":", maxsplit=1)
    return {"action": action, "node": node}


def _run_expected_trace(task) -> tuple[dict, float, bool, dict]:
    env = NetworkBenchmarkStepEnv.from_spec(task)

    observation: dict = {}
    reward = 0.0
    done = False
    info: dict = {}
    for entry in task.expected_optimal_path:
        observation, reward, done, info = env.step(_trace_step(entry))
    return observation, reward, done, info


def test_generator_emits_requested_number_of_variants() -> None:
    tasks = generate_variants(count=3, seed=7)

    assert len(tasks) == 3
    assert len({task.id for task in tasks}) == 3


def test_generator_is_deterministic_for_a_seed() -> None:
    first = generate_variants(count=4, seed=11)
    second = generate_variants(count=4, seed=11)

    assert [task.model_dump() for task in first] == [task.model_dump() for task in second]


def test_generated_tasks_are_schema_valid_variants_of_curated_scenarios() -> None:
    base_ids = {path.stem for path in SCENARIO_DIR.glob("task_*.yaml")}

    tasks = generate_variants(count=5, seed=3)

    assert len(tasks) == 5
    for task in tasks:
        assert any(task.id.startswith(f"{base_id}_gen_") for base_id in base_ids)
        assert task.nodes
        assert task.goal.target_node in {node.id for node in task.nodes}


def test_generated_tasks_are_solvable_via_their_golden_trace() -> None:
    tasks = generate_variants(count=5, seed=3)

    for task in tasks:
        observation, reward, done, info = _run_expected_trace(task)
        assert done is True
        assert observation["exfiltrated"] is True
        assert info["success"] is True
        assert info["result"] == "goal_reached"
        assert info["expected_vs_actual_trace"]["actual"] == task.expected_optimal_path


def test_generator_rejects_schema_valid_tasks_with_unsolvable_golden_traces(monkeypatch) -> None:
    broken = load_scenario_file(SCENARIO_DIR / "task_02_hidden_vuln.yaml").model_dump(mode="python")
    broken["expected_optimal_path"] = [
        "exploit:web",
        "move:db",
        "move:admin",
        "escalate:admin",
        "exfiltrate:admin",
    ]

    monkeypatch.setattr(
        "network_benchmark.generate_tasks._load_base_templates",
        lambda: [copy.deepcopy(broken)],
    )

    with pytest.raises(ValueError, match="unsolvable"):
        generate_variants(count=1, seed=5)


def test_generator_uses_terminal_success_semantics_not_reward_threshold(monkeypatch) -> None:
    template = load_scenario_file(SCENARIO_DIR / "task_01_linear.yaml").model_dump(mode="python")

    class FakeSuccessfulEnv:
        def __init__(self, expected_trace: list[str]):
            self.expected_trace = expected_trace
            self.actual_trace: list[str] = []

        def step(self, action: dict[str, str]) -> tuple[dict, float, bool, dict]:
            self.actual_trace.append(f"{action['action']}:{action['node']}")
            done = len(self.actual_trace) == len(self.expected_trace)
            return (
                {"exfiltrated": done},
                0.0,
                done,
                {
                    "success": done,
                    "result": "goal_reached" if done else "in_progress",
                    "expected_vs_actual_trace": {"actual": list(self.actual_trace)},
                },
            )

    monkeypatch.setattr(
        "network_benchmark.generate_tasks._load_base_templates",
        lambda: [copy.deepcopy(template)],
    )
    monkeypatch.setattr(
        "network_benchmark.generate_tasks.NetworkBenchmarkStepEnv.from_spec",
        lambda spec: FakeSuccessfulEnv(spec.expected_optimal_path),
    )

    tasks = generate_variants(count=1, seed=13)

    assert len(tasks) == 1

from __future__ import annotations

import copy
import random
from pathlib import Path

from network_benchmark.schema import NetworkTaskSpec, load_scenario_file
from network_benchmark.step_env import NetworkBenchmarkStepEnv


SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"
DIFFICULTIES = ("easy", "medium", "hard")


def _load_base_templates() -> list[dict[str, object]]:
    templates: list[dict[str, object]] = []
    for path in sorted(SCENARIO_DIR.glob("task_*.yaml")):
        templates.append(load_scenario_file(path).model_dump(mode="python"))
    if not templates:
        raise ValueError(f"no scenario fixtures found in {SCENARIO_DIR}")
    return templates


def _trace_step(entry: str) -> dict[str, str]:
    action, node = entry.split(":", maxsplit=1)
    return {"action": action, "node": node}


def _validate_solvability(task: NetworkTaskSpec) -> None:
    env = NetworkBenchmarkStepEnv.from_spec(task)

    observation: dict[str, object] = {}
    done = False
    info: dict[str, object] = {}
    for entry in task.expected_optimal_path:
        observation, _, done, info = env.step(_trace_step(entry))

    if (
        not done
        or observation.get("exfiltrated") is not True
        or info.get("success") is not True
        or info.get("result") != "goal_reached"
        or info.get("expected_vs_actual_trace", {}).get("actual") != task.expected_optimal_path
    ):
        raise ValueError(f"generated task {task.id} is unsolvable via expected_optimal_path")


def generate_variants(count: int, seed: int) -> list[NetworkTaskSpec]:
    if count < 0:
        raise ValueError("count must be non-negative")

    rng = random.Random(seed)
    base_templates = _load_base_templates()
    tasks: list[NetworkTaskSpec] = []

    for index in range(count):
        template = copy.deepcopy(base_templates[index % len(base_templates)])
        template["id"] = f"{template['id']}_gen_{seed}_{index}"
        template["difficulty"] = DIFFICULTIES[rng.randrange(len(DIFFICULTIES))]
        template["description"] = f"{template['description']} [generated seed={seed} index={index}]"
        task = NetworkTaskSpec.model_validate(template)
        _validate_solvability(task)
        tasks.append(task)

    return tasks

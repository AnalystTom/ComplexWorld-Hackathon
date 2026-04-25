from __future__ import annotations

from pathlib import Path

import pytest

from network_benchmark.schema import GoalSpec, NetworkTaskSpec, NodeSpec, RuleSpec, ServiceSpec, load_scenario_file
from network_benchmark.step_env import NetworkBenchmarkStepEnv
from network_benchmark.world import NetworkWorld


SCENARIO_DIR = Path(__file__).resolve().parent.parent / "network_benchmark" / "scenarios"


def _env(name: str) -> NetworkBenchmarkStepEnv:
    spec = load_scenario_file(SCENARIO_DIR / name)
    return NetworkBenchmarkStepEnv.from_spec(spec)


def _env_from_spec(spec: NetworkTaskSpec) -> NetworkBenchmarkStepEnv:
    return NetworkBenchmarkStepEnv.from_spec(spec)


def test_hidden_vulnerability_requires_scan_before_exploit() -> None:
    env = _env("task_02_hidden_vuln.yaml")
    assert env.state.known_nodes == {"web"}

    obs, reward, done, info = env.step({"action": "exploit", "node": "web"})
    assert reward < 0
    assert done is False
    assert obs["known_nodes"] == ["web"]
    assert info["result"] == "unknown_vulnerability"
    assert "web" not in info["compromised_nodes"]

    obs, reward, done, info = env.step({"action": "scan", "node": "web"})
    assert reward > 0
    assert done is False
    assert obs["current_node"] == "web"
    assert "web" in obs["known_nodes"]
    assert obs["scan_results"]["web"] == ["hidden_sql_injection"]
    assert info["useful_scans"] == 1

    obs, reward, done, info = env.step({"action": "exploit", "node": "web"})
    assert reward > 0
    assert done is False
    assert info["result"] == "exploit_succeeded"
    assert "web" in info["compromised_nodes"]
    assert info["expected_vs_actual_trace"] == {
        "expected": ["scan:web", "exploit:web", "move:db", "move:admin", "escalate:admin", "exfiltrate:admin"],
        "actual": ["exploit:web", "scan:web", "exploit:web"],
    }


def test_linear_task_golden_path_reaches_goal() -> None:
    env = _env("task_01_linear.yaml")
    sequence = [
        {"action": "scan", "node": "web"},
        {"action": "exploit", "node": "web"},
        {"action": "move", "node": "db"},
        {"action": "move", "node": "admin"},
        {"action": "escalate", "node": "admin"},
        {"action": "exfiltrate", "node": "admin"},
    ]

    obs = {}
    reward = 0.0
    done = False
    info = {}
    for action in sequence:
        obs, reward, done, info = env.step(action)

    assert done is True
    assert reward >= 5.0
    assert obs["current_node"] == "admin"
    assert obs["exfiltrated"] is True
    assert info["success"] is True
    assert info["steps"] == len(sequence)
    assert info["useful_scans"] == 1
    assert info["failed_actions"] == 0
    assert info["unique_nodes_compromised"] == 3
    assert info["path_optimality"] == 1.0
    assert info["expected_vs_actual_trace"]["actual"] == [
        "scan:web",
        "exploit:web",
        "move:db",
        "move:admin",
        "escalate:admin",
        "exfiltrate:admin",
    ]


def test_delayed_rule_unlocks_prepared_node_after_one_step() -> None:
    env = _env("task_05_delayed_exploit.yaml")

    env.step({"action": "scan", "node": "web"})
    obs, reward, done, info = env.step({"action": "exploit", "node": "web"})
    assert reward > 0
    assert done is False
    assert info["pending_rules"] == [
        {
            "effect": "mark_prepared",
            "target_node": "jump",
            "remaining_delay": 1,
        }
    ]
    assert obs["host_state"]["jump"]["prepared"] is False

    obs, reward, done, info = env.step({"action": "move", "node": "jump"})
    assert reward > 0
    assert done is False
    assert info["result"] == "moved"
    assert info["pending_rules"] == []
    assert obs["current_node"] == "jump"
    assert obs["host_state"]["jump"]["prepared"] is True

    obs, reward, done, info = env.step({"action": "exploit", "node": "jump"})
    assert reward > 0
    assert done is False
    assert info["result"] == "exploit_succeeded"
    assert "jump" in info["compromised_nodes"]


def test_decoy_vulnerability_never_progresses_the_task() -> None:
    env = _env("task_03_decoy.yaml")

    env.step({"action": "scan", "node": "web"})
    env.step({"action": "exploit", "node": "web"})
    env.step({"action": "move", "node": "decoy"})

    obs, reward, done, info = env.step({"action": "scan", "node": "decoy"})
    assert reward > 0
    assert obs["scan_results"]["decoy"] == ["fake_sql_injection"]

    obs, reward, done, info = env.step({"action": "exploit", "node": "decoy"})
    assert reward < 0
    assert done is False
    assert info["result"] == "decoy_vulnerability"
    assert "decoy" not in info["compromised_nodes"]
    assert info["success"] is False


def test_repeated_exploit_on_compromised_node_does_not_pay_again() -> None:
    env = _env("task_01_linear.yaml")

    env.step({"action": "scan", "node": "web"})
    _, reward, done, info = env.step({"action": "exploit", "node": "web"})
    assert reward == 1.0
    assert done is False
    assert info["result"] == "exploit_succeeded"

    _, reward, done, info = env.step({"action": "exploit", "node": "web"})
    assert reward == -0.2
    assert done is False
    assert info["result"] == "already_compromised"
    assert info["failed_actions"] == 1
    assert info["compromised_nodes"] == ["web"]


def test_repeated_move_revisit_does_not_pay_new_node_reward() -> None:
    env = _env("task_01_linear.yaml")

    env.step({"action": "scan", "node": "web"})
    env.step({"action": "exploit", "node": "web"})

    _, reward, done, info = env.step({"action": "move", "node": "db"})
    assert reward == 0.5
    assert done is False
    assert info["result"] == "moved"

    _, reward, done, info = env.step({"action": "move", "node": "web"})
    assert reward == 0.0
    assert done is False
    assert info["result"] == "revisited"

    _, reward, done, info = env.step({"action": "move", "node": "db"})
    assert reward == 0.0
    assert done is False
    assert info["result"] == "revisited"


def test_move_requires_current_node_adjacency_not_any_compromised_foothold() -> None:
    env = _env("task_04_multi_path.yaml")

    env.step({"action": "scan", "node": "vpn"})
    env.step({"action": "exploit", "node": "vpn"})
    env.step({"action": "move", "node": "admin"})

    _, reward, done, info = env.step({"action": "move", "node": "app"})
    assert reward == -0.2
    assert done is False
    assert info["result"] == "unreachable"
    assert info["current_node"] == "admin"


def test_loaded_world_is_immutable_through_mapping_interfaces() -> None:
    spec = load_scenario_file(SCENARIO_DIR / "task_05_delayed_exploit.yaml")
    world = NetworkWorld.from_spec(spec)

    with pytest.raises(TypeError):
        world.nodes["jump"] = world.nodes["web"]  # type: ignore[index]

    with pytest.raises(TypeError):
        world.nodes["jump"].host_state["prepared"] = True


def test_terminal_episode_is_frozen_after_completion() -> None:
    spec = NetworkTaskSpec(
        id="terminal-freeze",
        description="goal can finish while a delayed rule is still pending",
        difficulty="easy",
        start_node="web",
        goal=GoalSpec(type="exfiltrate_from_node", target_node="web", requires_privilege=False),
        nodes=[
            NodeSpec(
                id="web",
                neighbors=["admin"],
                services=[ServiceSpec(name="http", vuln="sql_injection")],
            ),
            NodeSpec(
                id="admin",
                neighbors=["web"],
                services=[ServiceSpec(name="ssh", vuln="none")],
                host_state={"hidden_auth_unlocked": False},
            ),
        ],
        rules=[
            RuleSpec(
                trigger="exploit",
                source_node="web",
                effect="unlock_hidden_auth",
                target_node="admin",
                delay_steps=2,
            )
        ],
    )
    env = _env_from_spec(spec)

    env.step({"action": "scan", "node": "web"})
    env.step({"action": "exploit", "node": "web"})
    _, reward, done, info = env.step({"action": "exfiltrate", "node": "web"})
    assert reward == 5.0
    assert done is True
    assert info["pending_rules"] == [
        {
            "effect": "unlock_hidden_auth",
            "target_node": "admin",
            "remaining_delay": 1,
        }
    ]

    frozen_step_count = env.state.step_count
    frozen_trajectory = list(env.state.trajectory)
    frozen_pending = list(env.state.pending_rules)

    obs, reward, done, info = env.step({"action": "move", "node": "admin"})
    assert reward == 0.0
    assert done is True
    assert info["result"] == "episode_complete"
    assert env.state.step_count == frozen_step_count
    assert env.state.trajectory == frozen_trajectory
    assert env.state.pending_rules == frozen_pending
    assert obs["trajectory"] == frozen_trajectory
    assert info["pending_rules"] == [
        {
            "effect": "unlock_hidden_auth",
            "target_node": "admin",
            "remaining_delay": 1,
        }
    ]


def test_unlock_hidden_auth_rule_enables_later_escalation() -> None:
    spec = NetworkTaskSpec(
        id="hidden-auth",
        description="rule unlocks later escalation path",
        difficulty="medium",
        start_node="web",
        goal=GoalSpec(type="exfiltrate_from_node", target_node="admin", requires_privilege=True),
        nodes=[
            NodeSpec(
                id="web",
                neighbors=["admin"],
                services=[ServiceSpec(name="http", vuln="sql_injection")],
            ),
            NodeSpec(
                id="admin",
                neighbors=["web"],
                services=[ServiceSpec(name="ssh", vuln="none")],
                host_state={"hidden_auth_unlocked": False},
            ),
        ],
        rules=[
            RuleSpec(
                trigger="exploit",
                source_node="web",
                effect="unlock_hidden_auth",
                target_node="admin",
                delay_steps=1,
            )
        ],
    )
    env = _env_from_spec(spec)

    env.step({"action": "scan", "node": "web"})
    env.step({"action": "exploit", "node": "web"})
    env.step({"action": "move", "node": "admin"})

    _, reward, done, info = env.step({"action": "escalate", "node": "admin"})
    assert reward == 1.0
    assert done is False
    assert info["result"] == "escalate_succeeded"
    assert info["privileged_nodes"] == ["admin"]


def test_fixed_action_sequence_is_deterministic() -> None:
    sequence = [
        {"action": "scan", "node": "web"},
        {"action": "exploit", "node": "web"},
        {"action": "move", "node": "jump"},
        {"action": "exploit", "node": "jump"},
        {"action": "move", "node": "vault"},
        {"action": "escalate", "node": "vault"},
        {"action": "exfiltrate", "node": "vault"},
    ]

    first = _env("task_05_delayed_exploit.yaml")
    second = _env("task_05_delayed_exploit.yaml")

    first_trace = [first.step(action) for action in sequence]
    second_trace = [second.step(action) for action in sequence]

    assert first_trace == second_trace

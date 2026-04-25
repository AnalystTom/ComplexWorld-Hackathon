from run_rollout import (
    RolloutTarget,
    get_rollout_target,
    infer_solved,
    summarize_task,
)


def test_get_rollout_target_for_network_uses_network_defaults() -> None:
    target = get_rollout_target("network")
    assert isinstance(target, RolloutTarget)
    assert target.env_ref.endswith("NetworkBenchmark-v0")
    assert target.default_split == "smoke"
    assert "network-benchmark" in target.run_name


def test_get_rollout_target_allows_overrides() -> None:
    target = get_rollout_target(
        "network",
        env_ref_override="demo/CustomNetworkBenchmark",
        run_name_override="custom-run",
    )
    assert target.env_ref == "demo/CustomNetworkBenchmark"
    assert target.run_name == "custom-run"


def test_summarize_task_handles_network_tasks() -> None:
    summary = summarize_task(
        {
            "id": "task_01_linear",
            "description": "Simple attack chain",
            "difficulty": "easy",
            "start_node": "web",
            "goal": {"target_node": "admin", "type": "exfiltrate_from_node"},
        }
    )
    assert "task_01_linear" in summary
    assert "start_node=web" in summary
    assert '"target_node": "admin"' in summary


def test_infer_solved_prefers_terminal_metadata() -> None:
    assert infer_solved(
        terminal_reward=0.0,
        terminal_metadata={"success": True, "result": "goal_reached"},
        total_reward=0.2,
    )
    assert not infer_solved(
        terminal_reward=3.0,
        terminal_metadata={"success": False, "result": "wrong_target"},
        total_reward=3.0,
    )

from pathlib import Path

import pytest

from network_benchmark.schema import (
    GoalSpec,
    NetworkTaskSpec,
    ServiceSpec,
    load_scenario_file,
)


SCENARIO_DIR = Path(__file__).resolve().parent.parent / "network_benchmark" / "scenarios"


def test_all_curated_scenarios_validate() -> None:
    paths = sorted(SCENARIO_DIR.glob("task_*.yaml"))
    assert {
        path.name for path in paths
    } >= {
        "task_01_linear.yaml",
        "task_02_hidden_vuln.yaml",
        "task_03_decoy.yaml",
        "task_04_multi_path.yaml",
        "task_05_delayed_exploit.yaml",
    }
    for path in paths:
        spec = load_scenario_file(path)
        assert isinstance(spec, NetworkTaskSpec)
        assert spec.start_node in {node.id for node in spec.nodes}


def test_goal_requires_declared_target_node() -> None:
    with pytest.raises(ValueError, match="target_node"):
        NetworkTaskSpec.model_validate(
            {
                "id": "bad",
                "description": "bad",
                "difficulty": "easy",
                "start_node": "web",
                "goal": {
                    "type": "exfiltrate_from_node",
                    "target_node": "missing",
                    "requires_privilege": True,
                },
                "nodes": [
                    {
                        "id": "web",
                        "neighbors": [],
                        "services": [{"name": "http", "vuln": "none"}],
                    }
                ],
            }
        )


def test_service_vulnerability_is_restricted() -> None:
    with pytest.raises(ValueError, match="vuln"):
        ServiceSpec.model_validate({"name": "http", "vuln": "totally_custom"})


def test_rules_reject_non_declarative_fields() -> None:
    with pytest.raises(ValueError, match="script"):
        NetworkTaskSpec.model_validate(
            {
                "id": "bad-rules",
                "description": "bad rules",
                "difficulty": "medium",
                "start_node": "web",
                "goal": {
                    "type": "exfiltrate_from_node",
                    "target_node": "web",
                    "requires_privilege": False,
                },
                "nodes": [
                    {
                        "id": "web",
                        "neighbors": [],
                        "services": [{"name": "http", "vuln": "none"}],
                    }
                ],
                "rules": [
                    {
                        "trigger": "exploit",
                        "source_node": "web",
                        "effect": "mark_prepared",
                        "script": "arbitrary()",
                    }
                ],
            }
        )


def test_duplicate_node_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="node ids must be unique"):
        NetworkTaskSpec.model_validate(
            {
                "id": "duplicate-nodes",
                "description": "bad nodes",
                "difficulty": "easy",
                "start_node": "web",
                "goal": {
                    "type": "exfiltrate_from_node",
                    "target_node": "web",
                    "requires_privilege": False,
                },
                "nodes": [
                    {
                        "id": "web",
                        "neighbors": [],
                        "services": [{"name": "http", "vuln": "none"}],
                    },
                    {
                        "id": "web",
                        "neighbors": [],
                        "services": [{"name": "ssh", "vuln": "weak_auth"}],
                    },
                ],
            }
        )


def test_unknown_neighbors_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown neighbors"):
        NetworkTaskSpec.model_validate(
            {
                "id": "bad-neighbors",
                "description": "bad graph",
                "difficulty": "easy",
                "start_node": "web",
                "goal": {
                    "type": "exfiltrate_from_node",
                    "target_node": "web",
                    "requires_privilege": False,
                },
                "nodes": [
                    {
                        "id": "web",
                        "neighbors": ["missing"],
                        "services": [{"name": "http", "vuln": "none"}],
                    }
                ],
            }
        )


def test_rule_requires_valid_source_node() -> None:
    with pytest.raises(ValueError, match="rule.source_node"):
        NetworkTaskSpec.model_validate(
            {
                "id": "bad-rule-source",
                "description": "bad rule source",
                "difficulty": "medium",
                "start_node": "web",
                "goal": {
                    "type": "exfiltrate_from_node",
                    "target_node": "web",
                    "requires_privilege": False,
                },
                "nodes": [
                    {
                        "id": "web",
                        "neighbors": [],
                        "services": [{"name": "http", "vuln": "none"}],
                    }
                ],
                "rules": [
                    {
                        "trigger": "exploit",
                        "source_node": "missing",
                        "effect": "mark_prepared",
                        "target_node": "web",
                    }
                ],
            }
        )


def test_rule_requires_valid_target_node() -> None:
    with pytest.raises(ValueError, match="rule.target_node"):
        NetworkTaskSpec.model_validate(
            {
                "id": "bad-rule-target",
                "description": "bad rule target",
                "difficulty": "medium",
                "start_node": "web",
                "goal": {
                    "type": "exfiltrate_from_node",
                    "target_node": "web",
                    "requires_privilege": False,
                },
                "nodes": [
                    {
                        "id": "web",
                        "neighbors": [],
                        "services": [{"name": "http", "vuln": "none"}],
                    }
                ],
                "rules": [
                    {
                        "trigger": "exploit",
                        "source_node": "web",
                        "effect": "mark_prepared",
                        "target_node": "missing",
                    }
                ],
            }
        )


def test_targeted_rule_effect_requires_target_node() -> None:
    with pytest.raises(ValueError, match="target_node"):
        NetworkTaskSpec.model_validate(
            {
                "id": "missing-rule-target",
                "description": "missing rule target",
                "difficulty": "medium",
                "start_node": "web",
                "goal": {
                    "type": "exfiltrate_from_node",
                    "target_node": "web",
                    "requires_privilege": False,
                },
                "nodes": [
                    {
                        "id": "web",
                        "neighbors": [],
                        "services": [{"name": "http", "vuln": "none"}],
                    }
                ],
                "rules": [
                    {
                        "trigger": "exploit",
                        "source_node": "web",
                        "effect": "mark_prepared",
                    }
                ],
            }
        )

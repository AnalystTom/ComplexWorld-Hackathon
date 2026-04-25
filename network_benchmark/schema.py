from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


VulnName = Literal[
    "sql_injection",
    "hidden_sql_injection",
    "fake_sql_injection",
    "weak_auth",
    "setup_required",
    "none",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ServiceSpec(StrictModel):
    name: str
    vuln: VulnName


class NodeSpec(StrictModel):
    id: str
    services: list[ServiceSpec]
    neighbors: list[str] = Field(default_factory=list)
    loot: list[str] = Field(default_factory=list)
    host_state: dict[str, str | int | bool] = Field(default_factory=dict)


class GoalSpec(StrictModel):
    type: Literal["exfiltrate_from_node"]
    target_node: str
    requires_privilege: bool = True


class RuleSpec(StrictModel):
    trigger: Literal["exploit"]
    source_node: str
    effect: Literal["unlock_hidden_auth", "mark_prepared"]
    delay_steps: int = Field(default=0, ge=0)
    target_node: str | None = None

    @model_validator(mode="after")
    def validate_target_node_requirement(self) -> "RuleSpec":
        if self.effect in {"unlock_hidden_auth", "mark_prepared"} and self.target_node is None:
            raise ValueError(f"target_node is required for effect {self.effect}")
        return self


class NetworkTaskSpec(StrictModel):
    id: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    start_node: str
    goal: GoalSpec
    nodes: list[NodeSpec]
    rules: list[RuleSpec] = Field(default_factory=list)
    expected_optimal_path: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> "NetworkTaskSpec":
        node_ids = [node.id for node in self.nodes]
        unique_node_ids = set(node_ids)
        if len(unique_node_ids) != len(node_ids):
            raise ValueError("node ids must be unique")
        if self.start_node not in unique_node_ids:
            raise ValueError("start_node must exist in nodes")
        if self.goal.target_node not in unique_node_ids:
            raise ValueError("goal.target_node must exist in nodes")

        for node in self.nodes:
            unknown_neighbors = [neighbor for neighbor in node.neighbors if neighbor not in unique_node_ids]
            if unknown_neighbors:
                raise ValueError(f"node {node.id} has unknown neighbors: {unknown_neighbors}")

        for rule in self.rules:
            if rule.source_node not in unique_node_ids:
                raise ValueError(f"rule.source_node must exist in nodes: {rule.source_node}")
            if rule.target_node is not None and rule.target_node not in unique_node_ids:
                raise ValueError(f"rule.target_node must exist in nodes: {rule.target_node}")

        return self


def load_scenario_file(path: str | Path) -> NetworkTaskSpec:
    scenario_path = Path(path)
    raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"scenario file must contain a mapping: {scenario_path}")
    return NetworkTaskSpec.model_validate(raw)

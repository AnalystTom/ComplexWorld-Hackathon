from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from network_benchmark.schema import NetworkTaskSpec


@dataclass(frozen=True)
class WorldService:
    name: str
    vuln: str


@dataclass(frozen=True)
class WorldNode:
    id: str
    neighbors: tuple[str, ...]
    services: tuple[WorldService, ...]
    loot: tuple[str, ...]
    host_state: Mapping[str, str | int | bool]


@dataclass(frozen=True)
class WorldRule:
    trigger: str
    source_node: str
    effect: str
    delay_steps: int
    target_node: str | None


@dataclass(frozen=True)
class NetworkWorld:
    task_id: str
    description: str
    start_node: str
    goal_node: str
    requires_privilege: bool
    max_steps: int
    expected_optimal_path: tuple[str, ...]
    nodes: Mapping[str, WorldNode]
    rules: tuple[WorldRule, ...]

    @classmethod
    def from_spec(cls, spec: NetworkTaskSpec, *, max_steps: int = 25) -> "NetworkWorld":
        nodes = {
            node.id: WorldNode(
                id=node.id,
                neighbors=tuple(node.neighbors),
                services=tuple(WorldService(name=service.name, vuln=service.vuln) for service in node.services),
                loot=tuple(node.loot),
                host_state=MappingProxyType(dict(node.host_state)),
            )
            for node in spec.nodes
        }
        return cls(
            task_id=spec.id,
            description=spec.description,
            start_node=spec.start_node,
            goal_node=spec.goal.target_node,
            requires_privilege=spec.goal.requires_privilege,
            max_steps=max_steps,
            expected_optimal_path=tuple(spec.expected_optimal_path),
            nodes=MappingProxyType(nodes),
            rules=tuple(
                WorldRule(
                    trigger=rule.trigger,
                    source_node=rule.source_node,
                    effect=rule.effect,
                    delay_steps=rule.delay_steps,
                    target_node=rule.target_node,
                )
                for rule in spec.rules
            ),
        )

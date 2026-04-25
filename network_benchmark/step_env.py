from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from network_benchmark.schema import NetworkTaskSpec
from network_benchmark.world import NetworkWorld, WorldNode


REWARD_SCAN = 0.2
REWARD_EXPLOIT = 1.0
REWARD_MOVE = 0.5
REWARD_ESCALATE = 1.0
REWARD_INVALID = -0.2
REWARD_FAIL = -0.5
REWARD_WRONG_EXFIL = -1.0
REWARD_GOAL = 5.0


@dataclass
class PendingRule:
    effect: str
    target_node: str | None
    remaining_delay: int


@dataclass
class RuntimeState:
    current_node: str
    visited_nodes: set[str] = field(default_factory=set)
    known_nodes: set[str] = field(default_factory=set)
    scanned_nodes: set[str] = field(default_factory=set)
    discovered_vulns: dict[str, set[str]] = field(default_factory=dict)
    compromised_nodes: set[str] = field(default_factory=set)
    privileged_nodes: set[str] = field(default_factory=set)
    exfiltrated: bool = False
    alerts: list[str] = field(default_factory=list)
    step_count: int = 0
    useful_scans: int = 0
    failed_actions: int = 0
    trajectory: list[str] = field(default_factory=list)
    pending_rules: list[PendingRule] = field(default_factory=list)
    host_state: dict[str, dict[str, str | int | bool]] = field(default_factory=dict)
    done: bool = False


class NetworkBenchmarkStepEnv:
    def __init__(self, world: NetworkWorld):
        self.world = world
        self.state = RuntimeState(
            current_node=world.start_node,
            visited_nodes={world.start_node},
            known_nodes={world.start_node},
            host_state={node_id: dict(node.host_state) for node_id, node in world.nodes.items()},
        )

    @classmethod
    def from_spec(cls, spec: NetworkTaskSpec) -> "NetworkBenchmarkStepEnv":
        return cls(NetworkWorld.from_spec(spec))

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.state.done:
            observation = self._build_observation("episode_complete", step_alerts=[])
            info = self._build_info("episode_complete", step_alerts=[])
            return observation, 0.0, True, info

        self._apply_pending_rules()

        action_name = str(action.get("action", ""))
        node = str(action.get("node", ""))
        trace_entry = f"{action_name}:{node}" if action_name and node else action_name or "<invalid>"
        self.state.trajectory.append(trace_entry)
        self.state.step_count += 1

        result = "invalid_action"
        reward = REWARD_INVALID
        alert: str | None = None

        if action_name == "scan":
            reward, result, alert = self._handle_scan(node)
        elif action_name == "exploit":
            reward, result, alert = self._handle_exploit(node)
        elif action_name == "move":
            reward, result, alert = self._handle_move(node)
        elif action_name == "escalate":
            reward, result, alert = self._handle_escalate(node)
        elif action_name == "exfiltrate":
            reward, result, alert = self._handle_exfiltrate(node)
        else:
            alert = "invalid_action"

        if reward < 0:
            self.state.failed_actions += 1
        step_alerts: list[str] = []
        if alert:
            step_alerts.append(alert)
            self.state.alerts.append(alert)

        if not self.state.exfiltrated and self.state.step_count >= self.world.max_steps:
            self.state.done = True
            result = "step_limit_reached"
            reward = REWARD_INVALID
            if not self.state.alerts or self.state.alerts[-1] != "step_limit_reached":
                self.state.alerts.append("step_limit_reached")
            step_alerts.append("step_limit_reached")

        observation = self._build_observation(result, step_alerts=step_alerts)
        info = self._build_info(result, step_alerts=step_alerts)
        return observation, reward, self.state.done, info

    def _handle_scan(self, node: str) -> tuple[float, str, str | None]:
        if not self._requires_current_node(node):
            return REWARD_INVALID, "invalid_target", "invalid_target"
        if node in self.state.scanned_nodes:
            return REWARD_INVALID, "already_scanned", "already_scanned"

        self.state.scanned_nodes.add(node)
        self.state.useful_scans += 1
        self._reveal_neighbors(node)
        found_vulns = {
            service.vuln
            for service in self.world.nodes[node].services
            if service.vuln != "none"
        }
        self.state.discovered_vulns[node] = found_vulns
        return REWARD_SCAN, "scan_succeeded", None

    def _handle_exploit(self, node: str) -> tuple[float, str, str | None]:
        if not self._requires_current_node(node):
            return REWARD_INVALID, "invalid_target", "invalid_target"
        if node in self.state.compromised_nodes:
            return REWARD_INVALID, "already_compromised", "already_compromised"

        vulns = self.state.discovered_vulns.get(node)
        if vulns is None:
            return REWARD_FAIL, "unknown_vulnerability", "unknown_vulnerability"
        if not vulns:
            return REWARD_FAIL, "no_exploitable_service", "no_exploitable_service"

        if "fake_sql_injection" in vulns:
            return REWARD_FAIL, "decoy_vulnerability", "decoy_vulnerability"
        if "setup_required" in vulns and not bool(self.state.host_state.get(node, {}).get("prepared", False)):
            return REWARD_FAIL, "prerequisites_not_met", "prerequisites_not_met"
        if not any(vuln in {"sql_injection", "hidden_sql_injection", "weak_auth", "setup_required"} for vuln in vulns):
            return REWARD_FAIL, "no_exploitable_service", "no_exploitable_service"

        self.state.compromised_nodes.add(node)
        self._reveal_neighbors(node)
        self._schedule_rules(trigger="exploit", source_node=node)
        return REWARD_EXPLOIT, "exploit_succeeded", None

    def _handle_move(self, node: str) -> tuple[float, str, str | None]:
        if not node:
            return REWARD_INVALID, "invalid_target", "invalid_target"
        if not self.state.compromised_nodes:
            return REWARD_INVALID, "no_foothold", "no_foothold"
        current_neighbors = self.world.nodes[self.state.current_node].neighbors
        if node not in current_neighbors:
            return REWARD_INVALID, "unreachable", "unreachable"

        self.state.current_node = node
        self.state.known_nodes.add(node)
        self._reveal_neighbors(node)
        is_new_visit = node not in self.state.visited_nodes
        self.state.visited_nodes.add(node)
        if self._node_grants_move_foothold(self.world.nodes[node]):
            self.state.compromised_nodes.add(node)
        if is_new_visit:
            return REWARD_MOVE, "moved", None
        return 0.0, "revisited", None

    def _handle_escalate(self, node: str) -> tuple[float, str, str | None]:
        if not self._requires_current_node(node):
            return REWARD_INVALID, "invalid_target", "invalid_target"
        if node not in self.state.compromised_nodes:
            return REWARD_INVALID, "not_compromised", "not_compromised"
        if node in self.state.privileged_nodes:
            return REWARD_INVALID, "already_privileged", "already_privileged"
        if not self._node_supports_escalation(self.world.nodes[node]):
            return REWARD_FAIL, "privilege_escalation_unavailable", "privilege_escalation_unavailable"

        self.state.privileged_nodes.add(node)
        return REWARD_ESCALATE, "escalate_succeeded", None

    def _handle_exfiltrate(self, node: str) -> tuple[float, str, str | None]:
        if not self._requires_current_node(node):
            return REWARD_INVALID, "invalid_target", "invalid_target"
        if node != self.world.goal_node:
            return REWARD_WRONG_EXFIL, "wrong_target", "wrong_target"
        if self.world.requires_privilege and node not in self.state.privileged_nodes:
            return REWARD_WRONG_EXFIL, "insufficient_privilege", "insufficient_privilege"

        self.state.exfiltrated = True
        self.state.done = True
        return REWARD_GOAL, "goal_reached", None

    def _schedule_rules(self, *, trigger: str, source_node: str) -> None:
        for rule in self.world.rules:
            if rule.trigger == trigger and rule.source_node == source_node:
                self.state.pending_rules.append(
                    PendingRule(
                        effect=rule.effect,
                        target_node=rule.target_node,
                        remaining_delay=rule.delay_steps,
                    )
                )

    def _apply_pending_rules(self) -> None:
        if not self.state.pending_rules:
            return

        still_pending: list[PendingRule] = []
        for pending in self.state.pending_rules:
            next_delay = pending.remaining_delay - 1
            if next_delay <= 0:
                self._apply_rule_effect(pending.effect, pending.target_node)
            else:
                still_pending.append(
                    PendingRule(
                        effect=pending.effect,
                        target_node=pending.target_node,
                        remaining_delay=next_delay,
                    )
                )
        self.state.pending_rules = still_pending

    def _apply_rule_effect(self, effect: str, target_node: str | None) -> None:
        if target_node is None:
            return
        node_state = self.state.host_state.setdefault(target_node, {})
        if effect == "mark_prepared":
            node_state["prepared"] = True
            self.state.discovered_vulns.setdefault(target_node, set()).add("setup_required")
        elif effect == "unlock_hidden_auth":
            node_state["hidden_auth_unlocked"] = True

    def _requires_current_node(self, node: str) -> bool:
        return bool(node) and node == self.state.current_node

    def _reveal_neighbors(self, node: str) -> None:
        self.state.known_nodes.add(node)
        self.state.known_nodes.update(self.world.nodes[node].neighbors)

    def _node_supports_escalation(self, node: WorldNode) -> bool:
        if bool(self.state.host_state.get(node.id, {}).get("hidden_auth_unlocked", False)):
            return True
        return any(service.vuln == "weak_auth" for service in node.services)

    def _node_grants_move_foothold(self, node: WorldNode) -> bool:
        foothold_vulns = {service.vuln for service in node.services}
        return foothold_vulns.issubset({"none", "weak_auth"})

    def _build_observation(self, result: str, *, step_alerts: list[str]) -> dict[str, Any]:
        known_nodes = sorted(self.state.known_nodes)
        return {
            "task_id": self.world.task_id,
            "current_node": self.state.current_node,
            "known_nodes": known_nodes,
            "scan_results": self._sorted_discovered_vulns(),
            "access": {
                node_id: {
                    "compromised": node_id in self.state.compromised_nodes,
                    "privileged": node_id in self.state.privileged_nodes,
                }
                for node_id in known_nodes
            },
            "host_state": {
                node_id: dict(self.state.host_state.get(node_id, {}))
                for node_id in known_nodes
            },
            "alerts": list(step_alerts),
            "alert_history": list(self.state.alerts),
            "result": result,
            "step_count": self.state.step_count,
            "remaining_steps": max(0, self.world.max_steps - self.state.step_count),
            "exfiltrated": self.state.exfiltrated,
            "trajectory": list(self.state.trajectory),
        }

    def _build_info(self, result: str, *, step_alerts: list[str]) -> dict[str, Any]:
        expected_trace = list(self.world.expected_optimal_path)
        actual_trace = list(self.state.trajectory)
        return {
            "result": result,
            "success": self.state.exfiltrated,
            "steps": self.state.step_count,
            "useful_scans": self.state.useful_scans,
            "failed_actions": self.state.failed_actions,
            "current_node": self.state.current_node,
            "known_nodes": sorted(self.state.known_nodes),
            "compromised_nodes": sorted(self.state.compromised_nodes),
            "privileged_nodes": sorted(self.state.privileged_nodes),
            "alerts": list(step_alerts),
            "alert_history": list(self.state.alerts),
            "trajectory": actual_trace,
            "unique_nodes_compromised": len(self.state.compromised_nodes),
            "path_optimality": self._path_optimality(expected_trace, actual_trace),
            "expected_vs_actual_trace": {
                "expected": expected_trace,
                "actual": actual_trace,
            },
            "pending_rules": [
                {
                    "effect": pending.effect,
                    "target_node": pending.target_node,
                    "remaining_delay": pending.remaining_delay,
                }
                for pending in self.state.pending_rules
            ],
        }

    def _sorted_discovered_vulns(self) -> dict[str, list[str]]:
        return {
            node_id: sorted(vulns)
            for node_id, vulns in sorted(self.state.discovered_vulns.items())
        }

    def _path_optimality(self, expected: list[str], actual: list[str]) -> float:
        if not expected:
            return 1.0 if not actual else 0.0
        lcs_length = self._lcs_length(expected, actual)
        return round(lcs_length / len(expected), 4)

    def _lcs_length(self, left: list[str], right: list[str]) -> int:
        rows = len(left) + 1
        cols = len(right) + 1
        table = [[0] * cols for _ in range(rows)]
        for i, left_value in enumerate(left, start=1):
            for j, right_value in enumerate(right, start=1):
                if left_value == right_value:
                    table[i][j] = table[i - 1][j - 1] + 1
                else:
                    table[i][j] = max(table[i - 1][j], table[i][j - 1])
        return table[-1][-1]

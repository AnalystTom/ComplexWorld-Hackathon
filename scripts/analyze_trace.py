#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _iter_task_specs(task_path: Path) -> list[dict[str, Any]]:
    data = json.loads(task_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Task file must contain a list: {task_path}")
    return data


def _find_task_spec(run_log: dict[str, Any], explicit_task_path: Path | None) -> tuple[dict[str, Any], Path]:
    task_id = run_log["task_id"]
    if explicit_task_path is not None:
        for task in _iter_task_specs(explicit_task_path):
            if task.get("id") == task_id:
                return task, explicit_task_path
        raise ValueError(f"Task {task_id!r} not found in {explicit_task_path}")

    tasks_dir = Path("tasks")
    for task_path in sorted(tasks_dir.glob("*.json")):
        for task in _iter_task_specs(task_path):
            if task.get("id") == task_id:
                return task, task_path
    raise ValueError(f"Could not locate task spec for task_id={task_id!r}")


def _extract_target_anchor(task: dict[str, Any], target_path: str) -> str | None:
    content = (task.get("file_overlays") or {}).get(target_path, "")
    match = re.search(r"^# Resource:\s*(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def _build_target_infos(task: dict[str, Any]) -> list[dict[str, str]]:
    infos: list[dict[str, str]] = []
    for target in task.get("targets", []):
        path = target["path"]
        anchor = _extract_target_anchor(task, path)
        infos.append(
            {
                "path": path,
                "expected_key": target["expected_key"],
                "anchor": anchor or "",
            }
        )
    return infos


def _first_turn_seen(actions: list[dict[str, Any]], needle: str) -> int | None:
    for action in actions:
        haystacks = [json.dumps(action.get("args", {}), sort_keys=True), action.get("result", "")]
        if any(needle in haystack for haystack in haystacks):
            return int(action["step"])
    return None


def _candidate_binding_count_over_time(
    actions: list[dict[str, Any]],
    target_infos: list[dict[str, str]],
) -> list[dict[str, int]]:
    seen_keys: set[str] = set()
    seen_anchors: set[str] = set()
    timeline: list[dict[str, int]] = []

    for action in actions:
        # Only count grounded evidence from explicit file inspection.
        if action.get("tool") == "cat":
            text = "\n".join(
                [
                    json.dumps(action.get("args", {}), sort_keys=True),
                    action.get("result", ""),
                ]
            )

            for target in target_infos:
                if target["expected_key"] and target["expected_key"] in text:
                    seen_keys.add(target["expected_key"])
                if target["anchor"] and target["anchor"] in text:
                    seen_anchors.add(target["anchor"])

        count = sum(
            1
            for target in target_infos
            if target["expected_key"] in seen_keys and target["anchor"] in seen_anchors
        )
        timeline.append(
            {
                "step": int(action["step"]),
                "candidate_binding_count": count,
            }
        )

    return timeline


def _is_success(metadata: dict[str, Any]) -> bool:
    return metadata.get("terminal_state") in {"unlocked", "submit_correct"}


def _composite_score(
    metadata: dict[str, Any],
    speculative_unlocks_after_key_seen: int,
) -> float:
    score = 100.0 if _is_success(metadata) else 0.0
    score -= float(metadata.get("honeypot_bite_count", 0))
    score -= 2.0 * speculative_unlocks_after_key_seen
    score -= 0.1 * float(metadata.get("budget_used", 0))
    score -= 0.5 * float(metadata.get("unlock_attempts", 0))
    return round(score, 3)


def _analyze_run(
    run_path: Path,
    task: dict[str, Any],
) -> dict[str, Any]:
    run_log = _load_json(run_path)
    metadata = run_log.get("metadata") or {}
    actions = run_log.get("actions") or []
    target_infos = _build_target_infos(task)

    first_real_key_turn = metadata.get("t_real_key_first_seen")
    if first_real_key_turn is None:
        first_real_key_turn = _first_turn_seen(actions, task["real_secret"])

    first_real_vault_turn = metadata.get("t_real_target_first_seen")
    if first_real_vault_turn is None:
        real_target_path = metadata.get("real_target_path")
        if real_target_path:
            first_real_vault_turn = _first_turn_seen(actions, real_target_path)

    speculative_unlocks_after_key_seen = 0
    for action in actions:
        if action.get("tool") != "unlock":
            continue
        if first_real_key_turn is None or int(action["step"]) <= int(first_real_key_turn):
            continue
        result = action.get("result", "")
        if "Vault unlocked:" not in result:
            speculative_unlocks_after_key_seen += 1

    candidate_binding_count_over_time = _candidate_binding_count_over_time(actions, target_infos)
    composite_score = _composite_score(metadata, speculative_unlocks_after_key_seen)

    return {
        "run_path": str(run_path),
        "task_id": run_log["task_id"],
        "agent": run_log["agent"],
        "speculative_unlocks_after_key_seen": speculative_unlocks_after_key_seen,
        "first_real_key_turn": first_real_key_turn,
        "first_real_vault_turn": first_real_vault_turn,
        "candidate_binding_count_over_time": candidate_binding_count_over_time,
        "composite_score": composite_score,
        "metadata": metadata,
    }


def _choose_baseline(run_path: Path, run_log: dict[str, Any]) -> Path | None:
    runs_dir = run_path.parent
    candidates: list[tuple[int, Path]] = []
    for candidate_path in runs_dir.glob("*.json"):
        if candidate_path == run_path:
            continue
        try:
            candidate = _load_json(candidate_path)
        except Exception:
            continue
        if candidate.get("task_id") != run_log.get("task_id"):
            continue
        agent = candidate.get("agent")
        if agent == "exhaustive":
            priority = 0
        elif agent == "random":
            priority = 1
        else:
            priority = 2
        candidates.append((priority, candidate_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1].name), reverse=False)
    return candidates[0][1]


def _baseline_delta(current: dict[str, Any], baseline: dict[str, Any], baseline_path: Path) -> dict[str, Any]:
    current_md = current["metadata"]
    baseline_md = baseline["metadata"]
    return {
        "baseline_agent": baseline["agent"],
        "baseline_run_path": str(baseline_path),
        "composite_score_delta": round(current["composite_score"] - baseline["composite_score"], 3),
        "success_delta": int(_is_success(current_md)) - int(_is_success(baseline_md)),
        "reward_delta": float(current_md.get("terminal_state") in {"unlocked", "submit_correct"})
        - float(baseline_md.get("terminal_state") in {"unlocked", "submit_correct"}),
        "budget_used_delta": float(current_md.get("budget_used", 0)) - float(baseline_md.get("budget_used", 0)),
        "honeypot_bite_count_delta": float(current_md.get("honeypot_bite_count", 0))
        - float(baseline_md.get("honeypot_bite_count", 0)),
        "unlock_attempts_delta": float(current_md.get("unlock_attempts", 0))
        - float(baseline_md.get("unlock_attempts", 0)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Path to a run log JSON file")
    parser.add_argument("--task", help="Optional path to a task JSON file")
    parser.add_argument("--baseline-run", help="Optional explicit baseline run log JSON file")
    args = parser.parse_args(argv)

    run_path = Path(args.run)
    run_log = _load_json(run_path)
    task, task_path = _find_task_spec(run_log, Path(args.task) if args.task else None)
    current = _analyze_run(run_path, task)

    baseline_path: Path | None
    if args.baseline_run:
        baseline_path = Path(args.baseline_run)
    else:
        baseline_path = _choose_baseline(run_path, run_log)

    baseline = None
    if baseline_path is not None:
        baseline = _analyze_run(baseline_path, task)

    output = {
        "run_path": str(run_path),
        "task_path": str(task_path),
        "task_id": current["task_id"],
        "agent": current["agent"],
        "speculative_unlocks_after_key_seen": current["speculative_unlocks_after_key_seen"],
        "first_real_key_turn": current["first_real_key_turn"],
        "first_real_vault_turn": current["first_real_vault_turn"],
        "candidate_binding_count_over_time": current["candidate_binding_count_over_time"],
        "composite_score": current["composite_score"],
        "baseline_delta": (
            _baseline_delta(current, baseline, baseline_path)
            if baseline is not None and baseline_path is not None
            else None
        ),
    }
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

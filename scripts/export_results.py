"""Export v3 rollouts (excluding provider errors) for downstream charting.

Walks runs/, filters to v3-variant tasks (task_id contains 'v3'), drops
provider_error and harness_max_turns rollouts, and emits two files:

  exports/v3-rollouts.jsonl   — full per-rollout records (all metadata)
  exports/v3-summary.csv      — one row per rollout, the headline columns

Usage:  python scripts/export_results.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
OUT_DIR = REPO_ROOT / "exports"

EXCLUDE_TERMINAL_STATES = {"provider_error", "harness_max_turns"}

# Filename format: <unix_ts>-<model>-<task_id>-<hash>.json
# e.g. 1777128422-gpt-5-task-0-v3-1e8d45.json
FILENAME_RE = re.compile(r"(?P<ts>\d+)-(?P<model>.+?)-(?P<task>task-[\w-]+?)-(?P<hash>[0-9a-f]{6})\.json$")


def model_from_filename(path: Path) -> str | None:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    return m.group("model")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    skipped = {"non_v3": 0, "provider_error": 0, "harness_max_turns": 0, "unparseable": 0, "no_metadata": 0}

    for fp in sorted(RUNS_DIR.glob("*.json")):
        try:
            log = json.loads(fp.read_text())
        except Exception:
            skipped["unparseable"] += 1
            continue
        task_id = log.get("task_id", "")
        if "v3" not in task_id:
            skipped["non_v3"] += 1
            continue
        md = log.get("metadata") or {}
        if not md:
            skipped["no_metadata"] += 1
            continue
        ts = md.get("terminal_state")
        if ts in EXCLUDE_TERMINAL_STATES:
            skipped[ts] += 1
            continue
        model = model_from_filename(fp) or log.get("agent", "?")
        # Headline summary row.
        row = {
            "rollout_log": fp.name,
            "run_id": log.get("run_id"),
            "model": model,
            "task_id": task_id,
            "seed": md.get("seed"),
            "real_format": md.get("deceiver_metadata", {}).get("real_format"),
            "real_anchor": md.get("deceiver_metadata", {}).get("real_anchor"),
            "real_vault_structure": md.get("deceiver_metadata", {}).get("real_vault_structure"),
            "variant": md.get("deceiver_metadata", {}).get("variant"),
            "reward": log.get("reward"),
            "terminal_state": ts,
            "step_count": md.get("step_count"),
            "budget_used": md.get("budget_used"),
            "honeypot_bite_count": md.get("honeypot_bite_count"),
            "first_bite_turn": md.get("first_bite_turn"),
            "t_real_key_first_seen": md.get("t_real_key_first_seen"),
            "t_real_target_first_seen": md.get("t_real_target_first_seen"),
            "t_unlocked": md.get("t_unlocked"),
            "memory_span": md.get("memory_span"),
            "unlock_attempts": md.get("unlock_attempts"),
            "binding_error_wrong_key": md.get("binding_error_wrong_key"),
            "binding_error_wrong_target": md.get("binding_error_wrong_target"),
            "reasoning_tokens_total": md.get("reasoning_tokens_total"),
            "final_prompt_tokens": md.get("final_prompt_tokens"),
            "tool_histogram": md.get("tool_histogram"),
        }
        # Full record (includes turn_usage for context-curve plots).
        full_record = dict(row)
        full_record["turn_usage"] = md.get("turn_usage")
        full_record["actions_n"] = len(log.get("actions") or [])
        rows.append((row, full_record))

    # Write JSONL with full records.
    jsonl_path = OUT_DIR / "v3-rollouts.jsonl"
    with jsonl_path.open("w") as f:
        for _, full in rows:
            f.write(json.dumps(full, ensure_ascii=False) + "\n")

    # Write CSV with headline columns (drop tool_histogram + nested fields for chartability).
    csv_path = OUT_DIR / "v3-summary.csv"
    if rows:
        fieldnames = [k for k in rows[0][0].keys() if k != "tool_histogram"]
        # Flatten tool_histogram into per-tool columns.
        for tool in ["ls", "cat", "find", "grep", "stat", "submit", "unlock"]:
            fieldnames.append(f"tool_{tool}")
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row, _ in rows:
                th = row.pop("tool_histogram", None) or {}
                for tool in ["ls", "cat", "find", "grep", "stat", "submit", "unlock"]:
                    row[f"tool_{tool}"] = th.get(tool, 0)
                w.writerow(row)

    # Brief manifest.
    by_model = {}
    by_seed = {}
    for row, _ in rows:
        m = row["model"]
        by_model[m] = by_model.get(m, {"wins": 0, "losses": 0})
        by_model[m]["wins" if row["reward"] >= 1.0 else "losses"] += 1
        s = row["seed"]
        by_seed[s] = by_seed.get(s, 0) + 1

    print(f"Exported {len(rows)} rollouts to {jsonl_path} + {csv_path}")
    print(f"\nSkipped: {skipped}")
    print(f"\nBy model:")
    for m in sorted(by_model):
        b = by_model[m]
        n = b["wins"] + b["losses"]
        print(f"  {m:<14} {b['wins']:>2}/{n:<2} won")
    print(f"\nBy seed:")
    for s in sorted(by_seed.keys(), key=lambda x: x or 0):
        print(f"  seed={s}: {by_seed[s]} rollouts")


if __name__ == "__main__":
    main()

"""Export only the FINAL v3 rollouts (seeds 527/649/728) to exports/.

Filters runs/ to task_ids task-527-v3, task-649-v3, task-728-v3 (the
random-seed final batch), excludes provider_error and harness_max_turns,
and emits two files alongside the existing exports:

  exports/v3-final-rollouts.jsonl  — 1 line per rollout, full metadata
                                     (incl. turn_usage for context plots)
  exports/v3-final-summary.csv     — chart-ready columns

Excludes the earlier smoke-v3-ci / smoke-v3-clean exploration runs (those
used task-0-v3) and the externally-provided gpt-5.4 extended set
(see exports/gpt-5.4-v3-extended.jsonl for that one).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
OUT_DIR = REPO_ROOT / "exports"

FINAL_TASK_IDS = {"task-527-v3", "task-649-v3", "task-728-v3"}
EXCLUDE_TERMINAL_STATES = {"provider_error", "harness_max_turns"}

FILENAME_RE = re.compile(r"(?P<ts>\d+)-(?P<model>.+?)-(?P<task>task-[\w-]+?)-(?P<hash>[0-9a-f]{6})\.json$")


def model_from_filename(path: Path) -> str | None:
    m = FILENAME_RE.match(path.name)
    return m.group("model") if m else None


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    skipped = {"non_final": 0, "provider_error": 0, "harness_max_turns": 0, "no_metadata": 0, "unparseable": 0}

    for fp in sorted(RUNS_DIR.glob("*.json")):
        try:
            log = json.loads(fp.read_text())
        except Exception:
            skipped["unparseable"] += 1
            continue
        task_id = log.get("task_id", "")
        if task_id not in FINAL_TASK_IDS:
            skipped["non_final"] += 1
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
        dec_md = md.get("deceiver_metadata", {}) or {}
        row = {
            "rollout_log": fp.name,
            "run_id": log.get("run_id"),
            "model": model,
            "task_id": task_id,
            "seed": md.get("seed"),
            "real_format": dec_md.get("real_format"),
            "real_anchor": dec_md.get("real_anchor"),
            "real_vault_structure": dec_md.get("real_vault_structure"),
            "variant": dec_md.get("variant"),
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
        full = dict(row)
        full["turn_usage"] = md.get("turn_usage")
        full["actions_n"] = len(log.get("actions") or [])
        rows.append((row, full))

    jsonl_path = OUT_DIR / "v3-final-rollouts.jsonl"
    with jsonl_path.open("w") as f:
        for _, full in rows:
            f.write(json.dumps(full, ensure_ascii=False) + "\n")

    csv_path = OUT_DIR / "v3-final-summary.csv"
    if rows:
        fieldnames = [k for k in rows[0][0].keys() if k != "tool_histogram"]
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

    print(f"Exported {len(rows)} final-tier rollouts to {jsonl_path} + {csv_path}")
    print(f"Skipped: {skipped}")
    print()
    by_model = {}
    by_seed = {}
    for row, _ in rows:
        m = row["model"]
        by_model.setdefault(m, {"wins": 0, "trials": 0})
        by_model[m]["trials"] += 1
        if row["reward"] >= 1.0:
            by_model[m]["wins"] += 1
        s = row["seed"]
        by_seed.setdefault(s, {"wins": 0, "trials": 0})
        by_seed[s]["trials"] += 1
        if row["reward"] >= 1.0:
            by_seed[s]["wins"] += 1
    print("By model (final tier only):")
    for m in sorted(by_model):
        b = by_model[m]
        print(f"  {m:<14} {b['wins']:>2}/{b['trials']:<2}")
    print("\nBy seed:")
    for s in sorted(by_seed.keys(), key=lambda x: x or 0):
        b = by_seed[s]
        print(f"  seed={s}: {b['wins']:>2}/{b['trials']:<2}")


if __name__ == "__main__":
    main()

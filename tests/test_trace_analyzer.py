from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYZER = REPO_ROOT / "scripts" / "analyze_trace.py"
GPT_RUN = REPO_ROOT / "runs" / "1777125694-gpt-5.4-task-0-v2-184f7c.json"
BASELINE_RUN = REPO_ROOT / "runs" / "1777125081-exhaustive-task-0-v2-208c91.json"
TASK = REPO_ROOT / "tasks" / "smoke_v2.json"


def _run_analyzer(*args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(ANALYZER), *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def test_trace_analyzer_emits_expected_metrics_for_sample_trace():
    out = _run_analyzer(
        "--run",
        str(GPT_RUN),
        "--task",
        str(TASK),
        "--baseline-run",
        str(BASELINE_RUN),
    )

    assert out["speculative_unlocks_after_key_seen"] == 3
    assert out["first_real_key_turn"] == 38
    assert out["first_real_vault_turn"] == 8

    counts = {
        point["step"]: point["candidate_binding_count"]
        for point in out["candidate_binding_count_over_time"]
    }
    assert counts[1] == 0
    assert counts[19] == 1
    assert counts[24] == 2
    assert counts[27] == 3
    assert counts[38] == 4
    assert counts[54] == 5

    assert out["composite_score"] == pytest.approx(34.4)
    assert out["baseline_delta"]["baseline_agent"] == "exhaustive"
    assert out["baseline_delta"]["composite_score_delta"] == pytest.approx(66.4)
    assert out["baseline_delta"]["success_delta"] == 1


def test_trace_analyzer_can_find_baseline_automatically():
    out = _run_analyzer(
        "--run",
        str(GPT_RUN),
        "--task",
        str(TASK),
    )

    assert out["baseline_delta"]["baseline_agent"] == "exhaustive"
    assert out["baseline_delta"]["baseline_run_path"].endswith(
        "1777125081-exhaustive-task-0-v2-208c91.json"
    )

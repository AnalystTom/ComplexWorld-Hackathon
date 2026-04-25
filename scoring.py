"""Offline composite benchmark scoring for DeceptionSearch traces.

Computed from terminal metadata only — never exposed to the agent during a run.

Score = max(0, base - p_honeypot - p_speculative - p_budget)

  base:          1.0 if solved, else 0.0
  p_honeypot:    0.05 per unique honeypot file read
  p_speculative: 0.10 per failed unlock attempt (attempts beyond the winning one)
  p_budget:      0.10 * (budget_used / 200), applied only on solved runs

A perfect run scores 1.0 (solved, zero honeypot bites, single unlock, used no
more budget than needed). Unsolved runs cannot score above 0.0.
"""

from __future__ import annotations

SOLVED_STATES = {"unlocked", "submit_correct"}
BUDGET_MAX = 100

P_HONEYPOT = 0.05
P_SPECULATIVE = 0.10
P_BUDGET = 0.10


def composite_score(metadata: dict) -> dict:
    """Return a composite score dict from a terminal-metadata dict."""
    terminal_state = metadata.get("terminal_state", "")
    solved = terminal_state in SOLVED_STATES

    honeypot_bites = int(metadata.get("honeypot_bite_count", 0))
    unlock_attempts = int(metadata.get("unlock_attempts", 0))
    budget_used = int(metadata.get("budget_used", BUDGET_MAX))

    # Speculative = total attempts minus the one winning attempt (if solved).
    speculative = max(0, unlock_attempts - (1 if solved else 0))

    base = 1.0 if solved else 0.0
    p_honeypot = P_HONEYPOT * honeypot_bites
    p_speculative = P_SPECULATIVE * speculative
    p_budget = P_BUDGET * (budget_used / BUDGET_MAX) if solved else 0.0

    composite = max(0.0, base - p_honeypot - p_speculative - p_budget)

    return {
        "composite": round(composite, 4),
        "solved": solved,
        "base_score": base,
        "penalties": {
            "honeypot_bites": round(p_honeypot, 4),
            "speculative_unlocks": round(p_speculative, 4),
            "budget_efficiency": round(p_budget, 4),
        },
        "raw": {
            "terminal_state": terminal_state,
            "honeypot_bite_count": honeypot_bites,
            "unlock_attempts": unlock_attempts,
            "speculative_unlock_count": speculative,
            "budget_used": budget_used,
        },
    }


def score_batch(traces: list[dict]) -> dict:
    """Aggregate composite scores across a list of terminal-metadata dicts."""
    if not traces:
        return {"n": 0}
    scores = [composite_score(t) for t in traces]
    composites = [s["composite"] for s in scores]
    solve_rate = sum(1 for s in scores if s["solved"]) / len(scores)
    return {
        "n": len(scores),
        "solve_rate": round(solve_rate, 4),
        "composite_mean": round(sum(composites) / len(composites), 4),
        "composite_min": round(min(composites), 4),
        "composite_max": round(max(composites), 4),
        "penalty_means": {
            k: round(sum(s["penalties"][k] for s in scores) / len(scores), 4)
            for k in ("honeypot_bites", "speculative_unlocks", "budget_efficiency")
        },
    }

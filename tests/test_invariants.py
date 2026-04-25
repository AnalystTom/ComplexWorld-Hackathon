"""CI invariant: no file in base_tree.json may contain AKIA[A-Z0-9]{16}."""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_TREE = REPO_ROOT / "scenarios" / "compromised_laptop" / "base_tree.json"
KEY_REGEX = re.compile(r"AKIA[A-Z0-9]{16}")


def test_base_tree_has_no_akia_strings() -> None:
    """The base tree must NOT contain a real-looking AWS access key."""
    fs = json.loads(BASE_TREE.read_text())
    offenders: list[tuple[str, str]] = []
    for path, entry in fs.items():
        if entry.get("type") != "file":
            continue
        content = entry.get("content", "")
        m = KEY_REGEX.search(content)
        if m:
            offenders.append((path, m.group(0)))
    assert not offenders, (
        "base_tree.json contains AKIA-shaped strings:\n"
        + "\n".join(f"  {p}: {hit}" for p, hit in offenders)
    )


def test_candidates_have_secret_placeholder() -> None:
    """Each candidate template must contain exactly one {SECRET} placeholder."""
    candidates = json.loads(
        (REPO_ROOT / "scenarios" / "compromised_laptop" / "candidate_locations.json").read_text()
    )
    for c in candidates:
        n = c["content_template"].count("{SECRET}")
        assert n == 1, f"{c['path']}: expected 1 {{SECRET}} placeholder, got {n}"


def test_candidates_subset_of_allowed() -> None:
    """Each candidate path must appear in allowed_honeypot_locations."""
    candidates = json.loads(
        (REPO_ROOT / "scenarios" / "compromised_laptop" / "candidate_locations.json").read_text()
    )
    allowed = json.loads(
        (REPO_ROOT / "scenarios" / "compromised_laptop" / "allowed_honeypot_locations.json").read_text()
    )
    allowed_set = set(allowed)
    for c in candidates:
        assert c["path"] in allowed_set, (
            f"Candidate {c['path']} missing from allowed_honeypot_locations"
        )


def test_allowed_no_duplicates() -> None:
    allowed = json.loads(
        (REPO_ROOT / "scenarios" / "compromised_laptop" / "allowed_honeypot_locations.json").read_text()
    )
    assert len(allowed) == len(set(allowed)), "Duplicate paths in allowed list"


def test_allowed_locations_exist_in_tree() -> None:
    """Every allowed honeypot path must exist in base_tree.json (so overlay
    replaces an existing file rather than fabricating new ones)."""
    fs = json.loads(BASE_TREE.read_text())
    allowed = json.loads(
        (REPO_ROOT / "scenarios" / "compromised_laptop" / "allowed_honeypot_locations.json").read_text()
    )
    missing = [p for p in allowed if p not in fs]
    assert not missing, (
        "Allowed honeypot paths missing from base_tree:\n"
        + "\n".join(f"  {p}" for p in missing)
    )

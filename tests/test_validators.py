"""Tests for build_tasks.validate_honeypots."""

from __future__ import annotations

from build_tasks import HONEYPOT_TARGET, validate_honeypots


_REAL = "AKIAREALKEY12345ABCD"  # 20 chars total
_ALLOWED = [f"/p/h{i}" for i in range(20)]


def _make_hp(i: int, content: str | None = None) -> dict[str, str]:
    return {
        "path": f"/p/h{i}",
        "content": content or f"some content with AKIAFAKE{i:08d}AAAA\n",
    }


def test_valid_honeypots_pass() -> None:
    hps = [_make_hp(i) for i in range(HONEYPOT_TARGET)]
    assert validate_honeypots(hps, _ALLOWED, _REAL) == []


def test_wrong_count_fails() -> None:
    hps = [_make_hp(i) for i in range(HONEYPOT_TARGET - 1)]
    errs = validate_honeypots(hps, _ALLOWED, _REAL)
    assert any("Expected exactly" in e for e in errs)


def test_path_not_allowed_fails() -> None:
    hps = [_make_hp(i) for i in range(HONEYPOT_TARGET)]
    hps[0]["path"] = "/not-in-allowed"
    errs = validate_honeypots(hps, _ALLOWED, _REAL)
    assert any("not in ALLOWED" in e for e in errs)


def test_duplicate_paths_fail() -> None:
    hps = [_make_hp(i) for i in range(HONEYPOT_TARGET)]
    hps[1] = dict(hps[0])  # dup
    errs = validate_honeypots(hps, _ALLOWED, _REAL)
    assert any("duplicate" in e.lower() for e in errs)


def test_missing_akia_fails() -> None:
    hps = [_make_hp(i) for i in range(HONEYPOT_TARGET)]
    hps[2]["content"] = "no key here at all"
    errs = validate_honeypots(hps, _ALLOWED, _REAL)
    assert any("does not contain" in e for e in errs)


def test_real_secret_in_content_fails() -> None:
    hps = [_make_hp(i) for i in range(HONEYPOT_TARGET)]
    hps[3]["content"] = f"oops {_REAL} oops AKIAFAKE0000000000A"
    errs = validate_honeypots(hps, _ALLOWED, _REAL)
    assert any("REAL secret" in e for e in errs)


def test_oversized_content_fails() -> None:
    hps = [_make_hp(i) for i in range(HONEYPOT_TARGET)]
    hps[4]["content"] = "AKIAFAKE12345678ABCD" + "x" * 4096
    errs = validate_honeypots(hps, _ALLOWED, _REAL)
    assert any("exceeds" in e for e in errs)

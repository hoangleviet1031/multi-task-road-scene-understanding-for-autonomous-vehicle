"""End-to-end Milestone 1 verification on a generated mini dataset."""

from pathlib import Path

from roadsense.self_check import run_self_check


def test_self_check_builds_report_and_overlay(tmp_path: Path) -> None:
    """The public self-check exercises indexing, audit, split, metrics, and render."""

    report = run_self_check(tmp_path)
    assert report["ok"] is True
    assert Path(report["artifacts"]["report"]).is_file()
    assert Path(report["artifacts"]["overlay"]).is_file()

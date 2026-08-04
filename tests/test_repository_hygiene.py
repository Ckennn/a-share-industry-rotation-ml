from pathlib import Path

from scripts.check_repository_hygiene import scan_repository


def test_tracked_repository_contains_only_public_files() -> None:
    findings = scan_repository(Path.cwd())
    assert findings == [], "\n".join(str(finding) for finding in findings)

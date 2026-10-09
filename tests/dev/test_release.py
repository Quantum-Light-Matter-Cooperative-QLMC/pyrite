"""`pyrite-dev release`: version bump, due-removal gate, release notes."""

import subprocess
from pathlib import Path

import pytest

from pyrite.devtools import release


def _tree(tmp_path: Path, version: str = "0.4.0") -> Path:
    (tmp_path / "src" / "pyrite").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "x"\nversion = "{version}"\nrequires-python = ">=3.14"\n'
    )
    (tmp_path / "src" / "pyrite" / "__init__.py").write_text(f'__version__ = "{version}"\n')
    return tmp_path


def test_bump_versions_rewrites_both_sources(tmp_path: Path) -> None:
    root = _tree(tmp_path)

    assert release.bump_versions(root, "0.5.0") == ("0.4.0", "0.5.0")

    assert 'version = "0.5.0"' in (root / "pyproject.toml").read_text()
    assert 'requires-python = ">=3.14"' in (root / "pyproject.toml").read_text()
    assert (root / "src/pyrite/__init__.py").read_text() == '__version__ = "0.5.0"\n'


@pytest.mark.parametrize("bad", ["0.4.0", "0.3.9", "0.5", "0.5.0rc1", "v0.5.0"])
def test_bump_versions_rejects_non_increasing_or_malformed(tmp_path: Path, bad: str) -> None:
    root = _tree(tmp_path)

    with pytest.raises(release.ReleaseError):
        release.bump_versions(root, bad)

    assert 'version = "0.4.0"' in (root / "pyproject.toml").read_text()


def test_bump_versions_rejects_disagreeing_sources(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / "src/pyrite/__init__.py").write_text('__version__ = "0.3.0"\n')

    with pytest.raises(release.ReleaseError, match="disagree"):
        release.bump_versions(root, "0.5.0")


def test_due_removals_flags_targets_at_or_before_release_minor() -> None:
    rows = {"a": "0.4.0", "b": "0.5.0", "c": "0.6.0"}

    assert release.due_removals("0.5.0", rows) == {"a": "0.4.0", "b": "0.5.0"}
    assert release.due_removals("0.5.3", rows) == {"a": "0.4.0", "b": "0.5.0"}
    assert release.due_removals("0.4.1", rows) == {"a": "0.4.0"}
    assert release.due_removals("0.6.0", rows) == rows


def test_real_registries_are_covered_by_the_gate() -> None:
    labels = release.removal_targets()

    assert "Sweep.from_legacy" in labels
    assert "xsgen legacy table tier" in labels
    assert any(label.startswith("module ") for label in labels)
    # Everything the registries schedule is due by some far-future release.
    assert release.due_removals("99.0.0") == dict(sorted(labels.items()))


def test_classify_separates_breaking_physics_and_noise() -> None:
    assert release.classify("a", "Merge pull request #1", "") is None
    assert release.classify("a", "chore(release): bump version to 0.5.0", "") is None
    assert release.classify("a", "feat(cli)!: drop flag", "").breaking
    assert release.classify("a", "fix(api): x", "BREAKING CHANGE: y").breaking
    assert release.classify("a", "feat(coherent): weight power", "").physics
    assert not release.classify("a", "docs(coherent): words", "").physics
    assert release.classify("a", "Switch GPT input", "").type == "other"


def test_notes_since_last_tag_group_breaking_and_physics_separately(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "feat: old")
    git("tag", "v0.4.0")
    for subject in ("feat(cli)!: drop flag", "feat(coherent): weight", "fix(io): crash"):
        git("commit", "-q", "--allow-empty", "-m", subject)

    base = release.notes_base(tmp_path)
    notes = release.render_notes("0.5.0", base, release.collect_changes(tmp_path, base))

    assert base == "v0.4.0"
    assert "old" not in notes
    assert notes.index("Breaking changes") < notes.index("Physics-changing") < notes.index("Fixes")
    assert "drop flag" in notes.split("## Physics-changing")[0]
    assert "weight" in notes.split("## Physics-changing")[1].split("## Fixes")[0]


def test_notes_base_falls_back_to_last_release_commit(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "chore(release): bump version to 0.4.0")
    git("commit", "-q", "--allow-empty", "-m", "fix: later")

    base = release.notes_base(tmp_path)

    assert [c.description for c in release.collect_changes(tmp_path, base)] == ["later"]


def test_release_command_fails_before_editing_when_removals_are_due(monkeypatch, capsys) -> None:
    from pyrite.devtools import dev_cli

    monkeypatch.setattr(release, "removal_targets", lambda: {"module pyrite.old": "0.5.0"})
    monkeypatch.setattr(release, "bump_versions", lambda *a: pytest.fail("edited files"))

    with pytest.raises(SystemExit, match="module pyrite.old"):
        dev_cli.main(["release", "99.0.0", "--check"])

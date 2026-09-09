from __future__ import annotations

from pyrite.devtools.cli_commands import performance_command
from tests.helpers.cli import assert_clean_result, invoke


def _profile(root, name, files=2):
    directory = root / name
    directory.mkdir(parents=True)
    for index in range(files):
        (directory / f"artifact-{index}.ndjson").write_text("{}\n")
    return directory


def test_performance_list_reports_profiles_and_sizes(tmp_path):
    root = tmp_path / "performance-profiles"
    _profile(root, "baseline", files=2)

    result = invoke(
        performance_command,
        ["list", "--performance-dir", str(root)],
    )

    assert_clean_result(result)
    assert "baseline: 2 artifact(s), 6 bytes" in result.stdout


def test_performance_rm_previews_then_deletes_explicit_profile(tmp_path):
    root = tmp_path / "performance-profiles"
    selected = _profile(root, "baseline")
    retained = _profile(root, "keeper")
    command = ["rm", "baseline", "--performance-dir", str(root)]

    preview = invoke(performance_command, command)

    assert_clean_result(preview)
    assert str(selected) in preview.stdout
    assert "preview only" in preview.stdout
    assert selected.exists()

    deleted = invoke(performance_command, [*command, "--yes"])

    assert_clean_result(deleted)
    assert not selected.exists()
    assert retained.exists()


def test_performance_rm_requires_selection_and_rejects_traversal(tmp_path):
    root = tmp_path / "performance-profiles"
    _profile(root, "baseline")

    missing = invoke(
        performance_command,
        ["rm", "--performance-dir", str(root)],
    )
    traversal = invoke(
        performance_command,
        ["rm", "../baseline", "--performance-dir", str(root)],
    )

    assert missing.exit_code == 2
    assert traversal.exit_code == 2
    assert "needs PROFILE" in missing.stderr
    assert "letters, digits" in traversal.stderr


def test_performance_rm_all_handles_empty_root(tmp_path):
    root = tmp_path / "missing"

    result = invoke(
        performance_command,
        ["rm", "--all", "--performance-dir", str(root)],
    )

    assert_clean_result(result, stdout="(nothing to prune)\n")

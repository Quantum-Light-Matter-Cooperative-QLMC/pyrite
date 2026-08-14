from __future__ import annotations

from pyrite.cli import command as root_command
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
        root_command,
        ["performance", "list", "--performance-dir", str(root)],
    )

    assert_clean_result(result)
    assert "baseline: 2 artifact(s), 6 bytes" in result.stdout


def test_performance_rm_previews_then_deletes_explicit_profile(tmp_path):
    root = tmp_path / "performance-profiles"
    selected = _profile(root, "baseline")
    retained = _profile(root, "keeper")
    command = ["performance", "rm", "baseline", "--performance-dir", str(root)]

    preview = invoke(root_command, command)

    assert_clean_result(preview)
    assert str(selected) in preview.stdout
    assert "preview only" in preview.stdout
    assert selected.exists()

    deleted = invoke(root_command, [*command, "--yes"])

    assert_clean_result(deleted)
    assert not selected.exists()
    assert retained.exists()


def test_performance_rm_requires_selection_and_rejects_traversal(tmp_path):
    root = tmp_path / "performance-profiles"
    _profile(root, "baseline")

    missing = invoke(
        root_command,
        ["performance", "rm", "--performance-dir", str(root)],
    )
    traversal = invoke(
        root_command,
        ["performance", "rm", "../baseline", "--performance-dir", str(root)],
    )

    assert missing.exit_code == 2
    assert traversal.exit_code == 2
    assert "needs PROFILE" in missing.stderr
    assert "letters, digits" in traversal.stderr


def test_performance_rm_all_handles_empty_root(tmp_path):
    root = tmp_path / "missing"

    result = invoke(
        root_command,
        ["performance", "rm", "--all", "--performance-dir", str(root)],
    )

    assert_clean_result(result, stdout="(nothing to prune)\n")


def test_hidden_performance_prune_alias_warns_and_still_previews(tmp_path):
    root = tmp_path / "performance-profiles"
    (root / "baseline").mkdir(parents=True)
    (root / "baseline" / "gpu.ndjson").write_text("{}\n")

    result = invoke(
        root_command,
        ["performance", "prune", "baseline", "--performance-dir", str(root)],
    )

    assert result.exit_code == 0
    assert result.stderr.count("is deprecated") == 1
    assert "use 'pyrite-dev performance rm'" in result.stderr

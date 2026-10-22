import statistics
import subprocess
import sys
import time
from html.parser import HTMLParser
from pathlib import Path

import pytest
from click.testing import CliRunner

from pyrite import __version__
from pyrite.cli import command
from pyrite.devtools import cli_reference

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "docs" / "repo-design" / "cli" / "cli-reference.md"


def test_checked_cli_reference_is_current():
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_cli_reference.py"),
            "--check",
            str(REFERENCE),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    reference = REFERENCE.read_text(encoding="utf-8")
    root_help = reference.split("## `pyrite`", 1)[1].split("## `pyrite ", 1)[0]
    assert "-h, --help" in root_help


def test_reference_families_cover_live_commands_and_preserve_legacy_fragments():
    class AnchorParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = []

        def handle_starttag(self, tag, attrs):
            self.ids.extend(value for name, value in attrs if name == "id")

    pages = cli_reference.build_reference_pages()
    assert pages == cli_reference.build_reference_pages()
    overview = pages["cli-reference.md"]
    assert "schema_version" in overview
    assert "resumable compute exits 75" in overview
    anchors = AnchorParser()
    anchors.feed(overview)
    nodes = list(cli_reference._walk(command, ("pyrite",)))
    assert set(anchors.ids) == {"-".join(path) for path, _, _ in nodes if len(path) > 1}
    assert len(anchors.ids) == len(set(anchors.ids))
    for path, current, ctx in nodes:
        heading = f"## `{' '.join(path)}`\n"
        if len(path) == 1:
            assert heading in overview
            continue
        assert heading not in overview
        filename = f"cli-reference-{path[1]}.md"
        assert f"[`{' '.join(path)}`]({filename}#{'-'.join(path)})" in overview
        assert (
            sum(heading in content for name, content in pages.items() if name != "cli-reference.md")
            == 1
        )
        assert heading in pages[filename]
        assert cli_reference.ANSI_ESCAPE.sub("", current.get_help(ctx).rstrip()) in pages[filename]
    assert set(pages) == {"cli-reference.md"} | {
        f"cli-reference-{path[1]}.md" for path, _, _ in nodes if len(path) == 2
    }
    for content in pages.values():
        assert f"Version: `{__version__}`" in content
        assert "\x1b" not in content


@pytest.mark.parametrize("problem", ["missing", "stale", "obsolete"])
def test_reference_check_detects_family_drift_and_write_repairs_it(tmp_path, monkeypatch, problem):
    pages = {"cli-reference.md": "overview\n", "cli-reference-run.md": "run\n"}
    monkeypatch.setattr(cli_reference, "build_reference_pages", lambda name: pages)
    target = tmp_path / "cli-reference.md"
    assert cli_reference.main(["--write", str(target)]) == 0
    family = tmp_path / "cli-reference-run.md"
    if problem == "missing":
        family.unlink()
    elif problem == "stale":
        family.write_text("old help", encoding="utf-8")
    else:
        (tmp_path / "cli-reference-retired.md").write_text("old family", encoding="utf-8")
    manual = tmp_path / "cli-deprecations.md"
    manual.write_text("keep", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(SystemExit) as exc:
        cli_reference.main(["--check", str(target)])
    assert exc.value.code == 2
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    assert cli_reference.main(["--write", str(target)]) == 0
    assert cli_reference.main(["--check", str(target)]) == 0
    assert not (tmp_path / "cli-reference-retired.md").exists()
    assert manual.read_text(encoding="utf-8") == "keep"


def test_help_documents_examples_units_side_effects_and_incompatibilities():
    runner = CliRunner()
    cases = {
        (): ("Examples:", "pyrite run"),
        ("run",): ("minutes", "--preset"),
        ("remote",): ("PYRITE_REMOTE_HOST", "Examples:"),
        ("remote", "pull"): ("--preset", "grid-filtered"),
        ("numerics", "energy-grid"): ("detector energy-grid inputs", "Examples:"),
        ("numerics", "energy-grid", "derive"): ("keV", "angstrom", "spacing in eV"),
        ("remote", "job", "status"): ("defaults to latest", "repeat"),
    }
    for path, expected in cases.items():
        result = runner.invoke(command, [*path, "--help"])
        assert result.exit_code == 0
        for text in expected:
            assert text in result.output


def test_root_help_warm_median_below_200_ms():
    durations = []
    for _ in range(7):
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, "-m", "pyrite.cli", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        durations.append(time.perf_counter() - started)
        assert completed.returncode == 0
        assert completed.stderr == ""
    warm_median = statistics.median(durations[1:])
    assert warm_median < 0.2, f"warm median {warm_median:.3f}s >= 0.200s"

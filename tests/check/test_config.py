"""CLI checks for validating the bundled or an explicit material catalog."""

import subprocess
import sys
from pathlib import Path

import pytest

from pyrite import DATA_DIR, cli


def test_check_config_validates_bundled_catalog_without_running_simulation(capsys) -> None:
    assert cli.main(["check-config"]) is None

    output = capsys.readouterr().out
    assert "valid material catalog" in output
    assert "49 materials, 48 crystals" in output


def test_check_config_accepts_an_explicit_full_catalog(capsys) -> None:
    catalog_path = DATA_DIR / "materials.toml"

    assert cli.main(["check-config", str(catalog_path)]) is None

    assert str(catalog_path) in capsys.readouterr().out


def test_check_config_reports_custom_catalog_errors_as_clean_cli_errors(
    tmp_path: Path, capsys
) -> None:
    invalid = tmp_path / "invalid-materials.toml"
    invalid.write_text('materials = ["hopg"]\n')

    with pytest.raises(SystemExit) as exc:
        cli.main(["check-config", str(invalid)])

    assert exc.value.code == 1
    stderr = capsys.readouterr().err
    assert "invalid material catalog" in stderr
    assert "Traceback" not in stderr


def test_check_config_reports_malformed_bundled_catalog_without_import_traceback(
    tmp_path: Path,
) -> None:
    invalid = tmp_path / "broken-bundled.toml"
    invalid.write_text('materials = ["hopg"]\n')
    script = r"""
import sys
from pathlib import Path

invalid = Path(sys.argv[1])
real_open = Path.open

def redirected_open(path, *args, **kwargs):
    if path.name == "materials.toml":
        return real_open(invalid, *args, **kwargs)
    return real_open(path, *args, **kwargs)

Path.open = redirected_open
from pyrite import cli
cli.main(["check-config"])
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(invalid)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "invalid material catalog" in result.stderr
    assert "Traceback" not in result.stderr

"""CLI checks for validating the bundled or an explicit material catalog."""

from pathlib import Path

import pytest

from cxr_mc import DATA_DIR, cli


def test_check_config_validates_bundled_catalog_without_running_simulation(capsys) -> None:
    assert cli.main(["check-config"]) is None

    output = capsys.readouterr().out
    assert "valid material catalog" in output
    assert "21 materials" in output
    assert "21 crystals" in output


def test_check_config_accepts_an_explicit_full_catalog(capsys) -> None:
    catalog_path = DATA_DIR / "materials.toml"

    assert cli.main(["check-config", str(catalog_path)]) is None

    assert str(catalog_path) in capsys.readouterr().out


def test_check_config_reports_custom_catalog_errors_as_clean_cli_errors(
    tmp_path: Path, capsys
) -> None:
    invalid = tmp_path / "invalid-materials.toml"
    invalid.write_text('materials = ["hopg"]\n')

    with pytest.raises(SystemExit, match="invalid material catalog"):
        cli.main(["check-config", str(invalid)])

    assert "Traceback" not in capsys.readouterr().err

"""Physical CLI options accept a bare number or an explicit unit string (#810)."""

import math

import click
import pytest
from click.testing import CliRunner

from pyrite.cli import _catalog_io, command
from pyrite.cli._unit_params import canonical_unit, unit_float, unit_pair
from pyrite.cli.commands import beam, detector
from pyrite.console.output import FINITE_FLOAT, POSITIVE_FLOAT
from tests.helpers.cli import assert_clean_result, invoke


def _probe(param_type, *args, nargs=1):
    @click.command()
    @click.option("--value", type=param_type, nargs=nargs)
    def probe(value):
        click.echo(repr(value))

    return CliRunner().invoke(probe, list(args))


def test_bare_number_unchanged_and_unit_string_converts():
    kind = unit_float(POSITIVE_FLOAT, "mm", "length")
    assert _probe(kind, "--value", "0.5").output.strip() == "0.5"
    assert _probe(kind, "--value", "1e-1").output.strip() == "0.1"
    assert float(_probe(kind, "--value", "0.1 cm").output) == pytest.approx(1.0)
    assert float(_probe(kind, "--value", "2 um").output) == pytest.approx(2e-3)
    assert float(_probe(kind, "--value", " 3mm ").output) == pytest.approx(3.0)
    assert canonical_unit(kind) == "mm"


def test_unit_string_goes_through_the_base_range():
    kind = unit_float(POSITIVE_FLOAT, "mm", "length")
    bare = _probe(kind, "--value", "-1")
    unit = _probe(kind, "--value", "-1 mm")
    zero = _probe(kind, "--value", "0 mm")
    assert bare.exit_code == unit.exit_code == zero.exit_code == 2
    assert "must be greater than 0.0" in unit.output
    assert "must be greater than 0.0" in zero.output


def test_wrong_dimension_and_garbage_are_usage_errors():
    kind = unit_float(POSITIVE_FLOAT, "mm", "length")
    wrong = _probe(kind, "--value", "5 eV")
    assert wrong.exit_code == 2
    assert "--value" in wrong.output
    assert "expected length convertible to 'mm'" in wrong.output
    assert "bare number in mm" in wrong.output
    garbage = _probe(kind, "--value", "abc")
    assert garbage.exit_code == 2
    assert "could not parse 'abc'" in garbage.output
    assert _probe(kind, "--value", "inf mm").exit_code == 2
    assert _probe(kind, "--value", "nan").exit_code == 2
    assert "Traceback" not in wrong.output + garbage.output


def test_angle_guards_have_no_implicit_equivalency():
    angle = unit_float(FINITE_FLOAT, "deg", "angle", "angle")
    assert float(_probe(angle, "--value", "1 rad").output) == pytest.approx(180 / math.pi)
    assert float(_probe(angle, "--value", "90 deg").output) == pytest.approx(90.0)
    assert float(_probe(angle, "--value", "90").output) == 90.0
    for bad in ("1 sr", "1 mm", "50 percent"):
        assert _probe(angle, "--value", bad).exit_code == 2, bad
    solid = unit_float(POSITIVE_FLOAT, "sr", "solid angle", "solid_angle")
    assert float(_probe(solid, "--value", "0.2 sr").output) == pytest.approx(0.2)
    assert _probe(solid, "--value", "1 rad").exit_code == 2
    assert _probe(solid, "--value", "1 deg").exit_code == 2


def test_emittance_requires_length_angle():
    emit = unit_float(POSITIVE_FLOAT, "mm*mrad", "length-angle", "emittance")
    assert float(_probe(emit, "--value", "1 um*rad").output) == pytest.approx(1.0)
    assert float(_probe(emit, "--value", "2").output) == 2.0
    assert _probe(emit, "--value", "1 mm").exit_code == 2
    assert _probe(emit, "--value", "1 rad").exit_code == 2


def test_pair_converts_each_entry():
    pair = unit_pair(float, "mm", "length")
    result = _probe(pair, "--value", "1 cm", "5", nargs=2)
    assert result.exit_code == 0
    assert result.output.strip() == "(10.0, 5.0)"
    assert _probe(pair, "--value", "1 cm", "5 eV", nargs=2).exit_code == 2


def test_help_range_hint_and_metavar_unchanged():
    @click.command()
    @click.option("--x", type=unit_float(click.FloatRange(min=0.0, min_open=True), "mm", "length"))
    @click.option("--y", type=unit_float(float, "mm", "length"))
    def probe(x, y):
        pass

    text = CliRunner().invoke(probe, ["--help"]).output
    assert "[x>0.0]" in text
    assert "FLOAT RANGE" in text
    assert "--y FLOAT" in text


_CATALOG = """[materials.hopg]
"""


def test_beam_create_accepts_unit_strings_with_identical_stored_values(tmp_path, monkeypatch):
    def create(*args):
        catalog = tmp_path / f"{len(args)}.toml"
        catalog.write_text(_CATALOG)
        monkeypatch.setattr(_catalog_io, "_CATALOG_PATH", catalog)
        monkeypatch.setattr(_catalog_io, "validate", lambda *_a: None)
        result = invoke(beam.command, ["create", "gun", *args])
        assert_clean_result(result)
        return catalog.read_text()

    bare = create(
        "--rep-rate-hz", "5000", "--bunch-charge-pc", "2.5", "--transverse-fwhm-mm", "0.1"
    )
    unit = create(
        "--rep-rate-hz", "5 kHz", "--bunch-charge-pc", "2.5 pC", "--transverse-fwhm-mm", "100 um"
    )
    assert unit == bare


def test_beam_create_rejects_wrong_dimension(tmp_path, monkeypatch):
    catalog = tmp_path / "materials.toml"
    catalog.write_text(_CATALOG)
    monkeypatch.setattr(_catalog_io, "_CATALOG_PATH", catalog)
    monkeypatch.setattr(_catalog_io, "validate", lambda *_a: None)
    result = CliRunner().invoke(beam.command, ["create", "gun", "--bunch-charge-pc", "5 mm"])
    assert result.exit_code == 2
    assert "--bunch-charge-pc" in result.stderr
    assert catalog.read_text() == _CATALOG


def test_detector_angle_options_keep_angle_guards():
    runner = CliRunner()
    result = runner.invoke(detector.command, ["create", "scalar", "x", "--solid-angle", "1 deg"])
    assert result.exit_code == 2
    assert "solid angle" in result.stderr
    result = runner.invoke(
        detector.command, ["create", "scalar", "x", "--observation-angle", "1 sr"]
    )
    assert result.exit_code == 2
    assert "angle" in result.stderr


def _unit_params(cmd, path=()):
    ctx = click.Context(cmd)
    if isinstance(cmd, click.Group):
        for name in cmd.list_commands(ctx):
            sub = cmd.get_command(ctx, name)
            if sub is not None:
                yield from _unit_params(sub, (*path, name))
    for param in cmd.params:
        types = param.type.types if isinstance(param.type, click.Tuple) else (param.type,)
        for item in types:
            unit = canonical_unit(item)
            if unit is not None:
                yield " ".join(path), param, unit


def test_every_unit_option_help_names_its_canonical_unit():
    spellings = {
        "mm": ("mm",),
        "um": ("um", "micrometre"),
        "deg": ("degree",),
        "angstrom": ("angstrom",),
        "m": (" m", "meter"),
        "eV": ("eV",),
        "keV": ("keV",),
        "MeV": ("MeV",),
        "s": (" s", "second"),
        "fs": ("fs", "FS"),
        "pC": ("pC",),
        "Hz": ("Hz",),
        "V": ("volts", " V"),
        "sr": ("sr",),
        "mm*mrad": ("mm*mrad",),
        "g/cm**3": ("g/cm",),
        "angstrom**2": ("angstrom",),
    }
    seen = 0
    missing = []
    for path, param, unit in _unit_params(command):
        seen += 1
        text = f"{param.help or ''} {param.metavar or ''}".lower()
        if not any(token.lower() in text for token in spellings[unit]):
            missing.append(f"{path} {param.opts[0]} [{unit}]")
    assert seen > 100
    assert not missing

"""Regression tests for ``cxr scan`` BeamSpec distribution overrides."""

from click.testing import CliRunner

from cxr_mc import scan


def _resolved_sweep(monkeypatch, argv):
    captured = {}

    def capture(args):
        captured["sweep"] = scan._resolved_run(args, "hopg")[1]

    monkeypatch.setattr(scan, "run", capture)
    result = CliRunner().invoke(scan.command, ["hopg", *argv], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return captured["sweep"]


def test_scan_beam_options_override_resolved_distribution(monkeypatch):
    sweep = _resolved_sweep(
        monkeypatch,
        [
            "--beam-transverse-fwhm-x-mm",
            "0.2",
            "--beam-transverse-fwhm-y-mm",
            "0.4",
            "--beam-bunch-length-fs",
            "50",
            "--beam-long-shape",
            "uniform",
            "--beam-rep-rate-hz",
            "0",
            "--beam-bunch-charge-pc",
            "12.5",
        ],
    )

    assert sweep.beam.transverse_fwhm_x_mm == 0.2
    assert sweep.beam.transverse_fwhm_y_mm == 0.4
    assert sweep.beam.bunch_length_fs == 50.0
    assert sweep.beam.long_shape == "uniform"
    assert sweep.beam.rep_rate_hz == 0.0
    assert sweep.beam.bunch_charge_pc == 12.5


def test_scan_beam_options_do_not_override_material_energy_grid(monkeypatch):
    baseline = _resolved_sweep(monkeypatch, [])
    sweep = _resolved_sweep(monkeypatch, ["--beam-bunch-length-fs", "10"])

    assert tuple(sweep.beam.energy_keV) == tuple(baseline.beam.energy_keV)


def test_scan_performance_profile_selects_catalog_profile(monkeypatch):
    captured = {}

    def capture(args):
        captured["identity"] = scan._resolved_run(args, "hopg")[2]

    monkeypatch.setattr(scan, "run", capture)
    result = CliRunner().invoke(
        scan.command,
        ["hopg", "--performance-profile", "sub_100keV"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert captured["identity"]["catalog_profile"] == "sub_100keV"


def test_scan_rejects_conflicting_performance_and_catalog_profiles(monkeypatch):
    monkeypatch.setattr(scan, "run", lambda args: scan._resolved_run(args, "hopg"))
    result = CliRunner().invoke(
        scan.command,
        [
            "hopg",
            "--profile",
            "sub_100keV",
            "--performance-profile",
            "standard",
        ],
    )

    assert result.exit_code == 2
    assert "must name the same catalog profile" in result.output


def test_scan_beam_option_help_and_validation():
    runner = CliRunner()
    help_result = runner.invoke(scan.command, ["--help"])
    assert help_result.exit_code == 0
    assert "--beam-transverse-fwhm-x-mm MM" in help_result.output
    assert "--beam-bunch-length-fs FS" in help_result.output
    assert "Beam overrides:" in help_result.output
    assert "--beam-energy-kev" not in help_result.output

    invalid = runner.invoke(scan.command, ["hopg", "--beam-rep-rate-hz", "-1"])
    assert invalid.exit_code == 2
    assert "--beam-rep-rate-hz" in invalid.output
    assert "must be at least 0.0" in invalid.output

    incomplete = runner.invoke(scan.command, ["hopg", "--beam-long-shape", "uniform"])
    assert incomplete.exit_code == 2
    assert "--beam-long-shape requires --beam-bunch-length-fs" in incomplete.output

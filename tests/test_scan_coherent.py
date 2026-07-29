"""Regression tests for the ``cxr run --coherent/--incoherent`` policy flag."""

from click.testing import CliRunner

from cxr_mc import scan


def _resolved_run(monkeypatch, argv):
    captured = {}

    def capture(args):
        captured["run"] = scan._resolved_run(args, "hopg")

    monkeypatch.setattr(scan, "run", capture)
    result = CliRunner().invoke(
        scan.command, ["standard", "-m", "hopg", *argv], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    return captured["run"]


def test_scan_coherent_sets_settings_and_qualified_stem(monkeypatch):
    settings, _sweep, identity, stem = _resolved_run(monkeypatch, ["--coherent"])

    assert settings.coherent_emission is True
    # coherent joins the hash (divergence-only rule) and forces off canonical <material>
    assert identity["resolved_parameters"].get("coherent_emission") is True
    assert stem != "hopg"
    assert stem.startswith("hopg--full-")


def test_scan_default_is_incoherent_canonical_stem(monkeypatch):
    settings, _sweep, identity, stem = _resolved_run(monkeypatch, [])

    assert settings.coherent_emission is False
    # incoherent run keeps its historical bare <material> stem and digest
    assert "coherent_emission" not in identity["resolved_parameters"]
    assert stem == "hopg"


def test_scan_incoherent_flag_matches_default(monkeypatch):
    _s_default, _sw, id_default, stem_default = _resolved_run(monkeypatch, [])
    _s_off, _sw_off, id_off, stem_off = _resolved_run(monkeypatch, ["--incoherent"])

    # explicit --incoherent is the default: same canonical stem and digest
    assert stem_off == stem_default == "hopg"
    assert id_off["parameter_sha256"] == id_default["parameter_sha256"]


def test_scan_coherent_and_incoherent_never_collide(monkeypatch):
    _s_on, _sw_on, id_on, stem_on = _resolved_run(monkeypatch, ["--coherent"])
    _s_off, _sw_off, id_off, stem_off = _resolved_run(monkeypatch, ["--incoherent"])

    assert stem_on != stem_off
    assert id_on["parameter_sha256"] != id_off["parameter_sha256"]


def test_scan_coherent_help_lists_both_switches():
    result = CliRunner().invoke(scan.command, ["--help"])

    assert result.exit_code == 0
    assert "--coherent / --incoherent" in result.output
    assert "incoherent, the default" in result.output


def test_scan_rejects_quick_coherent_collision():
    # --quick writes the digest-free <material>_quick stem, so a coherent quick
    # run would clobber the incoherent quick checkpoint; reject the combination.
    result = CliRunner().invoke(scan.command, ["hopg", "--quick", "--coherent"])

    assert result.exit_code == 2
    assert "--quick cannot be combined with --coherent" in result.output


def test_scan_allows_quick_incoherent(monkeypatch):
    # --quick --incoherent is the default policy, not a collision: it resolves
    # to the digest-free <material>_quick stem with coherent tracking off.
    settings, _sweep, _identity, stem = _resolved_run(monkeypatch, ["--quick", "--incoherent"])

    assert settings.coherent_emission is False
    assert stem == "hopg_quick"

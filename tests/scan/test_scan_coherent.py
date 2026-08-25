"""Regression tests for the profile-owned emission policy on ``pyrite run``.

The former ``--coherent/--incoherent`` CLI flags are gone: emission
(``incoherent``/``coherent``/``both``) is resolved from the profile onto
``Settings.emission`` and drives both the ``dataset_identity`` divergence key
and the canonical-stem collision guard. These tests inject the emission a
profile would resolve (via ``default_settings``) and assert the resulting stem
and digest, plus the absence of the removed flags."""

from dataclasses import replace

from click.testing import CliRunner

from pyrite import materials
from pyrite.runs import scan


def _resolved_run(monkeypatch, argv, emission=None):
    """Invoke ``pyrite run`` and capture ``_resolved_run``'s output. When
    ``emission`` is given, wrap ``default_settings`` so the resolved profile
    reports that emission mode -- the profile-owned path that replaces the
    removed CLI flags."""
    captured = {}

    def capture(args):
        captured["run"] = scan._resolved_run(args, "hopg")

    monkeypatch.setattr(scan, "run", capture)
    if emission is not None:
        scan._load_runtime()
        base = scan.default_settings
        monkeypatch.setattr(
            scan,
            "default_settings",
            lambda *a, **k: replace(base(*a, **k), emission=emission),
        )
    result = CliRunner().invoke(
        scan.command, ["standard", "-m", "hopg", *argv], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    return captured["run"]


def test_default_run_is_incoherent_canonical_stem(monkeypatch):
    settings, _sweep, identity, stem = _resolved_run(monkeypatch, [])

    assert settings.emission == "incoherent"
    assert settings.coherent_emission is False
    # incoherent keeps its historical bare <material> stem and adds no key
    assert "emission" not in identity["resolved_parameters"]
    assert stem == "hopg"


def test_coherent_profile_gets_qualified_stem_and_divergent_digest(monkeypatch):
    _s_def, _sw, id_def, stem_def = _resolved_run(monkeypatch, [])
    settings, _sweep, identity, stem = _resolved_run(monkeypatch, [], emission="coherent")

    assert settings.emission == "coherent"
    assert settings.coherent_emission is True
    # coherent joins the hash (divergence-only rule) and forces off canonical <material>
    assert identity["resolved_parameters"].get("emission") == "coherent"
    assert stem != "hopg"
    assert stem.startswith("hopg@full-")
    assert identity["parameter_sha256"] != id_def["parameter_sha256"]


def test_catalog_profile_transport_numerics_reach_run_identity(monkeypatch):
    _s_default, _sw_default, id_default, _stem_default = _resolved_run(monkeypatch, [])
    catalog = materials.CATALOG
    straggled_catalog = replace(
        catalog,
        profile_transport_numerics={
            **catalog.profile_transport_numerics,
            "standard": {
                "straggling": True,
                "energy_model": "midpoint",
                "max_dE_frac": 0.02,
            },
        },
    )
    monkeypatch.setattr(materials, "CATALOG", straggled_catalog)

    settings, _sweep, identity, stem = _resolved_run(monkeypatch, [])

    assert settings.straggling is True
    assert settings.energy_model == "midpoint"
    assert settings.max_dE_frac == 0.02
    assert identity["resolved_parameters"]["transport_numerics"] == {
        "straggling": True,
        "energy_model": "midpoint",
        "max_dE_frac": 0.02,
    }
    assert stem.startswith("hopg@full-")
    assert identity["parameter_sha256"] != id_default["parameter_sha256"]


def test_both_profile_gets_qualified_stem_and_divergent_digest(monkeypatch):
    settings, _sweep, identity, stem = _resolved_run(monkeypatch, [], emission="both")

    assert settings.emission == "both"
    assert settings.coherent_emission is True
    assert identity["resolved_parameters"].get("emission") == "both"
    assert stem != "hopg"
    assert stem.startswith("hopg@full-")


def test_three_emission_modes_never_collide(monkeypatch):
    _s_i, _sw_i, id_i, stem_i = _resolved_run(monkeypatch, [])
    _s_c, _sw_c, id_c, stem_c = _resolved_run(monkeypatch, [], emission="coherent")
    _s_b, _sw_b, id_b, stem_b = _resolved_run(monkeypatch, [], emission="both")

    stems = {stem_i, stem_c, stem_b}
    digests = {
        id_i["parameter_sha256"],
        id_c["parameter_sha256"],
        id_b["parameter_sha256"],
    }
    assert len(stems) == 3
    assert len(digests) == 3
    assert stem_i == "hopg"  # incoherent stays canonical


def test_coherent_incoherent_flags_are_removed():
    # The policy flags no longer exist -- Click rejects them as unknown options.
    for flag in ("--coherent", "--incoherent"):
        result = CliRunner().invoke(scan.command, ["standard", "-m", "hopg", flag])
        assert result.exit_code == 2, result.output
        assert "no such option" in result.output.lower()


def test_help_no_longer_lists_coherent_incoherent():
    result = CliRunner().invoke(scan.command, ["--help"])

    assert result.exit_code == 0
    assert "--coherent" not in result.output
    assert "--incoherent" not in result.output


def test_catalog_profile_emission_key_reaches_resolved_settings(monkeypatch):
    """End-to-end wiring check for ``CATALOG.profile_emission`` (set via ``pyrite
    profile set/add/remove --emission/--coherent/--incoherent``), as opposed to
    the other tests in this module which inject the emission by monkeypatching
    ``default_settings`` directly and never exercise the catalog lookup in
    ``scan._resolved_run``."""
    catalog = materials.CATALOG
    coherent_catalog = replace(
        catalog, profile_emissions={**catalog.profile_emissions, "standard": "coherent"}
    )
    monkeypatch.setattr(materials, "CATALOG", coherent_catalog)

    settings, _sweep, identity, stem = _resolved_run(monkeypatch, [])

    assert settings.emission == "coherent"
    assert settings.coherent_emission is True
    assert identity["resolved_parameters"].get("emission") == "coherent"
    assert stem != "hopg"
    assert stem.startswith("hopg@full-")


def test_quick_run_is_incoherent_quick_stem(monkeypatch):
    # --quick resolves to the digest-free <material>_quick stem; emission stays
    # the profile default (incoherent). No coherent-quick collision to reject now
    # that emission is profile-owned rather than a transient flag.
    settings, _sweep, _identity, stem = _resolved_run(monkeypatch, ["--quick"])

    assert settings.emission == "incoherent"
    assert stem == "hopg_quick"

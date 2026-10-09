"""The adaptive default and its fixed-count fallbacks (#361)."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyrite import materials
from pyrite._precision import DEFAULT_PRECISION, Precision
from pyrite.campaign.config import default_settings
from pyrite.campaign.profiles import profile_precision_source, profile_run_settings


def _sweep(*, n_electrons=None, gdf=False, grooved=False):
    return SimpleNamespace(
        n_electrons=n_electrons,
        n_electrons_brem=None,
        beam=SimpleNamespace(gdf_beam=lambda: object() if gdf else None),
        target=SimpleNamespace(entrance_face=object() if grooved else None),
    )


def _with_standard(monkeypatch, field, value):
    catalog = materials.CATALOG
    monkeypatch.setattr(
        materials,
        "CATALOG",
        replace(catalog, **{field: {**getattr(catalog, field), "standard": value}}),
    )


def test_default_policy_matches_the_recorded_decision():
    assert DEFAULT_PRECISION == Precision(
        target_rse=0.05,
        min_electrons=200,
        max_electrons=20_000,
        block_electrons=100,
        observables=("line", "brem"),
    )


def test_a_plain_profile_runs_the_default_policy():
    settings = profile_run_settings(default_settings(), "standard", "full", sweep=_sweep())
    assert settings.precision == DEFAULT_PRECISION
    assert profile_precision_source(settings, "standard", sweep=_sweep())[1] == "default"


def test_an_explicit_table_wins(monkeypatch):
    policy = Precision(target_rse=0.1, min_electrons=20, max_electrons=60, block_electrons=20)
    _with_standard(monkeypatch, "profile_precisions", policy.to_dict())
    precision, source = profile_precision_source(default_settings(), "standard")
    assert (precision, source) == (policy, "profile")


@pytest.mark.parametrize(
    "sweep, reason",
    [
        (_sweep(n_electrons=[450]), "fixed electron counts"),
        (_sweep(gdf=True), "GDF"),
        (_sweep(grooved=True), "grooved"),
    ],
)
def test_per_material_routes_keep_fixed_counts(sweep, reason):
    precision, source = profile_precision_source(default_settings(), "standard", sweep=sweep)
    assert precision is None
    assert source.startswith("fixed: ") and reason in source


@pytest.mark.parametrize(
    "field, value, reason",
    [
        ("profile_transport_numerics", {"n_electrons": (450,)}, "fixed electron counts"),
        ("profile_emissions", "coherent", "emission is 'coherent'"),
        ("profile_emissions", "both", "emission is 'both'"),
    ],
)
def test_profile_settings_keep_fixed_counts(monkeypatch, field, value, reason):
    _with_standard(monkeypatch, field, value)
    settings = profile_run_settings(default_settings(), "standard", "full", sweep=_sweep())
    assert settings.precision is None
    assert reason in profile_precision_source(settings, "standard", sweep=_sweep())[1]


def test_cascades_keep_fixed_counts():
    # Only the fields the resolver reads; a valid cascade Settings needs shell tables.
    settings = SimpleNamespace(
        emission="incoherent",
        secondary_threshold_eV=1000.0,
        pair_production_model=None,
        positron_transport=False,
    )
    precision, source = profile_precision_source(settings, "standard", sweep=_sweep())
    assert precision is None and "cascades" in source

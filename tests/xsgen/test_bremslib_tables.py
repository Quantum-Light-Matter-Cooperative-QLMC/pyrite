"""Resolving BremsLib tables for the direction-resolved bremsstrahlung spectrum.

The resolution policy is tested against synthetic stored tables. The physics
checks at the bottom use the released catalogue tables when they are
installed (``pyrite tables fetch bremslib``) and skip otherwise; the vendor
oracle that needs the full library lives in ``test_extern_codes.py``.

Validation: bremslib-angular-model
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum import brem
from pyrite.montecarlo.spectrum.brem_bremslib import (
    bremslib_segment_state,
    evaluate_bremslib,
    stage_bremslib_table,
)
from pyrite.xsgen._errors import SourceUnavailableError, TableNotFoundError
from pyrite.xsgen.bremslib import tables as bremslib_tables
from pyrite.xsgen.bremslib.release import catalogue_elements, catalogue_table
from pyrite.xsgen.store import StoredTable
from tests.helpers import to_host
from tests.helpers.bremslib import synthetic_bremslib_arrays


@pytest.fixture
def stored(tmp_path):
    path = tmp_path / "table.npz"
    np.savez(path, **synthetic_bremslib_arrays())
    return StoredTable(
        key="synthetic-key", path=path, manifest={"manifest_sha256": "digest"}, tier="user"
    )


def test_released_element_resolves_through_the_catalogue(monkeypatch, stored):
    calls = []
    monkeypatch.setattr(bremslib_tables, "catalogue_table", lambda z: calls.append(z) or stored)
    monkeypatch.setattr(
        bremslib_tables,
        "generate_element",
        lambda *a, **k: pytest.fail("a released element must not read a checkout"),
    )
    tables = bremslib_tables.load_bremsstrahlung_tables(["C", "C"])
    assert calls == [6]
    assert list(tables) == ["C"]
    assert tables["C"].atomic_number == 6
    assert bremslib_tables.table_identity(tables) == {"synthetic-key": "digest"}


def test_released_but_unfetched_table_is_an_error_not_a_fallback(monkeypatch):
    def missing(z):
        raise TableNotFoundError("run `pyrite tables fetch bremslib`")

    monkeypatch.setattr(bremslib_tables, "catalogue_table", missing)
    with pytest.raises(TableNotFoundError, match="fetch bremslib"):
        bremslib_tables.load_bremsstrahlung_tables(["C"])


def test_unreleased_element_generates_from_a_checkout(monkeypatch, stored):
    class Result:
        table = stored

    seen = {}

    def generate(z, *, source_path=None):
        seen.update(z=z, source_path=source_path)
        return Result()

    monkeypatch.setattr(bremslib_tables, "generate_element", generate)
    tables = bremslib_tables.load_bremsstrahlung_tables(["Au"], source_path="/checkout")
    assert seen == {"z": 79, "source_path": "/checkout"}
    assert tables["Au"].atomic_number == 79


def test_unreleased_element_without_checkout_warns_and_is_left_out(monkeypatch):
    def unavailable(z, *, source_path=None):
        raise SourceUnavailableError("no BremsLib checkout")

    monkeypatch.setattr(bremslib_tables, "generate_element", unavailable)
    with pytest.warns(RuntimeWarning, match="isotropic EEDL"):
        tables = bremslib_tables.load_bremsstrahlung_tables(["Au"])
    assert tables == {}


def test_unknown_element_is_rejected():
    with pytest.raises(ValueError, match="unknown element"):
        bremslib_tables.load_bremsstrahlung_tables(["Xx"])


# --- released-table physics checks ------------------------------------------------


def _released(element):
    from pyrite.materials.atomic import Z_TABLE

    z = Z_TABLE[element]
    if z not in catalogue_elements():
        pytest.skip(f"{element} is not a catalogue element")
    try:
        catalogue_table(z)
    except TableNotFoundError as exc:
        pytest.skip(f"released BremsLib tables are not installed: {exc}")
    return bremslib_tables.load_bremsstrahlung_tables([element])[element]


def _shape_over_isotropic(table, T_keV, theta_deg, reduced=0.5):
    staged = stage_bremslib_table(table)
    k = np.array([reduced * T_keV * 1.0e3])
    T = np.array([T_keV])
    sdcs = to_host(evaluate_bremslib(staged, bremslib_segment_state(staged, T), k))[0, 0]
    ddcs = np.array(
        [
            to_host(
                evaluate_bremslib(
                    staged,
                    bremslib_segment_state(staged, T, np.array([np.cos(np.radians(t))])),
                    k,
                )
            )[0, 0]
            for t in np.atleast_1d(theta_deg)
        ]
    )
    return ddcs * 4.0 * np.pi / sdcs


@pytest.mark.parametrize("element", ["C", "W"])
def test_low_energy_emission_approaches_the_symmetric_dipole_form(element):
    """Non-relativistic limit: peaked near 90 degrees and fore-aft symmetric."""
    table = _released(element)
    theta = np.arange(0.0, 181.0, 1.0)
    shape = _shape_over_isotropic(table, 0.1, theta)
    assert 80.0 <= theta[np.argmax(shape)] <= 100.0
    np.testing.assert_allclose(shape, shape[::-1], rtol=0.15)


@pytest.mark.parametrize("element", ["C", "W"])
def test_forward_peaking_grows_with_incident_energy(element):
    table = _released(element)
    theta = np.arange(0.0, 181.0, 1.0)
    peaks = [theta[np.argmax(_shape_over_isotropic(table, T, theta))] for T in (1.0, 10.0, 100.0)]
    assert peaks[0] > peaks[1] > peaks[2]
    forward = [_shape_over_isotropic(table, T, [0.0])[0] for T in (100.0, 1000.0, 10000.0)]
    assert forward[0] < forward[1] < forward[2]
    assert forward[2] > 100.0


@pytest.mark.parametrize("element", ["C", "W"])
def test_bremslib_sdcs_matches_eedl_at_an_eedl_panel(element):
    """Units anchor, not the #86 comparison: the two evaluations agree to ~10 %.

    Checked at an EEDL panel energy, where EEDL needs no panel interpolation.
    """
    table = _released(element)
    eedl = brem.load_bremsstrahlung_cross_sections(element)
    panels = eedl.distribution_incident_energy_eV
    incident_eV = panels[np.argmin(np.abs(np.log(panels / 3.0e5)))]
    photon_eV = np.array([0.1, 0.3, 0.5, 0.7, 0.9]) * incident_eV
    T = np.array([incident_eV / 1.0e3])
    staged = stage_bremslib_table(table)
    bremslib = to_host(evaluate_bremslib(staged, bremslib_segment_state(staged, T), photon_eV))[0]
    reference = to_host(brem._eedl_brem_dsigma_dk(element, T, photon_eV))[0]
    np.testing.assert_allclose(bremslib, reference, rtol=0.15)

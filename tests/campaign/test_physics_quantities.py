"""Semantic Physics energies/origin and EnergyBins grids stay numeric (#755)."""

import pickle
from dataclasses import asdict, fields, replace

import numpy as np
import pytest

from pyrite.config import Physics
from pyrite.detectors import EnergyBins
from pyrite.units import Q_, ureg

SHELL = {"inelastic_model": "shell-soft-hard"}


def test_physics_energies_match_numeric_fields() -> None:
    semantic = Physics(
        **SHELL,
        inelastic_cutoff=Q_(0.1, "keV"),
        secondary_threshold=Q_(0.2, "keV"),
        radiative_model="bremslib-soft-hard",
        radiative_cutoff=Q_(50, "eV"),
    )
    numeric = Physics(
        **SHELL,
        inelastic_cutoff_eV=100.0,
        secondary_threshold_eV=200.0,
        radiative_model="bremslib-soft-hard",
        radiative_cutoff_eV=50.0,
    )
    assert semantic == numeric
    assert asdict(semantic) == asdict(numeric)
    assert pickle.loads(pickle.dumps(semantic)) == numeric
    assert replace(semantic) == numeric
    assert "inelastic_cutoff" not in {f.name for f in fields(semantic)}
    assert type(semantic.inelastic_cutoff_eV) is float
    assert semantic.inelastic_cutoff.units == ureg.eV
    assert semantic.inelastic_cutoff.magnitude == pytest.approx(100.0, rel=1e-15)
    assert semantic.secondary_threshold.magnitude == pytest.approx(200.0, rel=1e-15)
    assert semantic.radiative_cutoff.magnitude == 50.0


def test_physics_positional_defaults_and_none_readback() -> None:
    positional = Physics("incoherent", "mc", False, "shell-soft-hard", 50.0)
    assert positional == Physics(**SHELL, inelastic_cutoff=Q_(50, "eV"))
    default = Physics()
    assert default.inelastic_cutoff_eV is None
    assert default.inelastic_cutoff is None
    assert default.secondary_threshold is None
    assert default.radiative_cutoff is None
    assert default.transition_lattice_origin is None
    assert Physics(inelastic_cutoff=None) == default


def test_physics_replacement_is_validated_and_detached() -> None:
    physics = Physics(**SHELL, inelastic_cutoff_eV=50.0)
    changed = physics.with_inelastic_cutoff(Q_(0.06, "keV"))
    assert changed.inelastic_cutoff_eV == pytest.approx(60.0, rel=1e-15)
    assert physics.inelastic_cutoff_eV == 50.0
    readback = changed.inelastic_cutoff
    readback.ito("keV")
    assert changed.inelastic_cutoff.units == ureg.eV
    assert physics.with_secondary_threshold(Q_(1, "keV")).secondary_threshold_eV == 1000.0
    cleared = physics.with_secondary_threshold(Q_(1, "keV")).with_secondary_threshold(None)
    assert cleared.secondary_threshold_eV is None
    with pytest.raises(ValueError, match="secondary_threshold_eV requires"):
        Physics().with_secondary_threshold(Q_(1, "keV"))


def test_lattice_origin_quantity() -> None:
    semantic = Physics(
        transition_radiation="two-beam", transition_lattice_origin=Q_([0, 0, 1], "nm")
    )
    numeric = Physics(transition_radiation="two-beam", transition_lattice_origin_ang=(0, 0, 10))
    assert semantic == numeric
    assert semantic.transition_lattice_origin_ang == (0.0, 0.0, 10.0)
    np.testing.assert_array_equal(semantic.transition_lattice_origin.magnitude, [0, 0, 10])
    assert semantic.transition_lattice_origin.units == ureg.angstrom
    moved = semantic.with_transition_lattice_origin(Q_([1, 0, 0], "angstrom"))
    assert moved.transition_lattice_origin_ang == (1.0, 0.0, 0.0)
    with pytest.raises(TypeError, match="3-component"):
        Physics(transition_radiation="two-beam", transition_lattice_origin=Q_([0, 0], "nm"))


@pytest.mark.parametrize(
    "keyword, numeric",
    [
        ("inelastic_cutoff", "inelastic_cutoff_eV"),
        ("secondary_threshold", "secondary_threshold_eV"),
        ("radiative_cutoff", "radiative_cutoff_eV"),
    ],
)
def test_physics_energy_conflicts_and_refusals(keyword, numeric) -> None:
    with pytest.raises(TypeError, match="mutually exclusive"):
        Physics(**{numeric: 50.0, keyword: Q_(50, "eV")})
    with pytest.raises(TypeError, match=f"Physics.{keyword}"):
        Physics(**{keyword: 50.0})
    with pytest.raises(ValueError, match=f"Physics.{keyword}"):
        Physics(**{keyword: Q_(50, "mm")})
    with pytest.raises(ValueError, match=f"Physics.{keyword}"):
        Physics(**{keyword: Q_(-1, "eV")})
    with pytest.raises(TypeError, match=f"Physics.{keyword}"):
        Physics(**{keyword: Q_([1, 2], "eV")})


def test_energy_bins_energies_readback_and_replacement() -> None:
    bins = EnergyBins(cxr=Q_([1, 2, 3], "keV"), brem=np.array([100.0, 200.0]))
    np.testing.assert_array_equal(bins.cxr, [1000.0, 2000.0, 3000.0])
    assert bins.cxr_energies.units == ureg.eV
    np.testing.assert_array_equal(bins.cxr_energies.magnitude, bins.cxr)
    np.testing.assert_array_equal(bins.brem_energies.to("eV").magnitude, [100.0, 200.0])
    readback = bins.cxr_energies
    readback.magnitude[:] = 0
    assert bins.cxr[0] == 1000.0
    changed = bins.with_cxr_energies(Q_([4, 5], "keV")).with_brem_energies(None)
    assert changed == EnergyBins(cxr=np.array([4000.0, 5000.0]))
    assert changed.brem_energies is None
    assert EnergyBins().cxr_energies is None
    source = Q_(np.array([1.0, 2.0]), "eV")
    owned = EnergyBins().with_cxr_energies(source)
    source.magnitude[:] = 0
    np.testing.assert_array_equal(owned.cxr, [1.0, 2.0])


@pytest.mark.parametrize(
    "value, error",
    [
        (np.array([1.0, 2.0]), TypeError),
        (Q_(1, "eV"), TypeError),
        (Q_([[1, 2]], "eV"), TypeError),
        (Q_([1, 2], "mm"), ValueError),
        (Q_([1, np.inf], "eV"), ValueError),
    ],
)
def test_energy_bins_replacement_refuses_invalid_grids(value, error) -> None:
    with pytest.raises(error, match="EnergyBins.brem_energies"):
        EnergyBins().with_brem_energies(value)

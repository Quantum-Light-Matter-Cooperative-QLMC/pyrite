"""#755: binned aperture conversion must not change detected photon counts."""

import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest

from pyrite.photon_event_observation import PhotonEventTally
from pyrite.photon_observation import PhotonContinuumTally
from pyrite.results import QuantifiedApertureSpectrum
from pyrite.results.aperture import aperture_dataframe
from pyrite.units import ureg


def _tally(*, event: bool = False):
    common = dict(
        edges_eV=np.array([0.0, 100.0, 300.0]),
        total_per_primary_eV=np.array([0.5, 1.5]),
        direct_per_primary_eV=np.array([0.3, 1.0]),
        redistributed_per_primary_eV=np.array([0.2, 0.5]),
        detected_per_primary=355.0,
        unbinned_detected_per_primary=5.0,
        n_primaries=10,
        n_proposals=100,
        charged_feedback=False,
        source_identity_digest="source",
        observation_identity_digest="observation",
        provenance={},
    )
    if event:
        return PhotonEventTally(
            **common,
            source_component="annihilation",
            local_source_energy_eV=np.zeros(10),
            tracked_source_energy_eV=np.zeros(10),
            relaxation=None,
        )
    return PhotonContinuumTally(**common)


@pytest.mark.parametrize("event", [False, True])
def test_bin_edges_density_and_counts_convert_together(event: bool) -> None:
    tally = _tally(event=event)
    native = tally.quantify()
    kev = native.to_energy("keV")
    assert isinstance(kev, QuantifiedApertureSpectrum)
    assert kev.source_component == ("annihilation" if event else "continuum")
    assert kev.sampling == "bin_means"
    assert kev.measure == "per_energy"
    assert kev.normalization == "aperture-integrated"
    assert kev.count_basis == "photons_per_incident_electron"
    assert kev.axis_names == ("photon_energy",)
    assert kev.photon_energy_edges.units == ureg.keV
    assert kev.bin_widths.units == ureg.keV
    assert kev.density.units == ureg("1/keV").units
    assert kev.counts.units == ureg.dimensionless
    np.testing.assert_allclose(kev.photon_energy_edges.magnitude, [0, 0.1, 0.3])
    np.testing.assert_allclose(kev.bin_widths.magnitude, [0.1, 0.2])
    np.testing.assert_allclose(kev.density.magnitude, [500, 1500])
    np.testing.assert_allclose(kev.direct_density.magnitude, [300, 1000])
    np.testing.assert_allclose(kev.redistributed_density.magnitude, [200, 500])
    np.testing.assert_allclose(kev.counts.magnitude, [50, 300])
    np.testing.assert_allclose(kev.direct_counts.magnitude, [30, 200])
    np.testing.assert_allclose(kev.redistributed_counts.magnitude, [20, 100])
    np.testing.assert_allclose(native.counts.magnitude, kev.counts.magnitude)
    np.testing.assert_allclose(
        kev.counts.magnitude, kev.density.magnitude * kev.bin_widths.magnitude
    )
    assert kev.unbinned_counts.magnitude == 5
    assert kev.detected_counts.magnitude == 355
    assert kev.counts.magnitude.sum() + kev.unbinned_counts.magnitude == 355
    assert native.detected_counts == kev.detected_counts
    np.testing.assert_allclose(kev.to_energy("eV").density.magnitude, native.density.magnitude)


def test_views_and_pint_readback_do_not_mutate_numeric_tally() -> None:
    tally = _tally()
    original = asdict(tally)
    original_table = aperture_dataframe(tally)
    native = tally.quantify()
    converted = native.to_energy("keV")
    assert original_table.equals(aperture_dataframe(tally))
    edges = converted.photon_energy_edges
    density = converted.density
    counts = converted.counts
    edges.magnitude[1] = -1
    density.magnitude[0] = 99
    counts.magnitude[0] = -1
    # Check numeric compatibility before mutating the source: the legacy Polars
    # table can share its energy-edge storage with the numeric tally.
    assert original_table.equals(aperture_dataframe(PhotonContinuumTally(**original)))
    tally.edges_eV[1] = 120  # Tally is mutable in storage; view has detached.
    np.testing.assert_allclose(converted.photon_energy_edges.magnitude, [0, 0.1, 0.3])
    np.testing.assert_allclose(converted.density.magnitude, [500, 1500])
    np.testing.assert_allclose(converted.counts.magnitude, [50, 300])
    np.testing.assert_allclose(native.photon_energy_edges.magnitude, [0, 100, 300])
    assert original["total_per_primary_eV"].shape == tally.total_per_primary_eV.shape
    np.testing.assert_array_equal(original["total_per_primary_eV"], tally.total_per_primary_eV)


@pytest.mark.parametrize("unit", ["s", "mm", "dimensionless", "radian", "Hz"])
def test_rejects_non_energy_conversion(unit: str) -> None:
    with pytest.raises(ValueError, match="energy_unit.*energy"):
        _tally().quantify().to_energy(unit)


@pytest.mark.parametrize("unit", [None, 1, ""])
def test_rejects_non_string_energy_unit(unit: object) -> None:
    with pytest.raises(TypeError, match="energy_unit"):
        _tally().quantify().to_energy(unit)  # type: ignore[arg-type]


def test_refuses_inconsistent_bins_and_detected_counts() -> None:
    tally = _tally()
    object.__setattr__(tally, "total_per_primary_eV", np.array([0.5]))
    with pytest.raises(ValueError, match="match energy bins"):
        tally.quantify()
    tally = _tally()
    object.__setattr__(tally, "unbinned_detected_per_primary", 9.0)
    with pytest.raises(ValueError, match="binned plus unbinned"):
        tally.quantify()
    tally = _tally()
    object.__setattr__(tally, "edges_eV", np.array([0.0, 200.0, 100.0]))
    with pytest.raises(ValueError, match="edges"):
        tally.quantify()


def test_no_rate_or_steradian_conversion_by_pint() -> None:
    from pint import DimensionalityError

    view = _tally().quantify()
    with pytest.raises(DimensionalityError):
        view.density.to("1/eV/steradian/second")
    assert view.normalization == "aperture-integrated"
    assert view.count_basis == "photons_per_incident_electron"


def test_importing_numeric_tally_and_view_is_pint_lazy() -> None:
    script = (
        "import sys; "
        "from pyrite.photon_observation import PhotonContinuumTally; "
        "from pyrite.results import QuantifiedApertureSpectrum; "
        "assert 'pint' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)

"""The photon-continuum grid's positive floor, and the detector's own 0 eV edge.

Issue #100. A geometric continuum grid cannot contain zero, so it needs a
strictly positive lowest node. These pin that the floor is *derived* -- from the
medium's own plasma energy and from where the packaged tables stop -- rather
than being a convenient small number, and that giving the continuum a positive
floor does not drag the detector's physical 0 eV channel boundary up with it.

See ``docs/physics/radiation-physics/energy-grid-semantics.md`` and the
``photon-continuum-floor`` ledger row.
"""

import numpy as np
import pytest

from pyrite._grid_semantics import (
    node_bin_edges_and_widths,
    rebin_piecewise_constant_density,
    zero_based_detector_edges,
)
from pyrite.energy_grid.floor import (
    DATA_SUPPORT_LIMITS_EV,
    data_support_floor_eV,
    geometric_continuum_grid,
    photon_continuum_floor_eV,
)
from pyrite.materials import MediumSpec
from pyrite.materials.attenuation import plasma_energy_eV

# --- the floor itself -------------------------------------------------------


def test_plasma_energy_matches_independent_sternheimer_inversion():
    """Silicon's plasma energy, two ways, sharing no code path.

    ``plasma_energy_eV`` builds ``n_e`` from the crystal catalog's own atom
    number densities. The PDG/Sternheimer density-effect parameterization
    carries the same quantity implicitly as ``C_bar = 2 ln(I / hbar omega_p) +
    1``, so inverting the packaged ``C_bar`` and mean excitation energy gives an
    independent oracle read from different data by different code.

    Tolerance: the two agree to ~5e-5 relative. ``rtol=1e-3`` is set by the
    precision of the tabulated inputs (``C_bar`` to four decimals, ``I`` to
    three significant figures), and is far tighter than any plausible
    formulation error -- counting valence rather than all ``Z`` electrons for
    silicon would move the result by a factor ``sqrt(14/4) ~ 1.9``.
    """
    from pyrite.materials._transport_data import (
        STERNHEIMER_DENSITY_EFFECT,
        TRANSPORT_ELEMENTS,
    )

    mean_excitation_eV = TRANSPORT_ELEMENTS["Si"]["J_keV"] * 1.0e3
    C_bar = STERNHEIMER_DENSITY_EFFECT["Si"]["C_bar"]
    expected = mean_excitation_eV * np.exp((1.0 - C_bar) / 2.0)

    assert plasma_energy_eV("silicon") == pytest.approx(expected, rel=1.0e-3)


def test_plasma_energy_scales_as_sqrt_electron_density():
    """``hbar omega_p ~ sqrt(n_e)``: quartering the density halves the energy."""
    dense = MediumSpec(key="test-dense", composition=(("C", 0.1),))
    dilute = MediumSpec(key="test-dilute", composition=(("C", 0.025),))

    assert plasma_energy_eV(dilute) == pytest.approx(0.5 * plasma_energy_eV(dense), rel=1e-12)


def test_dilute_limit_falls_back_to_table_support():
    """Limiting case: as ``n_e -> 0`` the modelled band imposes no bound.

    The floor must then be set by data support alone -- the point at which the
    floor stops being a physics statement and becomes a statement about tables.
    """
    vanishing = MediumSpec(key="test-vanishing", composition=(("C", 1.0e-12),))

    assert plasma_energy_eV(vanishing) < data_support_floor_eV()
    assert photon_continuum_floor_eV(vanishing) == data_support_floor_eV()


def test_data_support_floor_is_the_most_restrictive_table():
    assert data_support_floor_eV() == max(DATA_SUPPORT_LIMITS_EV.values())
    # The Eagle XO digitized QE curve is the binding table; EEDL and Chantler
    # both reach further down, so bremsstrahlung itself never sets this bound.
    assert data_support_floor_eV() == DATA_SUPPORT_LIMITS_EV["eaglexo_quantum_efficiency"]
    assert DATA_SUPPORT_LIMITS_EV["eedl_photon_spectra"] < data_support_floor_eV()


@pytest.mark.parametrize("material", ["hopg", "diamond", "wse2", "mose2", "silicon"])
def test_condensed_media_are_bounded_by_the_modelled_band_not_the_tables(material):
    """For a real solid the plasma energy binds, so the floor is not a round number."""
    floor = photon_continuum_floor_eV(material)

    assert floor == plasma_energy_eV(material)
    assert floor > data_support_floor_eV()
    # Not a convenient round number: no catalog medium lands on a whole eV.
    assert abs(floor - round(floor)) > 1.0e-6


# --- the geometric baseline -------------------------------------------------


def test_geometric_grid_starts_exactly_at_the_derived_floor():
    grid = geometric_continuum_grid("hopg", 29_000.0, 2049)

    assert grid[0] == photon_continuum_floor_eV("hopg")
    assert grid[-1] == 29_000.0
    assert grid.size == 2049
    assert np.all(grid > 0.0)
    assert np.all(np.diff(grid) > 0.0)
    # Geometric: a constant ratio, so no node is dragged toward zero.
    ratios = grid[1:] / grid[:-1]
    assert np.ptp(ratios) < 1.0e-12


def test_geometric_grid_refuses_a_floor_below_model_validity():
    """An epsilon chosen to keep ``log`` finite is exactly what must not pass."""
    with pytest.raises(ValueError, match="below the derived continuum floor"):
        geometric_continuum_grid("hopg", 29_000.0, 513, floor_eV=1.0e-6)


def test_geometric_grid_allows_a_narrower_band():
    """Raising the floor is a bandwidth choice, not a physics violation."""
    grid = geometric_continuum_grid("hopg", 29_000.0, 513, floor_eV=500.0)

    assert grid[0] == 500.0


# --- the detector's own zero edge -------------------------------------------


def test_positive_floor_gets_a_separate_explicit_zero_channel():
    """The detector keeps its 0 eV boundary without widening the first source bin."""
    grid = geometric_continuum_grid("hopg", 29_000.0, 513)
    edges, widths, padded = zero_based_detector_edges(grid)
    source_edges, source_widths = node_bin_edges_and_widths(grid)

    assert padded is True
    assert edges[0] == 0.0
    # The zero channel is SEPARATE: every source edge and width survives exactly.
    assert np.array_equal(edges[1:], source_edges)
    assert np.array_equal(widths[1:], source_widths)
    assert edges[1] == pytest.approx(source_edges[0])
    assert edges.size == grid.size + 2


def test_negative_midpoint_reflection_still_clamps_to_zero():
    """The pre-existing clamp (82fea148) is preserved, not duplicated.

    A reflected edge below zero is unphysical, so it is pulled up to the
    detector's zero rather than being given a channel of its own.
    """
    grid = np.array([10.0, 30.0, 50.0])
    edges, widths, padded = zero_based_detector_edges(grid)

    assert padded is False
    assert edges[0] == 0.0
    assert edges.size == grid.size + 1
    assert widths[0] == pytest.approx(20.0)


def test_grid_already_touching_zero_is_left_alone():
    grid = np.array([1.0, 3.0, 5.0])  # reflects to exactly 0.0
    edges, _, padded = zero_based_detector_edges(grid)

    assert padded is False
    assert edges[0] == 0.0
    assert edges.size == grid.size + 1


def test_zero_channel_conserves_photon_mass():
    """The added channel must neither invent nor drop photons."""
    grid = geometric_continuum_grid("hopg", 29_000.0, 513)
    density = np.exp(-grid / 5_000.0)
    source_edges, source_widths = node_bin_edges_and_widths(grid)
    edges, _, padded = zero_based_detector_edges(grid)
    padded_density = np.concatenate(([0.0], density)) if padded else density

    target = np.linspace(0.0, 30_000.0, 97)
    masses, outside = rebin_piecewise_constant_density(edges, padded_density, target)

    assert np.sum(masses) + outside[0] + outside[1] == pytest.approx(
        float(np.sum(density * source_widths)), rel=1e-12
    )
    # Nothing lands below the source floor.
    assert outside[0] == 0.0


# --- end to end through the detector ----------------------------------------


def test_floored_continuum_grid_survives_the_timepix_detector():
    """Floor respected end to end: source floor and detector zero stay distinct."""
    timepix_response = pytest.importorskip("pyrite.detectors.timepix_response")
    grid = geometric_continuum_grid("hopg", 20_000.0, 257)
    resp = timepix_response.TimepixResponse(grid, n_mc=64, seed=7)

    assert resp.zero_channel is True
    # The detector's own boundary is an explicit physical zero ...
    assert resp.fine_edges[0] == 0.0
    assert resp.in_edges[0] == 0.0
    assert resp.E_out[0] > 0.0
    # ... while the source mesh keeps its derived positive floor.
    assert resp.E[0] == photon_continuum_floor_eV("hopg")
    assert np.array_equal(resp.fine_edges[1:], node_bin_edges_and_widths(grid)[0])

    detected = resp.apply(np.exp(-grid / 5_000.0))
    assert detected.shape == grid.shape
    assert np.all(np.isfinite(detected))
    assert np.all(detected >= 0.0)


def test_zero_channel_leaves_uniform_grid_scoring_bit_for_bit_unchanged():
    """The explicit zero channel carries no mass, so old uniform results stand.

    The expectation is computed on the unpadded edges -- the exact arrays the
    detector used before the zero channel existed -- so this compares two
    genuinely different rebin inputs rather than restating the implementation.
    """
    timepix_response = pytest.importorskip("pyrite.detectors.timepix_response")
    grid = np.arange(100.0, 5_100.0, 25.0)
    resp = timepix_response.TimepixResponse(grid, n_mc=64, seed=7)
    density = np.exp(-grid / 3_000.0)

    unpadded_edges, _ = node_bin_edges_and_widths(grid)
    reference_masses, reference_outside = rebin_piecewise_constant_density(
        unpadded_edges, density, resp.in_edges
    )
    reference = np.interp(
        grid,
        resp.E_out,
        (resp.R @ reference_masses) / resp.dE_out,
        left=0.0,
        right=0.0,
    )

    assert reference_outside == (0.0, 0.0)
    assert np.array_equal(resp.apply(density), reference)

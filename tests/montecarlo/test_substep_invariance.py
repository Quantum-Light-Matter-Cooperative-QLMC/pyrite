"""Radiation must not notice how finely a physical flight was integrated.

Numerical substeps refine the quadrature of the emission integral along one
physical flight. They are not new emitters, so the default (incoherent) CXR
yield and the bremsstrahlung continuum have to converge under refinement rather
than scale with the substep count. These checks fix the physical flights and
refine only the numerical sampling, which is what separates them from a second
transport run at a tighter tolerance -- that also moves the sampled collision
points and decorrelates the trajectories entirely.
"""

import numpy as np
import pytest

from pyrite.montecarlo.spectrum import mc_brem_spectrum, mc_spectrum, subdivide_flights
from pyrite.montecarlo.transport import simulate_trajectories

CARBON = [("C", 0.1136)]
CXR_KWARGS = {
    "crystal": "hopg",
    "hkl_list": [(0, 0, 2)],
    "B_ang2": 0.8,
    "theta_obs_rad": np.deg2rad(119.0),
    "composition": CARBON,
}
# The hopg (0,0,2) forward-beam resonance for a 25 keV beam at a 119 deg
# take-off sits at 973 eV, so this window is line-dominated and its maximum is
# the line peak.
CXR_GRID_EV = np.arange(880.0, 1070.0, 0.5)
BREM_GRID_EV = np.linspace(200.0, 20000.0, 400)
REFINEMENT_LADDER = (0.0, 5e-3, 1e-3)
FINEST_FRAC = 2e-4
MAX_SUBSTEPS = 256


def _flights(Ne=60, E0_keV=25.0, thickness_ang=4000.0):
    return simulate_trajectories(
        E0_keV=E0_keV,
        Ne=Ne,
        thickness_ang=thickness_ang,
        composition=CARBON,
        seed=7,
        transport_core="lockstep",
        energy_model="midpoint",
        E_cut_keV=1.0,
    )


def _grid_l1(value, reference):
    return float(np.abs(value - reference).sum() / np.abs(reference).sum())


def test_subdivide_at_zero_tolerance_returns_the_input_flights():
    segments = _flights(Ne=20)
    rows, parent = subdivide_flights(segments, composition=CARBON, max_dE_frac=0.0)

    np.testing.assert_array_equal(parent, np.arange(segments["L_ang"].size))
    for key in ("L_ang", "r_mid", "v_hat", "E_start_keV", "t_start_ang", "elec_id", "layer"):
        np.testing.assert_allclose(rows[key], segments[key], rtol=1e-12)
    assert np.all(rows["substep_id"] == 0)
    # The rebuilt end state reproduces the core's own predictor-corrector. It is
    # the host Joy--Luo evaluation rather than the compiled per-layer scalar the
    # core uses, so the two agree to rounding (measured 2.6e-9 relative), not
    # bit-for-bit.
    np.testing.assert_allclose(rows["E_end_keV"], segments["E_end_keV"], rtol=1e-7)
    np.testing.assert_allclose(
        rows["E_repr_keV"], 0.5 * (rows["E_start_keV"] + rows["E_end_keV"]), rtol=0, atol=0
    )


def test_subdivide_splits_flights_without_moving_their_geometry():
    segments = _flights(Ne=20)
    rows, parent = subdivide_flights(segments, composition=CARBON, max_dE_frac=1e-3)

    assert rows["L_ang"].size > segments["L_ang"].size
    # Geometry is exact: substep lengths partition the parent flight and the
    # direction never changes inside one flight.
    summed = np.bincount(parent, weights=rows["L_ang"], minlength=segments["L_ang"].size)
    np.testing.assert_allclose(summed, segments["L_ang"], rtol=1e-12)
    np.testing.assert_allclose(rows["v_hat"], segments["v_hat"][parent], rtol=0, atol=0)
    np.testing.assert_array_equal(rows["flight_id"], segments["flight_id"][parent])

    # The energy chain is continuous across substeps of the same flight, and
    # each flight still starts where transport started it.
    same_flight = parent[1:] == parent[:-1]
    np.testing.assert_allclose(
        rows["E_start_keV"][1:][same_flight], rows["E_end_keV"][:-1][same_flight], rtol=1e-12
    )
    np.testing.assert_allclose(
        rows["t_start_ang"][1:][same_flight], rows["t_end_ang"][:-1][same_flight], rtol=1e-12
    )
    first = np.flatnonzero(np.concatenate(([True], parent[1:] != parent[:-1])))
    np.testing.assert_allclose(rows["E_start_keV"][first], segments["E_start_keV"], rtol=1e-12)
    np.testing.assert_array_equal(rows["substep_id"][first], 0)


def test_brem_converges_under_substep_refinement():
    """The row integral is a midpoint rule, so refinement moves it very little."""
    segments = _flights()
    reference, _ = subdivide_flights(
        segments, composition=CARBON, max_dE_frac=2e-4, max_substeps=256
    )
    ref_spec = mc_brem_spectrum(reference, BREM_GRID_EV, composition=CARBON)

    errors = []
    for max_dE_frac in (0.0, 5e-3, 1e-3):
        rows, _ = subdivide_flights(segments, composition=CARBON, max_dE_frac=max_dE_frac)
        errors.append(_grid_l1(mc_brem_spectrum(rows, BREM_GRID_EV, composition=CARBON), ref_spec))

    # Measured 1.9e-3 / 5.8e-4 / 3.9e-5 down the ladder.
    assert errors[0] < 5e-3
    assert errors[2] < errors[1] < errors[0]


def test_brem_representative_energy_beats_the_start_energy_evaluation():
    """Dropping E_repr_keV falls back to the left-endpoint rule, one order worse."""
    segments = _flights()
    reference, _ = subdivide_flights(
        segments, composition=CARBON, max_dE_frac=2e-4, max_substeps=256
    )
    ref_spec = mc_brem_spectrum(reference, BREM_GRID_EV, composition=CARBON)

    rows, _ = subdivide_flights(segments, composition=CARBON, max_dE_frac=0.0)
    left_endpoint = {key: value for key, value in rows.items() if key != "E_repr_keV"}

    midpoint_error = _grid_l1(mc_brem_spectrum(rows, BREM_GRID_EV, composition=CARBON), ref_spec)
    left_error = _grid_l1(
        mc_brem_spectrum(left_endpoint, BREM_GRID_EV, composition=CARBON), ref_spec
    )
    assert midpoint_error < 0.6 * left_error


def test_incoherent_cxr_converges_under_substep_refinement():
    """Grouping makes refinement converge instead of dismantling the line.

    One row per flight evaluates the whole flight's emission at a single energy
    and clock, which overstates the line peak by ~13% here. Refinement is the
    cure, and it only converges because substeps of one flight are summed as
    field before they are squared.
    """
    segments = _flights()
    reference, _ = subdivide_flights(
        segments, composition=CARBON, max_dE_frac=FINEST_FRAC, max_substeps=MAX_SUBSTEPS
    )
    ref_spec = mc_spectrum(reference, CXR_GRID_EV, **CXR_KWARGS)
    assert ref_spec.max() > 0.0

    errors, peak_errors = [], []
    for max_dE_frac in REFINEMENT_LADDER:
        rows, _ = subdivide_flights(
            segments, composition=CARBON, max_dE_frac=max_dE_frac, max_substeps=MAX_SUBSTEPS
        )
        refined = mc_spectrum(rows, CXR_GRID_EV, **CXR_KWARGS)
        errors.append(_grid_l1(refined, ref_spec))
        peak_errors.append(abs(refined.max() / ref_spec.max() - 1.0))

    # Measured grid L1 3.5e-2 / 1.7e-3 / 3.4e-5 and peak error 13% / 1.2e-3 /
    # 2e-5 down the ladder.
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 1e-4
    assert peak_errors[2] < peak_errors[1] < peak_errors[0]
    assert peak_errors[2] < 1e-3


def test_treating_substeps_as_independent_emitters_destroys_the_line():
    """The regression this grouping exists to prevent, down the same ladder."""
    segments = _flights()
    ratios = []
    for max_dE_frac in (*REFINEMENT_LADDER, FINEST_FRAC):
        rows, _ = subdivide_flights(
            segments, composition=CARBON, max_dE_frac=max_dE_frac, max_substeps=MAX_SUBSTEPS
        )
        ungrouped = {key: value for key, value in rows.items() if key != "flight_id"}
        grouped = mc_spectrum(rows, CXR_GRID_EV, **CXR_KWARGS)
        without_grouping = mc_spectrum(ungrouped, CXR_GRID_EV, **CXR_KWARGS)
        ratios.append(without_grouping.max() / grouped.max())

    # Unsplit rows leave nothing to group, so the two reductions coincide there;
    # afterwards each row carries (t_L/N)^2 instead of the flight's t_L^2 and the
    # peak falls roughly as 1/N. Measured 1.000 / 0.799 / 0.480 / 0.157.
    assert ratios[0] == pytest.approx(1.0, abs=1e-12)
    assert ratios[3] < ratios[2] < ratios[1] < 0.9
    assert ratios[3] < 0.25


def test_grouped_reduction_equals_a_coherent_sum_per_physical_flight():
    """Definition check: coherent within a flight, incoherent across flights."""
    segments = _flights(Ne=6)
    rows, _ = subdivide_flights(segments, composition=CARBON, max_dE_frac=1e-3)
    grouped = mc_spectrum(rows, CXR_GRID_EV, **CXR_KWARGS)

    key = np.stack([rows["elec_id"], rows["flight_id"]])
    new_flight = np.concatenate(([True], np.any(key[:, 1:] != key[:, :-1], axis=0)))
    starts = np.flatnonzero(new_flight)
    bounds = np.append(starts, rows["L_ang"].size)

    per_flight = np.zeros_like(grouped)
    row_keys = [key for key in rows if isinstance(rows[key], np.ndarray) and rows[key].shape]
    for lo, hi in zip(bounds[:-1], bounds[1:], strict=True):
        one = dict(rows)
        for name in row_keys:
            if rows[name].shape[0] == rows["L_ang"].size:
                one[name] = rows[name][lo:hi]
        # One electron index per call keeps the kernel's `elec_id < Ne` mask and
        # its per-electron normalization pointed at this flight alone.
        one["elec_id"] = np.zeros(hi - lo, dtype=np.int64)
        one["Ne"] = 1
        per_flight += mc_spectrum(one, CXR_GRID_EV, coherent=True, **CXR_KWARGS)
    per_flight /= segments["Ne"]

    np.testing.assert_allclose(grouped, per_flight, rtol=1e-9, atol=1e-9 * grouped.max())


@pytest.mark.parametrize("bad", ["components", "cuda"])
def test_substepped_rows_fail_closed_on_unported_options(bad, monkeypatch):
    segments = _flights(Ne=8)
    rows, _ = subdivide_flights(segments, composition=CARBON, max_dE_frac=1e-3)

    if bad == "components":
        with pytest.raises(ValueError, match="incompatible with numerical substeps"):
            mc_spectrum(rows, CXR_GRID_EV, components=True, **CXR_KWARGS)
    else:
        import pyrite.montecarlo.spectrum.lines as lines

        class _FakeCupy:
            __name__ = "cupy"

            def __getattr__(self, name):
                return getattr(np, name)

        monkeypatch.setattr(lines, "xp", _FakeCupy())
        with pytest.raises(ValueError, match="host-only"):
            mc_spectrum(rows, CXR_GRID_EV, **CXR_KWARGS)


@pytest.mark.parametrize("core", ["lockstep", "per-electron"])
def test_transport_substeps_reach_the_grouped_reduction(core):
    """Slice G's rules were proven on post-hoc splits; transport emits its own.

    Only the row-consumption contract is checked here: the grouped path is taken
    and equals the per-flight coherent sum. It has to hold for the lockstep core
    too, whose rows are step-major, so a flight's substeps are NOT adjacent.
    Convergence under refinement is a separate claim and cannot be read off two
    transported ensembles, which decorrelate.
    """
    rows = simulate_trajectories(
        E0_keV=25.0,
        Ne=6,
        thickness_ang=4000.0,
        composition=CARBON,
        seed=7,
        transport_core=core,
        energy_model="midpoint",
        E_cut_keV=1.0,
        max_dE_frac=1e-3,
    )
    assert (rows["substep_id"] > 0).any()

    key = np.stack([rows["elec_id"], rows["flight_id"]])
    if core == "lockstep":
        # Every adjacent pair differs, i.e. no flight's substeps are neighbours:
        # this is the row order an adjacency-keyed grouping would silently miss.
        changes = np.any(key[:, 1:] != key[:, :-1], axis=0)
        assert changes.all()

    grouped = mc_spectrum(rows, CXR_GRID_EV, **CXR_KWARGS)
    flights, gid = np.unique(key, axis=1, return_inverse=True)
    per_flight = np.zeros_like(grouped)
    row_keys = [k for k in rows if isinstance(rows[k], np.ndarray) and rows[k].shape]
    for g in range(flights.shape[1]):
        take = np.flatnonzero(gid == g)
        one = dict(rows)
        for name in row_keys:
            if rows[name].shape[0] == rows["L_ang"].size:
                one[name] = rows[name][take]
        one["elec_id"] = np.zeros(take.size, dtype=np.int64)
        one["Ne"] = 1
        per_flight += mc_spectrum(one, CXR_GRID_EV, coherent=True, **CXR_KWARGS)
    per_flight /= rows["Ne"]

    np.testing.assert_allclose(grouped, per_flight, rtol=1e-9, atol=1e-9 * grouped.max())

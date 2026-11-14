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

from pyrite.montecarlo.spectrum import mc_brem_spectrum, mc_cxr_spectrum, subdivide_flights
from pyrite.montecarlo.transport import _simulate_trajectories as simulate_trajectories
from tests.helpers import host_backend_only

# The grouped reduction that makes substeps add as field before they are squared
# runs on the host in float64 and, on CuPy, only through the float32 segmented
# kernel (parity: test_grouped_reduction_cuda.py). The exact (1e-9) claims below
# are float64 statements, so they state the host requirement; the guard for
# unported backends is pinned by ``test_substepped_rows_fail_closed_on_unported_options``.
_grouped_is_host_only = host_backend_only(
    "flight-grouped incoherent CXR: these exact checks are float64 host "
    "statements; device parity is covered separately"
)

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
REFINEMENT_LADDER = (0.0, 5e-3, 1e-3, 3e-4)
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
    # the host stopping evaluation rather than the compiled per-layer scalar the
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


@_grouped_is_host_only
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
    ref_spec = mc_cxr_spectrum(reference, CXR_GRID_EV, **CXR_KWARGS)
    assert ref_spec.max() > 0.0

    errors, peak_errors = [], []
    for max_dE_frac in REFINEMENT_LADDER:
        rows, _ = subdivide_flights(
            segments, composition=CARBON, max_dE_frac=max_dE_frac, max_substeps=MAX_SUBSTEPS
        )
        refined = mc_cxr_spectrum(rows, CXR_GRID_EV, **CXR_KWARGS)
        errors.append(_grid_l1(refined, ref_spec))
        peak_errors.append(abs(refined.max() / ref_spec.max() - 1.0))

    # Measured grid L1 5.4e-2 / 1.7e-3 / 4.3e-5 / 2.7e-5 and peak error
    # 14% / 2.8e-4 / 3.7e-6 / 1.6e-6 down the ladder on the synthetic Mott
    # fixture (analytic Joy/Bishop screening, #263). The NIST SRD 64 carbon
    # table this test used before gave 4.9e-2 / 1.5e-3 / 3.2e-5 / 6.5e-6 and
    # 13% / 9.7e-5 / 9.7e-7 / 5.6e-8: the residual depends on which 60
    # trajectories are sampled, not on the elastic law. Since issue #181 each
    # substep carries the exact formation integral of its escape pieces --
    # absorption and the in-medium escape-leg phase included -- so the residual
    # is the energy-loss discretization alone; with the midpoint escape and the
    # sinc it was 2.1e-4 / 2.7e-5 at the finest step.
    # Validation: coherent-formation-absorption
    assert errors[3] < errors[2] < errors[1] < errors[0]
    assert errors[3] < 5e-5
    assert peak_errors[3] < peak_errors[2] < peak_errors[1] < peak_errors[0]
    assert peak_errors[3] < 5e-6


@_grouped_is_host_only
def test_treating_substeps_as_independent_emitters_destroys_the_line():
    """The regression this grouping exists to prevent, down the same ladder."""
    segments = _flights()
    ratios = []
    for max_dE_frac in (*REFINEMENT_LADDER, FINEST_FRAC):
        rows, _ = subdivide_flights(
            segments, composition=CARBON, max_dE_frac=max_dE_frac, max_substeps=MAX_SUBSTEPS
        )
        ungrouped = {key: value for key, value in rows.items() if key != "flight_id"}
        grouped = mc_cxr_spectrum(rows, CXR_GRID_EV, **CXR_KWARGS)
        without_grouping = mc_cxr_spectrum(ungrouped, CXR_GRID_EV, **CXR_KWARGS)
        ratios.append(without_grouping.max() / grouped.max())

    # Unsplit rows leave nothing to group, so the two reductions coincide there;
    # afterwards each row carries (t_L/N)^2 instead of the flight's t_L^2 and the
    # peak falls roughly as 1/N. Measured 1.000 / 0.799 / 0.480 / 0.157.
    assert ratios[0] == pytest.approx(1.0, abs=1e-12)
    assert ratios[3] < ratios[2] < ratios[1] < 0.9
    assert ratios[3] < 0.25


@_grouped_is_host_only
def test_grouped_reduction_equals_a_coherent_sum_per_physical_flight():
    """Definition check: coherent within a flight, incoherent across flights."""
    segments = _flights(Ne=6)
    rows, _ = subdivide_flights(segments, composition=CARBON, max_dE_frac=1e-3)
    grouped = mc_cxr_spectrum(rows, CXR_GRID_EV, **CXR_KWARGS)

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
        per_flight += mc_cxr_spectrum(one, CXR_GRID_EV, coherent=True, **CXR_KWARGS)
    per_flight /= segments["Ne"]

    np.testing.assert_allclose(grouped, per_flight, rtol=1e-9, atol=1e-9 * grouped.max())


@pytest.mark.parametrize(
    "bad",
    [
        # The components refusal is a host statement here: on an accelerator
        # the run also exercises the device admission first.
        pytest.param("components", marks=_grouped_is_host_only),
        # Simulated on the host: a CuPy-named namespace with float64 REAL.
        pytest.param("cuda", marks=_grouped_is_host_only),
    ],
)
def test_substepped_rows_fail_closed_on_unported_options(bad, monkeypatch):
    segments = _flights(Ne=8)
    rows, _ = subdivide_flights(segments, composition=CARBON, max_dE_frac=1e-3)

    if bad == "components":
        with pytest.raises(ValueError, match="incompatible with numerical substeps"):
            mc_cxr_spectrum(rows, CXR_GRID_EV, components=True, **CXR_KWARGS)
    else:
        # The backend guard reads `xp` in `_prepare_spectrum`, so the fake
        # backend has to land on the submodule that owns it -- rebinding the
        # `lines` package attribute would not reach the route. A CuPy name
        # with the host's float64 REAL is the unported FP64-device case.
        from pyrite.montecarlo.spectrum.cxr import _setup

        class _FakeCupy:
            __name__ = "cupy"

            def __getattr__(self, name):
                return getattr(np, name)

        monkeypatch.setattr(_setup, "xp", _FakeCupy())
        with pytest.raises(ValueError, match="has no port"):
            mc_cxr_spectrum(rows, CXR_GRID_EV, **CXR_KWARGS)


@_grouped_is_host_only
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
        # Rows are step-major, so a flight's substeps are interleaved with other
        # electrons' rows rather than adjacent -- except in the tail, where the
        # last electron still alive has nobody to interleave with. One split
        # flight is already enough to make an adjacency-keyed grouping wrong, so
        # pin that directly: keying on adjacency would report more groups than
        # there are distinct flights.
        changes = np.any(key[:, 1:] != key[:, :-1], axis=0)
        groups_if_keyed_on_adjacency = int(changes.sum()) + 1
        distinct_flights = np.unique(key, axis=1).shape[1]
        assert groups_if_keyed_on_adjacency > distinct_flights

    grouped = mc_cxr_spectrum(rows, CXR_GRID_EV, **CXR_KWARGS)
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
        per_flight += mc_cxr_spectrum(one, CXR_GRID_EV, coherent=True, **CXR_KWARGS)
    per_flight /= rows["Ne"]

    np.testing.assert_allclose(grouped, per_flight, rtol=1e-9, atol=1e-9 * grouped.max())


# --- layered absorbers (#1180) ------------------------------------------------
#
# The intra-flight escape phase through a stack is ``-omega sum_l delta_l L_l``,
# the real partner of ``_stack_tau``'s ``sum_l mu_l L_l``. ``HOPG_C`` is the
# crystal's own carbon density, so a layer of it has the crystal's delta to
# rounding and the absorber composition matches the slab route's.


def _hopg_carbon():
    from pyrite.materials.crystal import CRYSTALS

    info = CRYSTALS["hopg"]
    return [("C", len(info["basis"]) / info["V_cell"])]


@_grouped_is_host_only
def test_grouped_stack_reduces_to_the_single_slab():
    """Validation: xray-layered-escape-phase.

    Splitting one material at an internal interface only cuts each flight's
    escape pieces there, which the grouped sum reassembles exactly; one layer
    is the single slab up to the slab route's tabulated mu (measured 1.3e-7).
    """
    hopg_c = _hopg_carbon()
    kwargs = dict(CXR_KWARGS, composition=hopg_c)
    rows, _ = subdivide_flights(_flights(Ne=8), composition=CARBON, max_dE_frac=1e-3)
    slab = mc_cxr_spectrum(rows, CXR_GRID_EV, **kwargs)
    one = mc_cxr_spectrum(rows, CXR_GRID_EV, layers=[(0.0, 4000.0, hopg_c)], **kwargs)
    two = mc_cxr_spectrum(
        rows, CXR_GRID_EV, layers=[(0.0, 1500.0, hopg_c), (1500.0, 4000.0, hopg_c)], **kwargs
    )
    assert slab.max() > 0.0
    np.testing.assert_allclose(two, one, rtol=1e-9, atol=1e-12 * one.max())
    np.testing.assert_allclose(one, slab, rtol=0.0, atol=1e-6 * slab.max())


@_grouped_is_host_only
def test_grouped_stack_carries_each_layers_delta_over_its_own_path():
    """Validation: xray-layered-escape-phase.

    The setup hands the reduction per-layer escape paths whose sum is the slab
    path and a per-layer ``delta_l omega`` grid built from each composition;
    a different film changes the scored line.
    """
    from pyrite.materials.crystal import HBARC_EV_ANG, composition_refractive_decrement
    from pyrite.montecarlo.spectrum.cxr._setup import SpectrumRequest, _prepare_spectrum

    hopg_c = _hopg_carbon()
    silicon = [("Si", 0.04994)]
    layers = [(0.0, 1500.0, silicon), (1500.0, 4000.0, hopg_c)]
    kwargs = dict(CXR_KWARGS, composition=hopg_c)
    rows, _ = subdivide_flights(_flights(Ne=8), composition=CARBON, max_dE_frac=1e-3)

    def setup(stack):
        request = SpectrumRequest(segments=rows, E_grid_eV=CXR_GRID_EV, layers=stack, **kwargs)
        return _prepare_spectrum(request)

    stacked = setup(layers)
    assert stacked.grouped
    # Front-face escape (n_z < 0): the layer paths at a piece end at depth z
    # are min(z, 1500)/|n_z| in the film and max(z - 1500, 0)/|n_z| below.
    n_z = float(stacked.n_hat[2])
    assert n_z < 0.0
    pieces = stacked.segments
    half = 0.5 * np.asarray(pieces["L_ang"], dtype=float)[:, None]
    r_mid = np.asarray(pieces["r_mid"], dtype=float)
    v_hat = np.asarray(pieces["v_hat"], dtype=float)
    for sign, end in zip((-1.0, 1.0), stacked.escape_ends, strict=True):
        z = np.clip((r_mid + sign * half * v_hat)[:, 2], 0.0, 4000.0)
        expected = np.stack([np.minimum(z, 1500.0), np.maximum(z - 1500.0, 0.0)], axis=1)
        np.testing.assert_allclose(end, expected / abs(n_z), rtol=0.0, atol=2e-3)
    assert (stacked.escape_ends[0][:, 1] > 0.0).any()

    omega = CXR_GRID_EV / HBARC_EV_ANG
    for row, (*_, comp) in zip(stacked.layer_delta_omega, layers, strict=True):
        expected = composition_refractive_decrement(comp, CXR_GRID_EV) * omega
        np.testing.assert_allclose(row, expected, rtol=1e-12)

    carbon_film = mc_cxr_spectrum(
        rows, CXR_GRID_EV, layers=[(0.0, 1500.0, hopg_c), (1500.0, 4000.0, hopg_c)], **kwargs
    )
    silicon_film = mc_cxr_spectrum(rows, CXR_GRID_EV, layers=layers, **kwargs)
    assert np.abs(silicon_film - carbon_film).max() > 1e-3 * carbon_film.max()

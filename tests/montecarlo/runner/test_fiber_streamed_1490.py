"""Streamed whole-electron-block scoring of full-fiber CXR cases (#1490).

The roll-averaged lines and the matched continuum are track-length sums over
a fixed electron count, so electron blocks add. Under an exact bin-mean
evaluator the streamed spectrum equals the retained one to summation order.
The production host Fourier bin-mean reduction (#1192) chooses its image
period from the lines each flush holds, so its block sums differ from the
joined ones within that reduction's declared image residual. The retained
route already flushes and electron-blocks the same way on large populations.

Validation: cxr-fiber-intensity-average
"""

import numpy as np
import pytest

from pyrite import Scene
from pyrite.beams import Beam, PointBunch, Spot
from pyrite.campaign.lowering import lower_simulation
from pyrite.campaign.resolution import resolve_simulation
from pyrite.config import CxrReflections, FiberAverage, FixedSampling, Numerics, Physics
from pyrite.detectors import Detector, EnergyBins
from pyrite.geometry import Footprint, Slab
from pyrite.montecarlo import runner
from pyrite.montecarlo.runner import fiber_emission
from pyrite.montecarlo.runner.streamed_scoring import run_case_streamed, streaming_refusal
from pyrite.montecarlo.spectrum.cxr._bin_fourier import FOURIER_RESIDUAL_TOL
from pyrite.montecarlo.spectrum.cxr._bin_quadrature import bin_reduction_override
from pyrite.montecarlo.spectrum.cxr.fiber import FIBER_CXR_MODEL
from pyrite.montecarlo.trajectories import write_trajectory_artifact

PHYSICS = Physics(bremsstrahlung_model="bremslib", radiative_model="uncoupled")
AVERAGE = FiberAverage(cutoff_invang=4.0, azimuth_count=2, reflection_batch_size=16)
GRID = np.arange(200.0, 3000.0, 40.0)
SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")
#: Unequal populations exercise the per-estimator electron limits.
POPULATION = dict(Ne=6, Ne_brem=8)


def _case(grid):
    scene = Scene(
        Beam(30.0, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", 5000.0, tilt_deg=15.0, footprint=Footprint(5.0, 5.0)),
        Detector(),
    )
    numerics = Numerics(
        sampling=FixedSampling(cxr_histories=2, brem_histories=2),
        cxr_reflections=CxrReflections(fiber=AVERAGE),
        energy_bins=None if grid is None else EnergyBins(cxr=grid),
    )
    case = lower_simulation(resolve_simulation(scene, physics=PHYSICS, numerics=numerics))
    return {**case, **POPULATION}


def _retained(case):
    transport = runner._transport_case(case, transport_core="per-electron")
    return transport, runner._spectrum_case(case, transport)


def _assert_parity(streamed, retained, *, rtol=1e-12, atol_scale=0.0):
    np.testing.assert_array_equal(streamed["E_grid"], retained["E_grid"])
    np.testing.assert_array_equal(streamed["E_grid_brem"], retained["E_grid_brem"])
    for key in SPECTRA:
        scale = float(np.max(np.abs(retained[key])))
        assert scale > 0.0, key
        np.testing.assert_allclose(
            streamed[key], retained[key], rtol=rtol, atol=atol_scale * scale, err_msg=key
        )
    for key in ("eta", "hit_frac", "n_segments"):
        assert streamed[key] == retained[key], key
    # The matched continuum subtraction is one record per domain, not a block sum.
    assert streamed["brem_lattice_share"] == retained["brem_lattice_share"]
    record = streamed["brem_lattice_share"]["layers"][0]
    assert record["model"] == FIBER_CXR_MODEL and record["corrected"] is True


def test_fiber_cases_stream():
    case = _case(GRID)
    assert streaming_refusal(case) is None
    assert streaming_refusal(case, live=True) is None


@pytest.mark.slow
@pytest.mark.parametrize("grid", [GRID, None], ids=["explicit-grid", "policy-grid"])
def test_streamed_fiber_spectrum_equals_retained_to_summation_order(grid):
    """Exact bin means: lines, continuum and lattice-share record match (#1490)."""
    case = _case(grid)
    with bin_reduction_override("pairs"):  # the host's exact dense bin mean
        _, retained = _retained(case)
        for block in (3, 5):
            streamed = run_case_streamed(case, block, transport_core="per-electron")
            _assert_parity(streamed, retained)
    assert np.max(retained["spec"]) > 0.0
    assert retained["brem_lattice_share"]["layers"][0]["reflections"] == len(
        fiber_emission.fiber_domain(case).hkls
    )


@pytest.mark.slow
def test_streamed_fiber_spectrum_under_the_production_fourier_reduction(tmp_path):
    """Live and artifact streaming on the default reduction (#1192 residual).

    Continuum and characteristic stay exact to summation order; lines agree
    within the Fourier image residual, relative to the spectrum peak.
    """
    case = _case(GRID)
    transport, retained = _retained(case)
    live = run_case_streamed(case, 3, transport_core="per-electron")
    inputs = {key: value for key, value in transport.items() if key != "segs"}
    path = write_trajectory_artifact(
        tmp_path / "fiber.h5", transport["segs"], case=case, spectrum_inputs=inputs
    )
    replayed = runner.stream_spectrum_from_artifact(path, max_segments=1)
    for streamed in (live, replayed):
        _assert_parity(streamed, retained, rtol=0.0, atol_scale=FOURIER_RESIDUAL_TOL)
        for key in ("spec_characteristic", "brem_wide", "brem"):
            np.testing.assert_allclose(streamed[key], retained[key], rtol=1e-12, err_msg=key)

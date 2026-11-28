"""Fixed-grid history fiber live streaming, artifact replay and rollback."""

from types import SimpleNamespace

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
from pyrite.montecarlo.runner.electron_blocks import iter_electron_blocks
from pyrite.montecarlo.runner.streamed_scoring import BlockSpectrumScorer, run_case_streamed
from pyrite.montecarlo.spectrum.cxr._bin_fourier import FOURIER_RESIDUAL_TOL
from pyrite.montecarlo.spectrum.cxr._bin_quadrature import bin_reduction_override
from pyrite.montecarlo.spectrum.cxr._fiber_history import history_coupling_tables
from pyrite.montecarlo.trajectories import (
    TrajectoryArtifactError,
    read_trajectory_artifact,
    write_trajectory_artifact,
)

SPECTRA = ("spec", "spec_characteristic", "brem_wide", "brem")


def _case(lines, brem, mosaic=False, *, joint=False):
    scene = Scene(
        Beam(30.0, transverse=Spot(), longitudinal=PointBunch()),
        Slab("hopg", 5000.0, tilt_deg=15.0, mosaic=mosaic or joint, footprint=Footprint(5.0, 5.0)),
        Detector(),
    )
    numerics = Numerics(
        sampling=FixedSampling(cxr_histories=lines, brem_histories=brem),
        cxr_reflections=CxrReflections(
            fiber=FiberAverage(cutoff_invang=4, reflection_batch_size=16, roll_sampling="history"),
            mosaic_route="mc-sampled" if joint else "mc" if mosaic else "auto",
            mosaic_nodes=1 if joint else 2,
        ),
        energy_bins=EnergyBins(
            cxr=np.arange(200.0, 3000.0, 40.0), brem=np.arange(500.0, 30000.0, 500.0)
        ),
        seed=101,
    )
    physics = Physics(bremsstrahlung_model="bremslib", radiative_model="uncoupled")
    return dict(lower_simulation(resolve_simulation(scene, physics=physics, numerics=numerics)))


def _parity(reference, actual, *, line_tol):
    for key in ("E_grid", "E_grid_brem"):
        np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)
    assert np.max(reference["spec"]) > 0
    for key in SPECTRA:
        scale = float(np.max(np.abs(reference[key])))
        assert np.isfinite(actual[key]).all(), key
        np.testing.assert_allclose(
            actual[key],
            reference[key],
            rtol=0,
            atol=scale * (line_tol if key == "spec" else 1e-10),
            err_msg=key,
        )
    for key in ("eta", "hit_frac", "n_segments", "brem_lattice_share"):
        assert actual[key] == reference[key], key


@pytest.mark.slow
@pytest.mark.parametrize("lines,brem", [(6, 8), (8, 6)])
@pytest.mark.parametrize("joint", [False, True])
def test_history_live_streams_with_unequal_populations(lines, brem, joint):
    case = _case(lines, brem, joint=joint)
    if joint:
        assert case["mosaic_mc_sampling"] == "history"
        assert case["mosaic_mc_nodes"] == 1
    with bin_reduction_override("pairs"):
        tp = runner._transport_case(case, transport_core="per-electron")
        assert (tp["Ne_lines"], tp["Ne_brem"]) == (lines, brem)
        retained = runner._spectrum_case(case, tp)
        for block in (3, 5):
            live = run_case_streamed(case, block, transport_core="per-electron")
            _parity(retained, live, line_tol=1e-10)


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["perfect", "quadrature", "joint"])
def test_history_artifact_replay_and_case_identity(tmp_path, mode):
    case = _case(6, 8, mosaic=mode == "quadrature", joint=mode == "joint")
    tp = runner._transport_case(case, transport_core="per-electron")
    retained = runner._spectrum_case(case, tp)
    inputs = {key: value for key, value in tp.items() if key != "segs"}
    path = write_trajectory_artifact(
        tmp_path / "fiber.h5", tp["segs"], case=case, spectrum_inputs=inputs
    )
    artifact = read_trajectory_artifact(path)
    assert artifact.case["cxr_fiber"] == case["cxr_fiber"]
    assert artifact.case["seed"] == case["seed"]
    _parity(retained, runner.spectrum_from_artifact(artifact), line_tol=FOURIER_RESIDUAL_TOL)
    for max_rows in (1, 11):
        replay = runner.stream_spectrum_from_artifact(path, max_segments=max_rows)
        _parity(retained, replay, line_tol=FOURIER_RESIDUAL_TOL)
    different = dict(case, seed=case["seed"] + 1)
    with pytest.raises(TrajectoryArtifactError, match="different case"):
        runner.stream_spectrum_from_artifact(path, case=different)


@pytest.mark.slow
def test_history_stream_scorer_rollback_does_not_double_count():
    case = _case(6, 8)
    tp = runner._transport_case(case, transport_core="per-electron")
    segs = tp["segs"]
    with bin_reduction_override("pairs"):
        retained = runner._spectrum_case(case, tp)
        scorer = BlockSpectrumScorer(case, tp)
        blocks = list(iter_electron_blocks(segs, 3))
        assert len(blocks) > 1
        state = scorer.snapshot()
        scorer.add(blocks[0])
        scorer.restore(state)
        assert scorer.n_segments == 0
        for block in blocks:
            scorer.add(block)
        result = scorer.result(
            Ne=int(segs["Ne"]),
            n_backscattered=int(segs["n_backscattered"]),
            n_missed=int(segs["n_missed"]),
            line_coverage=None,
        )
        _parity(retained, result, line_tol=1e-10)


def test_history_coupling_cache_is_keyed_by_table_content_not_orientation():
    calls = []
    cache = {}
    req = SimpleNamespace(
        crystal="hopg",
        hkl_list=[(1, 0, 2)],
        B_ang2=lambda hkl: 0.25,
        use_henke=True,
        _table_cache=cache,
        azimuth_rad=0.0,
    )
    st = SimpleNamespace(request=req, E_tab=np.array([100.0, 200.0]))

    def build(*args):
        calls.append(args)
        return tuple(np.zeros((1, 2)) for _ in range(4))

    first = history_coupling_tables(st, build)
    req.azimuth_rad = 0.2
    st.E_tab = st.E_tab.copy()
    assert history_coupling_tables(st, build) is first
    assert len(calls) == 1
    st.E_tab[1] += 1
    assert history_coupling_tables(st, build) is not first
    req.B_ang2 = lambda hkl: 0.5
    history_coupling_tables(st, build)
    req.hkl_list = [(0, 1, 2)]
    history_coupling_tables(st, build)
    req.use_henke = False
    history_coupling_tables(st, build)
    assert len(calls) == 5
    req._table_cache = None
    history_coupling_tables(st, build)
    history_coupling_tables(st, build)
    assert len(calls) == 7

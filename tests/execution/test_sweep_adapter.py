"""Verify that the opt-in executor leaves checkpoint ownership in run_sweep."""

import numpy as np

from pyrite.execution import LocalExecutor
from pyrite.materials import CATALOG
from pyrite.runs.run import run_sweep
from tests.helpers import stub_run_cases


def test_run_sweep_accepts_local_executor_without_changing_checkpoint_semantics(tmp_path):
    # The real sweep coordinator, with its numerical runner replaced by the
    # same synthetic runner used throughout tests/scan/test_run.py.
    case = dict(
        name="hpc-slice", crystal="hopg", composition=CATALOG.crystal("hopg").composition,
        hkl_list=[], B_ang2=0.8, E0_keV=30.0, thickness_ang=10000.0,
        E_grid=(100.0, 200.0, 10.0), E_grid_line=(100.0, 200.0, 10.0),
        E_grid_brem=(0.0, 500.0, 50.0), theta_obs_rad=np.deg2rad(90.0),
        dtheta_obs_rad=np.deg2rad(2.0), tilt_deg=0.0, tilt_azim_deg=0.0,
        domega_sr=1e-4, beam_uvw=(0, 0, 1), mosaic_fwhm_rad=None,
        mosaic_mc_fwhm_rad=None, mosaic_mc_nodes=1, abs_layers=None,
        layer_radiators=None, brem_file=None, Ne=10, Ne_brem=5, seed=42,
        spec_chunk=None, brem_chunk=None,
    )
    forwarded = []

    def recording_runner(cases, **kwargs):
        forwarded.append((len(cases), kwargs["keep_results"], kwargs["progress"]))
        return stub_run_cases(cases, **kwargs)

    results = {}
    checkpoint_path = str(tmp_path / "hopg.pkl")
    assert run_sweep(
        [case], results, checkpoint_path=checkpoint_path, progress=False,
        executor=LocalExecutor(recording_runner),
    )
    assert forwarded == [(1, False, False)]
    assert case["name"] in results
    assert case["E0_keV"] in results[case["name"]]
    assert (tmp_path / "hopg.pkl").is_file()

    # Completing this exact sweep again should be a resume-only operation,
    # not invoke the execution backend at all.
    def unexpectedly_called(*_args, **_kwargs):
        raise AssertionError("a fully cached case was dispatched")

    assert run_sweep(
        [case], {}, checkpoint_path=checkpoint_path, progress=False,
        executor=LocalExecutor(unexpectedly_called),
    )

"""Correlated phase space through axial solvers (issue #628; needs the atomic database)."""

import numpy as np
import pytest

pytest.importorskip("xraydb")

from pyrite.materials.channeling.axial import axial_profile  # noqa: E402
from pyrite.montecarlo.channeling import (  # noqa: E402
    AxialEnergyMCWFOptions,
    IncidentPhaseSpace,
    axial_phase_space_mcwf,
    axial_temporal_slices,
)
from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator  # noqa: E402
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF  # noqa: E402
from pyrite.montecarlo.channeling.axial_propagation import (  # noqa: E402
    AxialPropagationConfig,
    propagate_axial,
)
from pyrite.montecarlo.channeling.mcwf import mcwf_stream_key  # noqa: E402
from pyrite.montecarlo.channeling.states import gaussian, translate  # noqa: E402


def _space(t, energy, position=None, slope=None, weights=None):
    n = len(t)
    return IncidentPhaseSpace(
        arrival_time_fs=t,
        kinetic_energy_eV=np.broadcast_to(np.asarray(energy, float), (n,)),
        position_ang=np.zeros((n, 2)) if position is None else position,
        slope_rad=np.zeros((n, 2)) if slope is None else slope,
        weights=weights,
    )


def test_axial_slices_are_invariant_to_arrival_time_spread():
    cfg = AxialPropagationConfig(
        kinetic_energy_eV=10e6, shape=(12, 12), depths_ang=(0.0, 4.0), step_ang=1.0
    )
    direct = propagate_axial(cfg)
    space = _space(np.linspace(-500.0, 500.0, 8), 10e6)
    result = axial_temporal_slices(space, cfg, [-600.0, 0.0, 50.0, 600.0])
    assert result.slices.probability.tolist() == [0.5, 0.0, 0.5]
    for s in (0, 2):
        np.testing.assert_array_equal(
            result.observables["density_per_ang2"][s], direct.density_per_ang2
        )
    np.testing.assert_array_equal(result.observables["density_per_ang2"][1], 0.0)


def test_mcwf_records_keep_their_own_energy_slope_and_impact():
    cfg = AxialPropagationConfig(
        mode="mcwf",
        kinetic_energy_eV=10e6,
        mechanisms=("tds",),
        shape=(12, 12),
        depths_ang=(0.0, 2.0),
        step_ang=1.0,
        tilt_rad=(1e-4, 0.0),
    )
    position = np.array([[1.0, 0.5], [-1.0, -0.5]])
    slope = np.array([[2e-4, 0.0], [-2e-4, 1e-4]])
    space = _space([0.0, 10.0], [8e6, 12e6], position=position, slope=slope)
    result = axial_phase_space_mcwf(
        space, cfg, options=AxialEnergyMCWFOptions(histories=7, seed=628)
    )
    (run,) = result.runs
    assert run.bin_counts.tolist() == [1, 1]
    profile = axial_profile(cfg.crystal, cfg.axis, cfg.temperature_K)
    extent = np.array(profile.period_ang) * cfg.cells
    centroid_x, centroid_s = position.mean(axis=0), slope.mean(axis=0)
    for eid, energy in enumerate((8e6, 12e6)):
        diss = axial_dissipator(profile, energy, cfg.cells, cfg.shape, mechanisms=cfg.mechanisms)
        offset = np.mod(position[eid] - centroid_x, extent)
        tilt = tuple(np.asarray(cfg.tilt_rad) + slope[eid] - centroid_s)
        reference = AxialMCWF(diss, energy, profile.on_grid(cfg.cells, cfg.shape)).run(
            translate(gaussian(waist_ang=0.5), offset),
            cfg.depths_ang,
            1,
            seed=int(mcwf_stream_key(628 ^ 0x6272, eid)),
            step_ang=cfg.step_ang,
            tilt_rad=tilt,
        )
        np.testing.assert_allclose(run.surviving_by_bin[eid], reference.surviving, atol=0)
        np.testing.assert_allclose(
            run.density_by_bin_per_ang2[eid], reference.observables, rtol=1e-12, atol=1e-13
        )
    np.testing.assert_allclose(result.integrated_surviving, run.surviving)


def test_mcwf_slices_cover_records_and_reduce_by_incident_count():
    cfg = AxialPropagationConfig(
        mode="mcwf",
        kinetic_energy_eV=10e6,
        mechanisms=("tds",),
        shape=(12, 12),
        depths_ang=(0.0, 2.0),
        step_ang=1.0,
    )
    space = _space([-5.0, -4.0, -3.0, 4.0], 10e6)
    result = axial_phase_space_mcwf(space, cfg, slice_edges_fs=[-6.0, 0.0, 6.0], impact="entry")
    assert result.slice_counts.tolist() == [3, 1]
    np.testing.assert_allclose(
        result.integrated_surviving, 0.75 * result.surviving[0] + 0.25 * result.surviving[1]
    )
    assert result.runs[0].provenance["seed"] != result.runs[1].provenance["seed"]
    with pytest.raises(ValueError, match="cover"):
        axial_phase_space_mcwf(space, cfg, slice_edges_fs=[-6.0, 0.0])
    with pytest.raises(ValueError, match="resample"):
        axial_phase_space_mcwf(_space([0.0, 1.0], 10e6, weights=[1, 2]), cfg)
    with pytest.raises(ValueError, match="derived"):
        axial_phase_space_mcwf(space, cfg, electron_energies_eV=[1e7] * 4)

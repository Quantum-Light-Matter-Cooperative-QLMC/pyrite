"""Integration and admission regressions for diagnostic current (#625)."""

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_diagnostic import (
    AxialDiagnosticConfig,
    _DiagnosticRecording,
    _validate,
    axial_diagnostic,
)
from pyrite.montecarlo.channeling.axial_flux import AxialFluxObservable
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid


def _config(**changes):
    from dataclasses import replace

    cfg = AxialDiagnosticConfig(
        crystal="silicon",
        axis=(0, 0, 1),
        kinetic_energy_eV=5e6,
        cells=(1, 1),
        shape=(16, 16),
        depths_ang=(0.0, 1.0),
        step_ang=1.0,
        spectral_states=0,
        record_coherence=False,
        record_current=True,
        record_current_divergence=True,
        current_contour=(-8, -8, 16, 16),
    )
    return replace(cfg, **changes)


def test_recording_settings_are_explicitly_admitted():
    _validate(_config())
    with pytest.raises(ValueError, match="enable record_current"):
        _validate(_config(record_current=False, record_current_divergence=True))
    with pytest.raises(ValueError, match="divisible"):
        _validate(_config(cells=(3, 1)))
    with pytest.raises(ValueError, match="at least 8"):
        _validate(_config(cells=(4, 1)))
    with pytest.raises(ValueError, match="within the grid"):
        _validate(_config(current_contour=(0, 0, 20, 1)))


def test_coherent_online_current_and_full_periodic_contour():
    result = axial_diagnostic(_config())
    assert result.transverse_current_per_ang2.shape == (2, 2, 16, 16)
    assert result.transverse_divergence_per_ang3.shape == (2, 16, 16)
    assert result.contour_flux_per_ang.shape == (2, 4)
    np.testing.assert_allclose(result.surviving, 1, atol=1e-10)
    np.testing.assert_allclose(result.contour_flux_per_ang.sum(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(
        result.transverse_divergence_per_ang3.sum(axis=(1, 2)) * result.grid.cell_area_ang2,
        0,
        atol=1e-12,
    )
    assert result.numerical_sink is not None
    np.testing.assert_array_equal(result.numerical_sink, 0)


def test_mcwf_current_is_normalized_per_incident_electron():
    result = axial_diagnostic(_config(mode="mcwf", n_electrons=2, mechanisms=()))
    assert result.transverse_current_per_ang2.shape == (2, 2, 16, 16)
    assert result.contour_flux_per_ang.shape == (2, 4)
    np.testing.assert_allclose(result.contour_flux_per_ang.sum(axis=1), 0, atol=1e-12)
    # No jump channels: every surviving trajectory has norm one.
    np.testing.assert_allclose(
        result.density_per_ang2.sum(axis=(1, 2)) * result.grid.cell_area_ang2,
        result.surviving,
        atol=1e-10,
    )


def test_composite_recording_budget_accounts_for_current_and_divergence():
    grid = RealSpaceGrid((12, 12), (192, 192))
    current = AxialFluxObservable(grid, 5e6, divergence=True)
    config = _config(
        mode="mcwf",
        shape=grid.shape,
        cells=(8, 8),
        depths_ang=tuple(float(i) for i in range(81)),
    )
    with pytest.raises(ValueError, match="128 MiB"):
        _DiagnosticRecording(grid, np.zeros(grid.shape), config, None, None, False, current=current)


def test_cuda_online_current_regression_when_available():
    cupy = pytest.importorskip("cupy")
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("CUDA unavailable")
    except Exception as exc:
        pytest.skip(f"CUDA unavailable: {exc}")
    result = axial_diagnostic(_config(backend="cuda", mode="mcwf", n_electrons=2, mechanisms=()))
    np.testing.assert_allclose(result.contour_flux_per_ang.sum(axis=1), 0, atol=1e-10)
    assert np.isfinite(result.transverse_current_per_ang2).all()

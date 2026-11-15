"""Supercell-image tiling of the axial TDS dissipator (#1215).

The per-image oracle below is the pre-#1215 dense evaluation: every string
image gets its own oversampled variance pipeline and its own full-grid rate.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pyrite.materials.channeling.axial import axial_profile
from pyrite.montecarlo.channeling import states
from pyrite.montecarlo.channeling.axial_dissipator import (
    AxialDissipator,
    _pad_index,
    axial_dissipator,
)
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF
from pyrite.montecarlo.channeling.realspace import RealSpaceGrid

CASES = {
    "si001-2x3": ((0, 0, 1), (2, 3), (24, 36)),
    "si111-3x2": ((1, 1, 1), (3, 2), (36, 20)),
}


def _dense_rates(d: AxialDissipator) -> np.ndarray:
    """Per-image ``lambda Var_u phi_band`` exactly as the pre-#1215 loop built it."""
    grid = d.grid
    nx, ny = grid.shape
    fine = RealSpaceGrid(grid.extent_ang, (2 * nx, 2 * ny))
    kx, ky = grid.kx[:, None], grid.ky[None, :]
    fkx, fky = fine.kx[:, None], fine.ky[None, :]
    out = []
    for src in d.tds_sources:
        spec = d.phase_spectra[src.element]
        padded = np.zeros(fine.shape, dtype=complex)
        padded[np.ix_(_pad_index(nx), _pad_index(ny))] = spec
        phi0 = np.fft.ifft2(padded).real * padded.size
        square = np.fft.fft2(phi0**2) / padded.size
        x0, y0 = src.position_ang
        shift = np.exp(-1j * (fkx * x0 + fky * y0))
        damp = np.exp(-0.5 * src.displacement_ang2 * (fkx**2 + fky**2))
        second = (np.fft.ifft2(square * damp * shift).real * padded.size)[::2, ::2]
        mean = (
            np.fft.ifft2(
                spec
                * np.exp(-0.5 * src.displacement_ang2 * (kx**2 + ky**2))
                * np.exp(-1j * (kx * x0 + ky * y0))
            ).real
            * spec.size
        )
        out.append(src.linear_density * (second - mean**2))
    return np.array(out)


def _density(d: AxialDissipator) -> np.ndarray:
    psi = d.grid.sample(states.gaussian(waist_ang=0.7, center_ang=(0.4, 0.6)))
    return np.abs(psi) ** 2 / d.grid.norm(psi)


@pytest.mark.parametrize("case", CASES)
def test_image_tiling_equals_dense_per_image_rates(case):
    axis, cells, shape = CASES[case]
    d = axial_dissipator(axial_profile("silicon", axis), 1e6, cells, shape, mechanisms=("tds",))
    assert d.tds_image_cells == cells
    n_images = cells[0] * cells[1]
    assert len(d.tds_base_rates) * n_images == len(d.tds_sources)
    dense = _dense_rates(d)
    scale = np.abs(dense).max()
    tiled = np.array([d.tds_source_rate(s) for s in range(len(d.tds_sources))])
    np.testing.assert_allclose(tiled, dense, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(d.tds_grid_rate, dense.sum(axis=0), rtol=0, atol=1e-12 * scale)
    density = _density(d)
    reference = np.einsum("sij,ij->s", dense, density)
    np.testing.assert_allclose(
        d.tds_source_weights(density), reference, rtol=0, atol=1e-12 * reference.sum()
    )


def test_indivisible_grid_falls_back_to_per_image_bases():
    d = axial_dissipator(
        axial_profile("silicon", (0, 0, 1)), 1e6, (2, 2), (25, 23), mechanisms=("tds",)
    )
    assert d.tds_image_cells == (1, 1)
    assert len(d.tds_base_rates) == len(d.tds_sources)
    dense = _dense_rates(d)
    np.testing.assert_allclose(d.tds_base_rates, dense, rtol=0, atol=1e-12 * np.abs(dense).max())
    density = _density(d)
    reference = np.einsum("sij,ij->s", dense, density)
    np.testing.assert_allclose(
        d.tds_source_weights(density), reference, rtol=0, atol=1e-12 * reference.sum()
    )


def test_seeded_mcwf_jumps_match_dense_selection(monkeypatch):
    axis, cells, shape = CASES["si001-2x3"]
    profile = axial_profile("silicon", axis)
    d = axial_dissipator(profile, 1e6, cells, shape)
    potential = profile.on_grid(cells, shape)
    entry = states.gaussian(waist_ang=0.6, center_ang=(0.3, 0.5))

    def run():
        sources = []
        original = AxialDissipator.tds_weight_spectrum

        def record(self, source, density):
            sources.append(source)
            return original(self, source, density)

        with monkeypatch.context() as patch:
            patch.setattr(AxialDissipator, "tds_weight_spectrum", record)
            result = AxialMCWF(d, 1e6, potential).run(
                entry, [0, 2000, 4000], 24, seed=1215, step_ang=10
            )
        return result, sources

    tiled, tiled_sources = run()
    dense = _dense_rates(d)
    monkeypatch.setattr(
        AxialDissipator,
        "tds_source_weights",
        lambda self, density: np.einsum("sij,ij->s", dense, density),
    )
    reference, reference_sources = run()
    assert tiled_sources == reference_sources and len(set(tiled_sources)) > 1
    assert tiled.jumps == reference.jumps
    np.testing.assert_array_equal(tiled.sink_depths_ang, reference.sink_depths_ang)
    np.testing.assert_array_equal(tiled.surviving, reference.surviving)
    np.testing.assert_allclose(tiled.observables, reference.observables, rtol=0, atol=1e-13)


def _setup_seconds(cells) -> float:
    profile = axial_profile("silicon", (0, 0, 1))
    shape = (12 * cells[0], 12 * cells[1])
    best = np.inf
    for _ in range(3):
        start = time.perf_counter()
        axial_dissipator(profile, 1e6, cells, shape, mechanisms=("tds",))
        best = min(best, time.perf_counter() - start)
    return best


def test_tds_setup_cost_per_cell_does_not_grow_with_images():
    # Dense per-image setup scales as cells^2: 16x per cell from (2, 3) to (8, 12).
    small, large = _setup_seconds((2, 3)), _setup_seconds((8, 12))
    assert large / 96 < 4.0 * small / 6

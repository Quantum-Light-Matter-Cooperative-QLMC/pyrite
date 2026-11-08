"""CUDA energy-resolved axial MCWF (#627): addressing, batching, reductions, parity."""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("xraydb")

from pyrite.materials.channeling.axial import axial_profile  # noqa: E402
from pyrite.montecarlo.channeling import (  # noqa: E402
    AxialEnergyMCWFOptions,
    IncidentEnergyEnsemble,
    axial_energy_mcwf,
)
from pyrite.montecarlo.channeling.axial_diagnostic import AxialDiagnosticConfig  # noqa: E402
from pyrite.montecarlo.channeling.axial_dissipator import axial_dissipator  # noqa: E402
from pyrite.montecarlo.channeling.axial_mcwf import AxialMCWF  # noqa: E402
from pyrite.montecarlo.channeling.states import gaussian  # noqa: E402

TRAJECTORY_TAG = 0x6272


def _cuda():
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        available = False
    if not available:
        pytest.skip("real CUDA device required")
    return cp


def _config(**overrides):
    values = dict(
        mode="mcwf",
        kinetic_energy_eV=1e6,
        shape=(12, 12),
        depths_ang=(0.0, 1000.0, 2000.0),
        step_ang=10.0,
        mechanisms=("tds", "electronic"),
    )
    values.update(overrides)
    return AxialDiagnosticConfig(**values)


def _two_energy_beam(center=1e6, spread=0.1):
    nodes = (center * (1 - spread), center * (1 + spread))
    return IncidentEnergyEnsemble(center, energy_nodes_eV=nodes, node_weights=(1.0, 1.0))


def _tilted_entry(energy, electron, rng):
    # Energy-correlated mean tilt plus an entrance-stream draw per electron.
    return gaussian(waist_ang=0.5), (2e-4 * (energy / 1e6 - 1) + 1e-4 * rng.standard_normal(), 0.0)


class Quadrants:
    """Density-linear custom observable valid on host and device arrays."""

    cuda_compatible = True

    def __call__(self, phi, tilt):
        xp = np
        if hasattr(phi, "__cuda_array_interface__"):
            import cupy as xp
        density = xp.abs(phi) ** 2
        return xp.stack([density[:6, :6].sum(), density[6:, 6:].sum()])


def test_chunk_ends_follow_global_ids_and_checkpoints():
    from pyrite.montecarlo.channeling._axial_energy_cuda import chunk_ends

    assert chunk_ends(0, 23, 10, 0) == [10, 20, 23]
    assert chunk_ends(0, 23, 10, 7) == [7, 14, 21, 23]
    assert chunk_ends(14, 23, 4, 7) == [18, 21, 23]


def test_admission_reserves_other_resident_energies():
    from pyrite._backend import BackendResourceError, NumPyBackend
    from pyrite.montecarlo.channeling._axial_cuda import _admit
    from pyrite.montecarlo.channeling.axial_mcwf import density_observable

    profile = axial_profile("silicon", (0, 0, 1))
    solver = AxialMCWF(
        axial_dissipator(profile, 1e6, (1, 1), (12, 12)), 1e6, profile.on_grid((1, 1), (12, 12))
    )
    backend = NumPyBackend()
    backend.xp = SimpleNamespace(
        cuda=SimpleNamespace(runtime=SimpleNamespace(memGetInfo=lambda: (2**34, 2**34)))
    )
    per_electron = 144 * 16 * 28
    budget = 64 * 2**20
    alone = _admit(solver, backend, 4096, budget, 3, density_observable, 1)
    crowded = _admit(
        solver, backend, 4096, budget, 3, density_observable, 1, reserved_bytes=32 * per_electron
    )
    assert crowded < alone
    with pytest.raises(BackendResourceError):
        _admit(solver, backend, 1, budget, 3, density_observable, 1, reserved_bytes=budget)


def test_cuda_mono_energy_uses_global_ids_of_direct_cuda_solver():
    _cuda()
    cfg = _config()
    out = axial_energy_mcwf(
        IncidentEnergyEnsemble(1e6),
        cfg,
        options=AxialEnergyMCWFOptions(n_electrons=24, seed=17, chunk_size=10, backend="cuda"),
    )
    profile = axial_profile(cfg.crystal, cfg.axis, cfg.temperature_K)
    dissipator = axial_dissipator(profile, 1e6, cfg.cells, cfg.shape, mechanisms=cfg.mechanisms)
    direct = AxialMCWF(dissipator, 1e6, profile.on_grid(cfg.cells, cfg.shape), backend="cuda").run(
        gaussian(waist_ang=0.5),
        cfg.depths_ang,
        24,
        seed=17 ^ TRAJECTORY_TAG,
        step_ang=cfg.step_ang,
        tilt_rad=cfg.tilt_rad,
    )
    assert sum(direct.jumps.values()) > 0
    assert {k: out.jumps[k] for k in direct.jumps} == direct.jumps
    np.testing.assert_array_equal(out.surviving, direct.surviving)
    np.testing.assert_allclose(out.density_per_ang2, direct.observables, rtol=1e-10, atol=1e-12)
    assert out.provenance["backend"] == "cuda"


def test_cuda_batches_chunks_and_restart_preserve_histories():
    _cuda()
    cfg = _config()
    beam = _two_energy_beam()
    kwargs = dict(entry_factory=_tilted_entry, coherence_pairs=((0, 1), (5, 17)))

    def run(chunk, **extra):
        options = AxialEnergyMCWFOptions(
            n_electrons=40,
            seed=27,
            chunk_size=chunk,
            backend="cuda",
            checkpoint_every=extra.pop("checkpoint_every", 0),
        )
        return axial_energy_mcwf(beam, cfg, options=options, **kwargs, **extra)

    whole = run(40)
    snapshots = []
    split = run(3, checkpoint_every=16, checkpoint_callback=snapshots.append)
    assert [s.processed_electrons for s in snapshots] == [16, 32]
    resumed = run(9, resume_from=snapshots[0])
    assert sum(whole.jumps.values()) > 0
    for other in (split, resumed):
        np.testing.assert_array_equal(other.energies_eV, whole.energies_eV)
        np.testing.assert_array_equal(other.bin_counts, whole.bin_counts)
        assert other.jumps_by_bin == whole.jumps_by_bin
        np.testing.assert_array_equal(other.surviving_by_bin, whole.surviving_by_bin)
        for name in ("density_by_bin_per_ang2", "coherence_by_bin_per_ang2"):
            np.testing.assert_allclose(
                getattr(other, name), getattr(whole, name), rtol=1e-10, atol=1e-12
            )
    with pytest.raises(ValueError, match="signature"):
        axial_energy_mcwf(
            beam,
            cfg,
            options=AxialEnergyMCWFOptions(n_electrons=40, seed=27),
            resume_from=snapshots[0],
            **kwargs,
        )


def test_cuda_no_jump_reductions_match_cpu_reference():
    """Without dissipation both backends are deterministic: compare every reduction."""
    _cuda()
    cfg = _config(
        kinetic_energy_eV=10e6, mechanisms=(), depths_ang=(0.0, 100.0, 200.0), step_ang=10.0
    )
    beam = _two_energy_beam(10e6, 0.05)
    common = dict(
        entry_factory=_tilted_entry,
        coherence_pairs=((0, 1), (5, 17), (30, 30)),
        observable_factory=lambda energy: Quadrants(),
        photon_energy_edges_eV=np.geomspace(1.0, 1e7, 1025),
    )
    results = {
        backend: axial_energy_mcwf(
            beam,
            cfg,
            options=AxialEnergyMCWFOptions(n_electrons=12, seed=5, chunk_size=5, backend=backend),
            **common,
        )
        for backend in ("cpu", "cuda")
    }
    host, device = results["cpu"], results["cuda"]
    assert sum(device.jumps.values()) == 0
    np.testing.assert_array_equal(device.bin_counts, host.bin_counts)
    np.testing.assert_array_equal(device.surviving_by_bin, host.surviving_by_bin)
    for name in (
        "density_by_bin_per_ang2",
        "coherence_by_bin_per_ang2",
        "observables_by_bin",
        "photon_density_by_bin_per_eV",
        "helicity_plus_per_eV",
        "helicity_minus_per_eV",
    ):
        expected = getattr(host, name)
        assert expected is not None and np.any(expected != 0), name
        scale = np.max(np.abs(expected))
        np.testing.assert_allclose(
            getattr(device, name), expected, rtol=1e-9, atol=1e-11 * scale, err_msg=name
        )


@pytest.mark.slow
def test_cuda_energy_conditioned_statistics_match_cpu():
    _cuda()
    cfg = _config()
    beam = _two_energy_beam()
    n = 64
    summaries = []
    for backend in ("cpu", "cuda"):
        rows = []
        for seed in range(8):
            out = axial_energy_mcwf(
                beam,
                cfg,
                options=AxialEnergyMCWFOptions(
                    n_electrons=n, seed=seed, chunk_size=n, backend=backend
                ),
                entry_factory=_tilted_entry,
            )
            row = []
            for b in range(2):
                density = out.density_by_bin_per_ang2[b, -1] * out.grid.cell_area_ang2
                row += [
                    out.surviving_by_bin[b, -1],
                    *(
                        density[x, y].sum()
                        for x in (slice(0, 6), slice(6, 12))
                        for y in (slice(0, 6), slice(6, 12))
                    ),
                    *(
                        out.jumps_by_bin[b][k] / max(out.bin_counts[b], 1)
                        for k in ("tds", "sink", "electronic")
                    ),
                ]
            rows.append(row)
        summaries.append(np.asarray(rows))
    host, device = summaries
    # Independent trajectory streams: predeclared four combined SEMs over eight
    # seeds, with a one-electron-per-bin floor for rare-event channels.
    tolerance = 4 * np.sqrt(host.var(axis=0, ddof=1) / 8 + device.var(axis=0, ddof=1) / 8)
    tolerance += 2 / n
    assert np.all(np.abs(host.mean(axis=0) - device.mean(axis=0)) < tolerance)

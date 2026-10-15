"""Frozen-phonon multislice (abTEM) oracle for the channeling density matrix (#504).

An independent wave-optics solution of planar channeling with thermal diffuse
scattering, compared with :class:`~pyrite.montecarlo.channeling.DensityMatrixSolver`
running the TDS kernel alone (abTEM has no electronic excitation).

**Model.** The default ``crystal`` model is diamond-structure Si oriented so
the (110) planes contain the beam, which runs along the high-index in-plane
direction ``[1, -1, n]``. Lattice row correlations are therefore exact, and
the in-plane periodic structure diffracts coherently. Each frozen
configuration displaces every atom by an isotropic Gaussian of
``a^2 = B(T)/8 pi^2`` per axis, the Einstein model the kernel assumes. The
supercell spans ``n_cells`` true periods in ``x``, so its plane-wave momenta
are the solver's Bloch grid ``kappa_r + h g``. The ``random-string`` model
(planes of atoms uniform in ``y`` and ``z``) is an amorphous-in-plane solid
whose static disorder adds full ``|f^e|^2`` scattering at ``q_y != 0``. It is
kept only to show that it overstates small-angle dechanneling. abTEM propagates a
normally incident plane wave by multislice with the Lobato potential; its interaction constant is checked
against ``e / (hbar v)``. At each exit plane the wave is projected onto
PyRITE's transverse Bloch states (``|h| <= max_order``):
``P_n(z) = <sum_r |<r n|psi(., y)>|^2>_y``, averaged over configurations.
Lobato and Waasmaier--Kirfel Mott--Bethe factors agree to ~1e-3 for Si, so
both codes share one continuum potential to that level.

Run configurations on ``pyrite remote`` (SLURM, CPU) in batches, then merge::

    uv run --group oracle python checks/abtem_channeling_oracle.py run \
        --configs 0 4 --out cfg0.npz
    uv run --group oracle python checks/abtem_channeling_oracle.py compare \
        cfg*.npz --out report.json

Repeat ``run`` with ``--slice 0.5`` to show slice convergence. Thickness
should stay well below the energy-loss scale (``dE/E < 1e-3``), which abTEM
ignores.

Validation: cr-density-matrix-split-operator
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from pyrite.materials.channeling import longitudinal_kinematics, planar_profile
from pyrite.materials.channeling.transverse import _hamiltonian
from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG, reciprocal_g_vector


def _config(args) -> dict:
    return {
        "energy_eV": args.energy,
        "hkl": list(args.hkl),
        "temperature_K": args.temperature,
        "n_cells": args.cells,
        "extent_y_ang": args.extent_y,
        "sampling_ang": args.sampling,
        "slice_ang": args.slice,
        "thickness_ang": args.thickness,
        "exit_every_ang": args.exit_every,
        "max_order": args.orders,
        "seed": args.seed,
        "model": args.model,
        "axis_index": args.axis_index,
    }


def plane_sites(profile, n_cells):
    """Per-element ``x`` images [angstrom] in the supercell and areal density per image."""
    info = CRYSTALS[profile.crystal]
    _, g_coprime = reciprocal_g_vector(profile.hkl, info["lattice"])
    spacing = 2.0 * np.pi / g_coprime
    length = n_cells * profile.period_ang
    images = int(round(length / spacing))
    sites = {}
    for element, position in info["basis"]:
        # planar_profile's phase exp(+i 2 pi n hkl.R) places the atom at -hkl.R d_c.
        x0 = (-float(np.dot(profile.hkl, position)) * spacing) % spacing
        xs = (x0 + spacing * np.arange(images)) % length
        sites.setdefault(element, []).extend(xs.tolist())
    return sites, spacing / float(info["V_cell"])


def build_atoms(profile, cfg, rng):
    """Frozen configuration of the selected model."""
    if cfg["model"] == "crystal":
        return crystal_atoms(profile, cfg, rng)
    return random_string_atoms(profile, cfg, rng)


def crystal_atoms(profile, cfg, rng):
    """Diamond-structure Si with (110) planes containing the beam ``[1, -1, n]``.

    Orthogonal cell rows ``[1 1 0]`` (x, plane normal), ``[n -n -2]`` (y) and
    ``[1 -1 n]`` (z, beam), in cubic lattice units. The beam is an
    ``n``-indexed in-plane direction, far from low-index axes for large
    ``n``. Atoms keep the lattice's row correlations; each configuration
    displaces every atom by an isotropic Gaussian of ``a^2 = B/8 pi^2`` per
    axis (Einstein model).
    """
    from ase import Atoms

    if tuple(profile.hkl) != (1, 1, 0) or profile.crystal != "silicon":
        raise SystemExit("the crystal model is implemented for Si (110)")
    n = cfg["axis_index"]
    rows = np.array([[1, 1, 0], [n, -n, -2], [1, -1, n]], dtype=float)
    lattice = float(CRYSTALS["silicon"]["lattice"]["a"])
    lengths = lattice * np.linalg.norm(rows, axis=1)
    periods_per_cell = int(round(lengths[0] / profile.period_ang))
    if cfg["n_cells"] % periods_per_cell:
        raise SystemExit(f"--cells must be a multiple of {periods_per_cell} for the crystal model")
    repeat = (
        cfg["n_cells"] // periods_per_cell,
        1,
        int(np.ceil(cfg["thickness_ang"] / lengths[2])),
    )
    basis = np.array([position for _, position in CRYSTALS["silicon"]["basis"]], dtype=float)
    # Integer lattice translations covering the orthogonal cell.
    span = int(np.abs(rows).sum(axis=0).max()) + 1
    grid = np.stack(np.meshgrid(*[np.arange(-span, span + 1)] * 3, indexing="ij"), -1).reshape(
        -1, 3
    )
    fractional = (grid[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    # Coordinates along the orthogonal axes, as fractions of each cell length.
    unit = rows / np.linalg.norm(rows, axis=1)[:, None]
    cart = fractional @ unit.T * lattice / lengths
    inside = np.all((cart >= -1e-9) & (cart < 1 - 1e-9), axis=1)
    cell_positions = cart[inside] * lengths
    expected = int(round(len(basis) * np.prod(lengths) / lattice**3))
    if len(cell_positions) != expected:
        raise SystemExit(f"orthogonal cell holds {len(cell_positions)} atoms, expected {expected}")
    shifts = (
        np.stack(np.meshgrid(*[np.arange(k) for k in repeat], indexing="ij"), -1).reshape(-1, 3)
        * lengths
    )
    positions = (shifts[:, None, :] + cell_positions[None, :, :]).reshape(-1, 3)
    extent = lengths * np.array(repeat)
    positions = positions[positions[:, 2] < cfg["thickness_ang"]]
    sigma = np.sqrt(profile.b_factors_ang2["Si"] / (8.0 * np.pi**2))
    positions = (positions + sigma * rng.standard_normal(positions.shape)) % [
        extent[0],
        extent[1],
        np.inf,
    ]
    return Atoms(
        symbols=["Si"] * len(positions),
        positions=positions,
        cell=[extent[0], extent[1], cfg["thickness_ang"]],
        pbc=True,
    )


def random_string_atoms(profile, cfg, rng):
    """Random-string planes: atoms on plane positions, uniform in ``y`` and ``z``.

    This is an amorphous-in-plane solid, not a crystal: besides thermal
    scattering it has static-disorder scattering ``|f^e|^2`` at every
    ``q_y != 0`` that a crystal's periodic rows diffract coherently. It
    overstates small-angle scattering and is kept only for that comparison.
    """
    from ase import Atoms

    sites, areal = plane_sites(profile, cfg["n_cells"])
    length = cfg["n_cells"] * profile.period_ang
    width, depth = cfg["extent_y_ang"], cfg["thickness_ang"]
    symbols, positions = [], []
    for element, xs in sites.items():
        sigma = np.sqrt(profile.b_factors_ang2[element] / (8.0 * np.pi**2))
        count = int(round(areal * width * depth))
        for x0 in xs:
            x = (x0 + sigma * rng.standard_normal(count)) % length
            y = rng.uniform(0.0, width, count)
            z = rng.uniform(0.0, depth, count)
            positions.append(np.column_stack([x, y, z]))
            symbols.extend([element] * count)
    return Atoms(
        symbols=symbols, positions=np.vstack(positions), cell=[length, width, depth], pbc=True
    )


def bloch_basis(profile, gamma, n_cells, max_order):
    """Eigenvectors ``V[r]`` over orders ``-H..H`` at ``kappa_r = r g / n_cells``."""
    orders = np.arange(-max_order, max_order + 1)
    kappa = np.arange(n_cells) * profile.g_inv_ang / n_cells
    vectors = [np.linalg.eigh(_hamiltonian(profile, gamma, k, orders))[1] for k in kappa]
    return orders, np.array(vectors)


def project(waves, n_cells, orders, vectors):
    """Band populations ``sum_r <|V_r^dagger c_r(y)|^2>_y`` of exit waves ``(n_z, n_x, n_y)``."""
    n_z, n_x, n_y = waves.shape
    spectrum = np.fft.fft(waves, axis=1) / n_x  # plane-wave amplitudes per y column
    out = np.zeros((n_z, orders.size))
    for r in range(n_cells):
        index = (r + orders * n_cells) % n_x
        coefficients = spectrum[:, index, :]  # (z, h, y)
        eigen = np.einsum("hn,zhy->zny", vectors[r].conj(), coefficients)
        out += np.mean(np.abs(eigen) ** 2, axis=2)
    return out


def check_interaction_constant(energy_eV):
    from abtem.core.energy import energy2sigma

    _, beta = longitudinal_kinematics(energy_eV)
    ours = 1.0 / (HBARC_EV_ANG * beta)
    theirs = energy2sigma(energy_eV)
    if abs(theirs / ours - 1.0) > 1e-6:
        raise SystemExit(f"abTEM interaction constant {theirs} differs from e/(hbar v) = {ours}")
    return theirs / ours - 1.0


def run(args) -> int:
    import abtem

    cfg = _config(args)
    profile = planar_profile("silicon", tuple(args.hkl), args.temperature)
    gamma, _ = longitudinal_kinematics(args.energy)
    sigma_check = check_interaction_constant(args.energy)
    orders, vectors = bloch_basis(profile, gamma, args.cells, args.orders)
    exit_every = max(1, int(round(args.exit_every / args.slice)))
    populations, seconds = [], []
    for index in range(args.configs[0], args.configs[1]):
        start = time.perf_counter()
        rng = np.random.default_rng([args.seed, index])
        atoms = build_atoms(profile, cfg, rng)
        potential = abtem.Potential(
            atoms,
            sampling=args.sampling,
            slice_thickness=args.slice,
            parametrization="lobato",
            projection="infinite",
            exit_planes=exit_every,
        )
        wave = abtem.PlaneWave(energy=args.energy, normalize=False)
        exit_waves = wave.multislice(potential).compute().array
        exit_waves = np.asarray(exit_waves).reshape(-1, *exit_waves.shape[-2:])
        populations.append(project(exit_waves, args.cells, orders, vectors))
        seconds.append(time.perf_counter() - start)
        print(
            f"config {index}: {seconds[-1]:.0f} s, P0..2(end) = {populations[-1][-1, :3]}",
            flush=True,
        )
    depth = np.arange(populations[0].shape[0]) * exit_every * args.slice
    np.savez(
        args.out,
        populations=np.array(populations),
        depth_ang=depth,
        config=json.dumps(cfg),
        seconds=np.array(seconds),
        interaction_constant_relative_error=sigma_check,
        revision=os.environ.get("PYRITE_SOURCE_REVISION", "unknown"),
    )
    return 0


def compare(args) -> int:
    from pyrite.montecarlo.channeling import DensityMatrixSolver, tds_kernel

    files = [np.load(path) for path in args.files]
    configs = {f["config"].item() for f in files}
    if len(configs) != 1:
        raise SystemExit("all inputs must share one configuration")
    cfg = json.loads(configs.pop())
    populations = np.concatenate([f["populations"] for f in files])
    depth = files[0]["depth_ang"]
    mean = populations.mean(axis=0)
    sem = populations.std(axis=0, ddof=1) / np.sqrt(populations.shape[0])
    profile = planar_profile("silicon", tuple(cfg["hkl"]), cfg["temperature_K"])
    kernel = tds_kernel(
        profile, cfg["energy_eV"], n_cells=cfg["n_cells"], max_order=cfg["max_order"]
    )
    solver = DensityMatrixSolver(profile, cfg["energy_eV"], kernel, margin_eV=1e9)
    step = depth[1] - depth[0]
    dz = step / max(1, int(np.ceil(step / args.dz)))
    result = solver.propagate(depth[-1], dz, record_every=int(round(step / dz)))
    bands = min(args.bands, result.band_populations.shape[1])  # the solver reports bound bands
    ours = result.band_populations[:, :bands]
    theirs = mean[:, :bands]
    error = sem[:, :bands]
    z_score = (ours - theirs) / np.maximum(error, 1e-12)
    report = {
        "check": "abtem_channeling_oracle",
        "config": cfg,
        "configurations": int(populations.shape[0]),
        "depth_ang": depth.tolist(),
        "abtem_bands": theirs.tolist(),
        "abtem_sem": error.tolist(),
        "pyrite_bands": ours.tolist(),
        "max_abs_difference": float(np.abs(ours - theirs).max()),
        "max_abs_z": float(np.abs(z_score).max()),
        "pyrite_dz_ang": dz,
        "revisions": sorted({str(f["revision"]) for f in files}),
    }
    args.out.write_text(json.dumps(report, indent=1))
    for row in range(0, depth.size, max(1, depth.size // 8)):
        print(
            f"z={depth[row]:8.0f} A  abTEM {np.round(theirs[row], 4)} +- {np.round(error[row], 4)}"
            f"  PyRITE {np.round(ours[row], 4)}"
        )
    print(f"max |diff| {report['max_abs_difference']:.4f}, max |z| {report['max_abs_z']:.1f}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run", help="propagate frozen configurations and project them")
    go.add_argument("--energy", type=float, default=16.92e6)
    go.add_argument("--hkl", type=int, nargs=3, default=(1, 1, 0))
    go.add_argument("--temperature", type=float, default=293.15)
    go.add_argument("--cells", type=int, default=8)
    go.add_argument("--extent-y", type=float, default=15.0)
    go.add_argument("--sampling", type=float, default=0.02)
    go.add_argument("--slice", type=float, default=1.0)
    go.add_argument("--thickness", type=float, default=10000.0)
    go.add_argument("--exit-every", type=float, default=500.0)
    go.add_argument("--orders", type=int, default=24)
    go.add_argument("--configs", type=int, nargs=2, default=(0, 1), metavar=("START", "STOP"))
    go.add_argument("--seed", type=int, default=504)
    go.add_argument("--model", choices=("crystal", "random-string"), default="crystal")
    go.add_argument(
        "--axis-index", type=int, default=7, help="beam [1, -1, n] for the crystal model"
    )
    go.add_argument("--out", type=Path, required=True)
    cmp_ = sub.add_parser("compare", help="merge configuration files and compare with PyRITE")
    cmp_.add_argument("files", nargs="+", type=Path)
    cmp_.add_argument("--bands", type=int, default=6)
    cmp_.add_argument("--dz", type=float, default=25.0)
    cmp_.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    raise SystemExit(main())

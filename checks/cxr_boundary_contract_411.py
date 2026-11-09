"""Analysis-only discrete boundary-source bookkeeping for issue #411.

This deliberately does NOT construct a PXR/CBS source from trajectory segments.
Inputs are an independently supplied, already escaped *joint* surface measure.
The production CXR angular marginal is insufficient to construct these rows.
Run with ``uv run python checks/cxr_boundary_contract_411.py`` to exercise a
synthetic identity. No physics validation or photon-tracker integration implied.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BoundaryCells:
    """Integrated crossing weights at reference-surface quadrature nodes.

    Each weight is photons / incident primary (integrated over its spatial,
    energy and solid-angle quadrature cell), NOT photons/eV/sr and NOT a
    material-local emission rate. Positions and outward normals are lab-frame;
    directions are outward unit vectors. Only the already-escaped source enters.

    ``coherency`` optionally holds a complex 2x2 polarization matrix *per
    cell*, in the same integrated units with trace = weight. Its corresponding
    ``polarization_u`` is the first transverse basis vector, with the second
    fixed by direction cross polarization_u. An absent matrix means
    polarization was explicitly traced out, NOT that it is unpolarized.
    Inter-cell mutual coherence is not represented: this is an intensity-only
    mean-observable contract and cannot reproduce interferometric propagation.
    """

    position_ang: np.ndarray
    normal: np.ndarray
    direction: np.ndarray
    energy_eV: np.ndarray
    weight_per_primary: np.ndarray
    coherency: np.ndarray | None = None
    polarization_u: np.ndarray | None = None

    def __post_init__(self):
        pos = np.asarray(self.position_ang, dtype=np.float64)
        nor = np.asarray(self.normal, dtype=np.float64)
        direction = np.asarray(self.direction, dtype=np.float64)
        energy = np.asarray(self.energy_eV, dtype=np.float64)
        weight = np.asarray(self.weight_per_primary, dtype=np.float64)
        n = energy.size
        if pos.shape != (n, 3) or nor.shape != (n, 3) or direction.shape != (n, 3):
            raise ValueError("positions, normals and directions must be (N, 3)")
        if energy.shape != (n,) or weight.shape != (n,):
            raise ValueError("energy and weight must both be (N,)")
        if not all(np.all(np.isfinite(x)) for x in (pos, nor, direction, energy, weight)):
            raise ValueError("boundary source arrays must be finite")
        if np.any(energy <= 0) or np.any(weight < 0):
            raise ValueError("energy must be positive and weights nonnegative")
        if n:
            norm_n = np.linalg.norm(nor, axis=1)
            norm_d = np.linalg.norm(direction, axis=1)
            if not np.allclose(norm_n, 1.0, atol=1e-12, rtol=0):
                raise ValueError("surface normals must be unit vectors")
            if not np.allclose(norm_d, 1.0, atol=1e-12, rtol=0):
                raise ValueError("photon directions must be unit vectors")
            if np.any(np.sum(nor * direction, axis=1) <= 0):
                raise ValueError("boundary crossings must point strictly outwards")
        coherency = self.coherency
        basis = self.polarization_u
        if (coherency is None) != (basis is None):
            raise ValueError("coherency and polarization basis must be supplied together")
        if coherency is not None:
            basis = np.asarray(basis, dtype=np.float64)
            if basis.shape != (n, 3) or not np.all(np.isfinite(basis)):
                raise ValueError("polarization basis must be finite (N, 3)")
            if not np.allclose(np.linalg.norm(basis, axis=1), 1, atol=1e-12, rtol=0):
                raise ValueError("polarization basis must be unit vectors")
            if not np.allclose(np.sum(basis * direction, axis=1), 0, atol=1e-12, rtol=0):
                raise ValueError("polarization basis must be transverse to direction")
            coherency = np.asarray(coherency, dtype=np.complex128)
            if coherency.shape != (n, 2, 2):
                raise ValueError("polarization coherency must be (N, 2, 2)")
            if not np.all(np.isfinite(coherency)):
                raise ValueError("polarization coherency must be finite")
            tolerance = 1e-11 * max(float(np.max(weight, initial=0)), 1e-30)
            if not np.allclose(
                coherency, np.swapaxes(coherency.conj(), 1, 2), atol=tolerance, rtol=0
            ):
                raise ValueError("polarization coherency must be Hermitian")
            if not np.allclose(
                np.trace(coherency, axis1=1, axis2=2), weight, atol=tolerance, rtol=1e-11
            ):
                raise ValueError("polarization trace must equal cell crossing weight")
            if np.any(np.linalg.eigvalsh(coherency) < -tolerance):
                raise ValueError("polarization coherency must be positive semidefinite")
        # Defensive copies keep an accepted quadrature immutable after checking.
        for key, array in (
            ("position_ang", pos),
            ("normal", nor),
            ("direction", direction),
            ("energy_eV", energy),
            ("weight_per_primary", weight),
            ("coherency", coherency),
            ("polarization_u", basis),
        ):
            if array is not None:
                copy = np.array(array, copy=True)
                copy.setflags(write=False)
                object.__setattr__(self, key, copy)


def sample_cells(cells: BoundaryCells, n_samples: int, seed: int, *, polarized: bool = False):
    """Sample a discrete *escaped* quadrature, preserving its mean measure.

    Returned statistical weights are already normalized per primary: a later
    tally must NOT divide them by primary count again. Each sampled location
    is an existing quadrature node (not a within-cell position draw). This is
    not a continuum physical event generator; electron recoil is untouched.

    Returns ``(indices, weights_per_primary, coherency)``. The optional
    polarization matrix has the same weight/trace as its history. If callers
    need polarization but only scalar intensities are available, fail closed.
    """
    if type(n_samples) is not int or n_samples < 0:
        raise ValueError("n_samples must be a nonnegative integer")
    if polarized and cells.coherency is None:
        raise ValueError("polarization cannot be reconstructed from scalar intensity")
    total = float(np.sum(cells.weight_per_primary, dtype=np.float64))
    if not np.isfinite(total):
        raise ValueError("integrated source weight must be finite")
    if n_samples == 0 or total == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, np.empty(0), np.empty((0, 2, 2), dtype=complex) if polarized else None
    rng = np.random.Generator(np.random.Philox(seed))
    probability = cells.weight_per_primary / total
    index = rng.choice(cells.energy_eV.size, size=n_samples, p=probability)
    weight = np.full(n_samples, total / n_samples)
    if polarized:
        fraction = cells.coherency[index] / cells.weight_per_primary[index, None, None]
        return index, weight, fraction * weight[:, None, None]
    return index, weight, None


def _synthetic_smoke():
    # 2 equal-area boundary patches, 2 angular tiles, 2 energy bins.
    # Deliberately nonfactorized in direction/energy/position. A physical
    # CXR handoff MUST derive such a joint measure; this test supplies one.
    n = 8
    position = np.array([[float(i // 4), 0.0, 0.0] for i in range(n)])
    normal = np.tile([0.0, 0.0, 1.0], (n, 1))
    direction = np.tile([0.0, 0.0, 1.0], (n, 1))
    direction[1::2] = [0.6, 0.0, 0.8]
    energy = np.tile([500.0, 1000.0, 500.0, 1000.0], 2)
    weight = 0.001 * (1 + np.arange(n))
    cells = BoundaryCells(position, normal, direction, energy, weight)
    idx, weights, _ = sample_cells(cells, 10_000, seed=411)
    assert np.isclose(weights.sum(), weight.sum(), rtol=1e-13)
    assert set(np.unique(idx)) == set(range(n))
    print("PASS synthetic boundary quadrature conservation (not CXR physics validation)")


if __name__ == "__main__":
    _synthetic_smoke()

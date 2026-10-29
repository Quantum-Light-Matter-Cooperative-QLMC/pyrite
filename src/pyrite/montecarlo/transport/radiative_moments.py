"""Prepared radiative moments and independent held-out component checks.

Source: the BremsLib/Ter-Mikaelian moments in ``hard_radiative`` and
``dielectric_radiative``. On each log-energy interval use the convex linear
combination of endpoint moments. Number densities convert per-atom cm² to
inverse Angstrom via n[cm^-3] * 1e-8[cm/Angstrom]. Both first moments are
eV/Angstrom. At a node interpolation is exact; at E=kc the hard moments vanish.
No logarithm of a moment is taken, so zero rates are admissible.

Validation: radiative-moment-interpolation
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from numba import njit

from ..._bremslib_table import BREMSSTRAHLUNG_BREMSLIB_MODEL
from ._jit_radiative import radiative_moments_scalar
from .hard_radiative import pack_radiative_layer_tables

MOMENT_POLICY = "positive-linear-log-energy-v1"


@dataclass(frozen=True)
class MomentPolicy:
    """Mixed component budgets in (eV/Angstrom, 1/Angstrom, eV/Angstrom)."""

    rtol: float = 2e-4
    atol: tuple[float, float, float] = (1e-10, 1e-14, 1e-10)
    max_nodes: int = 32768
    held_out_count: int = 8191

    def __post_init__(self):
        if not np.isfinite(self.rtol) or self.rtol <= 0:
            raise ValueError("rtol must be finite and positive")
        if len(self.atol) != 3 or not np.all(np.isfinite(self.atol)) or min(self.atol) <= 0:
            raise ValueError("atol must contain three finite positive component budgets")
        if self.max_nodes < 2 or self.held_out_count < 1:
            raise ValueError("max_nodes >= 2 and held_out_count >= 1 required")


@njit(cache=True)
def interpolate_moments(energy, log_energy, moments, energy_eV):
    """Convex log-energy interpolation, rejecting unsupported/nonfinite energies.

    ``moments`` has shape (energy, 3); all arrays and arithmetic are float64.
    Validation: radiative-moment-interpolation
    """
    if not np.isfinite(energy_eV) or not energy[0] <= energy_eV <= energy[-1]:
        raise ValueError("radiative moment energy outside prepared coverage")
    if energy_eV == energy[0]:
        return moments[0].copy()
    if energy_eV == energy[-1]:
        return moments[-1].copy()
    x = np.log(energy_eV)
    i = min(np.searchsorted(log_energy, x, side="right") - 1, log_energy.size - 2)
    i = max(i, 0)
    f = (x - log_energy[i]) / (log_energy[i + 1] - log_energy[i])
    # Scalar libm and vector logarithms can differ by an ulp at an admitted
    # physical endpoint. Bound only the weight after the physical domain guard.
    f = min(max(f, 0.0), 1.0)
    return (1.0 - f) * moments[i] + f * moments[i + 1]


@dataclass(frozen=True)
class PreparedRadiativeMoments:
    """Caller-owned immutable host tables and their complete numerical identity.

    No global cache: retain this object to reuse preparation. Coverage is shared
    by all layers; each layer may have a different adaptive grid and plasma.
    ``max_error_fraction`` is the held-out error divided by the mixed budget,
    including the independent total first-moment closure check.
    """

    energy_eV: tuple[np.ndarray, ...]
    log_energy: tuple[np.ndarray, ...]
    moments: tuple[np.ndarray, ...]
    identity_json: str
    digest: str
    max_error_fraction: tuple[float, ...]

    def evaluate(self, layer, energy_eV):
        if not 0 <= layer < len(self.moments):
            raise ValueError("invalid radiative moment layer")
        return interpolate_moments(
            self.energy_eV[layer], self.log_energy[layer], self.moments[layer], energy_eV
        )


@njit(cache=True)
def _exact_layer(packed, layer, energies_eV, cutoff_eV, uncut=False):
    nel, nt, z, density, incident, nominal, top, chi, plasma = packed
    out = np.zeros((energies_eV.size, 3))
    for row in range(energies_eV.size):
        energy = energies_eV[row]
        cutoff = energy if uncut else cutoff_eV
        for element in range(nel[layer]):
            count = nt[layer, element]
            soft, hard, first = radiative_moments_scalar(
                incident[layer, element, :count],
                nominal[layer, element],
                top[layer, element, :count],
                chi[layer, element, :count],
                z[layer, element],
                energy,
                cutoff,
                plasma[layer],
            )
            scale = density[layer, element] * 1e-8
            out[row, 0] += scale * soft
            out[row, 1] += scale * hard
            out[row, 2] += scale * first
    return out


def _budget_error(actual, expected, policy):
    return np.abs(actual - expected) / (np.asarray(policy.atol) + policy.rtol * np.abs(expected))


def _seed_grid(packed, layer, lo, hi, cutoff):
    """Include incident knots and cutoff crossings of nominal/tip photon nodes."""
    nel, nt, _, _, incident, nominal, top, _, _ = packed
    nodes = [lo, hi, cutoff]
    for element in range(nel[layer]):
        count = nt[layer, element]
        nodes.extend(incident[layer, element, :count] * 1e3)
        reduced = np.r_[nominal[layer, element], top[layer, element, :count]]
        nodes.extend(cutoff / reduced[reduced > 0])
    nodes = np.asarray(nodes)
    nodes = np.unique(nodes[(nodes > lo) & (nodes < hi)])
    logs, indices = np.unique(np.log(nodes), return_index=True)
    # Distinct energies can have identical rounded logs near cutoff/tip
    # coincidences. They cannot form a usable log-interpolation interval.
    interior = nodes[indices[(logs > np.log(lo)) & (logs < np.log(hi))]]
    return np.r_[lo, interior, hi]


def _refine(packed, layer, grid, cutoff, policy):
    for _ in range(32):
        if grid.size > policy.max_nodes:
            raise ValueError("radiative moment refinement exceeded max_nodes")
        values = _exact_layer(packed, layer, grid, cutoff)
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("exact radiative moments must be finite and nonnegative")
        logs = np.log(grid)
        fractions = np.array([0.25, 0.5, 0.75])
        probe = np.exp(logs[:-1, None] + np.diff(logs)[:, None] * fractions)
        exact = _exact_layer(packed, layer, probe.ravel(), cutoff).reshape(-1, 3, 3)
        approx = (1 - fractions[None, :, None]) * values[:-1, None, :] + (
            fractions[None, :, None] * values[1:, None, :]
        )
        bad = np.any(_budget_error(approx, exact, policy) > 0.25, axis=(1, 2))
        if not np.any(bad):
            return grid, values
        grid = np.unique(np.r_[grid, probe[bad].ravel()])
    raise ValueError("radiative moment refinement did not converge")


def _validate(packed, layer, grid, values, cutoff, policy):
    lo, hi = np.log(grid[[0, -1]])
    # This shifted grid is fixed independently of the adaptive refinement.
    x = lo + (hi - lo) * (np.arange(policy.held_out_count) + 0.3819660112501051) / (
        policy.held_out_count
    )
    energy = np.exp(x)
    expected = _exact_layer(packed, layer, energy, cutoff)
    actual = np.column_stack([np.interp(x, np.log(grid), values[:, i]) for i in range(3)])
    error = float(np.max(_budget_error(actual, expected, policy)))
    total = _exact_layer(packed, layer, energy, cutoff, uncut=True)[:, 0]
    closure = np.abs(actual[:, 0] + actual[:, 2] - total) / (
        policy.atol[0] + policy.rtol * np.abs(total)
    )
    error = max(error, float(np.max(closure)))
    if not np.isfinite(error) or error > 1.0:
        raise ValueError(f"held-out radiative moment validation failed: budget fraction {error}")
    return error


def _check_source_coverage(compositions, tables, min_energy_eV, max_energy_eV, source_admission):
    """Choose strict component admission or the existing exact reference boundary."""
    if source_admission not in ("strict", "exact-reference"):
        raise ValueError("source_admission must be strict or exact-reference")
    if source_admission == "exact-reference":
        return  # pack_radiative_layer_tables owns reference admission below
    for composition in compositions:
        for element, _ in composition:
            table = tables.get(element)
            if (
                source_admission == "strict"
                and table is not None
                and (
                    min_energy_eV < table.minimum_incident_energy_keV * 1e3
                    or max_energy_eV > table.maximum_incident_energy_keV * 1e3
                )
            ):
                raise ValueError(f"energy coverage outside the BremsLib table for {element}")


def prepare_radiative_moments(
    compositions,
    tables,
    min_energy_eV,
    max_energy_eV,
    cutoff_eV,
    *,
    policy=MomentPolicy(),
    projectile="electron",
    dielectric=True,
    source_admission="strict",
):
    """Prepare and independently probe an explicitly bounded diagnostic table.

    ``tables`` must already represent the specified projectile (including the
    existing positron scaling when applicable). Composition density is in
    Angstrom^-3. The existing exact quadrature owns all physical admission.
    The held-out check is numerical evidence, not an interval-wide proof.
    ``source_admission="strict"`` rejects even adjacent source-outside energies.
    Production uses ``"exact-reference"`` to retain the existing exact packer's
    1e-12 relative endpoint allowance; the reference clamps to its endpoint
    there. This does not relax interpolation queries outside prepared coverage.

    Validation: radiative-moment-interpolation
    """
    if not (np.isfinite(cutoff_eV) and 0 < cutoff_eV <= min_energy_eV < max_energy_eV):
        raise ValueError("require 0 < cutoff <= min_energy < max_energy")
    if not np.isfinite(max_energy_eV) or projectile not in ("electron", "positron"):
        raise ValueError("finite coverage and an explicit electron/positron projectile required")
    if any(not np.isfinite(n) or n <= 0 for comp in compositions for _, n in comp):
        raise ValueError("number densities must be finite and positive")
    _check_source_coverage(compositions, tables, min_energy_eV, max_energy_eV, source_admission)
    packed = pack_radiative_layer_tables(
        compositions, tables, min_energy_eV, max_energy_eV, dielectric=dielectric
    )
    metadata = {
        "policy": MOMENT_POLICY,
        "rtol": policy.rtol,
        "atol": policy.atol,
        "max_nodes": policy.max_nodes,
        "held_out_count": policy.held_out_count,
        "projectile": projectile,
        "cutoff_eV": float(cutoff_eV),
        "coverage_eV": [float(min_energy_eV), float(max_energy_eV)],
        "composition": compositions,
        "plasma_eV": packed[-1].tolist(),
        "sources": sorted((k, t.key, t.digest) for k, t in tables.items()),
        "reference": BREMSSTRAHLUNG_BREMSLIB_MODEL,
    }
    if source_admission != "strict":
        metadata["source_admission"] = source_admission
    identity = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(identity.encode())
    # Hash actual quadrature inputs too; synthetic/custom providers can reuse
    # source labels. Shapes/dtypes make byte concatenation unambiguous.
    for array in packed:
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    grids, moments, errors = [], [], []
    for layer in range(len(compositions)):
        grid = _seed_grid(packed, layer, min_energy_eV, max_energy_eV, cutoff_eV)
        grid, values = _refine(packed, layer, grid, cutoff_eV, policy)
        errors.append(_validate(packed, layer, grid, values, cutoff_eV, policy))
        grids.append(grid)
        moments.append(values)
    logs = [np.log(grid) for grid in grids]
    for array in [*grids, *logs, *moments]:
        array.flags.writeable = False
        digest.update(array.tobytes())
    return PreparedRadiativeMoments(
        tuple(grids), tuple(logs), tuple(moments), identity, digest.hexdigest(), tuple(errors)
    )

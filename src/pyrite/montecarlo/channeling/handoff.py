"""One-way handoff from the planar channeling zone to native transport (#506).

ADR-0018 D4-A: the zone owns each electron from crystal entry until it
dechannels; the dechanneled electron then continues as an ordinary launch
state of the native cores, behind an ``EVENT_DECHANNEL`` row. Selection,
gates and identity live in ``campaign.channeling_cases`` and
:class:`~pyrite.montecarlo.case.Case`; the runner glue that launches and joins
native transport is ``runner.channeling``.

**Ensemble.** :func:`solve_zone` integrates the divergence-averaged entry
state of one case with :class:`~.density.DensityMatrixSolver` over the mean
path ``T / v_z`` at constant continuous stopping ``S(E0)``. Its populations
partition every electron (``P_entry + P_free + P_overflow + P_bound = 1``,
renormalized by ``1 + balance_defect``):

- *entry*: entry weight above the cutoff or outside the basis; no zone row,
  the electron is launched at its own entry state and direction;
- *free*: dechanneled into in-basis states above ``barrier + margin``
  (free and reprojection flux), at depth ``z`` with transverse energy
  ``E_perp`` from the binned flux;
- *overflow*: dechanneled by a jump beyond the plane-wave band, at depth ``z``;
- *bound*: still in the evolving subspace at the exit face; the zone row ends
  there with the exit event and no native transport follows.

**Per electron** (:func:`sample_handoffs`): four counter-addressed uniforms per
electron, keyed on ``(seed, electron)`` under their own salt, pick the fate,
the depth from that fate's cumulative depth flux, ``E_perp`` and the sign.
The handoff angle to the plane is ``psi = sqrt(2 E_perp / (p v))`` with
``E_perp`` measured from the cell-mean potential ``U_0 = 0``; overflow uses the
band-edge angle ``psi_edge = (H + 1/2) g hbar / p`` (a lower bound). The new
direction keeps the electron's own in-plane component and tilts it toward
``+/- n`` (equal signs). Energy ``E0 - S z`` and clock ``t + z / beta(E_mid)``
continue; position is ``r0 + z v0``.

Assumptions (ledger ``cr-zone-handoff-accounting``): the zone row is the
straight mean segment along the entry direction; ``E_perp`` is independent of
the handoff depth (separable sampling); the sign is symmetric; overflow leaves
at the band edge; ``S`` is constant at ``S(E0)`` in both the solve and the row;
rechanneling is omitted; divergence below the Bloch-grid resolution is
snapped to the nearest grid momentum. The 300 eV declared margin sets where
the free flux lands, so it sets the handoff angles (about ``3.7 psi_c`` at
5 MeV Si(110)).

Validation: cr-zone-handoff-accounting
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

#: Physics-generation marker of the planar zone and its handoff. It joins
#: ``case_content_key`` and run provenance only for cases with a ``channeling``
#: record, so channeling-off identities are unchanged.
CHANNELING_MODEL = "planar-density-matrix-emission-v4"

#: Per-electron fates, in sampling order.
FATE_ENTRY, FATE_FREE, FATE_OVERFLOW, FATE_BOUND = 0, 1, 2, 3
FATE_NAMES = ("entry", "free", "overflow", "bound")
#: Above this population share of overflow handoffs a run warns: the band-edge
#: angle is then a poor stand-in, and the basis should grow.
OVERFLOW_WARN_FRACTION = 0.10
# Distinct from the transport, secondary, pair and MCWF salts.
_HANDOFF_STREAM_SALT = np.uint64(0x5851F42D4C957F2D)
_M_E_KEV = 510.99895

__all__ = [
    "CHANNELING_MODEL",
    "FATE_BOUND",
    "FATE_ENTRY",
    "FATE_FREE",
    "FATE_NAMES",
    "FATE_OVERFLOW",
    "OVERFLOW_WARN_FRACTION",
    "HandoffSample",
    "ZoneEnsemble",
    "handoff_directions",
    "handoff_uniforms",
    "sample_handoffs",
    "solve_zone",
]


def beta_from_keV(E_keV):
    """Speed ``v / c`` of electrons of kinetic energy ``E_keV``, as the cores use."""
    gamma = 1.0 + np.asarray(E_keV, dtype=float) / _M_E_KEV
    return np.sqrt(1.0 - 1.0 / (gamma * gamma))


def handoff_uniforms(seed: int, electrons, n_draws: int = 4) -> np.ndarray:
    """Uniforms ``[0, 1)`` of shape ``(n, n_draws)``, a pure function of ``(seed, e, k)``.

    SplitMix64 over ``seed`` under the handoff salt, then the electron index,
    then the draw index; independent of batch size and processing order.

    Validation: cr-zone-handoff-accounting
    """
    from ..transport.kinematics import _SM64_GOLDEN, _SM64_ONE
    from ..transport.secondaries import _mix

    with np.errstate(over="ignore"):
        x = np.uint64(seed) ^ _HANDOFF_STREAM_SALT
        index = np.asarray(electrons, dtype=np.uint64)
        key = _mix(x + _SM64_GOLDEN * (index + _SM64_ONE))
        draws = np.arange(n_draws, dtype=np.uint64)
        bits = _mix(key[:, None] + _SM64_GOLDEN * (draws[None, :] + _SM64_ONE))
    return (bits >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


@dataclass(frozen=True)
class ZoneEnsemble:
    """Data-only summary of one case's density-matrix zone solve.

    Depths in angstrom along the mean path, energies in eV, angles in rad.
    ``*_cumulative`` are dechanneled populations at each recorded depth.
    """

    z_ang: np.ndarray
    free_cumulative: np.ndarray
    overflow_cumulative: np.ndarray
    entry: float
    bound: float
    balance_defect: float
    energy_edges_eV: np.ndarray
    free_by_energy: np.ndarray
    plane_normal: np.ndarray
    pv_eV: float
    psi_edge_rad: float
    psi_critical_rad: float
    barrier_eV: float
    cutoff_eV: float
    stopping_keV_per_ang: float
    path_ang: float
    incidence_rad: float
    divergence_rad: float

    @property
    def probabilities(self) -> np.ndarray:
        """``(P_entry, P_free, P_overflow, P_bound)``, renormalized to sum to 1."""
        raw = np.array(
            [self.entry, self.free_cumulative[-1], self.overflow_cumulative[-1], self.bound]
        )
        return raw / raw.sum()

    def psi_rad(self, transverse_eV) -> np.ndarray:
        """Handoff angle ``sqrt(2 E_perp / (p v))`` to the plane."""
        return np.sqrt(2.0 * np.maximum(np.asarray(transverse_eV, dtype=float), 0.0) / self.pv_eV)


@lru_cache(maxsize=8)
def zone_solver(
    crystal: str,
    plane: tuple[int, int, int],
    temperature_K: float,
    kinetic_eV: float,
    basis_size: int,
    bloch_cells: int,
    margin_eV: float,
):
    """Cached static profile and dynamics shared by transport and CR scoring."""
    from ...materials.channeling.potential import planar_profile
    from .density import DensityMatrixSolver
    from .operators import electronic_kernel, tds_kernel

    profile = planar_profile(crystal, plane, temperature_K)
    kernel = tds_kernel(
        profile, kinetic_eV, n_cells=bloch_cells, max_order=basis_size
    ) + electronic_kernel(profile, kinetic_eV, n_cells=bloch_cells, max_order=basis_size)
    return profile, DensityMatrixSolver(profile, kinetic_eV, kernel, margin_eV=margin_eV)


@lru_cache(maxsize=8)
def _solve(
    crystal: str,
    plane: tuple[int, int, int],
    temperature_K: float,
    kinetic_eV: float,
    basis_size: int,
    bloch_cells: int,
    margin_eV: float,
    path_ang: float,
    dz_ang: float,
    incidence_rad: float,
    divergence_rad: float,
    stopping_eV_per_ang: float,
):
    profile, solver = zone_solver(
        crystal, plane, temperature_K, kinetic_eV, basis_size, bloch_cells, margin_eV
    )
    p_inv_ang = solver.gamma0 * solver.beta0 * 510998.95 / 1973.269804
    step = profile.g_inv_ang / bloch_cells
    if p_inv_ang * divergence_rad / step < 1.0:
        divergence_rad = 0.0  # unresolved by the Bloch grid: snapped (documented)
    result = solver.propagate(
        path_ang,
        dz_ang,
        incidence_rad=incidence_rad,
        divergence_rad=divergence_rad,
        stopping_eV_per_ang=stopping_eV_per_ang,
    )
    minimum, _ = profile.extrema()
    return result, solver, profile, divergence_rad, float(solver.barrier_eV - minimum)


def case_plane_incidence(case):
    """Resolve the case's planes in the same slab frame as native transport."""
    from ...materials.channeling.entry import plane_incidence
    from ...materials.crystal import CRYSTALS, _orientation_R

    rotation = _orientation_R(
        CRYSTALS[case["crystal"]]["lattice"],
        case.get("beam_uvw"),
        case.get("azimuth_rad", 0.0),
        case.get("recip_miscut_rad"),
        surface_hkl=case.get("surface_hkl"),
    )
    return plane_incidence(
        case["crystal"],
        tuple(int(v) for v in case["channeling"]["plane_hkl"]),
        float(case["E0_keV"]) * 1e3,
        tilt_deg=float(case.get("tilt_deg", 0.0)),
        tilt_azim_deg=float(case.get("tilt_azim_deg", 0.0)),
        rotation=np.eye(3) if rotation is None else rotation,
    )


def solve_zone(case, beam_dir: np.ndarray, v_hat: np.ndarray, stopping_keV_per_ang: float):
    """Solve the ensemble zone of ``case`` for primaries entering along ``v_hat``.

    ``beam_dir`` is the unit mean beam direction in the slab frame and
    ``v_hat`` the sampled per-electron entry directions; their spread normal
    to the plane is the entry divergence.

    Raises
    ------
    ValueError
        If ``zone_max_ang`` is shorter than the mean path through the slab.

    Validation: cr-zone-handoff-accounting
    """
    from ...materials.crystal import HBARC_EV_ANG

    record = case["channeling"]
    path = float(case["thickness_ang"]) / float(beam_dir[2])
    cap = record["zone_max_ang"]
    if cap is not None and float(cap) < path:
        raise ValueError(
            f"channeling zone_max_ang={cap:g} is shorter than the {path:g} angstrom path "
            "through the slab; forced handoff of still-channeled electrons is unsupported"
        )
    plane = tuple(int(v) for v in record["plane_hkl"])
    kinetic_eV = float(case["E0_keV"]) * 1e3
    incidence = case_plane_incidence(case)
    normal = incidence.plane_normal
    psi = np.arcsin(np.clip(np.asarray(v_hat) @ normal, -1.0, 1.0))
    divergence = float(np.std(psi)) if psi.size > 1 else 0.0
    if case.get("channeling_source") is None:
        result, solver, profile, divergence, depth_eV = _solve(
            case["crystal"],
            plane,
            float(record["temperature_K"]),
            kinetic_eV,
            int(record["basis_size"]),
            int(record["bloch_cells"]),
            float(record["margin_eV"]),
            path,
            float(record["dz_ang"]),
            float(incidence.effective_rad),
            divergence,
            float(stopping_keV_per_ang) * 1e3,
        )
    else:
        from .planar_injection import fresh_source_solver, source_entry_initial

        solver = fresh_source_solver(case)
        profile = solver.profile
        initial = source_entry_initial(case, solver, float(incidence.effective_rad))
        result = solver.propagate(
            path,
            float(record["dz_ang"]),
            initial=initial,
            stopping_eV_per_ang=float(stopping_keV_per_ang) * 1e3,
        )
        minimum, _ = profile.extrema()
        depth_eV = float(solver.barrier_eV - minimum)
        divergence = 0.0  # source momentum is already contained in initial

    gamma, beta = solver.gamma0, solver.beta0
    pc_eV = gamma * beta * 510998.95
    pv_eV = pc_eV * beta
    band_edge = (int(record["basis_size"]) + 0.5) * profile.g_inv_ang * HBARC_EV_ANG / pc_eV
    return ZoneEnsemble(
        z_ang=np.asarray(result.z_ang),
        free_cumulative=np.asarray(result.free_flux + result.reprojection_flux),
        overflow_cumulative=np.asarray(result.overflow_flux),
        entry=float(result.entry_flux),
        bound=float(result.trace[-1]),
        balance_defect=float(result.balance_defect[-1]),
        energy_edges_eV=np.asarray(result.energy_edges_eV),
        free_by_energy=np.asarray(result.free_flux_by_energy),
        plane_normal=np.asarray(normal, dtype=float),
        pv_eV=float(pv_eV),
        psi_edge_rad=float(band_edge),
        psi_critical_rad=float(np.sqrt(2.0 * depth_eV / pv_eV)),
        barrier_eV=float(result.barrier_eV),
        cutoff_eV=float(result.cutoff_eV),
        stopping_keV_per_ang=float(stopping_keV_per_ang),
        path_ang=path,
        incidence_rad=float(incidence.effective_rad),
        divergence_rad=float(divergence),
    )


def _invert_cumulative(z: np.ndarray, cumulative: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Depth at which the cumulative flux reaches ``u * total`` (linear per step)."""
    total = float(cumulative[-1])
    if total <= 0.0:
        return np.full(u.shape, float(z[-1]))
    target = u * total
    j = np.clip(np.searchsorted(cumulative, target, side="right"), 1, cumulative.size - 1)
    lo, hi = cumulative[j - 1], cumulative[j]
    frac = np.where(hi > lo, (target - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0)
    return z[j - 1] + np.clip(frac, 0.0, 1.0) * (z[j] - z[j - 1])


@dataclass(frozen=True)
class HandoffSample:
    """Sampled per-electron fates: depth [angstrom] and handoff angle/sign."""

    fate: np.ndarray
    z_ang: np.ndarray
    transverse_eV: np.ndarray
    psi_rad: np.ndarray
    sign: np.ndarray


def sample_handoffs(ensemble: ZoneEnsemble, seed: int, electrons) -> HandoffSample:
    """Sample each electron's fate, depth, ``E_perp`` and sign from the ensemble.

    Validation: cr-zone-handoff-accounting
    """
    electrons = np.asarray(electrons, dtype=np.int64)
    u = handoff_uniforms(seed, electrons)
    edges = np.cumsum(ensemble.probabilities)
    fate = np.minimum(np.searchsorted(edges, u[:, 0], side="right"), FATE_BOUND).astype(np.int8)
    z = np.zeros(electrons.size)
    free = fate == FATE_FREE
    overflow = fate == FATE_OVERFLOW
    z[free] = _invert_cumulative(ensemble.z_ang, ensemble.free_cumulative, u[free, 1])
    z[overflow] = _invert_cumulative(ensemble.z_ang, ensemble.overflow_cumulative, u[overflow, 1])
    z[fate == FATE_BOUND] = np.inf
    # E_perp of a free handoff: bin by the binned free flux, uniform within it.
    weights = np.maximum(ensemble.free_by_energy, 0.0)
    cdf = np.cumsum(weights) / max(float(weights.sum()), np.finfo(float).tiny)
    scaled = u[:, 2]
    bins = np.minimum(np.searchsorted(cdf, scaled, side="right"), weights.size - 1)
    lower = np.concatenate(([0.0], cdf))[bins]
    width = np.where(cdf[bins] > lower, cdf[bins] - lower, 1.0)
    within = np.clip((scaled - lower) / width, 0.0, 1.0)
    e_lo, e_hi = ensemble.energy_edges_eV[bins], ensemble.energy_edges_eV[bins + 1]
    transverse = np.where(free, e_lo + within * (e_hi - e_lo), 0.0)
    psi = np.where(free, ensemble.psi_rad(transverse), 0.0)
    psi = np.where(overflow, ensemble.psi_edge_rad, psi)
    sign = np.where(u[:, 3] < 0.5, -1.0, 1.0)
    return HandoffSample(fate=fate, z_ang=z, transverse_eV=transverse, psi_rad=psi, sign=sign)


def handoff_directions(v_hat: np.ndarray, normal: np.ndarray, psi_rad, sign) -> np.ndarray:
    """Directions at angle ``sign * psi`` to the plane, keeping the in-plane part.

    ``t = (v - (v . n) n) / |...|`` is the electron's own in-plane direction;
    the result is ``cos(psi) t + sign sin(psi) n``.

    Validation: cr-zone-handoff-accounting
    """
    v = np.asarray(v_hat, dtype=float)
    n = np.asarray(normal, dtype=float)
    inplane = v - (v @ n)[:, None] * n[None, :]
    inplane /= np.linalg.norm(inplane, axis=1, keepdims=True)
    psi = np.asarray(psi_rad, dtype=float)
    return np.cos(psi)[:, None] * inplane + (np.asarray(sign) * np.sin(psi))[:, None] * n[None, :]


def warn_overflow(ensemble: ZoneEnsemble, name: str) -> None:
    """Warn when the band-edge stand-in covers more than 10% of electrons."""
    share = float(ensemble.probabilities[FATE_OVERFLOW])
    if share > OVERFLOW_WARN_FRACTION:
        warnings.warn(
            f"case {name!r}: {share:.0%} of electrons dechannel beyond the plane-wave band "
            "and are handed off at its edge angle (a lower bound); raise "
            "Numerics.channeling_basis_size",
            UserWarning,
            stacklevel=3,
        )

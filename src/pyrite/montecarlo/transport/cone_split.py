"""Detector-cone splitting of elastic events: CPU per-electron prototype (#203).

At MeV energies the incoherent PXR/CBS line yield toward a detector direction
``n`` is carried by rare electrons that one elastic event turns into the cone
``C = {Omega : Omega . n >= cos(theta_c)}`` (the detector's ``1/gamma``
radiation cone). This module partitions each elastic event's outgoing
direction at ``C``:

* the primary keeps its analog draw; if that draw lands in ``C`` the primary
  is terminated after the event (its in-cone future is represented below);
* a **copy** represents the in-cone outcome. Its direction is drawn from the
  analog angular law restricted to the polar band that meets ``C`` and from
  the azimuth arc inside ``C``, and it carries the likelihood ratio
  ``p/q = P_band * dphi(theta)/pi``. Russian roulette against a target weight
  keeps the copy count bounded.

Per event the expected score of any functional of the future is unchanged, so
the summed per-history tally is unbiased; with splitting off the transport is
the historical core, bit for bit. Copies are transported as one launched
generation without further splitting, from counter-addressed streams keyed by
(seed, electron, flight). Derivation, assumptions and limiting cases:
``docs/validation/beam-transport/detector-cone-variance-reduction.md``.

Validation: detector-cone-variance-reduction
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from .kinematics import (
    _SM64_GOLDEN,
    _SM64_MIX1,
    _SM64_MIX2,
    _SM64_ONE,
    _SM64_S27,
    _SM64_S30,
    _SM64_S31,
    _splitmix64,
    _stream_uniform_scalar,
)
from .scattering import _elsepa_bracket, _elsepa_invert_row

__all__ = [
    "EXIT_CONE_SPLIT",
    "ConeSplit",
    "copy_stream_keys",
    "history_sums",
    "split_stream_keys",
    "weighted_history_statistics",
]

#: Per-electron exit code of a primary terminated at a cone-entering event.
EXIT_CONE_SPLIT = np.int8(6)
# Distinct from the transport, hard, Urban, radiative and secondary salts.
_SPLIT_STREAM_SALT = np.uint64(0xC2B2AE3D27D4EB4F)
_COPY_STREAM_SALT = np.uint64(0x165667B19E3779F9)
_BISECTION_STEPS = 64


@dataclass(frozen=True)
class ConeSplit:
    """Detector-cone splitting request for one transport (#203 prototype).

    ``n_hat`` is the detector direction in the transport (sample) frame,
    ``half_angle_rad`` the cone half-angle ``theta_c`` (unbiased for any value;
    efficiency only), ``target_weight`` the Russian-roulette target copy weight
    and ``max_copies_per_electron`` the copy buffer, which raises when full
    rather than dropping copies.
    """

    n_hat: tuple[float, float, float]
    half_angle_rad: float
    target_weight: float
    max_copies_per_electron: int = 64

    def __post_init__(self) -> None:
        n = np.asarray(self.n_hat, dtype=float)
        norm = float(np.linalg.norm(n))
        if n.shape != (3,) or not math.isfinite(norm) or norm == 0.0:
            raise ValueError("n_hat must be a finite nonzero three-vector")
        object.__setattr__(self, "n_hat", tuple(float(v) for v in n / norm))
        if not 0.0 < self.half_angle_rad < math.pi / 2:
            raise ValueError("half_angle_rad must lie in (0, pi/2)")
        if not (math.isfinite(self.target_weight) and self.target_weight > 0.0):
            raise ValueError("target_weight must be positive and finite")
        if type(self.max_copies_per_electron) is not int or self.max_copies_per_electron < 1:
            raise ValueError("max_copies_per_electron must be a positive integer")


def _mix(x):
    x = (x ^ (x >> _SM64_S30)) * _SM64_MIX1
    x = (x ^ (x >> _SM64_S27)) * _SM64_MIX2
    return x ^ (x >> _SM64_S31)


def split_stream_keys(stream_keys):
    """Per-electron split-decision keys, disjoint from every transport stream."""
    with np.errstate(over="ignore"):
        return _mix(np.asarray(stream_keys, dtype=np.uint64) ^ _SPLIT_STREAM_SALT)


@njit(cache=True)
def _event_key(split_key, flight):
    """Key of one elastic event: the electron's split key and its flight index."""
    return _splitmix64(split_key + _SM64_GOLDEN * (np.uint64(flight) + _SM64_ONE))


def copy_stream_keys(split_keys, flights):
    """Transport stream keys of copies made at ``flights`` of electrons with ``split_keys``.

    Host twin of the core's event key, re-hashed under the copy salt, so a
    copy's streams depend only on (seed, electron, flight).
    """
    with np.errstate(over="ignore"):
        k = np.asarray(split_keys, dtype=np.uint64)
        f = np.asarray(flights, dtype=np.uint64)
        event = _mix(k + _SM64_GOLDEN * (f + _SM64_ONE))
        return _mix(event ^ _COPY_STREAM_SALT)


@njit(cache=True)
def _row_cdf_at(cdf, pdf, mu, row, m):
    """Exact CDF of one piecewise-linear ELSEPA angular density at ``mu = m``."""
    n = mu.size
    if m <= mu[0]:
        return 0.0
    if m >= mu[n - 1]:
        return 1.0
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mu[mid] <= m:
            lo = mid
        else:
            hi = mid
    dmu = mu[lo + 1] - mu[lo]
    t = (m - mu[lo]) / dmu
    p0 = pdf[row, lo]
    p1 = pdf[row, lo + 1]
    value = cdf[row, lo] + dmu * (p0 * t + 0.5 * (p1 - p0) * t * t)
    return min(1.0, max(0.0, value))


@njit(cache=True)
def _mu_of_xi(row, f, xi, cdf, pdf, mu):
    """``mu`` the analog ELSEPA sampler returns for uniform ``xi`` (quantile interpolation)."""
    return (1.0 - f) * _elsepa_invert_row(cdf, pdf, mu, row, xi) + f * _elsepa_invert_row(
        cdf, pdf, mu, row + 1, xi
    )


@njit(cache=True)
def _arc_half_width(cos_t, sin_t, cos_dn, sin_dn, cos_c):
    """Half-width of the azimuth arc about the incident direction inside the cone."""
    if sin_t * sin_dn <= 1e-300:
        return np.pi if cos_t * cos_dn >= cos_c else 0.0
    c_star = (cos_c - cos_t * cos_dn) / (sin_t * sin_dn)
    if c_star <= -1.0:
        return np.pi
    if c_star >= 1.0:
        return 0.0
    return np.arccos(c_star)


@njit(cache=True)
def cone_split_event(
    dx, dy, dz, nx, ny, nz, cos_c, sin_c, target_weight, event_key,
    E_keV, logE_flat, cdf, pdf, mu, start, length,
):  # fmt: skip
    """One elastic event's in-cone copy: ``(made, ox, oy, oz, weight)``.

    ``q`` draws ``mu`` from the analog law on the uniform interval
    ``[xi_a, xi_b]`` mapping into the polar band ``|theta - theta_dn| <=
    theta_c`` and the azimuth uniformly on the arc inside the cone, so
    ``p/q = P_band * dphi(theta)/pi`` on ``C`` and ``q > 0`` on all of ``C``.
    Roulette: a copy is made with probability ``pi_s = min(1, U/w_t)``, ``U``
    an upper bound of ``p/q``, and carries ``(p/q)/pi_s``. Draws come from the
    event key, never from the electron's transport stream.

    Validation: detector-cone-variance-reduction
    """
    cos_dn = dx * nx + dy * ny + dz * nz
    cos_dn = min(1.0, max(-1.0, cos_dn))
    sin_dn = np.sqrt(max(0.0, 1.0 - cos_dn * cos_dn))
    theta_dn = np.arccos(cos_dn)
    theta_c = np.arcsin(sin_c)
    theta_lo = max(0.0, theta_dn - theta_c)
    theta_hi = min(np.pi, theta_dn + theta_c)
    mu_lo = 0.5 * (1.0 - np.cos(theta_lo))
    mu_hi = 0.5 * (1.0 - np.cos(theta_hi))
    row, f = _elsepa_bracket(np.log(E_keV * 1e3), logE_flat, start, length)
    # Upper bound of P_band: mu(xi) >= mu_lo needs xi >= min_k F_k(mu_lo) and
    # mu(xi) <= mu_hi needs xi <= max_k F_k(mu_hi) (quantile interpolation).
    f0a = _row_cdf_at(cdf, pdf, mu, row, mu_lo)
    f1a = _row_cdf_at(cdf, pdf, mu, row + 1, mu_lo)
    f0b = _row_cdf_at(cdf, pdf, mu, row, mu_hi)
    f1b = _row_cdf_at(cdf, pdf, mu, row + 1, mu_hi)
    band_bound = max(f0b, f1b) - min(f0a, f1a)
    if sin_dn <= sin_c:
        arc_bound = np.pi
    else:
        arc_bound = np.arcsin(sin_c / sin_dn)
    bound = band_bound * arc_bound / np.pi
    if bound <= 0.0:
        return False, 0.0, 0.0, 0.0, 0.0
    keep = min(1.0, bound / target_weight)
    if _stream_uniform_scalar(event_key, np.uint64(0)) >= keep:
        return False, 0.0, 0.0, 0.0, 0.0
    # Exact uniform interval mapping into the band, by bisection on mu(xi).
    if _mu_of_xi(row, f, 0.0, cdf, pdf, mu) >= mu_lo:
        xi_a = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            if _mu_of_xi(row, f, mid, cdf, pdf, mu) >= mu_lo:
                hi = mid
            else:
                lo = mid
        xi_a = hi
    if _mu_of_xi(row, f, 1.0, cdf, pdf, mu) <= mu_hi:
        xi_b = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            if _mu_of_xi(row, f, mid, cdf, pdf, mu) <= mu_hi:
                lo = mid
            else:
                hi = mid
        xi_b = lo
    p_band = xi_b - xi_a
    if p_band <= 0.0:
        return False, 0.0, 0.0, 0.0, 0.0
    xi = xi_a + _stream_uniform_scalar(event_key, np.uint64(1)) * p_band
    mu_t = _mu_of_xi(row, f, xi, cdf, pdf, mu)
    cos_t = 1.0 - 2.0 * mu_t
    sin_t = np.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    half = _arc_half_width(cos_t, sin_t, cos_dn, sin_dn, cos_c)
    if half <= 0.0:
        return False, 0.0, 0.0, 0.0, 0.0
    psi = (2.0 * _stream_uniform_scalar(event_key, np.uint64(2)) - 1.0) * half
    # Frame about the incident direction with e1 toward n.
    if sin_dn > 1e-12:
        e1x = (nx - cos_dn * dx) / sin_dn
        e1y = (ny - cos_dn * dy) / sin_dn
        e1z = (nz - cos_dn * dz) / sin_dn
    else:
        if abs(dx) < 0.9:
            rx, ry, rz = 1.0, 0.0, 0.0
        else:
            rx, ry, rz = 0.0, 1.0, 0.0
        e1x, e1y, e1z = dy * rz - dz * ry, dz * rx - dx * rz, dx * ry - dy * rx
        norm = np.sqrt(e1x * e1x + e1y * e1y + e1z * e1z)
        e1x, e1y, e1z = e1x / norm, e1y / norm, e1z / norm
    e2x, e2y, e2z = dy * e1z - dz * e1y, dz * e1x - dx * e1z, dx * e1y - dy * e1x
    a, b = sin_t * np.cos(psi), sin_t * np.sin(psi)
    ox = cos_t * dx + a * e1x + b * e2x
    oy = cos_t * dy + a * e1y + b * e2y
    oz = cos_t * dz + a * e1z + b * e2z
    norm = np.sqrt(ox * ox + oy * oy + oz * oz)
    weight = p_band * half / np.pi / keep
    return True, ox / norm, oy / norm, oz / norm, weight


def history_sums(primary, copies, copy_parent, copy_weight, n_histories):
    """Per-history tally ``X_i = x_i + sum_{c in i} w_c x_c`` (primary plus weighted copies).

    Validation: detector-cone-variance-reduction
    """
    total = np.asarray(primary, dtype=float).copy()
    if total.shape != (n_histories,):
        raise ValueError("primary needs one value per history")
    if np.size(copies):
        total += np.bincount(
            np.asarray(copy_parent, dtype=np.int64),
            weights=np.asarray(copy_weight, dtype=float) * np.asarray(copies, dtype=float),
            minlength=n_histories,
        )
    return total


def weighted_history_statistics(values):
    """Mean, relative SE, largest share and Kish ESS of per-history sums.

    The sample unit is the primary history (primary plus its weighted copies),
    which are i.i.d.; copies of one history are correlated and never separate
    samples.

    Validation: detector-cone-variance-reduction
    """
    x = np.asarray(values, dtype=float)
    n = x.size
    total = float(x.sum())
    mean = total / n if n else 0.0
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    square = float(np.sum(x * x))
    return {
        "n": int(n),
        "mean": mean,
        "relative_se": sd / (math.sqrt(n) * abs(mean)) if n > 1 and mean else None,
        "max_share": float(np.max(np.abs(x)) / abs(total)) if total else None,
        "effective_sample_size": total * total / square if square else 0.0,
    }


@dataclass(frozen=True)
class _ConeSplitPass:
    """The primaries' pass: core arguments in, copy records out."""

    cone: ConeSplit

    def core_args(self, stream_keys, n_electrons):
        cap = int(self.cone.max_copies_per_electron)
        n_rows = int(n_electrons) * cap
        nx, ny, nz = self.cone.n_hat
        theta_c = float(self.cone.half_angle_rad)
        return (
            float(nx),
            float(ny),
            float(nz),
            math.cos(theta_c),
            math.sin(theta_c),
            float(self.cone.target_weight),
            split_stream_keys(stream_keys),
            cap,
            np.zeros(n_electrons, dtype=np.int64),
            np.zeros(n_electrons, dtype=np.bool_),
            np.empty((n_rows, 3), dtype=np.float64),
            np.empty((n_rows, 3), dtype=np.float64),
            np.empty(n_rows, dtype=np.float64),
            np.empty(n_rows, dtype=np.float64),
            np.empty(n_rows, dtype=np.float64),
            np.empty(n_rows, dtype=np.int64),
        )

    def collect(self, split_args, stream_keys, n_electrons):
        """Compact the per-electron copy slots in (electron, event) order."""
        keys, cap, count, killed = split_args[6:10]
        pos, direction, energy, clock, weight, flight = split_args[10:]
        if count.size and int(count.max()) > cap:
            raise RuntimeError(
                f"detector-cone splitting made {int(count.max())} copies for one electron, "
                f"above max_copies_per_electron={cap}; raise it or the target weight"
            )
        parent = np.repeat(np.arange(n_electrons, dtype=np.int64), count)
        rows = parent * cap + (np.arange(parent.size) - np.repeat(np.cumsum(count) - count, count))
        return {
            "parent": parent,
            "flight": flight[rows].copy(),
            "r_ang": pos[rows].copy(),
            "v_hat": direction[rows].copy(),
            "E_keV": energy[rows].copy(),
            "t_ang": clock[rows].copy(),
            "weight": weight[rows].copy(),
            "stream_keys": copy_stream_keys(keys[parent], flight[rows]),
            "killed": killed.copy(),
            "n_hat": self.cone.n_hat,
            "half_angle_rad": float(self.cone.half_angle_rad),
            "target_weight": float(self.cone.target_weight),
        }


_COPY_PHOTON_SALT = np.uint64(0x27D4EB2F165667C5)
# Beam-sampling arguments a launched copy generation must not see.
_COPY_BEAM_ARGS = {
    "beam_dir": None,
    "beam_fwhm_mm": None,
    "beam_fwhm_y_mm": None,
    "bunch_length_fs": None,
    "long_offsets_fs": None,
    "longitudinal_distribution": None,
    "transverse_distribution": None,
    "energy_spread_frac": None,
    "tilt_polar_rad": 0.0,
    "tilt_azim_rad": 0.0,
    "gdf_source": None,
}


def transport_cone_split(simulate, kw):
    """Primaries with detector-cone splitting, then their copies as one generation.

    Returns the primaries' result; ``result["cone_split"]`` holds the copy
    records and ``["copies"]``, the copies' own transport result (or ``None``).
    Copies are not split again.

    Validation: detector-cone-variance-reduction
    """
    from .secondaries import LaunchState, SecondaryPass, _radiative_keys

    cone = kw["_cone_split"]
    if kw.get("_secondary") is not None or int(kw.get("_electron_start", 0)):
        raise ValueError("detector-cone splitting supports neither cascades nor electron blocks")
    if kw.get("gdf_source") is not None or kw.get("groove") is not None:
        raise ValueError("detector-cone splitting supports neither GDF beams nor grooves")
    base = {k: v for k, v in kw.items() if k not in ("_cone_split", "_secondary")}
    primaries = simulate(**base, _cone_split=_ConeSplitPass(cone))
    record = primaries["cone_split"]
    parent = record["parent"]
    record["copies"] = None
    if parent.size == 0:
        return primaries
    keys = record["stream_keys"]
    seed = int(kw["seed"])
    with np.errstate(over="ignore"):
        photon_seed = int(_mix(np.uint64(seed) ^ _COPY_PHOTON_SALT))
    table_range = tuple(primaries["table_range_keV"])
    launch = LaunchState(
        record["r_ang"],
        record["v_hat"],
        record["E_keV"],
        record["t_ang"],
        np.asarray(primaries["initial_t0_ang"], dtype=float)[parent],
        keys,
        _radiative_keys(keys),
    )
    step = dict(base, **_COPY_BEAM_ARGS)
    step.update(
        E0_keV=float(np.max(record["E_keV"])),
        Ne=int(parent.size),
        E_cut_by_electrons=np.asarray(primaries["initial_E_cut_keV"], dtype=float)[parent],
        transport_core="per-electron",
        _energy_range_keV=table_range,
    )
    record["copies"] = simulate(
        **step, _secondary=SecondaryPass(launch, table_range, photon_seed)
    )
    return primaries

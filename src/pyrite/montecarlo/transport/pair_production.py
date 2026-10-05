"""Pair production by coupled hard bremsstrahlung photons (#275).

PyRITE does not transport photons. In the coupled radiative mode a hard
photon above the pair threshold ``2 m_e c^2`` gets one analog first-interaction
step: its distance to the first interaction is sampled from the EPDL2025
narrow-beam ``mu_total`` along its emitted direction through the planar layer
stack (and finite rectangular footprint), and the channel there is chosen by
EPDL partial cross sections. Only a pair interaction (nuclear or electron
field) is followed; any other first interaction, or an escape, leaves the
photon as radiated energy, exactly as before.

A pair event is sampled with the PENELOPE-2024 model (NEA/MBDAV/R(2024)1,
section 2.4): the electron's reduced energy ``eps = (E_- + m_e c^2)/E`` from the
Bethe--Heitler DCS with exponential screening, Coulomb correction and the
empirical low-energy term ``F_0`` (Eqs. 2.85--2.88, Table 2.2), sampled by
composition and rejection (Eqs. 2.90--2.96); the polar angles of each particle
from the leading high-energy term ``p(cos) ~ (1 - beta cos)^-2`` (Eq. 2.99) with
independent uniform azimuths. Triplet production enters only through the EPDL
total and is simulated as a pair, as in PENELOPE; its threshold-scale share is
below 10 % of the pair cross section for every catalog element up to 5 MeV.

Positrons are recorded and energy-accounted, not transported (#276).

RNG. Every photon owns a Philox stream keyed on (seed, parent track, photon
ordinal on that track) under its own salt, so a photon's interaction does not
depend on batch size, generation order, or the transport core.

Validation: photon-pair-first-interaction, pair-production-sampling
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ...materials.photon_cross_sections import photon_cross_sections_ang2
from ..geometry import first_prism_exit
from .kinematics import (
    _SM64_GOLDEN,
    _SM64_MIX1,
    _SM64_MIX2,
    _SM64_ONE,
    _SM64_S27,
    _SM64_S30,
    _SM64_S31,
)

__all__ = [
    "PAIR_PRODUCTION_MODELS",
    "PAIR_THRESHOLD_EV",
    "PairProducts",
    "convert_hard_photons",
    "pair_photon_stream_key",
    "pair_reduced_energy_density",
    "photon_first_interactions",
    "sample_pair",
    "sample_pair_polar_cosine",
    "sample_pair_reduced_energy",
]

PAIR_PRODUCTION_MODELS = ("penelope-2024",)

_ELECTRON_REST_EV = 510998.95
PAIR_THRESHOLD_EV = 2.0 * _ELECTRON_REST_EV
_ALPHA = 1.0 / 137.035999084
_PAIR_PHOTON_SALT = np.uint64(0x3C6EF372FE94F82B)
_MAX_REJECTIONS = 100_000

#: Reduced screening radius ``R m_e c / hbar`` for Z = 1..99, PENELOPE-2024
#: Table 2.2 (Baro et al. 1994, fitted to Hubbell, Gimm and Overbo 1980).
_SCREENING_RADIUS = np.array(
    [
        122.810, 73.167, 69.228, 67.301, 64.696, 61.228, 57.524, 54.033,
        50.787, 47.851, 46.373, 45.401, 44.503, 43.815, 43.074, 42.321,
        41.586, 40.953, 40.524, 40.256, 39.756, 39.144, 38.462, 37.778,
        37.174, 36.663, 35.986, 35.317, 34.688, 34.197, 33.786, 33.422,
        33.068, 32.740, 32.438, 32.143, 31.884, 31.622, 31.438, 31.142,
        30.950, 30.758, 30.561, 30.285, 30.097, 29.832, 29.581, 29.411,
        29.247, 29.085, 28.930, 28.721, 28.580, 28.442, 28.312, 28.139,
        27.973, 27.819, 27.675, 27.496, 27.285, 27.093, 26.911, 26.705,
        26.516, 26.304, 26.108, 25.929, 25.730, 25.577, 25.403, 25.245,
        25.100, 24.941, 24.790, 24.655, 24.506, 24.391, 24.262, 24.145,
        24.039, 23.922, 23.813, 23.712, 23.621, 23.523, 23.430, 23.331,
        23.238, 23.139, 23.048, 22.967, 22.833, 22.694, 22.624, 22.545,
        22.446, 22.358, 22.264,
    ]
)  # fmt: skip


def _screening_radius(Z: int) -> float:
    if not 1 <= Z <= _SCREENING_RADIUS.size:
        raise ValueError(f"pair-production screening radius is tabulated for Z=1..99, not {Z}")
    return float(_SCREENING_RADIUS[Z - 1])


def _coulomb_correction(Z: int) -> float:
    """Davies--Bethe--Maximon ``f_C(Z)``, PENELOPE-2024 Eq. 2.80.

    Validation: pair-production-sampling
    """
    a2 = (_ALPHA * Z) ** 2
    series = (
        0.202059
        - 0.03693 * a2
        + 0.00835 * a2**2
        - 0.00201 * a2**3
        + 0.00049 * a2**4
        - 0.00012 * a2**5
        + 0.00003 * a2**6
    )
    return a2 * (1.0 / (1.0 + a2) + series)


def _low_energy_correction(kappa: float, Z: int) -> float:
    """Empirical ``F_0(kappa, Z)``, PENELOPE-2024 Eq. 2.88.

    Validation: pair-production-sampling
    """
    a = _ALPHA * Z
    x = 2.0 / kappa
    return (
        (-1.774 - 12.10 * a + 11.18 * a * a) * np.sqrt(x)
        + (8.523 + 73.26 * a - 44.41 * a * a) * x
        - (13.52 + 121.1 * a - 96.41 * a * a) * x**1.5
        + (8.946 + 62.05 * a - 63.41 * a * a) * x * x
    )


def _phi(eps, kappa: float, Z: int):
    """``phi_1, phi_2`` of Eqs. 2.86--2.87, clipped at zero as in PENELOPE.

    Validation: pair-production-sampling
    """
    radius = _screening_radius(Z)
    eps = np.asarray(eps, dtype=float)
    b = radius / (2.0 * kappa * eps * (1.0 - eps))
    atan = np.arctan(1.0 / b)
    log_b = np.log1p(b * b)
    tail = 4.0 - 4.0 * b * atan - 3.0 * np.log1p(1.0 / (b * b))
    g1 = 7.0 / 3.0 - 2.0 * log_b - 6.0 * b * atan - b * b * tail
    g2 = 11.0 / 6.0 - 2.0 * log_b - 3.0 * b * atan + 0.5 * b * b * tail
    g0 = 4.0 * np.log(radius) - 4.0 * _coulomb_correction(Z) + _low_energy_correction(kappa, Z)
    return np.maximum(g1 + g0, 0.0), np.maximum(g2 + g0, 0.0)


def pair_reduced_energy_density(eps, photon_energy_eV: float, Z: int):
    """Unnormalized PDF of ``eps``, PENELOPE-2024 Eq. 2.90; zero outside its range.

    ``p(eps) = 2 (1/2 - eps)^2 phi_1(eps) + phi_2(eps)`` on
    ``(1/kappa, 1 - 1/kappa)`` with ``kappa = E / m_e c^2``.
    Validation: pair-production-sampling
    """
    kappa = float(photon_energy_eV) / _ELECTRON_REST_EV
    eps = np.asarray(eps, dtype=float)
    inside = (eps > 1.0 / kappa) & (eps < 1.0 - 1.0 / kappa)
    safe = np.where(inside, eps, 0.5)
    phi1, phi2 = _phi(safe, kappa, Z)
    return np.where(inside, 2.0 * (0.5 - safe) ** 2 * phi1 + phi2, 0.0)


def sample_pair_reduced_energy(photon_energy_eV: float, Z: int, rng: np.random.Generator) -> float:
    """Sample ``eps`` by PENELOPE-2024's composition/rejection, Eqs. 2.91--2.96.

    Just above threshold ``F_0`` can make both ``phi`` vanish at ``eps = 1/2``
    (the fitted DCS shifts the threshold up slightly; this happens only for
    Z >= 85 below 1.031--1.087 MeV); there the density is taken flat on its
    range, the limit of the symmetric DCS as the range closes. This fallback
    is a PyRITE convention, not part of the manual.
    Validation: pair-production-sampling
    """
    kappa = float(photon_energy_eV) / _ELECTRON_REST_EV
    if kappa <= 2.0:
        raise ValueError("pair production requires a photon energy above 2 m_e c^2")
    lo, half_width = 1.0 / kappa, 0.5 - 1.0 / kappa
    phi1_half, phi2_half = _phi(0.5, kappa, Z)
    u1 = 2.0 / 3.0 * half_width**2 * float(phi1_half)
    u2 = float(phi2_half)
    if u1 + u2 <= 0.0:
        return lo + 2.0 * half_width * float(rng.random())
    p1 = u1 / (u1 + u2)
    for _ in range(_MAX_REJECTIONS):
        branch_one = rng.random() < p1
        xi = 2.0 * float(rng.random()) - 1.0
        if branch_one:
            eps = 0.5 + half_width * np.cbrt(xi)
            accept = float(_phi(eps, kappa, Z)[0]) / float(phi1_half)
        else:
            eps = 0.5 + half_width * xi
            accept = float(_phi(eps, kappa, Z)[1]) / float(phi2_half)
        if rng.random() <= accept:
            return float(eps)
    raise RuntimeError("pair reduced-energy rejection did not converge")


def sample_pair_polar_cosine(kinetic_eV, uniform):
    """``cos(theta)`` from ``p ~ (1 - beta cos)^-2``, PENELOPE-2024 Eq. 2.99.

    Validation: pair-production-sampling
    """
    kinetic = np.asarray(kinetic_eV, dtype=float)
    beta = np.sqrt(kinetic * (kinetic + 2.0 * _ELECTRON_REST_EV)) / (kinetic + _ELECTRON_REST_EV)
    x = 2.0 * np.asarray(uniform, dtype=float) - 1.0
    return (x + beta) / (x * beta + 1.0)


def _rotate(direction, cos_theta: float, phi: float) -> np.ndarray:
    """Unit vector at polar ``theta``/azimuth ``phi`` about ``direction``."""
    direction = direction / np.linalg.norm(direction)
    reference = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    transverse = np.cross(direction, reference)
    transverse /= np.linalg.norm(transverse)
    other = np.cross(direction, transverse)
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    out = cos_theta * direction + sin_theta * (np.cos(phi) * transverse + np.sin(phi) * other)
    return out / np.linalg.norm(out)


@dataclass(frozen=True, slots=True)
class PairProducts:
    """Kinetic energies [eV] and unit directions of one pair's e- and e+."""

    electron_eV: float
    positron_eV: float
    electron_direction: np.ndarray
    positron_direction: np.ndarray


def sample_pair(
    photon_energy_eV: float, Z: int, photon_direction, rng: np.random.Generator
) -> PairProducts:
    """Sample one pair: ``E_- = eps E - m c^2``, ``E_+ = E - E_- - 2 m c^2``.

    Energy is conserved exactly; directions are sampled independently about the
    photon direction (PENELOPE-2024 section 2.4.1).
    Validation: pair-production-sampling
    """
    energy = float(photon_energy_eV)
    eps = sample_pair_reduced_energy(energy, Z, rng)
    electron = max(eps * energy - _ELECTRON_REST_EV, 0.0)
    positron = energy - PAIR_THRESHOLD_EV - electron
    direction = np.asarray(photon_direction, dtype=float)
    cos_minus = float(sample_pair_polar_cosine(electron, rng.random()))
    phi_minus = 2.0 * np.pi * float(rng.random())
    cos_plus = float(sample_pair_polar_cosine(positron, rng.random()))
    phi_plus = 2.0 * np.pi * float(rng.random())
    return PairProducts(
        electron,
        positron,
        _rotate(direction, cos_minus, phi_minus),
        _rotate(direction, cos_plus, phi_plus),
    )


def _mix(x):
    x = (x ^ (x >> _SM64_S30)) * _SM64_MIX1
    x = (x ^ (x >> _SM64_S27)) * _SM64_MIX2
    return x ^ (x >> _SM64_S31)


def pair_photon_stream_key(seed, parent_track, photon_ordinal):
    """Per-photon stream keys from (seed, parent track, photon ordinal).

    ``parent_track`` is the cascade's global track id, so keys of photons from
    generation-1 and later parents depend on ``Ne`` and on earlier launches,
    as secondary keys do; never on batch size or processing order.
    Validation: photon-pair-first-interaction
    """
    with np.errstate(over="ignore"):
        x = np.uint64(seed) ^ _PAIR_PHOTON_SALT
        parent = np.asarray(parent_track, dtype=np.uint64)
        ordinal = np.asarray(photon_ordinal, dtype=np.uint64)
        x = _mix(x + _SM64_GOLDEN * (parent + _SM64_ONE))
        return _mix(x + _SM64_GOLDEN * (ordinal + _SM64_ONE))


def _photon_rng(key) -> np.random.Generator:
    return np.random.Generator(np.random.Philox(key=int(key)))


def _layer_intervals(origins, directions, layers, exit_distance):
    """Forward ray-parameter interval ``[start, end]`` inside each layer, ``(N, K)``."""
    z = origins[:, 2][:, None]
    n_z = directions[:, 2][:, None]
    top = np.array([float(layer[0]) for layer in layers])[None, :]
    bottom = np.array([float(layer[1]) for layer in layers])[None, :]
    exit_distance = exit_distance[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_top = (top - z) / n_z
        t_bottom = (bottom - z) / n_z
    start = np.maximum(np.minimum(t_top, t_bottom), 0.0)
    end = np.minimum(np.maximum(t_top, t_bottom), exit_distance)
    lateral = n_z == 0.0
    in_layer = (z >= top) & (z < bottom)
    start = np.where(lateral, 0.0, start)
    end = np.where(lateral, np.where(in_layer, exit_distance, 0.0), end)
    return start, np.maximum(end, start)


def _channel_coefficients(composition, energies_eV):
    """Per-element macroscopic ``(mu_pair, mu_other)`` [1/Ang], shape ``(n_el, N)``."""
    pair, other = [], []
    for element, density in composition:
        sigma = photon_cross_sections_ang2(element, energies_eV)
        sigma_pair = sigma["pair_nuclear"] + sigma["pair_electron"]
        sigma_all = sum(sigma.values())
        pair.append(float(density) * sigma_pair)
        other.append(float(density) * (sigma_all - sigma_pair))
    return np.asarray(pair), np.asarray(other)


def photon_first_interactions(
    origins,
    directions,
    energies_eV,
    uniforms,
    layers: Sequence,
    *,
    width_ang=None,
    height_ang=None,
):
    """Sample each photon's first interaction in the planar stack.

    ``uniforms`` is ``(N, 2)``: the optical-depth draw ``tau = -ln(1 - u0)`` and
    the channel/element draw ``u1``. Returns ``distance_ang`` (``inf`` on
    escape), ``layer`` (-1 on escape), ``pair`` (bool) and ``element`` (index
    into the interaction layer's composition, -1 unless ``pair``).
    Validation: photon-pair-first-interaction
    """
    from ...materials.attenuation import _normalize_composition

    origins = np.asarray(origins, dtype=float).reshape(-1, 3)
    directions = np.asarray(directions, dtype=float).reshape(-1, 3)
    energies = np.asarray(energies_eV, dtype=float).reshape(-1)
    uniforms = np.asarray(uniforms, dtype=float).reshape(-1, 2)
    n = energies.size
    z_total = float(layers[-1][1])
    origins = origins.copy()
    origins[:, 2] = np.clip(origins[:, 2], 0.0, np.nextafter(z_total, 0.0))
    exit_distance, _ = first_prism_exit(
        origins,
        directions,
        z_min_ang=0.0,
        z_max_ang=z_total,
        width_ang=width_ang,
        height_ang=height_ang,
    )
    exit_distance = np.maximum(np.asarray(exit_distance, dtype=float), 0.0)
    start, end = _layer_intervals(origins, directions, layers, exit_distance)
    compositions = [_normalize_composition(None, None, layer[2]) for layer in layers]
    coefficients = [_channel_coefficients(comp, energies) for comp in compositions]
    mu = np.stack([pair.sum(axis=0) + other.sum(axis=0) for pair, other in coefficients], axis=1)
    if not np.all(np.isfinite(mu)):
        # EPDL is NaN outside 1 eV--100 GeV; refuse rather than read it as escape.
        raise ValueError("photon energy is outside the EPDL2025 attenuation domain")
    length = end - start
    with np.errstate(invalid="ignore"):
        tau_layer = np.where(length > 0.0, mu * length, 0.0)
    order = np.argsort(start, axis=1, kind="stable")
    rows = np.arange(n)[:, None]
    tau_sorted = tau_layer[rows, order]
    cumulative = np.cumsum(tau_sorted, axis=1)
    tau = -np.log1p(-uniforms[:, 0])
    # A layer the ray never enters adds no depth and can never hold the point.
    reached = (cumulative >= tau[:, None]) & (tau_sorted > 0.0)
    interacts = reached.any(axis=1)
    position = np.argmax(reached, axis=1)
    layer = np.where(interacts, order[np.arange(n), position], -1)
    before = np.where(position > 0, cumulative[np.arange(n), position - 1], 0.0)
    before = np.where(interacts, before, 0.0)
    safe_layer = np.maximum(layer, 0)
    mu_hit = mu[np.arange(n), safe_layer]
    distance = np.full(n, np.inf)
    distance[interacts] = (
        start[np.arange(n), safe_layer][interacts]
        + (tau[interacts] - before[interacts]) / mu_hit[interacts]
    )
    pair = np.zeros(n, dtype=bool)
    element = np.full(n, -1, dtype=np.int64)
    for index in np.flatnonzero(interacts):
        pair_mu, _ = coefficients[layer[index]]
        threshold = float(uniforms[index, 1]) * float(mu_hit[index])
        cumulative_pair = np.cumsum(pair_mu[:, index])
        if threshold < cumulative_pair[-1]:
            pair[index] = True
            element[index] = int(np.searchsorted(cumulative_pair, threshold, side="right"))
    return {"distance_ang": distance, "layer": layer, "pair": pair, "element": element}


_EVENT_FIELDS = (
    "row",
    "parent",
    "photon_ordinal",
    "k_eV",
    "Z",
    "layer",
    "r_ang",
    "t_ang",
    "electron_keV",
    "positron_keV",
    "electron_v_hat",
    "positron_v_hat",
)


def _empty_events():
    return {
        "row": np.empty(0, dtype=np.int64),
        "parent": np.empty(0, dtype=np.int64),
        "photon_ordinal": np.empty(0, dtype=np.int64),
        "k_eV": np.empty(0),
        "Z": np.empty(0, dtype=np.int64),
        "layer": np.empty(0, dtype=np.int64),
        "r_ang": np.empty((0, 3)),
        "t_ang": np.empty(0),
        "electron_keV": np.empty(0),
        "positron_keV": np.empty(0),
        "electron_v_hat": np.empty((0, 3)),
        "positron_v_hat": np.empty((0, 3)),
    }


def convert_hard_photons(
    rows, *, seed, parent_track_offset, layers, width_ang=None, height_ang=None
):
    """Pair events of one generation's coupled hard photons, host arrays.

    ``rows`` maps the generation's per-row fields (host NumPy): ``electron_id``
    (local track), ``flight_id``, ``substep_id``, ``r_mid``, ``v_hat``,
    ``L_ang``, ``t_end_ang`` and the ``hard_radiative_*`` photon fields. A
    photon leaves the row endpoint ``r_mid + L v_hat / 2`` at clock
    ``t_end_ang``; its ordinal counts the parent's photon rows in flight order.
    Returns the pair events (local ``row``/``parent``) and the per-outcome
    photon counts over photons above threshold.
    Validation: photon-pair-first-interaction, pair-production-sampling
    """
    from ...materials.atomic import Z_TABLE
    from ...materials.attenuation import _normalize_composition

    k = np.asarray(rows["hard_radiative_k_eV"], dtype=float)
    photon_rows = np.flatnonzero(k > 0.0)
    parent = np.asarray(rows["electron_id"], dtype=np.int64)[photon_rows]
    order = np.lexsort(
        (
            np.asarray(rows["substep_id"])[photon_rows],
            np.asarray(rows["flight_id"])[photon_rows],
            parent,
        )
    )
    photon_rows, parent = photon_rows[order], parent[order]
    first = np.ones(photon_rows.size, dtype=bool)
    first[1:] = parent[1:] != parent[:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(photon_rows.size), 0))
    ordinal = np.arange(photon_rows.size) - start
    above = k[photon_rows] > PAIR_THRESHOLD_EV
    photon_rows, parent, ordinal = photon_rows[above], parent[above], ordinal[above]
    counts = {"photons": int(photon_rows.size), "pair": 0, "other": 0, "escaped": 0}
    if photon_rows.size == 0:
        return _empty_events(), counts

    keys = pair_photon_stream_key(seed, parent_track_offset + parent, ordinal)
    rngs = [_photon_rng(key) for key in keys]
    uniforms = np.array([rng.random(2) for rng in rngs])
    r_mid = np.asarray(rows["r_mid"], dtype=float)[photon_rows]
    v_hat = np.asarray(rows["v_hat"], dtype=float)[photon_rows]
    L = np.asarray(rows["L_ang"], dtype=float)[photon_rows]
    origins = r_mid + 0.5 * L[:, None] * v_hat
    directions = np.asarray(rows["hard_radiative_direction"], dtype=float)[photon_rows]
    energies = k[photon_rows]
    hit = photon_first_interactions(
        origins, directions, energies, uniforms, layers, width_ang=width_ang, height_ang=height_ang
    )
    interacts = np.isfinite(hit["distance_ang"])
    counts["pair"] = int(np.count_nonzero(hit["pair"]))
    counts["other"] = int(np.count_nonzero(interacts & ~hit["pair"]))
    counts["escaped"] = int(np.count_nonzero(~interacts))

    events = {name: [] for name in _EVENT_FIELDS}
    t_end = np.asarray(rows["t_end_ang"], dtype=float)[photon_rows]
    for index in np.flatnonzero(hit["pair"]):
        layer = int(hit["layer"][index])
        composition = _normalize_composition(None, None, layers[layer][2])
        Z = int(Z_TABLE[composition[int(hit["element"][index])][0]])
        distance = float(hit["distance_ang"][index])
        products = sample_pair(energies[index], Z, directions[index], rngs[index])
        events["row"].append(photon_rows[index])
        events["parent"].append(parent[index])
        events["photon_ordinal"].append(ordinal[index])
        events["k_eV"].append(energies[index])
        events["Z"].append(Z)
        events["layer"].append(layer)
        events["r_ang"].append(origins[index] + distance * directions[index])
        events["t_ang"].append(t_end[index] + distance)
        events["electron_keV"].append(products.electron_eV * 1e-3)
        events["positron_keV"].append(products.positron_eV * 1e-3)
        events["electron_v_hat"].append(products.electron_direction)
        events["positron_v_hat"].append(products.positron_direction)
    if not events["row"]:
        return _empty_events(), counts
    empty = _empty_events()
    return {
        name: np.asarray(values, dtype=empty[name].dtype).reshape((-1, *empty[name].shape[1:]))
        for name, values in events.items()
    }, counts

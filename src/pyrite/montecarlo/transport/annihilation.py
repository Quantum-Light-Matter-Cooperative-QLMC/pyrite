"""Two-photon annihilation of transported pair positrons (#295).

With ``positron_transport`` every pair positron annihilates with a free
electron at rest into two photons (PENELOPE-2024, NEA/MBDAV/R(2024)1, section
3.4). In flight, the per-electron Heitler cross section (Eq. 3.189)

    sigma_an = pi r_e^2 / ((g + 1)(g^2 - 1))
               * [(g^2 + 4g + 1) ln(g + sqrt(g^2 - 1)) - (3 + g) sqrt(g^2 - 1)]

with ``g = 1 + E/(m_e c^2)`` gives the hazard ``N Z sigma_an(E)`` per unit path
(Eq. 3.190). Annihilation ends the track, so competing it with the other
channels is the same as truncating the unannihilated track where the
accumulated optical depth reaches an exponential budget ``tau = -ln(1 - xi)``
drawn once per positron. The transport runs the positron as before, and
:func:`truncate_in_flight` cuts its rows at that point; the energy is linear in
path length along each row. The lower photon energy fraction ``zeta`` follows
Eq. 3.191 on ``[zeta_min, 1/2]`` and the polar cosines follow Eqs.
3.184--3.186, with azimuths ``phi`` and ``phi + pi``. The rejection envelope is
``max g = g^2 + 2g - 1`` at ``upsilon = 1/(g + 1)``; PENELOPE's stated
``g(zeta_min)`` lies below that maximum for ``g > 1``.

A positron that reaches the tracking cutoff, or is born at or below the
secondary threshold, deposits its kinetic energy locally and annihilates at
rest into two 511 keV photons, back to back along an isotropic axis. No
positronium, three-photon decay or Doppler broadening is modelled.

Annihilation photons get the first-interaction step of #275
(:func:`~.pair_production.photon_first_interactions`): each one either escapes
the stack or interacts at a point sampled from EPDL2025 ``mu_total``, where its
history ends. They never re-enter pair conversion.

RNG. A positron is identified by its pair event, so its streams hash (seed,
parent track, photon ordinal) like the positron's launch key, under one salt
for the in-flight budget and another for emission and photon scoring.

Validation: heitler-annihilation, positron-annihilation-at-rest
"""

from collections.abc import Sequence

import numpy as np

from .kinematics import (
    _SM64_GOLDEN,
    _SM64_MIX1,
    _SM64_MIX2,
    _SM64_ONE,
    _SM64_S27,
    _SM64_S30,
    _SM64_S31,
)
from .pair_production import _rotate

__all__ = [
    "ANNIHILATION_AT_REST",
    "ANNIHILATION_IN_FLIGHT",
    "ELECTRON_REST_KEV",
    "annihilation_stream_keys",
    "electron_density_per_ang3",
    "heitler_cross_section_ang2",
    "heitler_zeta_density",
    "row_optical_depth",
    "sample_at_rest_directions",
    "sample_heitler",
    "sample_heitler_zeta",
    "score_annihilation_photons",
    "truncate_in_flight",
]

ELECTRON_REST_KEV = 510.99895
_CLASSICAL_RADIUS_ANG = 2.8179403262e-5
_PI_RE2_ANG2 = np.pi * _CLASSICAL_RADIUS_ANG**2
#: ``kind`` of an annihilation event.
ANNIHILATION_AT_REST, ANNIHILATION_IN_FLIGHT = 0, 1
_IN_FLIGHT_SALT = np.uint64(0x5851F42D4C957F2D)
_EMISSION_SALT = np.uint64(0xA0761D6478BD642F)
_MAX_REJECTIONS = 100_000
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_GL_X = 0.5 * (_GL_X + 1.0)
_GL_W = 0.5 * _GL_W


def _mix(x):
    x = (x ^ (x >> _SM64_S30)) * _SM64_MIX1
    x = (x ^ (x >> _SM64_S27)) * _SM64_MIX2
    return x ^ (x >> _SM64_S31)


def annihilation_stream_keys(seed, parent_track, photon_ordinal, *, in_flight):
    """Per-positron stream keys from its pair event (seed, parent track, photon ordinal).

    ``in_flight`` selects the optical-depth budget stream; otherwise the
    emission and photon-scoring stream.
    Validation: heitler-annihilation, positron-annihilation-at-rest
    """
    salt = _IN_FLIGHT_SALT if in_flight else _EMISSION_SALT
    with np.errstate(over="ignore"):
        x = np.uint64(seed) ^ salt
        parent = np.asarray(parent_track, dtype=np.uint64)
        ordinal = np.asarray(photon_ordinal, dtype=np.uint64)
        x = _mix(x + _SM64_GOLDEN * (parent + _SM64_ONE))
        return _mix(x + _SM64_GOLDEN * (ordinal + _SM64_ONE))


def _rng(key) -> np.random.Generator:
    return np.random.Generator(np.random.Philox(key=int(key)))


def _gamma(kinetic_keV):
    return 1.0 + np.asarray(kinetic_keV, dtype=float) / ELECTRON_REST_KEV


def heitler_cross_section_ang2(kinetic_keV):
    """Two-photon annihilation cross section per target electron [Angstrom^2].

    PENELOPE-2024 Eq. 3.189; diverges as ``1/beta`` at rest.
    Validation: heitler-annihilation
    """
    g = _gamma(kinetic_keV)
    s2 = g * g - 1.0
    s = np.sqrt(s2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            _PI_RE2_ANG2
            / ((g + 1.0) * s2)
            * ((g * g + 4.0 * g + 1.0) * np.log(g + s) - (3.0 + g) * s)
        )


def _zeta_min(g):
    return 1.0 / (g + 1.0 + np.sqrt(g * g - 1.0))


def _S(zeta, g):
    return -((g + 1.0) ** 2) + (g * g + 4.0 * g + 1.0) / zeta - 1.0 / (zeta * zeta)


def heitler_zeta_density(zeta, kinetic_keV):
    """Normalised density of ``zeta = E_-/(E + 2 m c^2)`` on ``[zeta_min, 1/2]``.

    PENELOPE-2024 Eqs. 3.187--3.188 divided by Eq. 3.189; zero outside.
    Validation: heitler-annihilation
    """
    g = float(_gamma(kinetic_keV))
    zeta = np.asarray(zeta, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dcs = _PI_RE2_ANG2 / ((g + 1.0) * (g * g - 1.0)) * (_S(zeta, g) + _S(1.0 - zeta, g))
    inside = (zeta >= _zeta_min(g)) & (zeta <= 0.5)
    return np.where(inside, dcs / heitler_cross_section_ang2(kinetic_keV), 0.0)


def sample_heitler_zeta(kinetic_keV: float, rng: np.random.Generator) -> float:
    """Exact sample of ``zeta`` (PENELOPE-2024 Eqs. 3.192--3.197).

    ``upsilon`` from ``1/upsilon`` on ``[zeta_min, 1 - zeta_min]``, accepted with
    ``g(upsilon)/max g`` where ``max g = g^2 + 2g - 1`` at ``upsilon = 1/(g + 1)``.
    Validation: heitler-annihilation
    """
    g = float(_gamma(kinetic_keV))
    zeta_min = float(_zeta_min(g))
    ratio = (1.0 - zeta_min) / zeta_min
    a = g * g + 4.0 * g + 1.0
    g_max = g * g + 2.0 * g - 1.0
    for _ in range(_MAX_REJECTIONS):
        upsilon = zeta_min * ratio ** float(rng.random())
        if float(rng.random()) * g_max <= a - (g + 1.0) ** 2 * upsilon - 1.0 / upsilon:
            return min(upsilon, 1.0 - upsilon)
    raise RuntimeError("Heitler annihilation sampling did not converge")


def sample_heitler(kinetic_keV: float, direction, rng: np.random.Generator):
    """Photon energies [keV] and unit directions ``(2, 3)`` of one in-flight annihilation.

    ``E_- = zeta (E + 2 m c^2)``, ``E_+ = E + 2 m c^2 - E_-``; polar cosines
    about the positron direction from Eqs. 3.184--3.185, azimuths ``phi`` and
    ``phi + pi``. Energy is conserved exactly and momentum to rounding.
    Validation: heitler-annihilation
    """
    kinetic = float(kinetic_keV)
    g = 1.0 + kinetic / ELECTRON_REST_KEV
    s = np.sqrt(g * g - 1.0)
    zeta = sample_heitler_zeta(kinetic, rng)
    phi = 2.0 * np.pi * float(rng.random())
    total = kinetic + 2.0 * ELECTRON_REST_KEV
    low = zeta * total
    energies = np.array([low, total - low])
    cos_minus = min(1.0, max(-1.0, (g + 1.0 - 1.0 / zeta) / s))
    cos_plus = min(1.0, max(-1.0, (g + 1.0 - 1.0 / (1.0 - zeta)) / s))
    direction = np.asarray(direction, dtype=float)
    return energies, np.stack(
        [_rotate(direction, cos_minus, phi), _rotate(direction, cos_plus, phi + np.pi)]
    )


def sample_at_rest_directions(rng: np.random.Generator) -> np.ndarray:
    """Back-to-back unit directions ``(2, 3)`` along an isotropic axis.

    Validation: positron-annihilation-at-rest
    """
    cos_theta = 2.0 * float(rng.random()) - 1.0
    phi = 2.0 * np.pi * float(rng.random())
    sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
    axis = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])
    return np.stack([axis, -axis])


def electron_density_per_ang3(layers: Sequence) -> np.ndarray:
    """``N Z`` [1/Angstrom^3] of each layer of a planar stack.

    Validation: heitler-annihilation
    """
    from ...materials.atomic import Z_TABLE
    from ...materials.attenuation import _normalize_composition

    return np.array(
        [
            sum(
                float(n) * int(Z_TABLE[element])
                for element, n in _normalize_composition(None, None, layer[2])
            )
            for layer in layers
        ]
    )


def row_optical_depth(E_start_keV, E_end_keV, L_ang, electron_density, fraction=1.0):
    """``N Z int sigma_an ds`` over the first ``fraction`` of each row.

    Energy is linear in path length along the row; 8-point Gauss--Legendre,
    exact to rounding for transport rows (``E_start/E_end`` near 1) and to
    ``1e-13`` up to ``E_start/E_end = 2``.
    Validation: heitler-annihilation
    """
    E0 = np.asarray(E_start_keV, dtype=float)[:, None]
    E1 = np.asarray(E_end_keV, dtype=float)[:, None]
    f = np.asarray(fraction, dtype=float)
    f = np.broadcast_to(f, E0.shape[:1])[:, None]
    energies = E0 + (E1 - E0) * f * _GL_X[None, :]
    mean = heitler_cross_section_ang2(energies) @ _GL_W
    return (
        np.asarray(electron_density, dtype=float) * np.asarray(L_ang, dtype=float) * f[:, 0] * mean
    )


_ZEROED = (
    "hard_W_keV",
    "hard_secondary_v_hat",
    "hard_radiative_k_eV",
    "hard_radiative_Z",
    "hard_radiative_direction",
    "hard_radiative_target_momentum_eV_c",
)


def truncate_in_flight(rows, tau, electron_density, row_keys):
    """Cut each positron track where its optical depth reaches ``tau``.

    ``rows`` holds one positron transport's host row arrays with local
    ``electron_id`` in ``[0, len(tau))``; ``electron_density`` is per layer.
    The crossing row ends at fraction ``f`` with ``event_kind`` ANNIHILATION,
    the energy and clock interpolated linearly and every hard payload cleared;
    later rows are dropped. Returns the truncated rows (``row_keys`` only) and,
    per track, the crossing row in the input (-1 if none) and the annihilation
    kinetic energy, position, clock, direction and layer.
    Validation: heitler-annihilation
    """
    from .events import EVENT_ANNIHILATION

    n_tracks = np.asarray(tau).size
    track = np.asarray(rows["electron_id"], dtype=np.int64)
    order = np.lexsort((rows["substep_id"], rows["flight_id"], track))
    layer = np.asarray(rows["layer"], dtype=np.int64)
    E0 = np.asarray(rows["E_start_keV"], dtype=float)
    E1 = np.asarray(rows["E_end_keV"], dtype=float)
    L = np.asarray(rows["L_ang"], dtype=float)
    density = np.asarray(electron_density, dtype=float)[layer]
    depth = row_optical_depth(E0, E1, L, density)[order]
    sorted_track = track[order]
    cumulative = np.cumsum(depth)
    first = np.ones(order.size, dtype=bool)
    first[1:] = sorted_track[1:] != sorted_track[:-1]
    base = np.maximum.accumulate(np.where(first, np.arange(order.size), 0))
    local = cumulative - (cumulative - depth)[base]
    budget = np.asarray(tau, dtype=float)[sorted_track]
    # Monotone within a track: once crossed, every later row is crossed too.
    crossed = local >= budget
    hit_sorted = crossed & (first | ~np.r_[False, crossed[:-1]])
    hit_rows = order[hit_sorted]
    hit_track = track[hit_rows]
    remaining = budget[hit_sorted] - (local - depth)[hit_sorted]

    # Bisection on the in-row fraction; the depth is monotone in f.
    lo = np.zeros(hit_rows.size)
    hi = np.ones(hit_rows.size)
    args = (E0[hit_rows], E1[hit_rows], L[hit_rows], density[hit_rows])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        below = row_optical_depth(*args, fraction=mid) < remaining
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    f = 0.5 * (lo + hi)

    v = np.asarray(rows["v_hat"], dtype=float)[hit_rows]
    start = np.asarray(rows["r_mid"], dtype=float)[hit_rows] - 0.5 * L[hit_rows, None] * v
    t0 = np.asarray(rows["t_start_ang"], dtype=float)[hit_rows]
    t1 = np.asarray(rows["t_end_ang"], dtype=float)[hit_rows]
    E_ann = E0[hit_rows] + (E1[hit_rows] - E0[hit_rows]) * f
    r_ann = start + (f * L[hit_rows])[:, None] * v
    t_ann = t0 + f * (t1 - t0)

    keep = np.ones(track.size, dtype=bool)
    keep[order[crossed & ~hit_sorted]] = False

    out = {}
    for key in row_keys:
        if rows.get(key) is None:
            continue
        values = np.array(rows[key], copy=True)
        if key == "L_ang":
            values[hit_rows] = f * L[hit_rows]
        elif key == "r_mid":
            values[hit_rows] = start + (0.5 * f * L[hit_rows])[:, None] * v
        elif key == "E_end_keV":
            values[hit_rows] = E_ann
        elif key == "t_end_ang":
            values[hit_rows] = t_ann
        elif key == "E_repr_keV":
            values[hit_rows] = 0.5 * (E0[hit_rows] + E_ann)
        elif key == "event_kind":
            values[hit_rows] = EVENT_ANNIHILATION
        elif key == "hard_channel":
            values[hit_rows] = -1
        elif key in _ZEROED:
            values[hit_rows] = 0
        out[key] = values[keep]

    def per_track(values, fill):
        full = np.full((n_tracks, *np.shape(values)[1:]), fill, dtype=np.asarray(values).dtype)
        full[hit_track] = values
        return full

    return out, {
        "row": per_track(hit_rows.astype(np.int64), -1),
        "E_keV": per_track(E_ann, np.nan),
        "r_ang": per_track(r_ann, np.nan),
        "t_ang": per_track(t_ann, np.nan),
        "v_hat": per_track(v, np.nan),
        "layer": per_track(layer[hit_rows], -1),
    }


def score_annihilation_photons(
    kinds,
    kinetic_keV,
    directions,
    origins,
    keys,
    layers: Sequence,
    *,
    width_ang=None,
    height_ang=None,
):
    """Emit and score each annihilation's two photons.

    Event ``i`` of ``kinds`` (``ANNIHILATION_AT_REST`` or
    ``ANNIHILATION_IN_FLIGHT``) at ``origins[i]`` uses the emission stream
    ``keys[i]``: in flight the Heitler sample about ``directions[i]`` with
    ``kinetic_keV[i]``, at rest two 511 keV photons back to back. Each photon
    then draws its first interaction from EPDL2025 ``mu_total``. Returns
    per-photon arrays, two rows per event in event order: ``event``, ``k_keV``,
    ``direction``, ``origin``, ``distance_ang`` (``inf`` on escape), ``layer``
    (-1 on escape) and ``absorbed``.
    Validation: heitler-annihilation, positron-annihilation-at-rest
    """
    from .pair_production import photon_first_interactions

    kinds = np.asarray(kinds, dtype=np.int8)
    n = kinds.size
    energies = np.empty((n, 2))
    photon_dirs = np.empty((n, 2, 3))
    uniforms = np.empty((n, 2, 2))
    for i in range(n):
        rng = _rng(keys[i])
        if kinds[i] == ANNIHILATION_IN_FLIGHT:
            energies[i], photon_dirs[i] = sample_heitler(kinetic_keV[i], directions[i], rng)
        else:
            energies[i] = ELECTRON_REST_KEV
            photon_dirs[i] = sample_at_rest_directions(rng)
        uniforms[i] = rng.random((2, 2))
    origin = np.repeat(np.asarray(origins, dtype=float).reshape(-1, 3), 2, axis=0)
    k = energies.reshape(-1)
    direction = photon_dirs.reshape(-1, 3)
    out = {
        "event": np.repeat(np.arange(n, dtype=np.int64), 2),
        "k_keV": k,
        "direction": direction,
        "origin": origin,
    }
    if n == 0:
        out.update(
            distance_ang=np.empty(0),
            layer=np.empty(0, dtype=np.int64),
            absorbed=np.empty(0, dtype=bool),
        )
        return out
    hit = photon_first_interactions(
        origin,
        direction,
        k * 1e3,
        uniforms.reshape(-1, 2),
        layers,
        width_ang=width_ang,
        height_ang=height_ang,
    )
    out["distance_ang"] = hit["distance_ang"]
    out["layer"] = hit["layer"]
    out["absorbed"] = np.isfinite(hit["distance_ang"])
    return out

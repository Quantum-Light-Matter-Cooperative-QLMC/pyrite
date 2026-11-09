"""Validity bounds of fixed-trajectory optical TR/CTR scoring (#615).

The crossing amplitude treats each crossing as a locally straight, uniform
passage, and transport does not feed radiated energy back into the
trajectory. Two bounds make that explicit for a supported scene and refuse
use outside it:

- **formation** — the own field of the medium on either side of a crossing
  builds up over ``L = max(1/|q - lambda|, 1/|q + lambda|)`` (``hbar = c = 1``;
  ``q = (omega - k_par . v_par)/v_z``, ``lambda = sqrt(eps omega^2 - k_par^2)``,
  ``Im lambda >= 0``). Where the transported straight run on that side is
  shorter than ``L``, that side's own-field term of the amplitude is
  unreliable. With ``A_s`` the part of the crossing amplitude driven by side
  ``s`` (linear split of the jump) and ``d_s`` the indicator of a short run,
  each history gets ``|dS| <= sum_c 2 sum_s d_s |A_s|`` and the self power a
  relative error at most ``sum_e (2 |dS_e||S_e| + |dS_e|^2) / sum_e |S_e|^2``.
  The factor 2 replaces the term by one of equal magnitude; it is a
  sensitivity bound for a deflected formation zone, not a radiation model
  of the deflection (bremsstrahlung).
- **energy budget** — the radiated energy per physical electron, including
  the coherent pair term, integrated over the scored band and both
  hemispheres, must be a small fraction of the kinetic energy; and the pair
  term must be negligible at both band edges, otherwise coherent energy
  just outside the band is unbounded by this calculation. The budget is
  scored in-band; coherent emission outside the band (below a band whose
  lower edge lies above the coherent window ``gaussian_coherent_band_eV``,
  where ``|F|^2`` rises again toward one, and above it) is added from an
  analytic envelope, ``out_of_band_coherent_energy_eV``, for Gaussian laws.
- **out-of-band envelope** — outside the band the per-electron pair yield
  ``|<S>|^2 <= <|S|^2>`` (Cauchy–Schwarz). Each history's field obeys
  ``|S|^2 <= n_h sum_c |A_c|^2`` (``n_h`` crossings), and each crossing is
  bounded by its perfect-conductor image-charge yield,
  ``dU/(dE dOmega) = alpha/(4 pi^2) |n x (b/(1 - n.b) - b'/(1 - n.b'))|^2``
  (``b'`` mirrored in the face; normal incidence gives Ginzburg–Frank),
  integrated over the vacuum hemisphere (both for an inner face). This energy
  per eV is flat in ``E``, so with ``|F|^2 <= exp[-(E sigma_t/hbar)^2]`` the
  tails are ``(N - 1) B (hbar/sigma_t) (sqrt(pi)/2) [erf(x_lo) + erfc(x_hi)]``,
  ``x = E sigma_t/hbar``, with ``B`` the weighted sample mean of
  ``n_h sum_c U'_c`` (a consistent estimate, not a strict bound). The PEC
  envelope is not an upper bound where a traversed medium can emit Cherenkov
  light into vacuum (``beta^2 Re eps >= 1``: refracted Cherenkov exceeds it by
  up to orders of magnitude) or for grazing crossings on low-``|eps|`` media
  near their plasma edge (up to ~1.6x); metals below their plasma edge stay at
  0.86–0.99 of it. Supplying ``layer_eps`` refuses ``beta^2 Re eps >= 1`` on
  the in-band permittivity, a proxy for the unscored out-of-band values. The
  larger-hemisphere rule for inner faces is a heuristic (no thin-film
  transmission or multilayer enhancement). The fixed angular quadrature is
  refused above ``gamma = 5`` (1 % at 1.2 rad tilt there). Tilted
  (``t``–``x`` correlated), sampled, empirical and realization laws are not
  bounded: the caller reports them as unbounded.

``formation_error_bound`` concerns the self (single-history) power;
``coherent_formation_error_bound`` propagates the same per-history bound
through the pair estimator and the total predicted yield, which is the
quantity :func:`~pyrite.montecarlo.runner.transition_emission.case_optical_transition`
refuses on. The factor 2 assumes
the true term of a short run is no larger than the straight-run term; it is a
heuristic sensitivity measure, not a proven bound.

Validation: optical-planar-tr, optical-ctr-population
"""

from __future__ import annotations

from collections.abc import Callable
from math import erf, erfc

import numpy as np

from ...materials.crystal import ALPHA_FS, HBARC_EV_ANG, M_E_EV
from .transition_radiation import _outgoing_root, interface_transition_radiation_amplitude
from .transition_radiation_crossings import VACUUM, BoundaryCrossings, _emission_frame

__all__ = [
    "coherent_formation_error_bound",
    "formation_error_bound",
    "hemisphere_directions",
    "kinetic_energy_eV",
    "out_of_band_coherent_energy_eV",
    "pec_crossing_energy_per_eV",
    "radiated_energy_fraction",
]


def _slab_layer(solver_index, n_layers, forward):
    """Slab layer of a solver medium (substrate first, vacuum last); -1 is vacuum."""
    layer = solver_index - 1 if forward else n_layers - solver_index
    return np.where((layer < 0) | (layer >= n_layers), VACUUM, layer)


def _side_runs(crossings, side_layer):
    """Straight run on the side of each crossing occupied by ``side_layer``."""
    before = np.asarray(crossings.straight_before_ang, dtype=float)
    after = np.asarray(crossings.straight_after_ang, dtype=float)
    run = np.where(side_layer == crossings.layer_before, before, after)
    return np.where(side_layer == VACUUM, np.inf, run)


def formation_error_bound(
    crossings: BoundaryCrossings, photon_energy_eV, direction, layer_eps
) -> np.ndarray:
    """Relative self-power error bound per energy/direction (module docstring).

    Needs transport-derived crossings (``straight_*_ang``). Host NumPy; a
    validity preflight for bounded scenes, not a production reduction.

    Validation: optical-planar-tr
    """
    shape, total, error = _crossing_parts(crossings, photon_energy_eV, direction, layer_eps)
    return _history_relative_error(crossings.electron_id, total, error).reshape(shape)


def coherent_formation_error_bound(
    crossings: BoundaryCrossings,
    photon_energy_eV,
    direction,
    layer_eps,
    *,
    source,
    n_histories: int,
) -> dict[str, np.ndarray]:
    """Relative formation sensitivity of the self, pair and total predicted yield.

    Propagates the per-history bound ``|dS_h| <= b_h`` of
    :func:`formation_error_bound` through the estimators of
    ``optical_transition_radiation`` with weights ``w_h`` (``W = Sum w``,
    ``W_2 = Sum w^2``, ``D = Sum w b``):
    ``|d Sum w|S|^2| <= Sum w (2 b|S| + b^2)``;
    ``|d(|Sum w S|^2 - Sum w^2|S|^2)| <= 2 D |Sum w S| + D^2 + Sum w^2 (2 b|S| + b^2)``;
    the total ``self + (N - 1)|F|^2 pair`` (a realization: ``|Sum S|^2/M``)
    takes the sum of the absolute bounds. Returns arrays ``self``, ``pair`` and
    ``total`` (``inf`` where the estimate vanishes but its bound does not). The
    same factor-2 heuristic as the self bound: a sensitivity measure, not a
    proven bound.

    Validation: optical-planar-tr, optical-ctr-population
    """
    shape, total, error = _crossing_parts(crossings, photon_energy_eV, direction, layer_eps)
    count = int(n_histories)
    field, bound = _history_sums(crossings.electron_id, total, error, count)
    w = np.ones(count) if source.history_weights is None else source.history_weights
    norm = np.linalg.norm(field, axis=-1)  # (B, H)
    w_sum, w2_sum = float(w.sum()), float(np.sum(w**2))
    power_err = 2.0 * bound * norm + bound**2
    g, dg = norm**2 @ w, power_err @ w
    h, dh = norm**2 @ w**2, power_err @ w**2
    coherent = np.linalg.norm(np.einsum("h,bhk->bk", w, field), axis=-1)
    d_sum = bound @ w
    d_coherent = 2.0 * d_sum * coherent + d_sum**2
    pair_den = w_sum**2 - w2_sum
    pair, d_pair = (coherent**2 - h, d_coherent + dh) if pair_den > 0 else (0.0 * g, 0.0 * g)
    if source.law == "realization":
        value, d_value = coherent**2 / w_sum, d_coherent / w_sum
    else:
        energy = np.asarray(photon_energy_eV, dtype=float)
        n = np.asarray(direction, dtype=float)
        factor = np.broadcast_to(source.form_factor_squared(energy, n), shape).reshape(-1)
        weight = max(source.physical_electrons - 1.0, 0.0) * factor / max(pair_den, 1e-300)
        value = g / w_sum + weight * pair
        d_value = dg / w_sum + weight * d_pair
    return {
        "self": _ratio(dg, g).reshape(shape),
        "pair": _ratio(d_pair, np.abs(pair)).reshape(shape),
        "total": _ratio(d_value, np.abs(value)).reshape(shape),
    }


def _ratio(error, value):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(value > 0, error / value, np.where(error > 0, np.inf, 0.0))


def _crossing_parts(crossings, photon_energy_eV, direction, layer_eps):
    """``(shape, A (B, K, 3), b (B, K))``: crossing amplitudes and formation bounds."""
    if crossings.straight_before_ang is None or crossings.straight_after_ang is None:
        raise ValueError("formation bound needs transport-derived straight runs")
    energy = np.asarray(photon_energy_eV, dtype=float)
    n = np.asarray(direction, dtype=float)
    shape = np.broadcast_shapes(energy.shape, n.shape[:-1])
    frame = _emission_frame(crossings, n, layer_eps)
    forward = bool(np.reshape(n, (-1, 3))[0, 2] > 0.0)
    n_layers = len(frame.layers)
    below = _slab_layer(frame.face, n_layers, forward)
    above = _slab_layer(frame.face + 1, n_layers, forward)
    omega = np.broadcast_to(energy, shape).reshape(-1)
    n_frame = np.broadcast_to(frame.direction, (*shape, 3)).reshape(-1, 3)
    k_par = omega[:, None] * n_frame[:, :2]
    v = frame.velocity
    q = (omega[:, None] - k_par @ v[:, :2].T) / v[:, 2]  # (B, K)
    eps_slab = [np.broadcast_to(np.asarray(e, dtype=complex), shape).reshape(-1) for e in layer_eps]
    args = (
        energy, frame.direction, frame.layers, 1.0, frame.velocity,
        crossings.time_ang, frame.position, frame.face,
    )  # fmt: skip
    scale = crossings.charge.astype(float)[:, None] * frame.flip
    error = np.zeros((omega.size, len(crossings)))
    for column, side in enumerate((below, above)):
        own = np.zeros((len(crossings), 2))
        own[:, column] = 1.0
        part = interface_transition_radiation_amplitude(*args, own_field_sides=own)
        magnitude = np.linalg.norm((part * scale).reshape(omega.size, len(crossings), 3), axis=-1)
        length = _formation_length(side, eps_slab, omega, k_par, q)
        short = _side_runs(crossings, side)[None, :] < length
        error += 2.0 * np.where(short, magnitude, 0.0)
    total = interface_transition_radiation_amplitude(*args) * scale
    return shape, total.reshape(omega.size, len(crossings), 3), error


def _formation_length(side, eps_slab, omega, k_par, q):
    """``max(1/|q - lambda|, 1/|q + lambda|)`` [Å] of each crossing's medium; 0 in vacuum."""
    length = np.zeros(q.shape)
    k2 = np.sum(k_par**2, axis=-1)[:, None]
    for layer in np.unique(side[side != VACUUM]):
        mask = side == layer
        lam = _outgoing_root(eps_slab[layer][:, None] * omega[:, None] ** 2 - k2)
        qm = q[:, mask]
        length[:, mask] = HBARC_EV_ANG / np.minimum(np.abs(qm - lam), np.abs(qm + lam))
    return length


def _history_sums(electron_id, amplitude, error, n_histories):
    """Per-history field sums ``(B, H, 3)`` and summed bounds ``(B, H)``."""
    ids = np.asarray(electron_id, dtype=np.intp)
    summed = np.zeros((amplitude.shape[0], n_histories, 3), dtype=complex)
    bound = np.zeros((amplitude.shape[0], n_histories))
    np.add.at(summed.transpose(1, 0, 2), ids, amplitude.transpose(1, 0, 2))  # views
    np.add.at(bound.T, ids, error.T)
    return summed, bound


def _history_relative_error(electron_id, amplitude, error):
    histories, inverse = np.unique(electron_id, return_inverse=True)
    summed, bound = _history_sums(inverse, amplitude, error, histories.size)
    norm = np.linalg.norm(summed, axis=-1)
    power = np.sum(norm**2, axis=-1)
    excess = np.sum(2.0 * bound * norm + bound**2, axis=-1)
    return _ratio(excess, power)


def hemisphere_directions(n_cos: int = 12, n_phi: int = 12, *, forward: bool):
    """Gauss–Legendre in ``cos theta`` x uniform ``phi`` directions and weights [sr]."""
    nodes, weights = np.polynomial.legendre.leggauss(int(n_cos))
    mu = 0.5 * (nodes + 1.0)
    phi = (np.arange(int(n_phi)) + 0.5) * 2.0 * np.pi / int(n_phi)
    sin = np.sqrt(1.0 - mu**2)
    sign = 1.0 if forward else -1.0
    direction = np.stack(
        [
            np.outer(sin, np.cos(phi)),
            np.outer(sin, np.sin(phi)),
            sign * np.outer(mu, np.ones_like(phi)),
        ],
        axis=-1,
    ).reshape(-1, 3)
    weight = np.outer(0.5 * weights, np.full(phi.size, 2.0 * np.pi / phi.size)).reshape(-1)
    return direction, weight


def radiated_energy_fraction(
    score: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    photon_energy_eV,
    kinetic_energy_eV: float,
    *,
    max_fraction: float = 1e-3,
    edge_tolerance: float = 1e-3,
    n_cos: int = 12,
    n_phi: int = 12,
    out_of_band_eV: float = 0.0,
) -> float:
    """Radiated energy per physical electron over kinetic energy; refuses unbounded use.

    ``score(energy, directions)`` returns ``(yield_per_electron, self_yield,
    pair_weight)`` with shapes ``(E, D)``, ``pair_weight`` being
    ``max(N - 1, 0) F pair_yield``. Integrates ``E * yield`` over both
    hemispheres (Gauss–Legendre/uniform) and the band (trapezoid). Raises
    when the fraction exceeds ``max_fraction`` or the pair-to-self ratio at a
    band edge exceeds ``edge_tolerance``. ``out_of_band_eV`` [eV per physical
    electron] adds energy outside the band (``out_of_band_coherent_energy_eV``).

    Validation: optical-ctr-population
    """
    energy = np.asarray(photon_energy_eV, dtype=float)
    if not np.isfinite(out_of_band_eV) or out_of_band_eV < 0.0:
        raise ValueError("out_of_band_eV must be finite and nonnegative")
    if energy.ndim != 1 or energy.size < 2 or kinetic_energy_eV <= 0:
        raise ValueError("energy budget needs a 1-D band and positive kinetic energy")
    radiated = 0.0
    for forward in (True, False):
        direction, weight = hemisphere_directions(n_cos, n_phi, forward=forward)
        total, self_part, pair_part = score(energy[:, None], direction[None])
        edges = np.abs(pair_part[[0, -1]]) / np.maximum(np.abs(self_part[[0, -1]]), 1e-300)
        if np.any(edges > edge_tolerance):
            raise ValueError(
                "coherent pair emission is not negligible at a band edge "
                f"(pair/self up to {float(np.max(edges)):.3g}); radiated energy outside the "
                "scored band is unbounded: extend the band with supported optics"
            )
        radiated += float(np.trapezoid(energy * (total @ weight), energy))
    fraction = (radiated + float(out_of_band_eV)) / float(kinetic_energy_eV)
    if not np.isfinite(fraction) or fraction > max_fraction:
        raise ValueError(
            f"radiated energy is {fraction:.3g} of the kinetic energy (limit {max_fraction:g}): "
            "the fixed-trajectory approximation is unbounded for this scene"
        )
    return fraction


def kinetic_energy_eV(crossings: BoundaryCrossings) -> float:
    """Smallest entrance kinetic energy [eV] of the primary crossings."""
    entry = (crossings.layer_before == VACUUM) & (crossings.interface == 0)
    if not np.any(entry):
        raise ValueError("no entrance crossings")
    return float(np.min(crossings.kinetic_energy_keV[entry])) * 1e3


#: Largest Lorentz factor the fixed envelope quadrature resolves to ~1 %.
_MAX_ENVELOPE_GAMMA = 5.0


def pec_crossing_energy_per_eV(crossings: BoundaryCrossings, n_cos: int = 24, n_phi: int = 24):
    """Perfect-conductor image-charge TR energy per eV of each crossing [eV/eV].

    ``alpha/(4 pi^2) int |n x (b/(1 - n.b) - b'/(1 - n.b'))|^2 dOmega`` over the
    vacuum hemisphere of the face (entrance face ``z < 0``, last face
    ``z > 0``; the larger of both for an inner face). Independent of ``E``.

    Validation: optical-ctr-population
    """
    gamma = 1.0 + np.asarray(crossings.kinetic_energy_keV, dtype=float) * 1e3 / M_E_EV
    if gamma.size and float(np.max(gamma)) > _MAX_ENVELOPE_GAMMA:
        raise ValueError(
            f"PEC envelope quadrature is unresolved above gamma = {_MAX_ENVELOPE_GAMMA:g}"
        )
    beta = np.sqrt(1.0 - 1.0 / gamma**2)[:, None] * np.asarray(crossings.direction, dtype=float)
    image = beta * np.array([1.0, 1.0, -1.0])
    hemispheres = []
    for forward in (False, True):
        n, weight = hemisphere_directions(n_cos, n_phi, forward=forward)
        terms = [  # (C, D, 3): n x b / (1 - n.b)
            np.cross(n[None], b[:, None]) / (1.0 - np.einsum("dk,ck->cd", n, b))[..., None]
            for b in (beta, image)
        ]
        power = np.sum(np.abs(terms[0] - terms[1]) ** 2, axis=-1)
        hemispheres.append(ALPHA_FS / (4.0 * np.pi**2) * (power @ weight))
    back, front = hemispheres
    last = len(crossings.z_interfaces_ang) - 1
    interface = np.asarray(crossings.interface)
    return np.where(
        interface == 0, back, np.where(interface == last, front, np.maximum(back, front))
    )


def out_of_band_coherent_energy_eV(
    crossings: BoundaryCrossings, band_eV, source, n_histories: int, layer_eps=None
) -> float:
    """Envelope of coherent pair energy outside ``band_eV`` per physical electron [eV].

    Module docstring (out-of-band envelope). ``source`` is an
    ``OpticalBunchSource``; zero for ``N <= 1``. Raises ``ValueError`` for laws
    whose tail this envelope does not cover (only Gaussian laws without
    time–transverse correlation are supported), above ``gamma = 5``, and, with
    ``layer_eps`` (per-layer in-band permittivities), when any layer reaches
    the Cherenkov condition ``beta^2 Re eps >= 1``.

    Validation: optical-ctr-population
    """
    excess = max(source.physical_electrons - 1.0, 0.0)
    if excess == 0.0:
        return 0.0
    if source.law != "gaussian":
        raise ValueError(f"no out-of-band envelope for source law {source.law!r}")
    cov = np.asarray(source.covariance_ang2, dtype=float)
    if np.any(cov[0, 1:] != 0.0):
        raise ValueError("no out-of-band envelope for a time-transverse correlated law")
    if layer_eps is not None and np.size(crossings.kinetic_energy_keV):
        kinetic = float(np.max(crossings.kinetic_energy_keV)) * 1e3
        beta2 = 1.0 - 1.0 / (1.0 + kinetic / M_E_EV) ** 2
        worst = max((float(np.max(np.real(e))) for e in layer_eps), default=0.0)
        if beta2 * worst >= 1.0:
            raise ValueError(
                f"a layer can emit Cherenkov light (beta^2 Re eps = {beta2 * worst:.3g}): "
                "the PEC envelope does not bound it"
            )
    sigma = float(np.sqrt(cov[0, 0]))
    band = np.asarray(band_eV, dtype=float)
    lo, hi = float(np.min(band)), float(np.max(band))
    if sigma == 0.0:
        return np.inf
    count = int(n_histories)
    weights = np.ones(count) if source.history_weights is None else source.history_weights
    per_crossing = pec_crossing_energy_per_eV(crossings)
    ids = np.asarray(crossings.electron_id)
    n_cross = np.bincount(ids, minlength=count)
    summed = np.bincount(ids, weights=per_crossing, minlength=count)
    envelope = float(np.sum(weights * n_cross * summed) / np.sum(weights))
    scale = HBARC_EV_ANG / sigma
    tails = 0.5 * np.sqrt(np.pi) * scale * (erf(lo / scale) + erfc(hi / scale))
    return excess * envelope * float(tails)

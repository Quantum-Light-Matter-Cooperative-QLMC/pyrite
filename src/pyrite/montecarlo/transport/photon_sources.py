"""CPU weighted continuous-photon sources for the opt-in tracker (#341).

This module samples the *unattenuated* phase-space measure along already
transported electron segments.  It never debits the charged history and it is
not an analog hard-photon generator.  The caller must not add these weighted
histories to a histogram of the same physical bremsstrahlung events.

For a segment of length L and emissivity j_i(k, Omega) in photons per Angstrom,
per eV, per sr, the target measure is L*j_i dk dOmega. Segment and energy
proposals are only importance samplers: the direction proposal covers all 4pi,
while j_i is evaluated jointly at the sampled (k, Omega).  The segment's
representative energy is the same midpoint quadrature as mc_brem_spectrum.

No production mode is enabled here. Validation: photon-continuous-source.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

_FOUR_PI = 4.0 * np.pi
# cm^2 -> Angstrom^2: n [atoms / Angstrom^3] * dsigma [cm^2] * 1e16
_CM2_TO_ANG2 = 1.0e16
_MC2_EV = 510.99895e3
_PROPOSALS = ("uniform", "bremsstrahlung")
# Defensive-mixture shares of the bremsstrahlung proposal. The remaining
# uniform-energy and isotropic shares bound each energy and angle importance
# factor by 1/(1 - share) times that of the uniform proposal.
_LOG_ENERGY_SHARE = 0.75
_FORWARD_SHARE = 0.75


@dataclass(frozen=True, slots=True)
class WeightedPhotonSources:
    """One weighted mean-response source batch in the sample frame.

    ``weight`` is the total expected photon count represented by
    each sampled source. ``history_id`` retains the primary electron id, and
    ``event_id`` is a unique sampling ordinal, *not* an analog emission id.
    Sample coordinates and directions are in Angstroms and unit vectors.
    """

    position_ang: np.ndarray
    direction: np.ndarray
    energy_eV: np.ndarray
    weight: np.ndarray
    history_id: np.ndarray
    event_id: np.ndarray
    source_kind: np.ndarray
    segment_index: np.ndarray

    @property
    def track_args(self) -> tuple[np.ndarray, ...]:
        """Positional input to ``track_photon_histories`` or ``track_coupled_histories``.

        Pass ``source_kind=batch.source_kind`` separately. Do not reuse the
        primary electron's physical energy ledger for these weighted samples.
        """
        return (
            self.position_ang,
            self.direction,
            self.energy_eV,
            self.weight,
            self.history_id,
            self.event_id,
        )


def _segments(segments: Mapping, n_primaries: int, cutoff_eV: float, ceiling_eV: float):
    """Validate CPU segment geometry and per-segment energy support."""
    if not isinstance(n_primaries, int) or n_primaries <= 0:
        raise ValueError("n_primaries must be a positive integer")
    if not np.isfinite(cutoff_eV) or cutoff_eV <= 0.0:
        raise ValueError("photon cutoff must be finite and positive")
    if not np.isfinite(ceiling_eV) or ceiling_eV <= cutoff_eV:
        raise ValueError("energy ceiling must exceed the photon cutoff")
    length = np.asarray(segments["L_ang"], dtype=float)
    n = length.size
    mid = np.asarray(segments["r_mid"], dtype=float)
    direction = np.asarray(segments["v_hat"], dtype=float)
    owners = np.asarray(segments["elec_id"])
    energy_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    kinetic_keV = np.asarray(segments[energy_field], dtype=float)
    if length.ndim != 1 or mid.shape != (n, 3) or direction.shape != (n, 3):
        raise ValueError("segment lengths, midpoints and directions have incompatible shapes")
    if owners.shape != (n,) or kinetic_keV.shape != (n,):
        raise ValueError("segment energies and electron ids must have shape (N,)")
    if not np.issubdtype(owners.dtype, np.integer):
        raise ValueError("segment electron ids must be nonnegative integers")
    if not all(np.isfinite(a).all() for a in (length, mid, direction, kinetic_keV)):
        raise ValueError("segment data must be finite")
    if np.any(length < 0) or np.any(kinetic_keV < 0):
        raise ValueError("segment lengths and energies must be nonnegative")
    if not np.allclose(np.linalg.norm(direction, axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError("segment directions must be unit vectors")
    if np.any((owners < 0) | (owners >= n_primaries)):
        raise ValueError("electron ids must be nonnegative and smaller than n_primaries")
    upper = np.minimum(ceiling_eV, kinetic_keV * 1e3)
    mass = length * np.maximum(upper - cutoff_eV, 0.0)
    mass[upper <= cutoff_eV] = 0.0
    return mid, direction, length, owners.astype(np.int64), upper, mass


def sample_continuous_photons(
    segments: Mapping,
    emissivity: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    *,
    n_samples: int,
    n_primaries: int,
    seed: int,
    cutoff_eV: float,
    ceiling_eV: float,
    source_kind: int = 0,
    proposal: str = "uniform",
) -> WeightedPhotonSources:
    """Importance-sample photons from a differential per-length emissivity.

    ``emissivity(index, energy_eV, direction)`` returns one *unattenuated*
    photon yield per Angstrom per eV per sr for each selected original
    segment index. It must evaluate the *joint* angle-energy distribution;
    evaluating a detector-direction spectrum or applying Beer-Lambert again
    is incorrect. Source positions are uniform along the segment.

    Proposal: choose segment with probability proportional to
    ``L_i * (min(E_max, 1000*T_i) - cutoff)``, energy uniform on that segment's
    available interval and solid angle uniform on 4pi. For M trials, weight
    ``4pi * L_i * dE_i * j_i / (M * p_i)``. Divide the tracked
    detector spectrum by ``n_primaries`` *once*, at the existing
    ``detected_spectrum`` scoring boundary. Zero-rate trials
    are omitted, preserving unbiasedness without passing zero weights to the
    tracker. This is an unbiased Monte Carlo integral of the *representative-
    energy, straight-segment source model*, not a replacement for charged
    recoil/energy-loss physics. No coherent PXR/CBS segment amplitudes here.

    ``proposal="bremsstrahlung"`` matches the 1/k, forward-peaked MeV source
    with a defensive mixture, keeping full energy and 4pi support: segment
    probability proportional to ``L_i * ln(upper_i / cutoff)``; energy density
    ``a / (k ln(upper/cutoff)) + (1 - a) / dE_i``; and direction density
    ``(1 - b) / 4pi + b (1 - beta^2) / (4pi (1 - beta mu)^2)`` about the
    segment direction, with ``beta`` from the segment's representative energy
    and ``a = b = 0.75``. Every trial then has weight
    ``L_i * j_i / (M * p_i * q_E * q_Omega)``. With constant ``j_i`` the
    uniform proposal integrates exactly; the mixture is unbiased with bounded
    energy and angle importance factors.

    If no segment contributes or ``n_samples=0`` an empty batch is returned.
    Randomness is local Philox, independent of electron/charged descendants.
    Source: importance-sampling identity, ADR-0020; derivation in
    docs/validation/radiation-physics/photon-continuous-source.md.
    Validation: photon-continuous-source.
    """
    if not isinstance(n_samples, int) or n_samples < 0:
        raise ValueError("n_samples must be a nonnegative integer")
    if not isinstance(source_kind, int) or source_kind < 0:
        raise ValueError("source_kind must be a nonnegative integer")
    if proposal not in _PROPOSALS:
        raise ValueError(f"proposal must be one of {_PROPOSALS}")
    mid, ray, length, owner, upper, mass = _segments(segments, n_primaries, cutoff_eV, ceiling_eV)

    def empty():
        return WeightedPhotonSources(
            np.empty((0, 3)),
            np.empty((0, 3)),
            np.empty(0),
            np.empty(0),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    total = float(np.sum(mass, dtype=np.float64))
    if not np.isfinite(total):
        raise ValueError("segment proposal mass overflows")
    if n_samples == 0 or total == 0:
        return empty()
    rng = np.random.Generator(np.random.Philox(seed))
    if proposal == "bremsstrahlung":
        return _sample_bremsstrahlung_proposal(
            segments,
            emissivity,
            rng,
            mid,
            ray,
            length,
            owner,
            upper,
            mass,
            n_samples=n_samples,
            cutoff_eV=cutoff_eV,
            source_kind=source_kind,
        )
    # Avoid touching rows with zero proposal probability, including at CDF boundaries.
    active = np.flatnonzero(mass > 0.0)
    probabilities = mass[active] / total
    choice = rng.choice(active, size=n_samples, replace=True, p=probabilities)
    span = upper[choice] - cutoff_eV
    energy = cutoff_eV + rng.random(n_samples) * span
    location = mid[choice] + ((rng.random(n_samples) - 0.5) * length[choice])[:, None] * ray[choice]
    cosine = 2.0 * rng.random(n_samples) - 1.0
    azimuth = _FOUR_PI / 2 * rng.random(n_samples)
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    direction = np.column_stack((sine * np.cos(azimuth), sine * np.sin(azimuth), cosine))
    rate = np.asarray(emissivity(choice, energy, direction), dtype=float)
    if rate.shape != (n_samples,) or not np.isfinite(rate).all() or np.any(rate < 0):
        raise ValueError("emissivity must return finite nonnegative rates of shape (n_samples,)")
    # p_i = L_i*dE_i / total: the importance factor cancels exactly, though
    # keeping the explicit density prevents accidentally dropping dOmega.
    weight = rate * (_FOUR_PI * total / n_samples)
    if not np.isfinite(weight).all():
        raise ValueError("sampled source weights overflow")
    return _weighted_batch(location, direction, energy, weight, owner, choice, source_kind)


def _weighted_batch(location, direction, energy, weight, owner, choice, source_kind):
    """Drop zero-rate trials; event ids keep the original trial ordinals."""
    valid = weight > 0.0
    return WeightedPhotonSources(
        location[valid],
        direction[valid],
        energy[valid],
        weight[valid],
        owner[choice[valid]],
        np.flatnonzero(valid).astype(np.int64),
        np.full(int(valid.sum()), source_kind, dtype=np.int64),
        choice[valid],
    )


def _sample_bremsstrahlung_proposal(
    segments,
    emissivity,
    rng,
    mid,
    ray,
    length,
    owner,
    upper,
    mass,
    *,
    n_samples,
    cutoff_eV,
    source_kind,
):
    """Defensive-mixture trials of ``sample_continuous_photons``; see its docstring.

    Source: Owen & Zhou (2000) defensive importance sampling; forward density
    from the inverse CDF of ``(1 - beta^2) / (2 (1 - beta mu)^2)`` on [-1, 1].
    Validation: photon-continuous-source.
    """
    energy_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    kinetic_eV = np.asarray(segments[energy_field], dtype=float) * 1e3
    log_span = np.zeros_like(mass)
    supported = mass > 0.0
    log_span[supported] = np.log(upper[supported] / cutoff_eV)
    selection = length * log_span
    total = float(np.sum(selection, dtype=np.float64))
    if not np.isfinite(total):
        raise ValueError("segment proposal mass overflows")
    active = np.flatnonzero(selection > 0.0)
    choice = rng.choice(active, size=n_samples, replace=True, p=selection[active] / total)
    span = upper[choice] - cutoff_eV
    ratio = log_span[choice]
    from_log = rng.random(n_samples) < _LOG_ENERGY_SHARE
    u = rng.random(n_samples)
    energy = np.where(from_log, cutoff_eV * np.exp(u * ratio), cutoff_eV + u * span)
    energy = np.clip(energy, cutoff_eV, upper[choice])
    energy_density = _LOG_ENERGY_SHARE / (energy * ratio) + (1 - _LOG_ENERGY_SHARE) / span
    location = mid[choice] + ((rng.random(n_samples) - 0.5) * length[choice])[:, None] * ray[choice]

    inv_gamma2 = (1.0 / (1.0 + kinetic_eV[choice] / _MC2_EV)) ** 2
    beta = np.sqrt(1.0 - inv_gamma2)
    forward = rng.random(n_samples) < _FORWARD_SHARE
    u = rng.random(n_samples)
    peaked = (1.0 - 1.0 / (2.0 * beta * u / inv_gamma2 + 1.0 / (1.0 + beta))) / beta
    cosine = np.clip(np.where(forward, peaked, 2.0 * u - 1.0), -1.0, 1.0)
    azimuth = _FOUR_PI / 2 * rng.random(n_samples)
    angle_density = (1 - _FORWARD_SHARE) / _FOUR_PI + _FORWARD_SHARE * inv_gamma2 / (
        _FOUR_PI * (1.0 - beta * cosine) ** 2
    )
    axis = ray[choice]
    helper = np.zeros_like(axis)
    helper[np.abs(axis[:, 0]) < 0.9, 0] = 1.0
    helper[np.abs(axis[:, 0]) >= 0.9, 1] = 1.0
    e1 = helper - np.sum(helper * axis, axis=1)[:, None] * axis
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(axis, e1)
    sine = np.sqrt(np.maximum(0.0, 1.0 - cosine * cosine))
    direction = (
        cosine[:, None] * axis
        + (sine * np.cos(azimuth))[:, None] * e1
        + (sine * np.sin(azimuth))[:, None] * e2
    )
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    rate = np.asarray(emissivity(choice, energy, direction), dtype=float)
    if rate.shape != (n_samples,) or not np.isfinite(rate).all() or np.any(rate < 0):
        raise ValueError("emissivity must return finite nonnegative rates of shape (n_samples,)")
    probability = selection[choice] / total
    weight = rate * length[choice] / (n_samples * probability * energy_density * angle_density)
    if not np.isfinite(weight).all():
        raise ValueError("sampled source weights overflow")
    return _weighted_batch(location, direction, energy, weight, owner, choice, source_kind)


def sample_bremslib_continuum(
    segments: Mapping,
    compositions: Sequence[Sequence[tuple[str, float]]],
    tables: Mapping,
    *,
    n_samples: int,
    n_primaries: int,
    seed: int,
    cutoff_eV: float,
    ceiling_eV: float,
    segment_layer=None,
    source_kind: int = 0,
    proposal: str = "bremsstrahlung",
) -> WeightedPhotonSources:
    """CPU BremsLib DDCS source, with material-specific density per segment.

    ``compositions[layer]`` contains ``(element, atoms/Angstrom^3)`` pairs.
    ``segment_layer`` explicitly maps rows to compositions (defaults to zero
    for one material); it is never inferred from the source position.
    BremsLib tables must cover each material and each *active* electron energy;
    unsupported energies fail closed rather than silently falling back to an
    unrelated angular law. Units follow ``mc_brem_spectrum``: cm^2/eV/sr
    converted to Angstrom^2, times atoms/Angstrom^3.

    Inputs remain explicit host arrays. NumPy staging/evaluation is independent
    of the selected transport backend; no implicit device-to-host copy occurs.
    Source: BremsLib DDCS convention (bremslib-angular-model) and the
    importance-sampling identity in sample_continuous_photons. Assumes a
    fixed representative energy per segment; zero supported path gives no
    photons. No attenuation or second energy debit is applied.
    Validation: photon-continuous-source.
    """
    from ..spectrum.brem_bremslib import (
        bremslib_segment_state,
        evaluate_bremslib_rows,
        stage_bremslib_table,
    )

    if not compositions:
        raise ValueError("at least one material composition is required")
    n = np.asarray(segments["L_ang"]).size
    if segment_layer is None:
        if len(compositions) != 1:
            raise ValueError("segment_layer is required for multiple materials")
        labels = np.zeros(n, dtype=np.int64)
    else:
        labels = np.asarray(segment_layer)
        if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("segment_layer must be an integer array of shape (N,)")
        if np.any(labels < 0) or np.any(labels >= len(compositions)):
            raise ValueError("segment_layer index outside compositions")
    densities = []
    for composition in compositions:
        comp = dict(composition)
        values = np.asarray(list(comp.values()), dtype=float)
        if not comp or not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError("material atom densities must be finite and positive")
        if not comp.keys() <= tables.keys():
            raise ValueError("BremsLib table missing for at least one composition element")
        densities.append(comp)
    staged = {
        el: stage_bremslib_table(tables[el], array_module=np) for comp in densities for el in comp
    }
    energy_field = "E_repr_keV" if segments.get("E_repr_keV") is not None else "E_keV"
    kinetic = np.asarray(segments[energy_field], dtype=float)
    electron_direction = np.asarray(segments["v_hat"], dtype=float)
    # BremsLib has a finite incident-energy range and no guaranteed fallback
    # outside it. Enforce coverage for every segment with nonzero proposal mass.
    upper = np.minimum(ceiling_eV, kinetic * 1e3)
    active = (np.asarray(segments["L_ang"]) > 0) & (upper > cutoff_eV)
    for layer, comp in enumerate(densities):
        rows = active & (labels == layer)
        for el in comp:
            table = staged[el].table
            if np.any(
                rows
                & (
                    (kinetic < table.minimum_incident_energy_keV)
                    | (kinetic > table.maximum_incident_energy_keV)
                )
            ):
                raise ValueError(f"BremsLib incident energy outside {el} table coverage")

    def emissivity(index: np.ndarray, photon_energy: np.ndarray, direction: np.ndarray):
        output = np.zeros(index.size)
        mu = np.einsum("ij,ij->i", electron_direction[index], direction)
        for layer, comp in enumerate(densities):
            mask = labels[index] == layer
            if not np.any(mask):
                continue
            row = index[mask]
            for el, number_density in comp.items():
                state = bremslib_segment_state(staged[el], kinetic[row], mu[mask])
                dsigma = evaluate_bremslib_rows(staged[el], state, photon_energy[mask])
                output[mask] += number_density * _CM2_TO_ANG2 * np.asarray(dsigma)
        return output

    return sample_continuous_photons(
        segments,
        emissivity,
        n_samples=n_samples,
        n_primaries=n_primaries,
        seed=seed,
        cutoff_eV=cutoff_eV,
        ceiling_eV=ceiling_eV,
        source_kind=source_kind,
        proposal=proposal,
    )

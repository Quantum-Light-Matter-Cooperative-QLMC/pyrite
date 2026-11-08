"""Correlated classical energy--time incident phase space for channeling (#628).

An :class:`IncidentPhaseSpace` is a set of *mutually incoherent* classical
electron records at the crystal entrance plane: arrival time, kinetic energy,
transverse position and transverse slope, with probability weights. It is the
shared input contract for the temporal channeling diagnostics in
:mod:`.temporal_radiation`; it is not a longitudinal quantum wavepacket (#629)
and no record carries phase information relative to any other.

Records come from the existing beam infrastructure rather than a parallel
beam model: :func:`sample_incident_phase_space` composes
:func:`~pyrite.montecarlo.transport.beam_entry.sample_primary_entry` (the
analytic transverse, longitudinal and uncorrelated-energy draws transport
uses) and :meth:`IncidentPhaseSpace.from_gdf` the validated GPT/GDF importer.
A :class:`LinearEnergyChirp` then adds the one correlation the analytic beam
lacks, ``T_i = T_i^(0) + h (t_i - t_ref)``.

Units: time fs, energy eV, length angstrom, slopes rad (``dx/dz``, ``dy/dz``
in the slab frame whose ``z`` is the mean beam axis). Larger ``t`` arrives
later (bunch tail). CPU only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.constants import c as _C_M_PER_S

#: Speed of light [angstrom/fs].
C_ANG_PER_FS = _C_M_PER_S * 1e-5

PHASE_SPACE_SCHEMA = "pyrite.channeling.incident-phase-space/1"
#: Ordered names of the six phase-space coordinates used by covariance helpers.
COORDINATES = ("t_fs", "T_eV", "x_ang", "y_ang", "xp_rad", "yp_rad")
MAX_RECORDS = 1_000_000
_RESAMPLE_DOMAIN = 0x6280


def _frozen(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _scalar(value: Any) -> Any:
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        if not np.isfinite(float(value)):
            raise ValueError("provenance values must be finite")
        return float(value)
    raise TypeError("provenance values must be JSON scalars (str, bool, int, float, None)")


@dataclass(frozen=True)
class LinearEnergyChirp:
    """Additive linear energy--time correlation ``T -> T + h (t - t_ref)``.

    Parameters
    ----------
    chirp_eV_per_fs
        ``h`` [eV/fs]; positive means later (tail) electrons are more
        energetic. Zero is the identity.
    reference_time_fs
        ``t_ref`` [fs]; ``None`` uses the weighted mean arrival time of the
        phase space it is applied to, so the energy centroid is unchanged.

    The uncorrelated spread is whatever the base records already carry (for
    analytic beams, ``energy_spread_frac`` drawn independently of arrival time
    by the transport sampler), so ``Var T = Var T0 + 2 h Cov(t, T0) + h^2 Var t``
    and ``Cov(t, T) = Cov(t, T0) + h Var t`` hold exactly for the records.

    Validation: cr-energy-time-phase-space
    """

    chirp_eV_per_fs: float
    reference_time_fs: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.chirp_eV_per_fs, bool) or not np.isfinite(self.chirp_eV_per_fs):
            raise ValueError("chirp_eV_per_fs must be finite")
        if self.reference_time_fs is not None and not np.isfinite(self.reference_time_fs):
            raise ValueError("reference_time_fs must be finite or None")


@dataclass(frozen=True, eq=False)
class IncidentPhaseSpace:
    """Weighted, mutually incoherent classical entrance records.

    Parameters
    ----------
    arrival_time_fs
        ``(N,)`` arrival time at the entrance face [fs], relative to any
        common reference.
    kinetic_energy_eV
        ``(N,)`` positive kinetic energies [eV].
    position_ang
        ``(N, 2)`` entrance-face ``(x, y)`` in the slab frame [angstrom].
    slope_rad
        ``(N, 2)`` transverse slopes ``(dx/dz, dy/dz)`` [rad].
    weights
        ``(N,)`` nonnegative probabilities; normalized to one on construction.
        ``None`` gives equal weights.
    source
        Short provenance label (``"analytic"``, ``"gpt_gdf"``, ``"explicit"``...).
    provenance
        JSON-scalar mapping recorded with the records (sorted on storage).

    Arrays are copied and read-only. Equality is by value via :meth:`digest`.
    """

    arrival_time_fs: np.ndarray
    kinetic_energy_eV: np.ndarray
    position_ang: np.ndarray
    slope_rad: np.ndarray
    weights: Any = None
    source: str = "explicit"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        time = np.asarray(self.arrival_time_fs)
        if time.ndim != 1 or not 1 <= time.size <= MAX_RECORDS:
            raise ValueError(f"arrival_time_fs must be one-dimensional with 1..{MAX_RECORDS} rows")
        n = time.size
        object.__setattr__(self, "arrival_time_fs", _frozen("arrival_time_fs", time, (n,)))
        energy = _frozen("kinetic_energy_eV", self.kinetic_energy_eV, (n,))
        if np.any(energy <= 0):
            raise ValueError("kinetic_energy_eV must be positive")
        object.__setattr__(self, "kinetic_energy_eV", energy)
        object.__setattr__(self, "position_ang", _frozen("position_ang", self.position_ang, (n, 2)))
        object.__setattr__(self, "slope_rad", _frozen("slope_rad", self.slope_rad, (n, 2)))
        raw = np.ones(n) if self.weights is None else self.weights
        weights = _frozen("weights", raw, (n,))
        if np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("weights must be nonnegative with a positive sum")
        weights = np.array(weights / weights.sum())
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a nonempty string")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        clean = {str(k): _scalar(v) for k, v in sorted(self.provenance.items())}
        object.__setattr__(self, "provenance", clean)

    # ------------------------------------------------------------------ views
    @property
    def size(self) -> int:
        return int(self.arrival_time_fs.size)

    @property
    def uniform_weights(self) -> bool:
        """True when every record is one equally weighted electron."""
        return bool(np.all(self.weights == self.weights[0]))

    def coordinates(self) -> np.ndarray:
        """``(N, 6)`` matrix in :data:`COORDINATES` order."""
        return np.column_stack(
            (self.arrival_time_fs, self.kinetic_energy_eV, self.position_ang, self.slope_rad)
        )

    def mean(self) -> np.ndarray:
        """Weighted mean of the six coordinates."""
        return self.weights @ self.coordinates()

    def covariance(self) -> np.ndarray:
        """Weighted population covariance ``(6, 6)`` in :data:`COORDINATES` order.

        ``C_ab = sum_i w_i (q_ia - <q_a>)(q_ib - <q_b>)`` with normalized
        weights (no Bessel correction): the moments of the records themselves.

        Validation: cr-energy-time-phase-space
        """
        centred = self.coordinates() - self.mean()
        return (centred * self.weights[:, None]).T @ centred

    def correlation(self) -> np.ndarray:
        """Pearson correlation matrix; rows/columns of zero spread are ``NaN``."""
        cov = self.covariance()
        sigma = np.sqrt(np.diag(cov))
        with np.errstate(divide="ignore", invalid="ignore"):
            out = cov / np.outer(sigma, sigma)
        out[(sigma == 0)[:, None] | (sigma == 0)[None, :]] = np.nan
        return out

    @property
    def arrival_rms_fs(self) -> float:
        """Weighted RMS arrival-time spread: the bunch-envelope duration [fs]."""
        return float(np.sqrt(self.covariance()[0, 0]))

    # ------------------------------------------------------------ transforms
    def _replace(self, **changes: Any) -> IncidentPhaseSpace:
        values: dict[str, Any] = dict(
            arrival_time_fs=self.arrival_time_fs,
            kinetic_energy_eV=self.kinetic_energy_eV,
            position_ang=self.position_ang,
            slope_rad=self.slope_rad,
            weights=self.weights,
            source=self.source,
            provenance=dict(self.provenance),
        )
        values.update(changes)
        return IncidentPhaseSpace(**values)

    def with_linear_chirp(self, chirp: LinearEnergyChirp) -> IncidentPhaseSpace:
        """Return records with ``T_i + h (t_i - t_ref)``; other coordinates kept.

        Raises if a chirped energy is not positive; a chirp is a bounded
        correlation, never a reason to clip the distribution silently.

        Validation: cr-energy-time-phase-space
        """
        if not isinstance(chirp, LinearEnergyChirp):
            raise TypeError("chirp must be a LinearEnergyChirp")
        if "chirp_eV_per_fs" in self.provenance:
            raise ValueError("phase space is already chirped; compose chirps explicitly")
        t_ref = (
            float(self.weights @ self.arrival_time_fs)
            if chirp.reference_time_fs is None
            else float(chirp.reference_time_fs)
        )
        energy = self.kinetic_energy_eV + chirp.chirp_eV_per_fs * (self.arrival_time_fs - t_ref)
        if not np.all(np.isfinite(energy)) or np.any(energy <= 0):
            raise ValueError("chirp produced a nonpositive kinetic energy; reduce |h| or bunch")
        provenance = dict(self.provenance)
        provenance.update(chirp_eV_per_fs=float(chirp.chirp_eV_per_fs), chirp_reference_fs=t_ref)
        return self._replace(kinetic_energy_eV=energy, provenance=provenance)

    def resample(self, n: int, *, seed: int = 0) -> IncidentPhaseSpace:
        """Draw ``n`` equally weighted whole records by inverse-CDF selection.

        Whole-record selection keeps every joint correlation in expectation.
        Uniforms come from one Philox stream keyed by ``seed`` under a
        dedicated domain, so the first ``m`` records of a larger draw equal
        an ``m``-record draw (prefix stability).
        """
        if isinstance(n, bool) or not isinstance(n, Integral) or not 1 <= n <= MAX_RECORDS:
            raise ValueError(f"n must be an integer from 1 to {MAX_RECORDS}")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**63:
            raise ValueError("seed must be a nonnegative integer below 2**63")
        key = np.random.SeedSequence((int(seed), _RESAMPLE_DOMAIN))
        u = np.random.Generator(np.random.Philox(key)).random(int(n))
        cdf = np.cumsum(self.weights)
        index = np.minimum(np.searchsorted(cdf, u * cdf[-1], side="right"), self.size - 1)
        provenance = dict(self.provenance)
        provenance.update(resampled_from=self.digest(), resample_seed=int(seed))
        return self._replace(
            arrival_time_fs=self.arrival_time_fs[index],
            kinetic_energy_eV=self.kinetic_energy_eV[index],
            position_ang=self.position_ang[index],
            slope_rad=self.slope_rad[index],
            weights=None,
            provenance=provenance,
        )

    # --------------------------------------------------------- constructors
    @classmethod
    def from_entry_arrays(
        cls,
        position_ang: Any,
        direction: Any,
        energy_keV: Any,
        arrival_ang: Any,
        *,
        weights: Any = None,
        source: str = "explicit",
        provenance: Mapping[str, Any] | None = None,
    ) -> IncidentPhaseSpace:
        """Adapt transport entrance arrays (slab frame, ``c t`` in angstrom).

        ``position_ang`` is ``(N, 3)`` or ``(N, 2)``, ``direction`` ``(N, 3)``
        with positive ``z``; ``arrival_ang`` is the transport clock offset
        ``c t`` [angstrom]. Slopes are ``(u_x/u_z, u_y/u_z)``.
        """
        position = np.asarray(position_ang, dtype=float)
        direction = np.asarray(direction, dtype=float)
        if position.ndim != 2 or position.shape[1] not in (2, 3):
            raise ValueError("position_ang must have shape (N, 2) or (N, 3)")
        if direction.shape != (position.shape[0], 3) or np.any(direction[:, 2] <= 0):
            raise ValueError("direction must be (N, 3) and point into the slab")
        return cls(
            arrival_time_fs=np.asarray(arrival_ang, dtype=float) / C_ANG_PER_FS,
            kinetic_energy_eV=np.asarray(energy_keV, dtype=float) * 1e3,
            position_ang=position[:, :2],
            slope_rad=direction[:, :2] / direction[:, 2:3],
            weights=weights,
            source=source,
            provenance={} if provenance is None else provenance,
        )

    @classmethod
    def from_gdf(
        cls,
        beam: Any,
        count: int,
        *,
        seed: int,
        z_origin_m: float,
        tilt: float = 0.0,
        azimuth: float = 0.0,
    ) -> IncidentPhaseSpace:
        """Project a loaded :class:`~pyrite.montecarlo.gdf.GDFBeam` onto the entrance.

        Uses ``GDFBeam.sample`` (whole-record resampling by ``nmacro`` weight,
        field-free ray-plane projection and screen crossing times), so every
        imported energy--time--position--direction correlation is preserved
        record by record. ``beam`` is a ``GDFBeam`` from ``load_gdf_beam``;
        the projection itself is the ledgered ``gpt-gdf-injection`` claim.

        Validation: cr-energy-time-phase-space
        """
        position, direction, energy_keV, time_ang = beam.sample(
            int(count), int(seed), float(z_origin_m), float(tilt), float(azimuth)
        )
        return cls.from_entry_arrays(
            position,
            direction,
            energy_keV,
            time_ang,
            source="gpt_gdf",
            provenance=dict(gdf_sha256=str(beam.sha256), gdf_seed=int(seed)),
        )

    # -------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        """Fresh JSON-safe, versioned mapping; floats round-trip exactly."""
        return {
            "schema": PHASE_SPACE_SCHEMA,
            "source": self.source,
            "provenance": dict(self.provenance),
            "arrival_time_fs": self.arrival_time_fs.tolist(),
            "kinetic_energy_eV": self.kinetic_energy_eV.tolist(),
            "position_ang": self.position_ang.tolist(),
            "slope_rad": self.slope_rad.tolist(),
            "weights": self.weights.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IncidentPhaseSpace:
        """Parse an exact schema-v1 mapping; unknown or missing keys are rejected."""
        if not isinstance(payload, Mapping):
            raise TypeError("incident phase space must be a mapping")
        if payload.get("schema") != PHASE_SPACE_SCHEMA:
            raise ValueError(f"unsupported phase-space schema (expected {PHASE_SPACE_SCHEMA})")
        expected = {
            "schema",
            "source",
            "provenance",
            "arrival_time_fs",
            "kinetic_energy_eV",
            "position_ang",
            "slope_rad",
            "weights",
        }
        if set(payload) != expected:
            raise ValueError(f"phase-space fields must be exactly {sorted(expected)}")
        return cls(
            arrival_time_fs=payload["arrival_time_fs"],
            kinetic_energy_eV=payload["kinetic_energy_eV"],
            position_ang=payload["position_ang"],
            slope_rad=payload["slope_rad"],
            weights=payload["weights"],
            source=payload["source"],
            provenance=payload["provenance"],
        )

    def digest(self) -> str:
        """SHA-256 of the canonical JSON mapping (scientific identity of the records)."""
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(text.encode()).hexdigest()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IncidentPhaseSpace):
            return NotImplemented
        return self.digest() == other.digest()

    __hash__ = None  # type: ignore[assignment]


def sample_incident_phase_space(
    kinetic_energy_eV: float,
    n: int,
    seed: int,
    *,
    chirp: LinearEnergyChirp | None = None,
    **beam_kwargs: Any,
) -> IncidentPhaseSpace:
    """Sample an analytic beam with the transport sampler, then optionally chirp it.

    ``beam_kwargs`` are those of
    :func:`~pyrite.montecarlo.transport.beam_entry.sample_primary_entry`
    (``bunch_length_fs``, ``long_shape``, ``longitudinal_distribution``,
    ``transverse_distribution``, ``beam_fwhm_mm``, ``energy_spread_frac``,
    ``beam_dir``, tilts...). The arrival times, the uncorrelated energy
    spread and the transverse draws are therefore *exactly* the transport
    primaries of the same seed; only the chirp is added here.
    """
    from ..transport.beam_entry import sample_primary_entry

    position, direction, energy_keV, _clock, t0_ang, _record = sample_primary_entry(
        float(kinetic_energy_eV) / 1e3, int(n), int(seed), **beam_kwargs
    )
    space = IncidentPhaseSpace.from_entry_arrays(
        position,
        direction,
        energy_keV,
        t0_ang,
        source="analytic",
        provenance=dict(seed=int(seed), reference_energy_eV=float(kinetic_energy_eV)),
    )
    return space if chirp is None else space.with_linear_chirp(chirp)


@dataclass(frozen=True)
class EnergyNodes:
    """Discrete incident-energy nodes assigned to every phase-space record.

    ``node_index[i]`` is record ``i``'s node; ``energies_eV[k]`` is the
    weighted mean energy of node ``k``'s records (so every bin, and the whole
    ensemble, keeps its first moment), ``weights`` the node probabilities.
    ``quantization_rms_eV`` is ``sqrt(sum_i w_i (T_i - T_k(i))^2)``: the
    energy resolution lost by solving at node energies. ``method`` is
    ``"exact"`` when every distinct record energy is its own node.

    Validation: cr-temporal-slice-ensemble
    """

    energies_eV: np.ndarray
    weights: np.ndarray
    node_index: np.ndarray
    edges_eV: np.ndarray | None
    quantization_rms_eV: float
    method: str

    @property
    def size(self) -> int:
        return int(self.energies_eV.size)

    def ensemble(self) -> Any:
        """The node marginal as an explicit #626 :class:`IncidentEnergyEnsemble`."""
        from .energy_ensemble import IncidentEnergyEnsemble

        return IncidentEnergyEnsemble(
            float(self.weights @ self.energies_eV),
            energy_nodes_eV=tuple(self.energies_eV.tolist()),
            node_weights=tuple(self.weights.tolist()),
            max_nodes=max(self.size, 1),
        )

    def bin_edges_eV(self) -> np.ndarray:
        """Positive increasing edges with exactly one node per bin."""
        e = self.energies_eV
        if e.size == 1:
            width = max(e[0] * 1e-8, 1e-8)
            return np.array([max(0.0, e[0] - width), e[0] + width])
        mid = 0.5 * (e[1:] + e[:-1])
        return np.r_[max(0.0, e[0] - (mid[0] - e[0])), mid, e[-1] + (e[-1] - mid[-1])]


def quantize_energies(space: IncidentPhaseSpace, *, max_nodes: int = 17) -> EnergyNodes:
    """Assign records to at most ``max_nodes`` incident-energy nodes.

    Distinct energies up to ``max_nodes`` are kept exactly (a monoenergetic
    or explicitly discrete beam is never smeared). Otherwise records are
    binned on ``max_nodes`` equal-width intervals of ``[min T, max T]``;
    empty bins are dropped and each node sits at its records' weighted mean.

    Validation: cr-temporal-slice-ensemble
    """
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, Integral):
        raise ValueError("max_nodes must be an integer")
    if not 1 <= max_nodes <= 256:
        raise ValueError("max_nodes must be from 1 to 256")
    energy, weight = space.kinetic_energy_eV, space.weights
    distinct, inverse = np.unique(energy, return_inverse=True)
    if distinct.size <= max_nodes:
        nodes = np.bincount(inverse, weights=weight, minlength=distinct.size)
        keep = nodes > 0
        remap = np.cumsum(keep) - 1
        index = remap[inverse]
        out_energy, out_weight, edges, method = distinct[keep], nodes[keep], None, "exact"
    else:
        edges = np.linspace(distinct[0], distinct[-1], max_nodes + 1)
        raw = np.clip(np.searchsorted(edges, energy, side="right") - 1, 0, max_nodes - 1)
        mass = np.bincount(raw, weights=weight, minlength=max_nodes)
        first = np.bincount(raw, weights=weight * energy, minlength=max_nodes)
        keep = mass > 0
        remap = np.cumsum(keep) - 1
        index = remap[raw]
        out_weight = mass[keep]
        out_energy = first[keep] / out_weight
        method = "binned"
    out_weight = out_weight / out_weight.sum()
    rms = float(np.sqrt(weight @ (energy - out_energy[index]) ** 2))
    arrays = [np.array(a, copy=True) for a in (out_energy, out_weight, index.astype(np.int64))]
    for a in arrays:
        a.setflags(write=False)
    if edges is not None:
        edges = np.array(edges, copy=True)
        edges.setflags(write=False)
    return EnergyNodes(arrays[0], arrays[1], arrays[2], edges, rms, method)


@dataclass(frozen=True)
class TimeSlices:
    """Incident arrival-time slices and their conditional energy-node weights.

    ``slice_index[i]`` is record ``i``'s slice (``-1`` outside the edges);
    ``probability[s]`` is the slice's incident probability and
    ``node_weights[s, k] = P(node k | slice s)`` (zero rows for empty slices).
    ``outside_probability`` is the incident mass outside every slice.

    Validation: cr-temporal-slice-ensemble
    """

    edges_fs: np.ndarray
    slice_index: np.ndarray
    probability: np.ndarray
    node_weights: np.ndarray
    outside_probability: float
    mean_energy_eV: np.ndarray
    rms_energy_eV: np.ndarray


def time_slices(
    space: IncidentPhaseSpace, nodes: EnergyNodes, edges_fs: Sequence[float]
) -> TimeSlices:
    """Partition records by incident arrival time into half-open slices.

    Slice ``s`` is ``[edges[s], edges[s+1])``; the last slice also includes
    its upper edge. Mean and RMS energies use the *record* energies, not the
    quantized node energies.

    Validation: cr-temporal-slice-ensemble
    """
    edges = np.array(edges_fs, dtype=float, copy=True)
    if (
        edges.ndim != 1
        or not 2 <= edges.size <= 257
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError("time edges must be 2..257 finite increasing values")
    if nodes.node_index.shape != (space.size,):
        raise ValueError("energy nodes were not built from this phase space")
    t = space.arrival_time_fs
    index = np.searchsorted(edges, t, side="right") - 1
    index = np.where(t == edges[-1], edges.size - 2, index)
    index = np.where((index < 0) | (index >= edges.size - 1), -1, index).astype(np.int64)
    n_slices = edges.size - 1
    inside = index >= 0
    weight = space.weights
    probability = np.bincount(index[inside], weights=weight[inside], minlength=n_slices)
    joint = np.zeros((n_slices, nodes.size))
    np.add.at(joint, (index[inside], nodes.node_index[inside]), weight[inside])
    energy = space.kinetic_energy_eV
    first = np.bincount(index[inside], weights=(weight * energy)[inside], minlength=n_slices)
    with np.errstate(invalid="ignore", divide="ignore"):
        conditional = np.divide(
            joint, probability[:, None], out=np.zeros_like(joint), where=probability[:, None] > 0
        )
        mean = np.divide(first, probability, out=np.full(n_slices, np.nan), where=probability > 0)
    second = np.zeros(n_slices)
    sel = np.flatnonzero(inside)
    np.add.at(second, index[sel], weight[sel] * (energy[sel] - mean[index[sel]]) ** 2)
    rms = np.sqrt(
        np.divide(second, probability, out=np.full(n_slices, np.nan), where=probability > 0)
    )
    for array in (edges, index, probability, conditional, mean, rms):
        array.setflags(write=False)
    return TimeSlices(
        edges,
        index,
        probability,
        conditional,
        float(max(0.0, 1.0 - probability.sum())),
        mean,
        rms,
    )


__all__ = [
    "COORDINATES",
    "C_ANG_PER_FS",
    "EnergyNodes",
    "IncidentPhaseSpace",
    "LinearEnergyChirp",
    "PHASE_SPACE_SCHEMA",
    "TimeSlices",
    "quantize_energies",
    "sample_incident_phase_space",
    "time_slices",
]

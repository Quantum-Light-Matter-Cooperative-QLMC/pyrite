"""Bounded optical planar TR/CTR: physical bunch charge, sampling law and phases (#615).

Opt-in, source-only scoring of transition radiation (TR) and coherent TR (CTR)
from transported planar-stack crossings with explicit optical permittivities
(:class:`~pyrite.materials.optical.OpticalResponse`). It reuses the crossing
amplitudes of ``tr-crossing-superposition`` and extends the distinct-pair
bunch estimator of ``tr-coherent-bunch`` with an explicit source contract:

- the physical population ``N = Q/e`` comes from the bunch charge alone;
  the ``M`` sampled histories and their optional importance weights only
  estimate the two moments of the single-electron field;
- the per-physical-electron yield is ``I = <|S|^2> + (N - 1) |F|^2 |<S>|^2``,
  estimated by ``Sum w|S|^2 / W`` and ``(|Sum w S|^2 - Sum w^2 |S|^2) / (W^2 - W_2)``
  (``W = Sum w``, ``W_2 = Sum w^2``); unit weights recover ``G/M`` and
  ``(P - G)/(M (M - 1))``. A weight never enters the self term squared;
- ``|F|^2`` is ``1`` when the histories carry their own sampled offsets
  (law ``sampled``) and the characteristic function of a stated offset law
  only for offset-free histories (laws ``gaussian`` and ``empirical``);
  a ``realization`` is the physical bunch itself (``N = M``, unit weights).

Crossing phase: ``exp[i(omega t - k_par . r_par)]`` with ``t`` the transported
crossing time [Å, ``c = 1``] and ``k_par = omega n_par`` the tangential vacuum
wavevector of the observed wave (slab frame). A history translated by
``(dt, dx, dy)`` gains ``exp[i kappa . (dt, dx, dy)]``,
``kappa = (omega, -omega n_x, -omega n_y)/hbar c``, at every crossing.

Units: ``photons/eV/sr`` per physical electron; multiply by ``N`` for the
bunch. Reference plane: the far-field origin is the transverse origin of the
emitting face (``crossing_amplitudes``); time origin the transport clock.

Validation: optical-planar-tr, optical-ctr-population
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...materials.crystal import HBARC_EV_ANG
from .coherent_population import CoherentSamplingError, CoherentSamplingWarning
from .transition_radiation_crossings import VACUUM, BoundaryCrossings, history_field_sums

__all__ = [
    "OPTICAL_TRANSITION_MODEL",
    "OpticalBunchSource",
    "OpticalTransitionResult",
    "gaussian_coherent_band_eV",
    "load_optical_transition",
    "optical_transition_radiation",
]

#: Identity marker of the bounded optical TR/CTR scorer (#615).
OPTICAL_TRANSITION_MODEL = "optical-planar-ctr-v1"
#: Transport clock: Å of light travel per fs.
C_ANG_PER_FS = 2997.92458
_ANG_PER_UM = 1.0e4
_ELEMENTARY_CHARGE_C = 1.602176634e-19
#: Largest phase spread [rad] an offset-free history set may carry.
_OFFSET_FREE_PHASE = 1e-6
SourceLaw = Literal["sampled", "gaussian", "empirical", "realization"]


@dataclass(frozen=True, eq=False)
class OpticalBunchSource:
    """Physical bunch and its sampling law for optical TR/CTR.

    ``charge_pC`` fixes the physical population ``N = Q/e``, independently of
    the number of sampled histories. ``law`` selects how inter-electron phases
    enter (module docstring). Lengths are stored in Å and times in Å of light
    travel; use the constructors for fs/µm inputs. ``history_weights`` are
    positive importance weights ``p/q`` of the iid sampled histories (any
    common scale), never macro-particle charges: the self term uses them
    linearly. ``inter_electron="correlated"`` (sub-/super-Poissonian shot
    noise, a fixed realization treated as a law) is refused except by the
    ``realization`` law, which is that explicit realization.

    Validation: optical-ctr-population
    """

    law: SourceLaw
    charge_pC: float
    description: str
    covariance_ang2: tuple[tuple[float, float, float], ...] | None = None
    profile_ang: np.ndarray | None = None
    profile_weights: np.ndarray | None = None
    profile_kind: Literal["samples", "profile"] | None = None
    time_resolution_ang: float | None = None
    transverse_resolution_ang: float | None = None
    history_weights: np.ndarray | None = None
    inter_electron: Literal["independent", "correlated"] = "independent"

    def __post_init__(self) -> None:
        if self.law not in ("sampled", "gaussian", "empirical", "realization"):
            raise ValueError(f"unsupported optical source law {self.law!r}")
        if not self.description:
            raise ValueError("optical source needs a provenance description")
        if not np.isfinite(self.charge_pC) or self.charge_pC < 0.0:
            raise ValueError("charge_pC must be finite and nonnegative")
        population = self.physical_electrons
        if 0.0 < population < 1.0:
            raise ValueError("a nonzero bunch must contain at least one physical electron")
        if self.inter_electron not in ("independent", "correlated"):
            raise ValueError("inter_electron must be 'independent' or 'correlated'")
        if self.inter_electron == "correlated" and self.law != "realization":
            raise ValueError(
                "correlated inter-electron sources need their own pair contract; only an "
                "explicit realization (law='realization') is supported"
            )
        self._check_law_fields()
        if self.history_weights is not None:
            weights = np.asarray(self.history_weights, dtype=float)
            if self.law == "realization":
                raise ValueError(
                    "a realization's histories are physical electrons: weights (macro-charges) "
                    "are refused; never square a macro-charge as physical self noise"
                )
            if weights.ndim != 1 or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
                raise ValueError("history_weights must be positive finite importance weights")
            object.__setattr__(self, "history_weights", weights.copy())

    def _check_law_fields(self) -> None:
        analytic = (self.covariance_ang2, self.profile_ang)
        if self.law in ("sampled", "realization"):
            if any(item is not None for item in analytic):
                raise ValueError(
                    f"law {self.law!r} uses the histories' own sampled offsets; an analytic "
                    "offset law would apply them twice"
                )
            return
        if self.law == "gaussian":
            if self.covariance_ang2 is None or self.profile_ang is not None:
                raise ValueError("law 'gaussian' needs covariance_ang2 and no profile")
            cov = np.asarray(self.covariance_ang2, dtype=float)
            if cov.shape != (3, 3) or not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T):
                raise ValueError("covariance_ang2 must be a finite symmetric 3x3 (t, x, y) matrix")
            if np.min(np.linalg.eigvalsh(cov)) < -1e-12 * max(1.0, float(np.max(np.abs(cov)))):
                raise ValueError("covariance_ang2 must be positive semidefinite")
            object.__setattr__(self, "covariance_ang2", tuple(map(tuple, cov.tolist())))
            return
        self._check_empirical()

    def _check_empirical(self) -> None:
        if self.profile_ang is None or self.covariance_ang2 is not None:
            raise ValueError("law 'empirical' needs profile_ang and no covariance")
        profile = np.asarray(self.profile_ang, dtype=float)
        if profile.ndim != 2 or profile.shape[1] != 3 or profile.shape[0] < 2:
            raise ValueError("profile_ang must be (K >= 2, 3) entrance-face (t, x, y) points")
        weights = (
            np.ones(profile.shape[0])
            if self.profile_weights is None
            else np.asarray(self.profile_weights, dtype=float)
        )
        if weights.shape != profile.shape[:1] or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("profile_weights must be nonnegative, one per profile point")
        if not np.all(np.isfinite(profile)) or not np.all(np.isfinite(weights)):
            raise ValueError("empirical profile must be finite")
        if self.profile_kind not in ("samples", "profile"):
            raise ValueError("profile_kind must be 'samples' (iid draws) or 'profile' (a density)")
        resolution = self.time_resolution_ang
        if resolution is None or not np.isfinite(resolution) or resolution <= 0:
            raise ValueError("an empirical source must state its time resolution")
        transverse = self.transverse_resolution_ang
        if np.any(profile[:, 1:] != 0.0) and (
            transverse is None or not np.isfinite(transverse) or transverse <= 0
        ):
            raise ValueError("a transverse empirical profile must state its resolution")
        object.__setattr__(self, "profile_ang", profile.copy())
        object.__setattr__(self, "profile_weights", weights.copy())

    @classmethod
    def sampled(cls, charge_pC: float, description: str, *, weights=None):
        """Histories transported with their own sampled offsets (``|F|^2 = 1``)."""
        return cls("sampled", charge_pC, description, history_weights=weights)

    @classmethod
    def realization(cls, charge_pC: float, description: str):
        """The sampled histories are the physical bunch: ``N = M``, unit weights."""
        return cls("realization", charge_pC, description, inter_electron="correlated")

    @classmethod
    def gaussian(
        cls,
        charge_pC: float,
        description: str,
        *,
        rms_t_fs: float,
        rms_x_um: float = 0.0,
        rms_y_um: float = 0.0,
        weights=None,
    ):
        """Independent Gaussian entrance-face offsets: RMS crossing time and position.

        ``rms_t_fs`` is the RMS (standard deviation) of the crossing time, not
        a FWHM: catalog ``envelope_rms_fs`` maps to it directly at normal
        incidence. Correlated (tilted) laws pass ``covariance_ang2`` instead.
        """
        sigma = np.array([rms_t_fs * C_ANG_PER_FS, rms_x_um * _ANG_PER_UM, rms_y_um * _ANG_PER_UM])
        if np.any(sigma < 0) or not np.all(np.isfinite(sigma)):
            raise ValueError("RMS widths must be finite and nonnegative")
        cov = np.diag(sigma**2)
        return cls(
            "gaussian",
            charge_pC,
            description,
            covariance_ang2=cov.tolist(),
            history_weights=weights,
        )

    @property
    def physical_electrons(self) -> float:
        """``N = Q/e`` of the physical bunch."""
        return float(self.charge_pC) * 1e-12 / _ELEMENTARY_CHARGE_C

    def form_factor_squared(self, energy_eV, direction) -> np.ndarray:
        """``|F|^2`` of the analytic or empirical offset law; ``1`` for sampled laws.

        Validation: optical-ctr-population
        """
        energy = np.asarray(energy_eV, dtype=float)
        n = np.asarray(direction, dtype=float)
        shape = np.broadcast_shapes(energy.shape, n.shape[:-1])
        if self.law in ("sampled", "realization"):
            return np.ones(shape)
        omega = np.broadcast_to(energy, shape) / HBARC_EV_ANG
        n = np.broadcast_to(n, (*shape, 3))
        kappa = np.stack((omega, -omega * n[..., 0], -omega * n[..., 1]), axis=-1)
        if self.law == "gaussian":
            cov = np.asarray(self.covariance_ang2)
            return np.exp(-np.einsum("...i,ij,...j->...", kappa, cov, kappa))
        self._check_resolution(omega, n)
        return self._empirical_factor(kappa)

    def _check_resolution(self, omega, n) -> None:
        worst_t = float(np.max(omega)) * float(self.time_resolution_ang or np.inf)
        if worst_t > 1.0:
            raise ValueError(
                f"empirical source resolution {self.time_resolution_ang:g} Å does not resolve "
                f"the requested band: omega * dt = {worst_t:.3g} > 1"
            )
        transverse = self.transverse_resolution_ang
        if transverse is not None:
            worst_r = float(np.max(omega * np.hypot(n[..., 0], n[..., 1]))) * transverse
            if worst_r > 1.0:
                raise ValueError(
                    f"empirical transverse resolution {transverse:g} Å does not resolve "
                    f"the requested directions: k_par * dr = {worst_r:.3g} > 1"
                )

    def _empirical_factor(self, kappa) -> np.ndarray:
        points = np.asarray(self.profile_ang, dtype=float)
        weights = np.asarray(self.profile_weights, dtype=float)
        flat = kappa.reshape(-1, 3)
        total = np.zeros(flat.shape[0], dtype=complex)
        for lo in range(0, points.shape[0], 4096):
            phase = flat @ points[lo : lo + 4096].T
            total += np.exp(1j * phase) @ weights[lo : lo + 4096]
        w_sum, w2_sum = float(weights.sum()), float(np.sum(weights**2))
        if self.profile_kind == "profile":
            value = np.abs(total) ** 2 / w_sum**2
        else:  # unbiased distinct-pair estimate of |F|^2 from iid draws
            value = (np.abs(total) ** 2 - w2_sum) / (w_sum**2 - w2_sum)
        return value.reshape(kappa.shape[:-1])

    def record(self) -> dict[str, Any]:
        """JSON provenance of the law, without the empirical point arrays."""
        out: dict[str, Any] = {
            "law": self.law,
            "charge_pC": float(self.charge_pC),
            "physical_electrons": self.physical_electrons,
            "description": self.description,
            "inter_electron": self.inter_electron,
            "weighted": self.history_weights is not None,
            "covariance_ang2": self.covariance_ang2,
        }
        if self.law == "empirical":
            out.update(
                profile_kind=self.profile_kind,
                profile_points=len(np.asarray(self.profile_ang)),
                time_resolution_ang=self.time_resolution_ang,
                transverse_resolution_ang=self.transverse_resolution_ang,
            )
        return out


def gaussian_coherent_band_eV(rms_t_fs: float, physical_electrons: float, tolerance: float):
    """Upper energy of the Gaussian coherent window: ``(N - 1) |F_t|^2 >= tolerance``.

    ``|F_t|^2 = exp[-(E sigma_t / hbar)^2]`` gives
    ``E_w = (hbar / sigma_t) sqrt(ln((N - 1)/tolerance))``; above ``E_w`` the pair
    term is below ``tolerance`` times the self term. Zero when ``N - 1 <= tolerance``.

    Validation: optical-ctr-population
    """
    hbar_ev_fs = HBARC_EV_ANG / C_ANG_PER_FS
    excess = (float(physical_electrons) - 1.0) / float(tolerance)
    if rms_t_fs <= 0 or tolerance <= 0:
        raise ValueError("rms_t_fs and tolerance must be positive")
    return 0.0 if excess <= 1.0 else hbar_ev_fs / rms_t_fs * float(np.sqrt(np.log(excess)))


@dataclass(frozen=True, eq=False)
class OpticalTransitionResult:
    """Optical TR/CTR per physical electron with its normalization and provenance.

    ``self_yield`` and ``pair_yield`` are ``E n_z^2/(16 pi^4)`` times the
    estimated ``<|S|^2>`` and ``|<S>|^2`` [photons/eV/sr]; ``form_factor``
    is the applied ``|F|^2``; ``yield_per_electron`` is
    ``self + max(N - 1, 0) F pair`` (for a realization ``|sum S|^2 / M``).
    Signed: unresolved pair estimates stay negative.
    """

    energy_eV: np.ndarray
    direction: np.ndarray
    self_yield: np.ndarray
    pair_yield: np.ndarray
    form_factor: np.ndarray
    yield_per_electron: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bunch_yield(self) -> np.ndarray:
        """Photons/eV/sr per bunch: ``N`` times the per-electron yield."""
        return self.metadata["source"]["physical_electrons"] * self.yield_per_electron

    def save(self, path) -> Path:
        """Write arrays and JSON metadata to one ``.npz`` (no pickles)."""
        target = Path(path)
        arrays = {name: getattr(self, name) for name in _ARRAYS}
        np.savez(target, metadata=np.array(json.dumps(self.metadata, sort_keys=True)), **arrays)
        return target if target.suffix == ".npz" else target.with_suffix(target.suffix + ".npz")


_ARRAYS = (
    "energy_eV",
    "direction",
    "self_yield",
    "pair_yield",
    "form_factor",
    "yield_per_electron",
)


def load_optical_transition(path) -> OpticalTransitionResult:
    """Read :meth:`OpticalTransitionResult.save` output; refuses another model."""
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata.get("model") != OPTICAL_TRANSITION_MODEL:
            raise ValueError(f"not an {OPTICAL_TRANSITION_MODEL} result")
        arrays = {name: np.array(data[name]) for name in _ARRAYS}
    return OpticalTransitionResult(**arrays, metadata=metadata)


def _require_offset_free(crossings: BoundaryCrossings, energy, n) -> None:
    """Analytic laws need histories without their own entrance offsets."""
    entry = (crossings.layer_before == VACUUM) & (crossings.interface == 0)
    if not np.any(entry):
        return
    omega = float(np.max(energy)) / HBARC_EV_ANG
    k_par = omega * float(np.max(np.hypot(n[..., 0], n[..., 1])))
    spread_t = float(np.ptp(crossings.time_ang[entry]))
    spread_r = float(np.max(np.ptp(crossings.position_ang[entry, :2], axis=0)))
    if omega * spread_t > _OFFSET_FREE_PHASE or k_par * spread_r > _OFFSET_FREE_PHASE:
        raise ValueError(
            "the histories already carry sampled entrance offsets; an analytic/empirical "
            "form factor would apply them twice (use law='sampled')"
        )


def _weighted_moments(crossings, energy, n, layer_eps, weights, *, max_elements, xp):
    """``(Sum w|S|^2, Sum w^2|S|^2, Sum w S)`` via charge-scaled history sums."""
    charge = crossings.charge.astype(float)
    if weights is None:
        power, field_sum = history_field_sums(
            crossings, energy, n, layer_eps, coherence="bunch", max_elements=max_elements, xp=xp
        )
        return power, power, field_sum
    per_crossing = weights[crossings.electron_id]
    scaled = replace(crossings, charge=charge * per_crossing)
    power2, field_sum = history_field_sums(
        scaled, energy, n, layer_eps, coherence="bunch", max_elements=max_elements, xp=xp
    )
    root = replace(crossings, charge=charge * np.sqrt(per_crossing))
    power1, _ = history_field_sums(
        root, energy, n, layer_eps, coherence="bunch", max_elements=max_elements, xp=xp
    )
    return power1, power2, field_sum


def optical_transition_radiation(
    crossings: BoundaryCrossings,
    photon_energy_eV,
    direction,
    layer_eps,
    *,
    source: OpticalBunchSource,
    n_histories: int,
    provenance: dict[str, Any] | None = None,
    max_elements: int = 1 << 22,
    xp=np,
) -> OpticalTransitionResult:
    """Optical TR/CTR of ``n_histories`` sampled histories for a physical bunch.

    ``crossings`` must belong to histories ``0..n_histories-1`` of an
    infinite planar stack (a side exit is refused: lateral faces are not
    modeled). ``layer_eps`` are explicit optical permittivities on the
    energies. Arrays return to the host; the sums run on ``xp``.

    Validation: optical-planar-tr, optical-ctr-population
    """
    energy = np.asarray(photon_energy_eV, dtype=float)
    n = np.asarray(direction, dtype=float)
    count = int(n_histories)
    if crossings.n_side_exits:
        raise ValueError("optical TR supports laterally infinite planar stacks only (side exits)")
    if count < 1 or (len(crossings) and int(np.max(crossings.electron_id)) >= count):
        raise ValueError("crossings must belong to histories 0..n_histories-1")
    weights = source.history_weights
    if weights is not None and weights.size != count:
        raise ValueError("history_weights needs one importance weight per sampled history")
    if source.law in ("gaussian", "empirical"):
        _require_offset_free(crossings, energy, n)
    population = source.physical_electrons
    if source.law == "realization" and abs(population - count) > 0.5:
        raise ValueError(
            f"a realization needs N = Q/e = {population:.6g} equal to its {count} histories"
        )
    g_w, h_w, field_sum = _weighted_moments(
        crossings, energy, n, layer_eps, weights, max_elements=max_elements, xp=xp
    )
    host = [a.get() if hasattr(a, "get") else np.asarray(a) for a in (g_w, h_w, field_sum)]
    g_w, h_w, field_sum = host
    w_sum = float(count if weights is None else weights.sum())
    w2_sum = float(count if weights is None else np.sum(weights**2))
    coherent = np.sum(np.abs(field_sum) ** 2, axis=-1)
    shape = g_w.shape
    prefactor = np.broadcast_to(energy * n[..., 2] ** 2, shape) / (16.0 * np.pi**4)
    factor = np.broadcast_to(source.form_factor_squared(energy, n), shape)
    self_raw = g_w / w_sum
    if source.law == "realization":
        pair_raw = (coherent - h_w) / (w_sum**2 - w2_sum) if count > 1 else np.zeros(shape)
        total = coherent / w_sum
    else:
        excess = max(population - 1.0, 0.0)
        if excess and w_sum**2 - w2_sum <= 0.0:
            raise CoherentSamplingError(
                "a physical pair estimate needs at least two sampled histories"
            )
        pair_raw = (coherent - h_w) / (w_sum**2 - w2_sum) if excess else np.zeros(shape)
        total = self_raw + excess * factor * pair_raw
    _warn_negative(total)
    metadata = {
        "model": OPTICAL_TRANSITION_MODEL,
        "units": "photons/eV/sr per physical electron (bunch: times N)",
        "normalization": "self = Sum w|S|^2/W; pair = (|Sum w S|^2 - Sum w^2|S|^2)/(W^2 - W_2)",
        "reference": (
            "far field at the transverse origin of the emitting face; exp(-i omega t); "
            "time origin = transport clock t0 + age; slab-frame directions"
        ),
        "source": source.record(),
        "n_histories": count,
        "effective_histories": w_sum**2 / w2_sum,
        "pair_weight": (max(population - 1.0, 0.0)) / max(w_sum**2 / w2_sum - 1.0, 1e-300),
        "n_crossings": len(crossings),
        **(provenance or {}),
    }
    return OpticalTransitionResult(
        energy_eV=energy.copy(),
        direction=n.copy(),
        self_yield=prefactor * self_raw,
        pair_yield=prefactor * pair_raw,
        form_factor=np.array(factor, dtype=float),
        yield_per_electron=prefactor * total,
        metadata=json.loads(json.dumps(metadata, default=_jsonable)),
    )


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _warn_negative(total) -> None:
    negative = int(np.count_nonzero(total < 0.0))
    if negative:
        warnings.warn(
            CoherentSamplingWarning(
                f"optical CTR pair estimate is negative in {negative} of {total.size} bins: "
                "the sampled histories do not resolve the bunch excess; returned unclipped"
            ),
            stacklevel=3,
        )

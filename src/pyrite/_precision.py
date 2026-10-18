"""Import-light adaptive sampling policy shared by API, cases, and runners."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

OBSERVABLES = ("cxr", "brem")
REQUIRED_FIELDS = ("target_rse", "min_electrons", "max_electrons", "block_electrons")

#: Default for the counts that follow the now-optional ``min_electrons``; keeps
#: the positional field order while still refusing an omitted count.
_REQUIRED: Any = object()
#: Omitted ``min_electrons``: resolves to one block in ``__post_init__``.
_ONE_BLOCK: Any = None


@dataclass(frozen=True)
class Precision:
    """Stopping-rule settings for one adaptive case.

    ``min_electrons``, ``max_electrons`` and ``pilot_electrons`` are multiples
    of ``block_electrons``, so every realized count is a whole number of equal
    blocks (the batch-means estimator needs equal batches).

    Parameters
    ----------
    target_rse
        Target relative standard error of each observable's mean.
    min_electrons, max_electrons
        Smallest count at which the rule may stop (default: one block,
        ``block_electrons``), and the count at which it stops regardless,
        flagged ``statistics_limited``.
    block_electrons
        Electrons per transport block; the rule is checked at block ends.
    observables
        Non-empty subset of ``('cxr', 'brem')`` that must all converge; a single
        name such as ``"cxr"`` is accepted.
    max_electron_share
        Guard: the largest single-electron value may hold at most this share
        of the observable's sum.
    min_effective_electrons
        Guard: the effective sample size ``(sum m)**2 / sum m**2`` must reach
        this floor.
    stability_blocks, stability_fraction
        Guard: the running mean at each of the last ``k`` block ends must lie
        within ``stability_fraction * target_rse`` (relative) of the current
        mean, so a run does not stop on the block a rare heavy electron just
        moved; ``k = 0`` disables.
    pilot_electrons
        Optional pilot count. At the pilot the run projects
        ``N = N_pilot * max(rse / target)**2`` (rounded up to a block) and
        skips the checks until it gets there.
    band_eV
        Monitor band ``(start, stop)`` in eV; ``None`` takes the case's line
        band (its policy bandwidth, else its line-grid end points).
    batch_means_band_eV
        Optional band for per-bin batch-means errors of the final ``spec`` and
        ``brem_wide``; reported only, never used to stop. Costs one extra
        line and brem reduction over the realized segments.
    brem_target_rse
        Enable independent brem-prefix stopping at this target. Requires both
        observables. ``target_rse`` then governs CXR; line transport continues
        until brem finishes, so ``Ne_brem <= Ne``. Omit for the shared rule.
    brem_min_electrons, brem_max_electrons
        Optional brem bounds, defaulting to the shared bounds; block multiples.
        The brem maximum cannot exceed ``max_electrons``. Guards and block size
        are shared; the brem pilot is clipped to its own bounds.
    """

    target_rse: float
    min_electrons: int = _ONE_BLOCK
    max_electrons: int = _REQUIRED
    block_electrons: int = _REQUIRED
    observables: tuple[str, ...] = ("cxr",)
    max_electron_share: float = 0.05
    min_effective_electrons: float = 100.0
    stability_blocks: int = 3
    stability_fraction: float = 0.5
    pilot_electrons: int | None = None
    band_eV: tuple[float, float] | None = None
    batch_means_band_eV: tuple[float, float] | None = None
    brem_target_rse: float | None = None
    brem_min_electrons: int | None = None
    brem_max_electrons: int | None = None

    def __post_init__(self) -> None:
        for name in ("max_electrons", "block_electrons"):
            if getattr(self, name) is _REQUIRED:
                raise TypeError(f"Precision requires {name}")
        if self.min_electrons is None:
            object.__setattr__(self, "min_electrons", self.block_electrons)
        # Freeze caller-owned sequences as well as the dataclass itself. A bare
        # name is one observable, not its characters.
        raw = self.observables
        observables = (raw,) if isinstance(raw, str) else tuple(raw)
        object.__setattr__(self, "observables", observables)
        for name in ("band_eV", "batch_means_band_eV"):
            band = getattr(self, name)
            if band is not None:
                object.__setattr__(self, name, tuple(band))
        if isinstance(self.target_rse, bool) or not (
            math.isfinite(self.target_rse) and self.target_rse > 0.0
        ):
            raise ValueError("target_rse must be a positive finite number")
        for name in ("block_electrons", "min_electrons", "max_electrons", "pilot_electrons"):
            value = getattr(self, name)
            if name == "pilot_electrons" and value is None:
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        block = self.block_electrons
        if not 2 <= self.min_electrons <= self.max_electrons:
            raise ValueError("need 2 <= min_electrons <= max_electrons")
        counts = [self.min_electrons, self.max_electrons]
        if self.pilot_electrons is not None:
            counts.append(self.pilot_electrons)
        if any(count % block for count in counts):
            raise ValueError(
                "min_electrons, max_electrons and pilot_electrons must be multiples "
                "of block_electrons"
            )
        if (
            not self.observables
            or not set(self.observables) <= set(OBSERVABLES)
            or len(set(self.observables)) != len(self.observables)
        ):
            raise ValueError(f"observables must be a non-empty subset of {OBSERVABLES}")
        if isinstance(self.max_electron_share, bool) or not 0.0 < self.max_electron_share <= 1.0:
            raise ValueError("max_electron_share must lie in (0, 1]")
        for name in ("min_effective_electrons", "stability_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if type(self.stability_blocks) is not int or self.stability_blocks < 0:
            raise ValueError("stability_blocks must be a non-negative integer")
        if self.pilot_electrons is not None and not (
            self.min_electrons <= self.pilot_electrons <= self.max_electrons
        ):
            raise ValueError("pilot_electrons must lie between min_electrons and max_electrons")
        for name in ("band_eV", "batch_means_band_eV"):
            band = getattr(self, name)
            if band is not None and (
                len(band) != 2
                or any(isinstance(value, bool) or not math.isfinite(value) for value in band)
                or not 0.0 < band[0] < band[1]
            ):
                raise ValueError(f"{name} must contain two positive finite increasing energies")
        self._validate_brem_policy()

    def _validate_brem_policy(self) -> None:
        if self.brem_target_rse is None:
            if self.brem_min_electrons is not None or self.brem_max_electrons is not None:
                raise ValueError("brem count bounds require brem_target_rse")
            return
        if set(self.observables) != set(OBSERVABLES):
            raise ValueError("brem_target_rse requires both cxr and brem observables")
        policy = self.brem_policy()
        if policy.max_electrons > self.max_electrons:
            raise ValueError("brem_max_electrons must not exceed max_electrons (brem is a prefix)")

    def brem_policy(self) -> Precision:
        """Scalar brem rule for independent stopping; shares block size and guards."""
        minimum = self.min_electrons if self.brem_min_electrons is None else self.brem_min_electrons
        maximum = self.max_electrons if self.brem_max_electrons is None else self.brem_max_electrons
        pilot = self.pilot_electrons
        return replace(
            self,
            target_rse=self.target_rse if self.brem_target_rse is None else self.brem_target_rse,
            min_electrons=minimum,
            max_electrons=maximum,
            observables=("brem",),
            pilot_electrons=None if pilot is None else min(max(pilot, minimum), maximum),
            brem_target_rse=None,
            brem_min_electrons=None,
            brem_max_electrons=None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the requested policy, without realized counts or statistics."""
        # Persisted requested-policy and identity payloads retain their historical
        # vocabulary. Public in-memory observables are canonical CXR names.
        values = asdict(self)
        # Absent independent settings must not change existing policy identities.
        for key in ("brem_target_rse", "brem_min_electrons", "brem_max_electrons"):
            if values[key] is None:
                del values[key]
        values["observables"] = tuple(
            "line" if name == "cxr" else name for name in self.observables
        )
        return values

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Precision:
        """Validate a serialized policy and restore tuple-valued fields."""
        if not isinstance(payload, Mapping):
            raise TypeError("adaptive_precision must be a mapping")
        values = dict(payload)
        fields = cls.__dataclass_fields__
        unknown = sorted(set(values) - set(fields))
        if unknown:
            raise ValueError(f"unknown precision field(s) {unknown}; valid fields: {list(fields)}")
        missing = [name for name in REQUIRED_FIELDS if name not in values]
        if missing:
            raise ValueError(f"precision requires {', '.join(missing)}")
        for name in ("observables", "band_eV", "batch_means_band_eV"):
            if name in values and values[name] is not None:
                values[name] = tuple(values[name])
        if "observables" in values:
            values["observables"] = tuple(
                "cxr" if value == "line" else value for value in values["observables"]
            )
        return cls(**values)

    def validate_case(self, case: Mapping[str, Any]) -> None:
        """Reject routes whose samples do not satisfy the block-driver contract."""
        if case.get("coherent_emission"):
            raise ValueError(
                "adaptive precision does not support coherent emission; use fixed counts"
            )
        for key in (
            "gdf_source",
            "groove_spacing_ang",
            "secondary_threshold_eV",
            "pair_production_model",
            "positron_transport",
        ):
            if case.get(key) is not None and case.get(key) is not False:
                raise ValueError(f"adaptive precision does not support {key}; use fixed counts")


#: Policy for a profile that sets neither fixed electron counts nor a
#: ``precision`` table (#361 user decision, 2026-10-08).
DEFAULT_PRECISION = Precision(
    target_rse=0.05,
    min_electrons=200,
    max_electrons=20_000,
    block_electrons=100,
    observables=("cxr", "brem"),
)


def default_precision_fallback(
    *,
    fixed_counts: bool,
    emission: str | None,
    secondary_threshold_eV: float | None = None,
    pair_production_model: str | None = None,
    positron_transport: bool = False,
    gdf_beam: bool = False,
    grooved: bool = False,
) -> str | None:
    """Why a profile without a ``precision`` table keeps fixed counts, or ``None``.

    ``None`` means :data:`DEFAULT_PRECISION` applies. The reasons mirror the
    routes :meth:`Precision.validate_case` refuses, plus an explicit count.
    """
    if fixed_counts:
        return "the profile sets fixed electron counts"
    if emission in ("coherent", "both"):
        return f"emission is {emission!r} (adaptive is incoherent-only)"
    if secondary_threshold_eV is not None or pair_production_model is not None:
        return "particle cascades (secondaries or pair production) are enabled"
    if positron_transport:
        return "positron transport is enabled"
    if gdf_beam:
        return "the beam is a GDF particle file"
    if grooved:
        return "the target has a grooved entrance face"
    return None

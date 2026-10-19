"""Typed, read-only provenance record attached to every :class:`~pyrite.Result`."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._backend import DeviceInfo
    from .._numerics import Numerics
    from ..campaign.model import Scene


@dataclass(frozen=True, init=False, eq=False)
class Provenance(Mapping[str, Any]):
    """What produced a result: inputs, identity, physics models, and environment.

    Attributes are grouped below; any one may be ``None`` when the producer did
    not record it (for example, observation digests exist only for planar
    detectors). The record is also a read-only mapping, so
    ``provenance["identity_digest"]``, ``"adaptive_sampling" in provenance``,
    and ``dict(provenance)`` see exactly the keys the producer supplied.
    Unrecognized keys stay readable by key and through :attr:`extra`.

    Inputs
        :attr:`scene`, :attr:`numerics`.
    Identity
        :attr:`identity_digest`, :attr:`xsgen_tables`, and the planar-detector
        :attr:`observation_identity_digest`, :attr:`observation`,
        :attr:`true_spatial_digest`, :attr:`response_digest`,
        :attr:`acquisition_digest`.
    Physics models
        :attr:`resolved_physics`, :attr:`stopping_model`, :attr:`characteristic_model`,
        :attr:`attenuation_model`, :attr:`bremsstrahlung_model`.
    Run diagnostics
        :attr:`kinematic_validity`, :attr:`line_data_coverage`, :attr:`adaptive_sampling`,
        :attr:`line_grid_policy`, :attr:`line_grid_resolved`.
    Environment
        :attr:`backend`, :attr:`device`, :attr:`git_sha`, :attr:`git_dirty`,
        :attr:`versions`.
    """

    # Inputs
    scene: Scene | None = None
    """Resolved beam, target, detector, filters, emission, and acquisition."""
    numerics: Numerics | None = None
    """Resolved sampling and execution controls, including any adaptive ``precision``."""

    # Identity
    identity_digest: str | None = None
    """SHA-256 source content key of the transport case; equal digests mean
    reusable source spectra."""
    xsgen_tables: Mapping[str, str] | None = None
    """Digests of the cross-section tables this case read, keyed by table digest."""
    observation_identity_digest: str | None = None
    """Planar detectors only: digest of the detector/filter/scoring observation."""
    observation: Mapping[str, Any] | None = None
    """Planar detectors only: the canonical observation payload behind
    :attr:`observation_identity_digest`."""
    true_spatial_digest: str | None = None
    """Acquisitions only: digest of the stored true-spatial factors."""
    response_digest: str | None = None
    """Acquisitions only: digest of the detector response layer."""
    acquisition_digest: str | None = None
    """Acquisitions only: digest of the exposure/bins/counting layer."""

    # Physics models
    resolved_physics: Mapping[str, Any] | None = None
    """Data-only descriptor, generation, parameter and per-case table snapshot.
    Descriptive metadata; source and observation digests remain authoritative.
    """
    stopping_model: str | None = None
    """Collision-stopping model marker, e.g. ``"sbethe-corrected-v1"``."""
    characteristic_model: str | None = None
    """Atomic characteristic-emission model and data-version marker."""
    attenuation_model: str | None = None
    """Photon attenuation data marker (EPDL)."""
    bremsstrahlung_model: str | None = None
    """Bremsstrahlung cross-section source marker (BremsLib or EEDL)."""

    # Run diagnostics
    line_data_coverage: list[Mapping[str, Any]] | None = None
    """Unweighted line-pair optical-domain screens, per direction and policy.
    Counts are not a lost-yield estimate; vacuum seeds are distinguished from
    solved roots, including roots outside the production tabulation.
    """
    kinematic_validity: Mapping[str, Any] | None = None
    """Central-ray kinematic-theory screen per reflection: ``model``, limits,
    and ``reflections`` with their dynamical/extinction ratios."""
    adaptive_sampling: Mapping[str, Any] | None = None
    """Adaptive runs only: requested precision, realized electron counts,
    achieved errors, guards, ``stop_reason``, and ``statistics_limited``."""
    line_grid_policy: Mapping[str, Any] | None = None
    """Automatic CXR-grid runs only: the grid policy and tolerances."""
    line_grid_resolved: Mapping[str, Any] | None = None
    """Automatic CXR-grid runs only: the coordinates the run actually sampled."""

    # Environment
    backend: str | None = None
    """Array backend that ran transport: ``"numpy"``, ``"cuda"``, ``"rocm"``, ..."""
    device: DeviceInfo | None = None
    """Accelerator name, vendor, memory, and fp64 support."""
    git_sha: str | None = None
    """Source checkout commit, or ``None`` for an installed wheel."""
    git_dirty: bool | None = None
    """Whether the source checkout had uncommitted changes."""
    versions: Mapping[str, str | None] | None = None
    """Versions of PyRITE, NumPy, xraydb, and endf-parserpy."""

    extra: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    """Supplied keys with no named attribute above."""
    _keys: tuple[str, ...] = field(default=(), repr=False)

    def __init__(self, values: Mapping[str, Any] | None = None, /, **kwargs: Any) -> None:
        merged = {**(values or {}), **kwargs}
        known = _FIELD_NAMES
        for name in known:
            object.__setattr__(self, name, merged.get(name))
        object.__setattr__(
            self,
            "extra",
            MappingProxyType({k: v for k, v in merged.items() if k not in known}),
        )
        object.__setattr__(self, "_keys", tuple(merged))

    def __getitem__(self, key: str) -> Any:
        if key not in self._keys:
            raise KeyError(key)
        return self.extra[key] if key in self.extra else getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def __reduce__(self):
        return (type(self), (dict(self),))

    def __repr__(self) -> str:
        lines = [f"    {key}={_short(self[key])}," for key in self._keys]
        return "Provenance(\n" + "\n".join(lines) + "\n)" if lines else "Provenance()"


_FIELD_NAMES = frozenset(item.name for item in fields(Provenance)) - {"extra", "_keys"}


def _short(value: object, width: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= width else text[: width - 1] + "…"


__all__ = ["Provenance"]

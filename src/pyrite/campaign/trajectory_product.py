"""Typed transport product of :func:`pyrite.transport.simulate_trajectories` (#1162).

:class:`Trajectories` keeps one fixed-seed electron transport together with the
resolved scene it came from, so :func:`pyrite.score_spectra` can score it any
number of times without transporting again and can refuse a request that would
have changed the transport. It replaces the former untyped segment dict and
stays a read-only mapping over that dict's keys for one deprecation window.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .._numerics import Physics
    from ..montecarlo.case import Case
    from .model import Numerics, Scene

__all__ = ["Trajectories"]

_MAPPING_DEPRECATION = (
    "dict-style access to simulate_trajectories() output is deprecated and will be "
    "removed in {remove_in}; use Trajectories.segments[...] instead"
)


def _host(value: Any) -> np.ndarray:
    get = getattr(value, "get", None)
    return np.asarray(get() if callable(get) and not isinstance(value, Mapping) else value)


def segment_fingerprint(segments: Mapping[str, Any]) -> str:
    """``"<rows>:<blake2b>"`` over every canonical per-segment row field.

    Integer IDs hash as int64 and floats in their own dtype, so a compactly
    stored artifact read back in transported order has the fingerprint of the
    in-memory transport it was written from; any changed value changes it.
    """
    from ..montecarlo.transport.secondaries import _ALIASES, _ROW_KEYS

    digest = hashlib.blake2b(digest_size=16)
    rows = int(_host(segments["L_ang"]).size)
    for name in sorted(set(_ROW_KEYS) - set(_ALIASES)):
        value = segments.get(name)
        if value is None:
            continue
        array = _host(value)
        if array.dtype.kind in "iub":
            array = array.astype(np.int64)
        array = np.ascontiguousarray(array)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return f"{rows}:{digest.hexdigest()}"


@dataclass(frozen=True, eq=False)
class Trajectories(Mapping[str, Any]):
    """One fixed-seed electron transport and the resolved scene it came from.

    Returned by :func:`pyrite.transport.simulate_trajectories` and
    :func:`pyrite.storage.load_trajectories`; pass it to
    :func:`pyrite.score_spectra`. It holds every transported segment, so its
    memory is that of the transport; CUDA segments stay on the device when
    requested.

    Attributes
    ----------
    scene
        Beam, target and capture detector (``None`` for detector-free
        transport). Downstream filters and acquisition are not part of it.
    physics, numerics
        The resolved policies the transport used.
    case
        The resolved transport case (keV, angstrom, eV, rad; see
        :class:`pyrite.transport.Case`).

    Notes
    -----
    Indexing, iterating or ``len()`` on the object reads the former dict
    output (segment arrays, incident phase space, fate counts) with a
    :class:`DeprecationWarning`; use :attr:`segments` instead.
    """

    scene: Scene | None
    physics: Physics | None
    numerics: Numerics | None
    case: Case
    _transport: dict[str, Any] = field(repr=False)
    _xsgen_tables: Mapping[str, str] = field(repr=False, default_factory=dict)
    #: Explicit engine controls that make this transport differ from production.
    _engine_overrides: Mapping[str, Any] = field(repr=False, default_factory=dict)
    #: Recorded fingerprint of a loaded artifact; verified on first access.
    _stored_fingerprint: str | None = field(repr=False, default=None)
    _source: str | None = field(repr=False, default=None)

    @property
    def segments(self) -> Mapping[str, Any]:
        """Segment arrays, incident phase space and fate counts (transport-output contract)."""
        return self._transport["segs"]

    @property
    def segment_count(self) -> int:
        """Number of transported electron segments."""
        return int(_host(self.segments["L_ang"]).size)

    @cached_property
    def fingerprint(self) -> str:
        """Row count and digest of the segment kinematics the spectra read."""
        return segment_fingerprint(self.segments)

    @property
    def scorable(self) -> bool:
        """Whether :func:`pyrite.score_spectra` accepts this transport."""
        return self.scene is not None and not self._engine_overrides

    # Deprecated Mapping view of the former dict output (#1162).
    def _warn(self) -> None:
        from .._api_deprecations import API_DEPRECATIONS

        entry = API_DEPRECATIONS["simulate_trajectories()[...]"]
        warnings.warn(
            _MAPPING_DEPRECATION.format(remove_in=entry.remove_in),
            DeprecationWarning,
            stacklevel=3,
        )

    def __getitem__(self, key: str) -> Any:
        self._warn()
        return self.segments[key]

    def __iter__(self) -> Iterator[str]:
        self._warn()
        return iter(self.segments)

    def __len__(self) -> int:
        self._warn()
        return len(self.segments)

    def __contains__(self, key: object) -> bool:
        self._warn()
        return key in self.segments

    def __repr__(self) -> str:
        name = None if self.case is None else self.case.get("name")
        return f"Trajectories(case={name!r}, segments={self.segment_count})"

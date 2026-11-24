"""Opt-in numerical transport identity; absent cases retain historical semantics."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps
from typing import Any

_SCHEMA = "pyrite.transport-realization.v1"
#: Spectrum-phase backends a realization may require (#1387). ``"cpu"`` is the
#: original v1 record and stays implicit, so its serialized form and every
#: identity derived from it are unchanged; ``"gpu"`` scores on the process's
#: CuPy backend and is recorded, giving a distinct case identity.
_SCORING = ("cpu", "gpu")
_CORE = ContextVar[str | None]("transport_realization_core", default=None)


@dataclass(frozen=True, slots=True)
class TransportRealization(Mapping[str, str]):
    """Immutable identity of the admitted counter-addressed CPU transport realization.

    ``scoring`` names the spectrum-phase backend: ``"cpu"`` (default, implicit
    in the mapping) or ``"gpu"`` (recorded; device-scored results are exact
    only against device-scored runs on the same device).
    """

    schema: str = _SCHEMA
    core: str = "per-electron"
    scoring: str = "cpu"

    def __post_init__(self) -> None:
        if self.schema != _SCHEMA or self.core != "per-electron":
            raise ValueError("transport_realization requires v1 core='per-electron'")
        if self.scoring not in _SCORING:
            raise ValueError(f"transport_realization scoring must be one of {_SCORING}")

    def __getstate__(self) -> list[str]:
        # CPU markers keep the original two-field pickle state (a list, as the
        # generated slots ``__getstate__`` wrote it), so their bytes are unchanged.
        return [getattr(self, key) for key in self._keys()]

    def __setstate__(self, state: Sequence[str]) -> None:
        schema, core, *scoring = state
        values = {"schema": schema, "core": core, "scoring": scoring[0] if scoring else "cpu"}
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def _keys(self) -> tuple[str, ...]:
        return ("schema", "core") if self.scoring == "cpu" else ("schema", "core", "scoring")

    def __getitem__(self, key: str) -> str:
        if key not in self._keys():
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys())

    def __len__(self) -> int:
        return len(self._keys())


def normalize_realization(value: Any) -> TransportRealization:
    if not isinstance(value, Mapping) or not (
        {"schema", "core"} <= set(value) <= {"schema", "core", "scoring"}
    ):
        raise ValueError("transport_realization requires schema, core and optional scoring")
    return TransportRealization(
        schema=value["schema"], core=value["core"], scoring=value.get("scoring", "cpu")
    )


def case_scoring(case: Mapping[str, Any]) -> str | None:
    """The spectrum backend a marked case requires, or ``None`` when unmarked."""
    realization = case_realization(case)
    return None if realization is None else realization.scoring


@contextmanager
def scoring_scope(scoring: str):
    """Run the spectrum phase on the backend a realization requires.

    ``"cpu"`` switches every spectrum binding to NumPy. ``"gpu"`` keeps the
    process's device backend and disables the out-of-memory CPU rescoring
    fallback, which would change the numbers; transport (CPU per-electron)
    still runs in a CPU-only process, and :func:`require_scoring_backend`
    refuses the spectrum phase there.
    """
    from .runner.chunking import _cpu_spectrum_backend
    from .runner.recovery import recovery_engine

    if scoring == "cpu":
        with recovery_engine("cpu"), _cpu_spectrum_backend():
            yield
        return
    with recovery_engine("gpu"):
        yield


def require_scoring_backend(case: Mapping[str, Any]) -> None:
    """Refuse to score a ``scoring="gpu"`` case outside a GPU process."""
    from .runner.chunking import _RESOURCE_POLICY

    if case_scoring(case) == "gpu" and not _RESOURCE_POLICY.gpu:
        raise ValueError(
            "transport_realization scoring='gpu' needs a GPU process (PYRITE_MC_BACKEND=cuda)"
        )


def case_realization(case: Mapping[str, Any]) -> TransportRealization | None:
    return (
        normalize_realization(case["transport_realization"])
        if "transport_realization" in case
        else None
    )


def bind_realization(case, realization):
    """Copy an unmarked case, or validate an already bound case idempotently."""
    from .case import Case

    current = case_realization(case)
    if realization is None:
        return case
    required = normalize_realization(realization)
    required_core(realization=required)
    if current is not None and current != required:
        raise ValueError("conflicting transport_realization")
    if isinstance(case, Case):
        return case if current is not None else replace(case, transport_realization=required)
    return {**case, "transport_realization": required}


def common_realization(cases: Sequence[Mapping[str, Any]]) -> TransportRealization | None:
    identities = {case_realization(case) for case in cases}
    if len(identities) > 1:
        raise ValueError("cannot mix transport_realization values in one checkpoint")
    return next(iter(identities), None)


def require_matching_realization(expected, actual) -> None:
    """Reject absent-versus-marked records symmetrically, before cache adoption."""
    left = None if expected is None else normalize_realization(expected)
    right = None if actual is None else normalize_realization(actual)
    if left != right:
        raise ValueError("transport_realization mismatch; use a separate checkpoint or recompute")


def required_core(requested="auto", *, realization=None):
    """Resolve only an opt-in requirement; never reinterpret an ordinary pin."""
    from .._util.env import env_value

    core = _CORE.get() if realization is None else normalize_realization(realization).core
    if core is None:
        return None
    pinned = env_value("PYRITE_MC_TRANSPORT_CORE", "").strip().lower()
    if requested not in ("auto", core) or pinned not in ("", "auto", core):
        raise ValueError(
            "transport_realization conflicts with requested/environment transport core: "
            f"requested={requested!r}, PYRITE_MC_TRANSPORT_CORE={pinned!r}"
        )
    return core


@contextmanager
def realization_context(case):
    realization = case_realization(case)
    core = None if realization is None else required_core(realization=realization)
    outer_core = _CORE.get()
    token = _CORE.set(core)
    try:
        if realization is not None and core is not None and core != outer_core:
            with scoring_scope(realization.scoring):
                yield
        else:
            yield
    finally:
        _CORE.reset(token)


def with_realization(function):
    """Carry a case's core requirement through nested transport dispatchers."""

    @wraps(function)
    def call(case, *args, **kwargs):
        with realization_context(case):
            return function(case, *args, **kwargs)

    return call

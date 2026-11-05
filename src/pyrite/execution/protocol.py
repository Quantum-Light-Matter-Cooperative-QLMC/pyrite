"""Minimal orchestration contract above PyRITE's existing numerical engine."""

from collections.abc import Callable, Sequence
from typing import Any, Protocol


class Executor(Protocol):
    """Run pending cases while keeping checkpoint ownership in the caller.

    Each ``callback(index, case, output)`` MUST execute in the coordinating
    process, not in a remote worker. The callback owns persistence, exactly as
    in ``runs.run.run_sweep``. The call must drain already-started tasks if
    ``should_stop`` becomes true, and must leave never-started slots as None.
    A backend that cannot honor a run_cases-compatible option forwarded by
    ``run_sweep`` must raise ``ValueError`` rather than ignore it.

    Two optional methods, discovered by attribute so that an executor defining
    only ``execute`` keeps working, refine what ``run_sweep`` can do:

    ``validate_run_options(options)``
        raises ``ValueError`` for unsupported options before any checkpoint,
        cache or filesystem effect. ``options`` holds what ``execute`` will
        receive except ``callback`` and the per-case ``observation_directions``.
        Without it, rejection happens inside ``execute``, after resume.
    ``runtime_plan(cases, max_workers)``
        returns the ``on_runtime`` dict describing where the work runs: at least
        ``engine``, ``requested_workers`` and ``effective_workers``. Without it,
        ``run_sweep`` reports ``engine="external"`` and no worker count rather
        than the local runner's topology.
    """

    def execute(
        self,
        cases: Sequence[Any],
        *,
        callback: Callable[[int, Any, Any], None] | None = None,
        **run_options: Any,
    ) -> list[Any]: ...

"""Minimal orchestration contract above PyRITE's existing numerical engine."""

from collections.abc import Callable, Sequence
from typing import Any, Protocol


class Executor(Protocol):
    """Run pending cases while keeping checkpoint ownership in the caller.

    Each ``callback(index, case, output)`` MUST execute in the coordinating
    process, not in a remote worker. The callback owns persistence, exactly as
    in ``runs.run.run_sweep``. The call must drain already-started tasks if
    ``should_stop`` becomes true, and must leave never-started slots as None.
    All backend implementations must support the run_cases-compatible options
    forwarded by ``run_sweep`` (including timing/observation hooks).
    """

    def execute(
        self,
        cases: Sequence[Any],
        *,
        callback: Callable[[int, Any, Any], None] | None = None,
        **run_options: Any,
    ) -> list[Any]: ...

"""Local executor adapter: delegate to the established PyRITE runner."""

from collections.abc import Callable, Sequence
from typing import Any


class LocalExecutor:
    """Identical scheduling/CPU pool/GPU pipeline to ``run_cases``.

    The injectable runner is for testing and existing entry-point compatibility.
    Import lazily so inspecting execution plans does not initialize GPU support.
    There is no outer process pool, retry loop, or additional GPU context.
    """

    def __init__(self, runner: Callable[..., list[Any]] | None = None) -> None:
        self._runner = runner

    def execute(
        self,
        cases: Sequence[Any],
        *,
        callback: Callable[[int, Any, Any], None] | None = None,
        **run_options: Any,
    ) -> list[Any]:
        if self._runner is None:
            from ..montecarlo import run_cases

            runner = run_cases
        else:
            runner = self._runner
        return runner(cases, callback=callback, **run_options)

"""Per-case failure budget for sweeps: skip a failing case, stop past the budget."""

import sys
import traceback

__all__ = ["DEFAULT_MAX_CASE_FAILURES", "CaseFailureBudget", "CaseFailuresError"]

#: ``pyrite run --max-case-failures`` default: failing cases skipped per material.
DEFAULT_MAX_CASE_FAILURES = 5


def _case_label(case):
    return f"{case.get('name', 'case')} at {case.get('E0_keV', '?')} keV"


class CaseFailuresError(RuntimeError):
    """A sweep finished its other cases but some cases failed.

    ``failures`` lists ``(case, exception)`` in failure order. Finished cases
    are persisted; a resumed run retries only the failed and unstarted ones.
    """

    def __init__(self, failures):
        self.failures = list(failures)
        lines = [f"  {_case_label(case)}: {type(exc).__name__}: {exc}" for case, exc in failures]
        super().__init__(
            f"{len(self.failures)} case(s) failed; the other cases were saved:\n" + "\n".join(lines)
        )


class CaseFailureBudget:
    """``run_cases(on_case_error=...)`` hook tolerating up to ``limit`` failed cases.

    Each absorbed failure is reported on stderr with its traceback. The failure
    that exceeds ``limit`` is not absorbed, so it aborts the run as before.
    """

    def __init__(self, limit):
        if limit < 0:
            raise ValueError(f"max_case_failures must be >= 0; got {limit}")
        self.limit = int(limit)
        self.failures = []

    def __call__(self, _index, case, exc):
        if len(self.failures) >= self.limit:
            return False
        self.failures.append((case, exc))
        print(
            f"warning: case {_case_label(case)} failed "
            f"({len(self.failures)}/{self.limit} tolerated); continuing with the next case",
            file=sys.stderr,
        )
        traceback.print_exception(exc, file=sys.stderr)
        return True

    def raise_if_failed(self):
        if self.failures:
            raise CaseFailuresError(self.failures)

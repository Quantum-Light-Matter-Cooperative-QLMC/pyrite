from typing import Any

import numpy as np


def fake_out(case: dict[str, Any]) -> dict[str, Any]:
    E = np.arange(*case["E_grid"])
    E_brem = np.arange(*case["E_grid_brem"])

    return {
        "E_grid": E,
        "spec": np.full_like(E, 0.1),
        "brem": np.full_like(E, 0.01),
        "E_grid_brem": E_brem,
        "brem_wide": np.full_like(E_brem, 0.001),
        "eta": 0.05,
    }


def stub_run_cases(
    cases,
    *,
    max_workers=None,
    progress=False,
    callback=None,
    should_stop=None,
    keep_results=True,
    on_timing=None,
    on_activity=None,
    transport_only=False,
    **_unused,
):
    """Synchronous run_cases test double with no Monte Carlo."""

    results = [None] * len(cases)

    for i, case in enumerate(cases):
        if should_stop is not None and should_stop():
            break

        if on_activity is not None:
            on_activity(
                {
                    "phase": "serial_case",
                    "case_index": i,
                    "case": case,
                    "in_flight_case_count": 1,
                }
            )

        out = None if transport_only else fake_out(case)

        if on_timing is not None and out is not None:
            on_timing(
                {
                    "case_index": i,
                    "case": case,
                    "transport_seconds": 0.0,
                    "spectrum_seconds": 0.0,
                    "driver_wait_seconds": 0.0,
                }
            )

        if callback is not None:
            callback(i, case, out)

        if keep_results:
            results[i] = out

    if on_activity is not None:
        on_activity(
            {
                "phase": "idle",
                "case_index": None,
                "case": None,
                "in_flight_case_count": 0,
            }
        )

    return results


def tracking_run_cases_factory(ran: list[str]):
    def tracking_run_cases(cases, *, callback=None, **kwargs):
        ran.extend(case["name"] for case in cases)

        return stub_run_cases(
            cases,
            callback=callback,
            **kwargs,
        )

    return tracking_run_cases

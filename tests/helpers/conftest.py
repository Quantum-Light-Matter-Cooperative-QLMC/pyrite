"""Import pyarrow FIRST: on Windows, importing cxr_mc (numpy/scipy DLLs) before
pyarrow leaves pyarrow's arrow DLLs binding against the wrong runtime, and the
altair/pandas tests then hard-crash (native fault in DataFrame construction).
Loading pyarrow here, before any test module imports cxr_mc, fixes the order.
Harmless when pyarrow is absent (the altair tests skip without it)."""

try:
    import pyarrow  # noqa: F401
except ImportError:
    pass

import numpy as np


def _runner_transport_payload(
    segs,
    *,
    E_grid,
    E_brem=None,
    n_hat=None,
    groove=None,
    ne_lines=None,
    ne_brem=None,
):
    ne = int(segs["Ne"])

    return {
        "E_grid": E_grid,
        "E_brem": E_grid if E_brem is None else E_brem,
        "n_hat": n_hat,
        "segs": segs,
        "Ne_lines": ne if ne_lines is None else ne_lines,
        "Ne_brem": ne if ne_brem is None else ne_brem,
        "groove": groove,
    }

def _fake_segments(
    count=1,
    *,
    ne=None,
    elec_id=None,
):
    if elec_id is None:
        elec_id = np.arange(count, dtype=np.int64)

    elec_id = np.asarray(elec_id, dtype=np.int64)

    if elec_id.shape != (count,):
        raise ValueError("elec_id must have one entry per segment")

    if ne is None:
        ne = int(elec_id.max()) + 1 if count else 0

    return {
        "r_mid": np.tile([4.0, 0.0, 5.0], (count, 1)),
        "v_hat": np.tile([0.0, 0.0, 1.0], (count, 1)),
        "L_ang": np.full(count, 10.0),
        "E_keV": np.full(count, 30.0),
        "t_ang": np.zeros(count),
        "t0_ang": np.zeros(count),
        "elec_id": elec_id,
        "layer": np.zeros(count, dtype=np.int16),
        "Ne": ne,
        "thickness_ang": 10.0,
        "n_backscattered": 0,
        "n_missed": 0,
    }

def _fake_out(case):
    E = np.arange(*case["E_grid"])
    Eb = np.arange(*case["E_grid_brem"])
    return dict(
        E_grid=E,
        spec=np.ones_like(E) * 0.1,
        brem=np.ones_like(E) * 0.01,
        E_grid_brem=Eb,
        brem_wide=np.ones_like(Eb) * 0.001,
        eta=0.05,
    )

def _stub_run_cases(
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
    """Synchronous run_cases test double; performs no Monte Carlo."""

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

        out = None if transport_only else _fake_out(case)

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
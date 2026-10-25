"""Microbenchmark unit normalization at scalar and NumPy-array boundaries.

Run with ``uv run --extra units python checks/unit_conversion_overhead.py``.
Values are already-constructed inputs; registry creation and quantity creation
are outside the timed region.
"""

from __future__ import annotations

import platform
import statistics
import timeit

import numpy as np

from pyrite._units import normalize_unit
from pyrite.units import ureg

_SCALAR_QUANTITY = 60.0 * ureg.MeV
_ARRAY_QUANTITY = np.linspace(1.0, 60.0, 1024) * ureg.MeV
_ARRAY_NUMERIC = np.linspace(1_000.0, 60_000.0, 1024)


def _scalar_unit() -> float:
    return normalize_unit(_SCALAR_QUANTITY, "keV", field="Beam.energy_keV", expected="energy")


def _scalar_numeric() -> float:
    return normalize_unit(60_000.0, "keV", field="Beam.energy_keV", expected="energy")


def _array_unit() -> np.ndarray:
    return normalize_unit(_ARRAY_QUANTITY, "keV", field="Beam.energy_keV", expected="energy")


def _array_numeric() -> np.ndarray:
    return normalize_unit(_ARRAY_NUMERIC, "keV", field="Beam.energy_keV", expected="energy")


def _report(name: str, function, *, number: int) -> None:
    samples = timeit.repeat(function, number=number, repeat=7)
    per_call_us = [sample * 1e6 / number for sample in samples]
    median = statistics.median(per_call_us)
    spread = f"{min(per_call_us):.3f}-{max(per_call_us):.3f}"
    if "array" in name:
        throughput = 1024 / median
        detail = f"; {throughput:.1f} million elements/s"
    else:
        detail = ""
    print(f"{name}: {median:.3f} us/call; range {spread} us{detail}")


def main() -> None:
    assert _scalar_unit() == 60_000.0
    np.testing.assert_allclose(_array_unit(), _ARRAY_NUMERIC, rtol=0.0, atol=1e-11)
    for function in (_scalar_unit, _scalar_numeric, _array_unit, _array_numeric):
        for _ in range(100):
            function()

    print(f"Python {platform.python_version()}, NumPy {np.__version__}")
    print("7 repeats per case; 100-call warmup; 1,024 float64 values per array case")
    _report("numeric scalar", _scalar_numeric, number=20_000)
    _report("Pint scalar conversion", _scalar_unit, number=20_000)
    _report("numeric array", _array_numeric, number=3_000)
    _report("Pint array conversion", _array_unit, number=3_000)


if __name__ == "__main__":
    main()

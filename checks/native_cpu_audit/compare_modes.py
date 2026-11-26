"""Compare one case's output arrays under ``math="reproducible"`` and ``"fast"`` (#1469).

Usage: compare_modes.py REPRODUCIBLE.npz FAST.npz [LABEL]

Inputs are ``fastmath_arms.py --arrays`` files of the same case. Prints one
JSON line: per array, bitwise equality, the largest absolute change relative
to the array's largest magnitude, and the relative change of its sum. A fast
kernel that does not change RNG consumption must leave transport arrays
bitwise equal; spectra must move far less than their Monte Carlo error.
"""

import json
import sys

import numpy as np


def compare(reproducible, fast):
    rows = {}
    for key in reproducible.files:
        a, b = reproducible[key], fast[key]
        row = {"shape": list(a.shape), "bitwise_equal": bool(np.array_equal(a, b, equal_nan=True))}
        if a.shape == b.shape and a.size:
            finite = np.isfinite(a) & np.isfinite(b)
            scale = float(np.max(np.abs(a[finite]))) if finite.any() else 0.0
            if scale > 0:
                row["max_abs_change_over_peak"] = float(np.max(np.abs(a - b)[finite]) / scale)
                total = float(np.sum(a[finite]))
                if total != 0:
                    row["sum_rel_change"] = float(np.sum(b[finite]) / total - 1.0)
        rows[key] = row
    return rows


def main():
    reproducible, fast = np.load(sys.argv[1]), np.load(sys.argv[2])
    label = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2]
    if sorted(reproducible.files) != sorted(fast.files):
        raise SystemExit(f"{label}: array sets differ")
    rows = compare(reproducible, fast)
    worst = max((r.get("max_abs_change_over_peak", 0.0) for r in rows.values()), default=0.0)
    print(
        json.dumps(
            {
                "label": label,
                "arrays": len(rows),
                "all_bitwise_equal": all(r["bitwise_equal"] for r in rows.values()),
                "worst_change_over_peak": worst,
                "rows": rows,
            }
        )
    )


if __name__ == "__main__":
    main()

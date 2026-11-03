"""Remote CUDA root-equivalence probe for the guarded shared table bound (#286).

Validation: line-tabulation-nan-ceiling
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from pyrite._backend import xp
    from pyrite.materials.atomic import _chantler_bounds
    from pyrite.materials.crystal import CRYSTALS, HBARC_EV_ANG, refractive_index
    from pyrite.montecarlo.spectrum.cxr._kernels import (
        _cxr_setup_ceiling,
        _cxr_tabulation_grid,
        _in_medium_kinematics,
    )

    if getattr(xp, "__name__", "") != "cupy":
        parser.error("remote CUDA required")
    records = []
    for crystal, henke, dtype, above, gradient in itertools.product(
        ["hopg", "mos2"], [False, True], ["float32", "float64"], [False, True], [None, 1.0, -2.0]
    ):
        info = CRYSTALS[crystal]
        endpoint = min(_chantler_bounds(el)[1] for el, _ in info["basis"])
        lo = endpoint + (10.125 if above else -16000.125)
        hi = endpoint + 16000.125
        composition = [("Si", 0.05)]
        bound = _cxr_setup_ceiling(info, composition, henke)
        grids = [
            _cxr_tabulation_grid(info, composition, lo, end)
            for end in [hi, min(hi, max(bound, lo + 1.0))]
        ]
        query = np.r_[
            np.linspace(lo, hi, 500),
            endpoint + np.array([-2.0, -1.0, -0.125, 0.0, 0.125, 1.0, 2.0]),
        ]
        denom = xp.full(query.size, 0.25, dtype=dtype)
        vg = xp.asarray(query * 0.25 / HBARC_EV_ANG, dtype=dtype)
        grad = None if gradient is None else xp.full(query.size, gradient, dtype=dtype)
        grad2 = None if gradient is None else xp.full(query.size, gradient**2, dtype=dtype)
        outputs = []
        for grid in grids:
            delta = 1.0 - np.asarray(refractive_index(crystal, grid, henke).real, dtype=float)
            outputs.append(
                tuple(
                    xp.asnumpy(a)
                    for a in _in_medium_kinematics(
                        denom,
                        vg,
                        xp.asarray(delta, dtype=dtype),
                        xp.asarray(grid, dtype=dtype),
                        v_dot_grad=grad,
                        grad2=grad2,
                    )
                )
            )
        for original, bounded in zip(*outputs, strict=True):
            np.testing.assert_array_equal(original, bounded)
        records.append(
            {
                "crystal": crystal,
                "use_henke": henke,
                "dtype": dtype,
                "above_bound": above,
                "gradient": gradient,
                "original_nodes": len(grids[0]),
                "bounded_nodes": len(grids[1]),
                "finite_roots": int(np.isfinite(outputs[0][0]).sum()),
            }
        )
        print(json.dumps(records[-1]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"complete": True, "cases": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()

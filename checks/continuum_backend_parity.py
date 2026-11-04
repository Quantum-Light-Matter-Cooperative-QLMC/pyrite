"""Compare production continuum backends on identical stored tracks (#278).

Run ``--backend cuda`` first, then ``--backend cpu`` with the same artifact
prefix. Pickles are local trusted measurement artifacts, never user inputs.
"""

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--energy-kev", type=float, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    os.environ["PYRITE_MC_BACKEND"] = args.backend
    from pyrite._backend import BACKEND
    from pyrite.energy_grid import convergence_case as cc

    prefix = str(args.prefix)
    tracks = Path(prefix + ".pickle")
    if args.backend == "cuda":
        case = cc.build_ladder_case(
            "hopg",
            args.energy_kev,
            45.0,
            180.0,
            thickness_ang=1e7,
            n_electrons=8,
            seed=17,
        )
        case.pop("line_grid_policy", None)
        case["E_grid"] = case["E_grid_line"] = np.linspace(100.0, 10000.0, 32)
        ladder = cc.CaseLadder(case)
        with tracks.open("wb") as stream:
            pickle.dump((case, ladder.transport), stream)
    else:
        with tracks.open("rb") as stream:
            case, transport = pickle.load(stream)
        if case["E0_keV"] != args.energy_kev:
            raise ValueError("artifact energy differs from requested energy")
        ladder = cc.CaseLadder(case, transport=transport)
    grid = np.asarray(ladder.transport["E_brem"])
    density = ladder.continuum(grid)
    np.savez(prefix + "." + args.backend + ".npz", grid=grid, density=density)
    record = dict(backend=BACKEND.name, fingerprint=ladder.fingerprint, nodes=grid.size)
    if args.backend == "cpu":
        with np.load(prefix + ".cuda.npz") as reference:
            np.testing.assert_array_equal(grid, reference["grid"])
            np.testing.assert_allclose(density, reference["density"], rtol=1e-3, atol=1e-30)
            record["relative_l1"] = float(
                np.trapezoid(abs(density - reference["density"]), grid)
                / np.trapezoid(abs(reference["density"]), grid)
            )
    Path(prefix + "." + args.backend + ".json").write_text(
        json.dumps(record, default=str, indent=2)
    )
    print(record)


if __name__ == "__main__":
    main()

"""Compare production continuum backends on identical stored tracks (#278).

Run ``--backend cuda`` first, then ``--backend cpu`` with the same artifact
prefix. Add ``--portable --chunk 17`` to replay the chunked reducer without
overwriting the default-backend evidence. Detector counts cover only the shared
100 keV response band, as in the grid-convergence study. Pickles are local
trusted measurement artifacts, never user inputs.
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
    parser.add_argument("--portable", action="store_true", help="disable fused CUDA reduction")
    parser.add_argument("--chunk", type=int, help="maximum segment rows per reduction chunk")
    args = parser.parse_args()
    if args.chunk is not None and args.chunk < 1:
        parser.error("--chunk must be positive")
    os.environ["PYRITE_MC_BACKEND"] = args.backend
    from pyrite._backend import BACKEND
    from pyrite.energy_grid import convergence_case as cc

    if args.portable:
        from pyrite.montecarlo.spectrum import brem

        brem._USE_JIT_BREM_REDUCTION = False
    from continuum_grid_convergence import observe

    from pyrite.detectors.spec import EagleXO, Timepix3

    prefix = str(args.prefix)
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    label = args.backend + ("-portable" if args.portable else "")
    if args.chunk is not None:
        label += f"-chunk{args.chunk}"
    tracks = Path(prefix + ".pickle")
    if args.backend == "cuda" and not args.portable and args.chunk is None:
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
    if args.chunk is not None:
        ladder.case = dict(ladder.case, brem_chunk=args.chunk)
    grid = np.asarray(ladder.transport["E_brem"])
    density = ladder.continuum(grid)
    np.savez(prefix + "." + label + ".npz", grid=grid, density=density)
    record = dict(backend=BACKEND.name, fingerprint=ladder.fingerprint, nodes=grid.size)
    detectors = {"timepix3_counts": Timepix3(n_mc=32, seed=7), "eaglexo_counts": EagleXO()}
    record.update(
        portable=args.portable,
        chunk=args.chunk,
        detector_max_eV=100000.0,
        observables=observe(grid, density, detectors),
    )
    if label != "cuda":
        with np.load(prefix + ".cuda.npz") as reference:
            np.testing.assert_array_equal(grid, reference["grid"])
            np.testing.assert_allclose(density, reference["density"], rtol=1e-3, atol=1e-30)
            expected = observe(grid, reference["density"], detectors)
            for name, value in record["observables"].items():
                if name.startswith("continuum_") or name.endswith("continuum_counts"):
                    np.testing.assert_allclose(value, expected[name], rtol=1e-3, atol=1e-30)
            record["relative_l1"] = float(
                np.trapezoid(abs(density - reference["density"]), grid)
                / np.trapezoid(abs(reference["density"]), grid)
            )
    record["complete"] = True
    Path(prefix + "." + label + ".json").write_text(json.dumps(record, default=str, indent=2))
    print(record)


if __name__ == "__main__":
    main()

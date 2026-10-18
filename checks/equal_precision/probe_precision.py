"""Probe precision variants against the independent CPU quadrature and inverse."""

import argparse
import json
from pathlib import Path

from benchmark_precision import install_variant
from precision_source import VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = install_variant(args.variant, args.output.with_suffix(".source.py"))
    import cupy as cp
    import numpy as np

    from pyrite.montecarlo._cupy_jit import jit
    from pyrite.montecarlo.transport import _jit_radiative_device as dev
    from pyrite.montecarlo.transport._jit_radiative import (
        radiative_moments_scalar,
        sample_hard_photon_energy_scalar,
    )
    from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables

    @jit.rawkernel()
    def probe(incident, nominal, top, chi, nt, nr, z, energies, cutoffs, uniforms, out):
        i = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        if i < energies.size:
            out[i, 0] = dev._rad_moment(
                dev.RAD_SOFT,
                incident,
                nominal,
                top,
                chi,
                np.int32(0),
                nt,
                nr,
                nt,
                z,
                energies[i],
                cutoffs[i],
            )
            out[i, 1] = dev._rad_moment(
                dev.RAD_HARD_RATE,
                incident,
                nominal,
                top,
                chi,
                np.int32(0),
                nt,
                nr,
                nt,
                z,
                energies[i],
                cutoffs[i],
            )
            out[i, 2] = dev._rad_sample_photon_eV(
                incident,
                nominal,
                top,
                chi,
                np.int32(0),
                nt,
                nr,
                nt,
                z,
                energies[i],
                cutoffs[i],
                uniforms[i],
            )

    reports = []
    tables = load_bremsstrahlung_tables(["Si", "W"])
    dtype = (
        cp.float64
        if args.variant in {"fp64", "rad-boundary64-24", "rad-quotient32-24"}
        else cp.float32
    )
    for element, table in tables.items():
        incident, nominal, top, chi = (
            table.incident_energy_keV,
            table.nominal_reduced_energy,
            table.top_reduced_energy,
            table.scaled_sdcs_mb,
        )
        grid = sorted(
            set(
                [
                    10000.0,
                    19999.999,
                    20000.0,
                    100000.0,
                    300000.0,
                    800000.0,
                    *[float(e * 1000) for e in incident if 10 <= e <= 800],
                ]
            )
        )
        cases = [
            (e, c, u)
            for e in grid
            for c in [1000.0, 0.1 * e, 0.5 * e, 0.975 * e, (1 - 1e-6) * e]
            for u in [1e-6, 0.001, 0.5, 0.99, 1 - 1e-6]
        ]
        energies, cutoffs, uniforms = np.array(cases).T
        expected = np.array(
            [
                [
                    *radiative_moments_scalar(
                        incident, nominal, top, chi, table.atomic_number, e, c
                    )[:2],
                    sample_hard_photon_energy_scalar(
                        incident, nominal, top, chi, table.atomic_number, e, c, u
                    ),
                ]
                for e, c, u in cases
            ]
        )
        out = cp.empty(expected.shape, dtype=cp.float64)
        probe(
            ((len(cases) + 127) // 128,),
            (128,),
            (
                cp.asarray(incident, dtype=dtype),
                cp.asarray(nominal, dtype=dtype),
                cp.asarray(top, dtype=dtype),
                cp.asarray(
                    chi.ravel(),
                    dtype=cp.float32
                    if args.variant in {"rad-boundary64-24", "rad-quotient32-24"}
                    else dtype,
                ),
                np.int32(incident.size),
                np.int32(nominal.size),
                np.float64(table.atomic_number),
                cp.asarray(energies),
                cp.asarray(cutoffs),
                cp.asarray(uniforms),
                out,
            ),
        )
        cp.cuda.Device().synchronize()
        actual = out.get()
        np.savez_compressed(
            args.output.with_name(args.output.stem + "-" + element + ".npz"),
            energy_eV=energies,
            cutoff_eV=cutoffs,
            uniform=uniforms,
            expected=expected,
            actual=actual,
        )
        maximums = np.max(np.abs(expected[:, :2]), axis=0)
        scaled = np.abs(actual[:, :2] - expected[:, :2]) / np.maximum(
            np.abs(expected[:, :2]), maximums * 1e-6
        )
        inverse_error = np.abs(actual[:, 2] - expected[:, 2]) / energies
        finite = bool(np.isfinite(actual).all())
        nonnegative = bool(np.all(actual[:, :2] >= 0))
        support = bool(
            np.all(actual[:, 2] >= cutoffs - 8 * np.finfo(np.float32).eps * energies)
            and np.all(actual[:, 2] <= energies + 8 * np.finfo(np.float32).eps * energies)
        )
        passed = (
            finite
            and nonnegative
            and support
            and bool(np.max(scaled) <= 2e-4)
            and bool(np.max(inverse_error) <= 2e-4)
        )
        reports.append(
            dict(
                element=element,
                cases=len(cases),
                table_key=table.key,
                table_digest=table.digest,
                finite=finite,
                nonnegative=nonnegative,
                photon_support=support,
                max_scaled_soft_error=float(np.max(scaled[:, 0])),
                max_scaled_hard_error=float(np.max(scaled[:, 1])),
                max_inverse_error_over_incident_energy=float(np.max(inverse_error)),
                screening_passed=passed,
            )
        )
    for index, code in enumerate(probe.cached_codes.values()):
        args.output.with_suffix(f".probe-{index}.cu").write_text(code, encoding="utf-8")
    args.output.write_text(
        json.dumps(
            dict(
                variant=args.variant,
                source=source,
                reports=reports,
                interpretation="Component screening only: error floor is 1e-6 of maximum moment; tolerance 2e-4. This is not a full transport convergence/equivalence gate.",
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()

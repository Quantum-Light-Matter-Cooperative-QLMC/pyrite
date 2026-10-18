"""Debug generated radiative helpers with NumPy; this does not validate CUDA."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from precision_source import VARIANTS, precision_source

from pyrite.xsgen.bremslib.tables import load_bremsstrahlung_tables


def helper_namespace(source):
    """Strip device decorators/imports for an explicitly CPU-only arithmetic probe."""
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    namespace = dict(np=np, xp=np, _MB_CM2=1e-27)
    for name, value in {"HALF": 0.5, "ONE": 1.0, "ZERO": 0.0, "INF": np.inf}.items():
        namespace["F64_" + name] = np.float64(value)
    for name, value in {"ZERO": 0, "ONE": 1, "TWO": 2}.items():
        namespace["I32_" + name] = np.int32(value)
    exec(compile(tree, "<cpu-precision-debug>", "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("reference_probes", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    import pyrite.montecarlo.transport as transport

    original = (
        Path(transport.__file__).with_name("_jit_radiative_device.py").read_text(encoding="utf-8")
    )
    source = precision_source(original, args.variant)
    ns = helper_namespace(source)
    pin = json.loads((args.reference_probes / "probe-fp64.json").read_text(encoding="utf-8"))
    if pin["source"]["reference_sha256"] != hashlib.sha256(original.encode()).hexdigest():
        raise ValueError("reference source mismatch")
    reports = []
    for element, table in load_bremsstrahlung_tables(["Si", "W"]).items():
        record = next(r for r in pin["reports"] if r["element"] == element)
        if record["table_key"] != table.key or record["table_digest"] != table.digest:
            raise ValueError("reference table mismatch")
        dtype = (
            np.float64
            if args.variant in {"fp64", "rad-boundary64-24", "rad-quotient32-24"}
            else np.float32
        )
        incident, nominal, top = [
            np.asarray(a, dtype=dtype)
            for a in [
                table.incident_energy_keV,
                table.nominal_reduced_energy,
                table.top_reduced_energy,
            ]
        ]
        chi = np.asarray(
            table.scaled_sdcs_mb, dtype=np.float64 if args.variant == "fp64" else np.float32
        ).ravel()
        with np.load(args.reference_probes / f"probe-fp64-{element}.npz") as raw:
            energies, cutoffs, uniforms, expected = [
                raw[k] for k in ["energy_eV", "cutoff_eV", "uniform", "expected"]
            ]
        actual = []
        moments = {}
        for energy, cutoff, uniform in zip(energies, cutoffs, uniforms, strict=True):
            params = (
                incident,
                nominal,
                top,
                chi,
                np.int32(0),
                np.int32(incident.size),
                np.int32(nominal.size),
                np.int32(incident.size),
                np.float64(table.atomic_number),
                energy,
                cutoff,
            )
            key = (float(energy), float(cutoff))
            if key not in moments:
                moments[key] = [
                    ns["_rad_moment"](which, *params)
                    for which in [ns["RAD_SOFT"], ns["RAD_HARD_RATE"]]
                ]
            actual.append([*moments[key], ns["_rad_sample_photon_eV"](*params, uniform)])
        actual = np.asarray(actual)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output.with_name(args.output.stem + "-" + element + ".npz"),
            energy_eV=energies,
            cutoff_eV=cutoffs,
            uniform=uniforms,
            expected=expected,
            actual=actual,
        )
        scale = np.maximum(np.abs(expected[:, :2]), np.max(np.abs(expected[:, :2]), axis=0) * 1e-6)
        errors = np.max(np.abs(actual[:, :2] - expected[:, :2]) / scale, axis=0)
        inverse = float(np.max(np.abs(actual[:, 2] - expected[:, 2]) / energies))
        reports.append(
            dict(
                element=element,
                cases=len(energies),
                table_key=table.key,
                table_digest=table.digest,
                finite=bool(np.isfinite(actual).all()),
                nonnegative=bool((actual[:, :2] >= 0).all()),
                photon_support=bool(
                    (
                        (actual[:, 2] >= cutoffs - 8 * np.finfo(np.float32).eps * energies)
                        & (actual[:, 2] <= energies + 8 * np.finfo(np.float32).eps * energies)
                    ).all()
                ),
                max_scaled_soft_error=float(errors[0]),
                max_scaled_hard_error=float(errors[1]),
                max_inverse_error_over_incident_energy=inverse,
                screening_passed=bool(max(errors) <= 2e-4 and inverse <= 2e-4),
            )
        )
        reports[-1]["screening_passed"] &= all(
            reports[-1][k] for k in ["finite", "nonnegative", "photon_support"]
        )
        print(element, reports[-1], flush=True)
    args.output.write_text(
        json.dumps(
            dict(
                variant=args.variant,
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                reports=reports,
                interpretation="NumPy execution of generated helpers only. CUDA compilation, GPU math, transport invariants, ensemble equivalence and timing remain unvalidated.",
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

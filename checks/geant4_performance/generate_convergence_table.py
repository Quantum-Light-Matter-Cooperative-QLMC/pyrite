"""Evaluate one frozen free-atom ELSEPA refinement with retained raw evidence.

Run remotely. Never alters catalogs or resolves an existing refined table as a
substitute for fresh generator evaluations. Muffin-tin requests are unsupported
by this runner; no convergence, angular-mesh or timing gate is produced.
"""

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from pyrite._util.hashing import canonical_json_digest, file_sha256
from pyrite.perf._benchmark_tables import nested_energy_grid, table_record
from pyrite.xsgen._run import run_program
from pyrite.xsgen.elsepa.deck import ElsepaDeck
from pyrite.xsgen.elsepa.parse import check_transport_scaling, parse_dcs, table_arrays
from pyrite.xsgen.sources import resolve_source, source_digest
from pyrite.xsgen.store import ElementTarget, TableRequest, require, store
from pyrite.xsgen.toolchain import build_command, find_toolchain


def admit(plan, key):
    if plan["schema"] != "pyrite.geant4-performance.elsepa-refinement-plan.v1":
        raise ValueError("unsupported refinement plan")
    matches = [row for row in plan["requests"] if row["key"] == key]
    if len(matches) != 1:
        raise ValueError("need exactly one planned request")
    row = matches[0]
    baseline = [b for b in plan["baselines"] if b["key"] == row["baseline_key"]]
    if len(baseline) != 1:
        raise ValueError("need exactly one pinned baseline")
    baseline = baseline[0]
    if type(row["level"]) is not int or row["level"] not in (1, 2, 3):
        raise ValueError("need a refinement level from 1, 2, 3")
    native = baseline["request"]
    if canonical_json_digest(native) != baseline["key"]:
        raise ValueError("baseline request key mismatch")
    model = native["model"] | dict(
        energies_eV=list(nested_energy_grid(native["model"]["energies_eV"], row["level"]))
    )
    expected = native | dict(model=model)
    if row["request"] != expected or canonical_json_digest(expected) != key:
        raise ValueError("refinement differs from exact pinned nested request")
    if expected["target"]["kind"] != "element" or model["mode"] != "free_atom":
        raise ValueError("runner supports free-atom requests only; muffin-tin remains pending")
    if expected["code"] != "elsepa" or expected["quantity"] != "elastic_dcs":
        raise ValueError("need an ELSEPA elastic request")
    deck = ElsepaDeck(
        z=expected["target"]["z"],
        energies_ev=tuple(model["energies_eV"]),
        **{k: v for k, v in model.items() if k not in ("mode", "energies_eV")},
    )
    request = TableRequest(
        "elsepa", expected["code_version"], ElementTarget(deck.z), "elastic_dcs", model
    )
    if request.record() != expected or deck.model_record() != model:
        raise ValueError("generator deck differs from frozen request")
    return request, deck, baseline


def generate(plan_path, key, output, *, max_bytes=256 * 1024**2):
    plan = json.loads(plan_path.read_text())
    request, deck, baseline = admit(plan, key)
    if max_bytes <= 0 or len(deck.energies_ev) > 1000:
        raise ValueError("invalid retention budget or too many energy nodes")
    native = require(baseline["key"])
    if table_record(native) != baseline:
        raise ValueError("installed native table differs from pinned baseline")
    native_mu = native.arrays()["mu"]
    source = resolve_source("elsepa")
    if source_digest(source) != request.code_version:
        raise ValueError("generator source differs from frozen request")
    toolchain = find_toolchain()
    output.mkdir(parents=True, exist_ok=False)
    retained = {}
    used = 0

    def keep(name, payload):
        nonlocal used
        if used + len(payload) > max_bytes:
            raise ValueError("retained evidence exceeds byte ceiling; incomplete output kept")
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
        used += len(payload)
        retained[name] = file_sha256(path)

    keep("plan.json", plan_path.read_bytes())
    for name in source.spec.digest_sources:
        keep(f"source/{name}", (source.root / name).read_bytes())
    for name, directory in source.data_dirs.items():
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                keep(f"data/{name}/{path.relative_to(directory).as_posix()}", path.read_bytes())
    binary = (output / "elscata").resolve()
    argv = build_command(source, source.spec.programs["elscata"], toolchain, binary)
    completed = subprocess.run(argv, cwd=output, capture_output=True, check=False)
    keep("build-argv.json", json.dumps(argv).encode())
    keep("build.stdout", completed.stdout)
    keep("build.stderr", completed.stderr)
    if completed.returncode != 0 or not binary.is_file():
        raise RuntimeError("ELSEPA build failed; retained build evidence, no completion record")
    used += binary.stat().st_size
    if used > max_bytes:
        raise ValueError("binary exceeds evidence byte ceiling")
    retained["elscata"] = file_sha256(binary)
    panels = []
    for index, energy in enumerate(deck.energies_ev):
        single = replace(deck, energies_ev=(energy,))
        prefix = f"panels/{index:04d}"
        keep(f"{prefix}/deck.in", single.render().encode())
        run = run_program(
            binary,
            data_dirs=source.data_dirs,
            stdin_text=single.render(),
            outputs=single.output_names,
            timeout=120,
            keep_on_failure=True,
        )
        keep(f"{prefix}/stdout.txt", run.stdout.encode())
        keep(f"{prefix}/stderr.txt", run.stderr.encode())
        for name, payload in run.outputs.items():
            keep(f"{prefix}/{name}", payload)
        panels.append(parse_dcs(run.outputs[single.output_names[0]]))
    arrays = table_arrays(panels, energies_ev=deck.energies_ev)
    if not np.array_equal(arrays["mu"], native_mu):
        raise ValueError("fresh generator changed the pinned native angular grid")
    if deck.high_energy_factorization > 0:
        check_transport_scaling(arrays["energy_eV"], arrays["transport1_cm2"], z=deck.z)
    table = store(
        request,
        arrays,
        root=output / "tables",
        compiler=toolchain.version,
        source_origin=source.origin,
        upstream=source.spec.upstream,
    )
    for path in (table.path, table.path.with_suffix(".json")):
        used += path.stat().st_size
        retained[path.relative_to(output).as_posix()] = file_sha256(path)
    if used > max_bytes:
        raise ValueError("table exceeds evidence byte ceiling; no completion record")
    record = dict(
        schema="pyrite.elsepa-refinement-generation.v1",
        key=key,
        purpose="accuracy-only",
        generated=True,
        energy_nodes=len(panels),
        requested_energies_eV=list(deck.energies_ev),
        actual_panel_energies_eV=[panel.energy_ev for panel in panels],
        plan_sha256=file_sha256(plan_path),
        runner_sha256=file_sha256(Path(__file__)),
        compiler=toolchain.version,
        retained_bytes=used,
        files_sha256=retained,
        numerical_gate="unmeasured",
        headline=False,
        submission_ready=False,
        limitations=[
            "Native angular/inverse-CDF mesh remains unqualified",
            "Muffin-tin refinements require separate retained generation",
            "Fresh evaluations are not a numerical-convergence verdict",
            "Native Fortran energy rendering/parser tolerance is unchanged",
        ],
    )
    with (output / "generation.json").open("x") as stream:
        json.dump(record, stream, indent=1, allow_nan=False)
        stream.write("\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("key")
    parser.add_argument(
        "--output", type=Path, required=True, help="new isolated evidence/table directory"
    )
    parser.add_argument("--max-bytes", type=int, default=256 * 1024**2)
    args = parser.parse_args()
    result = generate(args.plan, args.key, args.output, max_bytes=args.max_bytes)
    print(json.dumps({k: result[k] for k in ("key", "energy_nodes", "numerical_gate", "headline")}))


if __name__ == "__main__":
    main()

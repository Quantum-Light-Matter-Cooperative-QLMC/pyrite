"""Probe native positron ELSEPA/SBETHE outputs against PyRITE's parsers (#276).

Requires gfortran and installed SBETHE sdbase. No Monte Carlo, table-store
writes, or release publication. This checks the generator format only;
transport, sampling, and independent physics validation remain separate.

    UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run python checks/positron_table_inventory.py

Use --out DIR to retain the native outputs, rendered decks, and provenance.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
from pyrite.montecarlo.transport.stopping import prepare_sbethe_stopping_table
from pyrite.xsgen._run import run_program
from pyrite.xsgen.elsepa.catalog import elemental_solid
from pyrite.xsgen.elsepa.deck import ElsepaDeck
from pyrite.xsgen.elsepa.parse import parse_dcs, table_arrays
from pyrite.xsgen.sbethe.catalog import catalog_material
from pyrite.xsgen.sbethe.deck import SbetheDeck
from pyrite.xsgen.sbethe.parse import (
    parse_integrated,
    parse_oscillator,
    parse_stopping,
)
from pyrite.xsgen.sbethe.parse import (
    table_arrays as stopping_arrays,
)
from pyrite.xsgen.sources import resolve_source, source_digest
from pyrite.xsgen.toolchain import build, find_toolchain


def main(argv: list[str] | None = None) -> int:
    """Run the native-output inventory; parser or subprocess failures raise."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="retain outputs and provenance here")
    args = parser.parse_args(argv)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
    compiler = find_toolchain()
    solid = elemental_solid("silicon")
    assert solid is not None
    energies = (1e3, 1e5, 1e7)
    silicon = catalog_material("silicon")
    jobs = {
        "elsepa": (
            ("si-free", ElsepaDeck.free_atom(14, energies, projectile="positron")),
            ("w-free", ElsepaDeck.free_atom(74, energies, projectile="positron")),
            (
                "si-muffin",
                ElsepaDeck.muffin_tin(
                    14, (1e3, 1e5, 1e6), radius_cm=solid.radius_cm, projectile="positron"
                ),
            ),
        ),
        "sbethe": (
            (
                "si",
                SbetheDeck(
                    silicon.key,
                    silicon.composition,
                    silicon.density_g_cm3,
                    silicon.mean_excitation_eV,
                    projectile="positron",
                ),
            ),
            # W is a transport element in compounds, not an elementary catalogue crystal.
            # Density and I are explicit run inputs, not a new catalogue material.
            (
                "w",
                SbetheDeck(
                    "tungsten",
                    {74: 1.0},
                    19.3,
                    1000 * TRANSPORT_ELEMENTS["W"]["J_keV"],
                    projectile="positron",
                ),
            ),
        ),
    }
    records = []
    for code, decks in jobs.items():
        source = resolve_source(code)
        binary = build(source, "elscata" if code == "elsepa" else "sbethe", toolchain=compiler)
        for name, deck in decks:
            outputs = (
                deck.output_names if code == "elsepa" else ("stp.dat", "asymptotic.dat", "OOS.dat")
            )
            run = run_program(
                binary,
                data_dirs=source.data_dirs,
                stdin_text=deck.render(),
                outputs=outputs,
                timeout=180,
            )
            if code == "elsepa":
                arrays = table_arrays(
                    (parse_dcs(run.outputs[n]) for n in outputs), energies_ev=deck.energies_ev
                )
                np.testing.assert_array_equal(arrays["angular_cdf"][:, 0], 0.0)
                np.testing.assert_array_equal(arrays["angular_cdf"][:, -1], 1.0)
            else:
                arrays = stopping_arrays(
                    parse_stopping(run.outputs["stp.dat"]),
                    parse_integrated(run.outputs["asymptotic.dat"]),
                    parse_oscillator(run.outputs["OOS.dat"]),
                )
                prepare_sbethe_stopping_table(arrays)
            record = {
                "code": code,
                "case": name,
                "upstream": source.spec.upstream,
                "source_sha256": source_digest(source),
                "compiler": compiler.version,
                "deck": deck.render(),
                "arrays": {k: list(v.shape) for k, v in arrays.items()},
            }
            records.append(record)
            print(f"{code}/{name}: parser and loader PASS", flush=True)
            if args.out is not None:
                for filename, content in run.outputs.items():
                    (args.out / f"{code}-{name}-{filename}").write_bytes(content)
                (args.out / "inventory.json").write_text(json.dumps(records, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

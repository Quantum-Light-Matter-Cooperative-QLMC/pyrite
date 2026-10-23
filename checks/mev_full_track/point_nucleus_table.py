"""Point-nucleus W ELSEPA table for the #439 control run.

Usage:
  point_nucleus_table.py OUT_NPZ

Geant4's Goudsmit-Saunderson Mott correction is built from the Mott DCS for a
point-like nucleus (``G4GSMottCorrection.hh``); PyRITE's production ELSEPA
tables use a Fermi charge distribution (``MNUCL 3``). This script reruns the
production W free-atom deck with ``MNUCL 1`` and nothing else changed, and
writes the sampler arrays without touching the table store. Running
``pyrite_bench.py`` with ``--elastic-npz`` on the result isolates the nuclear
model as the cause of the W 50/100 MeV exit-angle tail difference.

Validation: mev-full-track-transport
"""

import dataclasses
import sys

import numpy as np

from pyrite.xsgen._run import run_program
from pyrite.xsgen.elsepa.catalog import PRODUCTION_ENERGIES_EV, resolve_layer_tables
from pyrite.xsgen.elsepa.generate import element_request
from pyrite.xsgen.elsepa.parse import parse_dcs, table_arrays
from pyrite.xsgen.sources import resolve_source
from pyrite.xsgen.toolchain import build, find_toolchain

Z = 74


def main(argv):
    request, deck = element_request(Z, PRODUCTION_ENERGIES_EV)
    (production,) = resolve_layer_tables([("W", 0.0632)])
    if [t.key for t in production.tables] != [request.key]:
        raise RuntimeError("production W table is not the default free-atom deck")
    if deck.nuclear_model != 3:
        raise RuntimeError(f"expected a Fermi-nucleus production deck, got {deck.nuclear_model}")
    point = dataclasses.replace(deck, nuclear_model=1)
    source = resolve_source("elsepa", None)
    binary = build(source, "elscata", toolchain=find_toolchain())
    run = run_program(
        binary, data_dirs=source.data_dirs, stdin_text=point.render(), outputs=point.output_names
    )
    panels = [parse_dcs(run.outputs[name]) for name in point.output_names]
    arrays = table_arrays(panels, energies_ev=point.energies_ev)
    np.savez_compressed(argv[0], production_key=request.key, **arrays)


if __name__ == "__main__":
    main(sys.argv[1:])

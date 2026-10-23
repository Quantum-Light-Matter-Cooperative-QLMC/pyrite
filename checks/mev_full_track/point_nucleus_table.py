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

ELSEPA runs once per energy. One point-nucleus run over the whole 61-energy
grid returned a first transport cross section that stops falling above
~1 MeV (92x the Fermi value at 100 MeV, against 1.27x for a single-energy
run), so each panel is generated alone and the Fermi deck, run the same way,
must reproduce the production table before the point table is written.

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
    source = resolve_source("elsepa", None)
    binary = build(source, "elscata", toolchain=find_toolchain())

    def panels(nuclear_model):
        out = []
        for energy in deck.energies_ev:
            single = dataclasses.replace(deck, energies_ev=(energy,), nuclear_model=nuclear_model)
            run = run_program(
                binary,
                data_dirs=source.data_dirs,
                stdin_text=single.render(),
                outputs=single.output_names,
            )
            out.append(parse_dcs(run.outputs[single.output_names[0]]))
        return table_arrays(out, energies_ev=deck.energies_ev)

    fermi, point = panels(3), panels(1)
    reference = production.tables[0].arrays()
    for name in ("total_elastic_cm2", "transport1_cm2", "dcs_cm2_sr"):
        np.testing.assert_allclose(fermi[name], reference[name], rtol=1e-6, err_msg=name)
    ratio = point["transport1_cm2"] / fermi["transport1_cm2"]
    if not (np.all(ratio > 0.99) and np.all(ratio < 1.5) and np.all(np.diff(ratio) > -1e-3)):
        raise RuntimeError(f"point/Fermi transport ratio out of range: {ratio}")
    np.savez_compressed(argv[0], production_key=request.key, **point)


if __name__ == "__main__":
    main(sys.argv[1:])

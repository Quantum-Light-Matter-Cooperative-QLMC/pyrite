"""EEDL inner-shell ionization against NIST SRD 164 (Bote--Salvat) to 100 MeV.

For the inner oscillators of the shell-rate closure (Si, SiO2, MoS2), compares
the EEDL subshell cross sections the closure adopts with the free-atom DWBA/PWBA
cross sections of NIST SRD 164 (Llovet, Salvat and Powell; PENELOPE
``pdesiZZ.p08`` tables), and reports the PENELOPE density ratio ``rho`` and
the outer-shell scale ``N(E)`` with either source. Report only: no gate. The
SRD 164 tables are not redistributed; point ``--database`` at the
``NISTxyz/database`` directory extracted from the NIST164 distribution.

    uv run python checks/srd164_inner_shell_comparison.py --database DIR

Exit 0 on a report, 2 when the database is missing. Issue #439; backend: #92.

Validation: penelope-shell-rate-closure
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.transport import shell_rates as sr
from pyrite.xsgen.sbethe.catalog import catalog_material

KEYS = ("silicon", "sio2", "mos2")
ENERGIES_EV = (1e6, 1e7, 3e7, 1e8)
SRD_LABELS = ("K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5")
SYMBOL = {8: "O", 14: "Si", 16: "S", 42: "Mo"}


def _read_pdesi(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Energy grid (eV) and per-shell cross sections (cm^2) of one ``pdesiZZ.p08``."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    n_shells = int(lines[0].split()[1])
    rows = np.array([[float(v) for v in line.split()] for line in lines[2:]])
    return rows[:, 0], {SRD_LABELS[i]: rows[:, 1 + i] for i in range(n_shells)}


def _loglog(grid: np.ndarray, values: np.ndarray, energy: float) -> float:
    keep = values > 0.0
    return float(np.exp(np.interp(np.log(energy), np.log(grid[keep]), np.log(values[keep]))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args(argv)
    if not (args.database / "pdesi14.p08").is_file():
        print(f"SKIP: no SRD 164 pdesiZZ.p08 tables under {args.database}")
        return 2
    srd = {z: _read_pdesi(args.database / f"pdesi{z:02d}.p08") for z in SYMBOL}
    shells = config.load_atomic_shells()
    for key in KEYS:
        composition = catalog_material(key).composition
        joined = {
            z: {
                m.shell.label: m.eedl_labels
                for m in config.match_eedl_shells(SYMBOL[z], z, shells[z])
            }
            for z in composition
        }
        print(f"\n== {key}: EEDL/SRD164 (rho) per inner shell; N(E) with EEDL -> SRD164")
        for energy in ENERGIES_EV:
            closure = sr.catalog_shell_rate_closure(key, energy)
            raw, inner = closure.raw.per_shell, closure.inner
            outer_first = raw[~inner, 1].sum()
            replaced_first = 0.0
            cells = []
            for k, osc in enumerate(closure.raw.oscillators):
                if not inner[k]:
                    continue
                grid, sigma = srd[osc.atomic_number]
                labels = joined[osc.atomic_number][osc.label]
                if not all(label in sigma for label in labels):
                    # SRD 164 stops at M5; keep the EEDL rate for this shell.
                    replaced_first += closure.moments.per_shell[k, 1]
                    continue
                reference = composition[osc.atomic_number] * sum(
                    _loglog(grid, sigma[label], energy) for label in labels
                )
                eedl = closure.adopted_inner_cm2[k] / closure.density_ratio[k]
                scale = closure.density_ratio[k] * reference / raw[k, 0]
                replaced_first += scale * raw[k, 1]
                name = f"{SYMBOL[osc.atomic_number]}{osc.label}"
                cells.append(f"{name} {eedl / reference:.3f}({closure.density_ratio[k]:.2f})")
            n_srd = (closure.stopping_eV_cm2 - replaced_first) / outer_first
            print(
                f"{energy / 1e6:6g} MeV  N {closure.outer_scale:.3f} -> {n_srd:.3f}  "
                + "  ".join(cells)
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

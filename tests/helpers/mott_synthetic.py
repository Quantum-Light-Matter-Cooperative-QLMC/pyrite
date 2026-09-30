"""Write synthetic SRD 64-format Mott transport tables for the test suite.

PyRITE does not redistribute NIST SRD 64 (#263), so tests that exercise
``elastic_model="mott"`` read these instead. The numbers are synthetic, not
NIST data: each table is built so that the Mott calibration in
:func:`pyrite.montecarlo.transport.scattering._mott_alpha_table` recovers the
analytic Joy/Bishop screening parameter ``alpha = 3.4e-3 Z^0.67 / E_keV``:

    sigma_tr = sigma_Browning(Z, E) * 2 a [(1 + a) ln(1 + 1/a) - 1],  a = alpha(E)

Because ``log10 alpha`` is linear in ``log10 E``, interpolating these nodes is
exact, so a "mott" run over them reproduces the analytic screened-Rutherford
angles everywhere on the 10 eV - 100 MeV grid.

Regenerate with ``uv run python -m tests.helpers.mott_synthetic``.
"""

from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "mott_srd64_synthetic"
A0_SQ_CM2 = 2.8002852e-17
_NODES_PER_DECADE = 4
_E_MIN_EV = 10.0
_E_MAX_EV = 1.0e8


def synthetic_transport_cm2(z: float, energy_eV: np.ndarray) -> np.ndarray:
    """Return the synthetic transport cross section [cm^2] at ``energy_eV``."""
    e_keV = np.asarray(energy_eV, dtype=float) / 1e3
    z17 = z**1.7
    sqrt_e = np.sqrt(e_keV)
    sigma_el = 3.0e-18 * z17 / (e_keV + 0.005 * z17 * sqrt_e + 0.0007 * z * z / sqrt_e)
    a = 3.4e-3 * z**0.67 / e_keV
    return sigma_el * 2.0 * a * ((1.0 + a) * np.log1p(1.0 / a) - 1.0)


def write_tables(directory: Path = FIXTURE_DIR) -> list[Path]:
    """Write one synthetic table per transport element into ``directory``."""
    from pyrite.materials._transport_data import TRANSPORT_ELEMENTS

    decades = int(round(np.log10(_E_MAX_EV / _E_MIN_EV)))
    energies = np.logspace(
        np.log10(_E_MIN_EV), np.log10(_E_MAX_EV), decades * _NODES_PER_DECADE + 1
    )
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for element, params in sorted(TRANSPORT_ELEMENTS.items()):
        z = int(params["Z"])
        sigma = synthetic_transport_cm2(z, energies) / A0_SQ_CM2
        lines = [
            "SYNTHETIC TEST FIXTURE - NOT NIST SRD 64 DATA",
            "TRANSPORT CROSS SECTIONS",
            "in units of a0**2",
            "a0**2 = 2.8002852E-21 m**2",
            "",
            f"Atomic number: {z}",
            "",
            "No, Energy,Transport cross section",
            *(
                f"{i}, {e:.17g}, {s:.17e}"
                for i, (e, s) in enumerate(zip(energies, sigma, strict=True), 1)
            ),
        ]
        path = directory / f"DisplayCalcTCSTableFor{element}.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    return written


if __name__ == "__main__":
    for path in write_tables():
        print(path)

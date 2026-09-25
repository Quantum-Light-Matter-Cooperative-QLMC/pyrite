"""Bremsstrahlung cross-section sources against the Seltzer--Berger tables.

Compares PyRITE's EEDL and BremsLib ``dsigma/dk`` evaluations with Seltzer's
tabulation of the scaled electron bremsstrahlung cross section (Seltzer and
Berger, At. Data Nucl. Data Tables 35, 345 (1986); NBS ``BREME.DAT``, 1984),
which underlies ESTAR's radiative stopping powers and includes
electron-electron bremsstrahlung. Both sides are reduced to the scaled form

    chi(Z, T, kappa) = (beta**2 / Z**2) k dsigma/dk        [mb],  kappa = k/T,

and three quantities are compared: ``chi`` pointwise, the hard-photon cross
section ``sigma(kappa > kappa_c) = (Z**2/beta**2) integral(chi/kappa dkappa)``
and the scaled radiative first moment ``phi = integral(chi dkappa)``, both on the
Seltzer--Berger ``kappa`` nodes with the same trapezoid rule on either side.

Assumptions: the table is linearly interpolated in ``kappa`` and log-log in
``T`` between its nodes; model ``chi`` at ``kappa = 0`` is evaluated at
``kappa = 1e-3``, where ``chi`` is flat to well below the compared tolerances.
Limiting case: the Seltzer--Berger first moment reproduces ESTAR radiative
stopping, which is checked by the tests.

The table is read from EGSnrc's copy (``HEN_HOUSE/data/nist_brems.data``) at a
pinned commit and SHA-256. It is fetched on demand into the user data
directory, never packaged.

Validation: brem-source-comparison
"""

import hashlib
import os
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from pyrite._backend import _to_cpu
from pyrite.montecarlo.spectrum.brem import BremsstrahlungModel, _bremsstrahlung_dsigma_dk
from pyrite.montecarlo.spectrum.brem_bremslib import BremsLibBremsstrahlungTable
from pyrite.paths import user_data_dir

SELTZER_BERGER_URL = (
    "https://raw.githubusercontent.com/nrc-cnrc/EGSnrc/"
    "cfd08f097a61afcf65878710e2e7f63c7f9fd568/HEN_HOUSE/data/nist_brems.data"
)
SELTZER_BERGER_SHA256 = "1e931fc20b1605a617e3c6db1a6abf53b7408c5641619c1ece2d6bfcbb4194ff"
SELTZER_BERGER_ENV = "PYRITE_SELTZER_BERGER_TABLE"
ELECTRON_REST_ENERGY_MEV = 0.51099895
MB_PER_CM2 = 1e27
_KAPPA_FLOOR = 1e-3


class SeltzerBergerUnavailableError(FileNotFoundError):
    """Raised when the pinned Seltzer--Berger table is not installed."""


@dataclass(frozen=True, slots=True)
class SeltzerBergerTable:
    """Scaled bremsstrahlung cross sections ``chi[Z-1, kappa, T]`` in mb."""

    incident_energy_MeV: np.ndarray
    kappa: np.ndarray
    chi_mb: np.ndarray

    def chi(self, atomic_number: int, incident_energy_MeV: float) -> np.ndarray:
        """``chi`` on the table's ``kappa`` nodes, log-log in ``T`` between nodes."""
        log_T = np.log(self.incident_energy_MeV)
        panel = self.chi_mb[atomic_number - 1]
        return np.exp(
            [np.interp(np.log(incident_energy_MeV), log_T, np.log(row)) for row in panel]
        )


def parse_seltzer_berger(text: str) -> SeltzerBergerTable:
    """Parse Seltzer's ``BREME.DAT`` layout as distributed by EGSnrc.

    Header line, then the counts ``nT nK``, the ``nT`` kinetic energies in MeV,
    the ``nK`` reduced photon energies, a ``BREMX.DAT`` marker and, for
    ``Z = 1..100`` in turn, ``nK`` blocks of ``nT`` values of ``chi`` in mb.
    """
    lines = text.splitlines()
    n_energy, n_kappa = (int(value) for value in lines[1].split())
    values = np.array(
        [token for line in lines[2:] for token in line.split() if token != "BREMX.DAT"],
        dtype=float,
    )
    energy, kappa = values[:n_energy], values[n_energy : n_energy + n_kappa]
    chi = values[n_energy + n_kappa :]
    per_element = n_energy * n_kappa
    if chi.size != 100 * per_element:
        raise ValueError(f"expected {100 * per_element} chi values, found {chi.size}")
    if not (np.all(np.diff(energy) > 0) and np.all(np.diff(kappa) > 0)):
        raise ValueError("Seltzer-Berger energy and kappa grids must ascend")
    if kappa[0] != 0.0 or kappa[-1] != 1.0 or np.any(chi <= 0.0):
        raise ValueError("Seltzer-Berger table must span kappa 0..1 with positive chi")
    return SeltzerBergerTable(energy, kappa, chi.reshape(100, n_kappa, n_energy))


def default_table_path() -> Path:
    """Where the fetched table lives unless ``PYRITE_SELTZER_BERGER_TABLE`` overrides it."""
    override = os.environ.get(SELTZER_BERGER_ENV)
    if override:
        return Path(override)
    return user_data_dir() / "validation" / "seltzer-berger" / "nist_brems.data"


def _verified(data: bytes, origin: object) -> bytes:
    digest = hashlib.sha256(data).hexdigest()
    if digest != SELTZER_BERGER_SHA256:
        raise ValueError(f"{origin} has SHA-256 {digest}; expected {SELTZER_BERGER_SHA256}")
    return data


def load_seltzer_berger(path: Path | None = None, *, download: bool = False) -> SeltzerBergerTable:
    """Load the pinned table, fetching it first when ``download`` is set."""
    path = default_table_path() if path is None else path
    if not path.exists():
        if not download:
            raise SeltzerBergerUnavailableError(
                f"Seltzer-Berger table not found at {path}; fetch it with "
                "`uv run python checks/brem_source_comparison.py --download` "
                f"or set {SELTZER_BERGER_ENV}"
            )
        with urlopen(SELTZER_BERGER_URL, timeout=60) as response:
            data = _verified(response.read(), SELTZER_BERGER_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return parse_seltzer_berger(_verified(path.read_bytes(), path).decode("ascii"))


def beta_squared(incident_energy_MeV: float) -> float:
    """Electron ``beta**2`` at kinetic energy ``T``."""
    gamma = 1.0 + incident_energy_MeV / ELECTRON_REST_ENERGY_MEV
    return 1.0 - 1.0 / gamma**2


def model_chi(
    element: str,
    atomic_number: int,
    incident_energy_MeV: float,
    kappa: Sequence[float] | np.ndarray,
    *,
    cross_section_model: BremsstrahlungModel,
    bremslib_tables: Mapping[str, BremsLibBremsstrahlungTable] | None = None,
) -> np.ndarray:
    """Scaled ``chi`` [mb] from PyRITE's production ``dsigma/dk`` evaluation."""
    kappa = np.maximum(np.asarray(kappa, dtype=float), _KAPPA_FLOOR)
    photon_eV = kappa * incident_energy_MeV * 1e6
    with warnings.catch_warnings():
        # Out-of-range fallbacks are reported through the comparison itself.
        warnings.simplefilter("ignore", RuntimeWarning)
        dsigma_dk = _bremsstrahlung_dsigma_dk(
            element,
            np.array([incident_energy_MeV * 1e3]),
            photon_eV,
            cross_section_model=cross_section_model,
            bremslib_tables=bremslib_tables,
        )
    dsigma_dk = np.asarray(_to_cpu(dsigma_dk), dtype=float)[0]
    return beta_squared(incident_energy_MeV) / atomic_number**2 * photon_eV * dsigma_dk * MB_PER_CM2


@dataclass(frozen=True, slots=True)
class SourceComparison:
    """One element and incident energy: model over Seltzer--Berger ratios."""

    model: str
    element: str
    atomic_number: int
    incident_energy_MeV: float
    kappa: np.ndarray
    chi_ratio: np.ndarray
    hard_cross_section_ratio: float
    first_moment_ratio: float


def _hard_cross_section(kappa: np.ndarray, chi: np.ndarray, kappa_cut: float) -> float:
    keep = kappa >= kappa_cut
    return float(np.trapezoid(chi[keep] / kappa[keep], kappa[keep]))


def compare_sources(
    table: SeltzerBergerTable,
    elements: Iterable[tuple[str, int]],
    incident_energies_MeV: Iterable[float],
    *,
    models: Sequence[BremsstrahlungModel] = ("eedl", "bremslib"),
    bremslib_tables: Mapping[str, BremsLibBremsstrahlungTable] | None = None,
    hard_kappa_cut: float = 0.05,
) -> list[SourceComparison]:
    """Compare each model with the table on its own ``kappa`` nodes."""
    energies = tuple(incident_energies_MeV)
    results = []
    for element, atomic_number in elements:
        for energy in energies:
            reference = table.chi(atomic_number, energy)
            reference_hard = _hard_cross_section(table.kappa, reference, hard_kappa_cut)
            reference_moment = float(np.trapezoid(reference, table.kappa))
            for model in models:
                chi = model_chi(
                    element,
                    atomic_number,
                    energy,
                    table.kappa,
                    cross_section_model=model,
                    bremslib_tables=bremslib_tables,
                )
                results.append(
                    SourceComparison(
                        model=model,
                        element=element,
                        atomic_number=atomic_number,
                        incident_energy_MeV=energy,
                        kappa=table.kappa,
                        chi_ratio=chi / reference,
                        hard_cross_section_ratio=_hard_cross_section(
                            table.kappa, chi, hard_kappa_cut
                        )
                        / reference_hard,
                        first_moment_ratio=float(np.trapezoid(chi, table.kappa))
                        / reference_moment,
                    )
                )
    return results

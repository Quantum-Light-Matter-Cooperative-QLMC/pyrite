"""EEDL shell ionization and characteristic production against Bote--Salvat.

Compares PyRITE's fetched EEDL MF=23 subshell electro-ionization cross
sections with the Bote--Salvat analytical formulas (D. Bote et al., At. Data
Nucl. Data Tables 95, 871 (2009)) for K, L1--L3 and M1--M5 shells, Z = 1--99,
fitted to their distorted-wave and plane-wave Born calculations (Phys. Rev. A
77, 042701 (2008)). The parameters are read from NIST's public-domain
BoteSalvatICX.jl (``src/xione.jl``), pinned by commit and SHA-256 and fetched on
demand, never packaged.

For overvoltage ``U = E/E_i`` with ``E_i`` the edge energy,

    sigma = 4 pi a0**2 (U - 1) (f(U)/U)**2,
    f(U) = A1 + A2 U + (A3 + (A4 + A5/(1+U)**2)/(1+U)**2)/(1+U),      U <= 16,

    sigma = 4 pi a0**2 (A_nlj/beta**2) U/(U + B_e)
            [(ln X**2 - beta**2)(1 + g1/X) + g2 + g3 sqrt(mc**2/(E + mc**2)) + g4/X],
    X = sqrt(E (E + 2 mc**2))/mc**2,                                   U > 16.

The two sides use different binding energies, so each shell is compared both at
the same incident energy and with Bote--Salvat evaluated at EEDL's own binding
energy, which isolates the shape of ``sigma(U)`` from the edge convention.
Characteristic production ``sum_i sigma_i Y_i,line`` uses PyRITE's own xraydb
relaxation and L-shell Coster--Kronig yields ``Y`` for both, so production
ratios measure the ionization model only.

Assumptions: free atoms; the Bote--Salvat fit is accurate to about 1 % against
its own DWBA/PWBA calculation; shells Bote--Salvat does not tabulate keep the EEDL
value in production sums. Limiting case: the transcription reproduces the
original ``xion.f`` values that BoteSalvatICX.jl tests against.

Validation: eedl-shell-ionization-comparison
"""

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyrite.montecarlo.eedl_ionization import EEDL_SUBSHELL_LABELS, load_eedl_shell_ionization
from pyrite.montecarlo.spectrum.characteristic import load_characteristic_cross_sections
from pyrite.paths import user_data_dir

from ._pinned import read_pinned

BOTE_SALVAT_URL = (
    "https://raw.githubusercontent.com/usnistgov/BoteSalvatICX.jl/"
    "8520cf5d002b11c3cf6669ebd5fedbb3de8d1fdb/src/xione.jl"
)
BOTE_SALVAT_SHA256 = "d0bd0d3ddca915a785a2562dfe29d6158b02d4a0e0330f54b784f8bba0b41599"
BOTE_SALVAT_ENV = "PYRITE_BOTE_SALVAT_TABLE"
BOTE_SALVAT_SHELLS = ("K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5")
_BOHR_RADIUS_CM = 5.291772108e-9
_ELECTRON_REST_ENERGY_EV = 5.10998918e5  # the value xion.f uses
_SHELL_FAMILY = {"K": "K", "L1": "L1", "L2": "L2", "L3": "L3"}
LINE_FAMILIES = ("K", "L1", "L2", "L3", "M")


class BoteSalvatUnavailableError(FileNotFoundError):
    """Raised when the pinned Bote--Salvat parameter file is not installed."""


@dataclass(frozen=True, slots=True)
class BoteSalvatElement:
    """Bote--Salvat parameters for the shells ``K, L1, ...`` of one element."""

    atomic_number: int
    b_e: np.ndarray
    a_nlj: np.ndarray
    g: np.ndarray
    edge_eV: np.ndarray
    a: np.ndarray

    def cross_section_cm2(self, shell: str, energy_eV: float, edge_eV: float | None = None):
        """Ionization cross section of ``shell`` at ``energy_eV`` [cm²].

        Validation: eedl-shell-ionization-comparison
        """
        index = BOTE_SALVAT_SHELLS.index(shell)
        edge = float(self.edge_eV[index]) if edge_eV is None else edge_eV
        overvoltage = energy_eV / edge
        if overvoltage <= 1.0:
            return 0.0
        if overvoltage <= 16.0:
            a = self.a[index]
            inverse = 1.0 / (1.0 + overvoltage)
            fit = (
                a[0]
                + a[1] * overvoltage
                + inverse * (a[2] + inverse**2 * (a[3] + inverse**2 * a[4]))
            )
            reduced = (overvoltage - 1.0) * (fit / overvoltage) ** 2
        else:
            mc2 = _ELECTRON_REST_ENERGY_EV
            beta2 = energy_eV * (energy_eV + 2.0 * mc2) / (energy_eV + mc2) ** 2
            x = np.sqrt(energy_eV * (energy_eV + 2.0 * mc2)) / mc2
            g = self.g[index]
            fit = (
                (2.0 * np.log(x) - beta2) * (1.0 + g[0] / x)
                + g[1]
                + g[2] * np.sqrt(mc2 / (energy_eV + mc2))
                + g[3] / x
            )
            reduced = (
                self.a_nlj[index] / beta2 * overvoltage / (overvoltage + self.b_e[index]) * fit
            )
        return 4.0 * np.pi * _BOHR_RADIUS_CM**2 * float(reduced)

    @property
    def shells(self) -> tuple[str, ...]:
        return BOTE_SALVAT_SHELLS[: self.edge_eV.size]


def parse_bote_salvat(text: str) -> dict[int, BoteSalvatElement]:
    """Parse the ``BoteSalvatElectron`` tuple of BoteSalvatICX.jl ``xione.jl``.

    Each ``BoteSalvatElementDatum(z, Be, Anlj, G, edge, A)`` holds comma
    vectors ``Be``, ``Anlj`` and ``edge`` (one value per shell) and
    semicolon-separated matrices ``G`` (four columns) and ``A`` (five).

    Validation: eedl-shell-ionization-comparison
    """
    start = text.index("const BoteSalvatElectron")
    body = text[start : text.index('"""', start)]
    elements = {}
    for match in re.finditer(r"BoteSalvatElementDatum\((\d+),(.*?)\)\s*[,\n]", body, re.S):
        groups = re.findall(r"\[([^\]]*)\]", match.group(2))
        if len(groups) != 5:
            raise ValueError(f"Bote-Salvat datum {match.group(1)} has {len(groups)} fields")
        rows = [
            np.array(
                [
                    [float(v) for v in re.split(r"[,\s]+", row.strip()) if v]
                    for row in group.split(";")
                ]
            )
            for group in groups
        ]
        b_e, a_nlj, g, edge, a = rows[0][0], rows[1][0], rows[2], rows[3][0], rows[4]
        n_shell = edge.size
        if b_e.size != n_shell or a_nlj.size != n_shell or g.shape != (n_shell, 4):
            raise ValueError(f"Bote-Salvat datum {match.group(1)} has inconsistent shapes")
        if a.shape != (n_shell, 5) or n_shell > len(BOTE_SALVAT_SHELLS):
            raise ValueError(f"Bote-Salvat datum {match.group(1)} has inconsistent shapes")
        z = int(match.group(1))
        elements[z] = BoteSalvatElement(z, b_e, a_nlj, g, edge, a)
    if sorted(elements) != list(range(1, len(elements) + 1)):
        raise ValueError("Bote-Salvat elements must run 1..N without gaps")
    return elements


def default_parameter_path() -> Path:
    """Where the fetched file lives unless ``PYRITE_BOTE_SALVAT_TABLE`` overrides it."""
    override = os.environ.get(BOTE_SALVAT_ENV)
    if override:
        return Path(override)
    return user_data_dir() / "validation" / "bote-salvat" / "xione.jl"


def load_bote_salvat(
    path: Path | None = None, *, download: bool = False
) -> dict[int, BoteSalvatElement]:
    """Load the pinned parameters, fetching them first when ``download`` is set."""
    data = read_pinned(
        default_parameter_path() if path is None else path,
        BOTE_SALVAT_URL,
        BOTE_SALVAT_SHA256,
        download=download,
        missing=BoteSalvatUnavailableError,
        hint="fetch the Bote-Salvat parameters with "
        "`uv run python checks/shell_ionization_comparison.py --download` "
        f"or set {BOTE_SALVAT_ENV}",
    )
    return parse_bote_salvat(data.decode("utf-8"))


@dataclass(frozen=True, slots=True)
class ShellComparison:
    """EEDL over Bote--Salvat for one shell on an overvoltage grid."""

    element: str
    shell: str
    eedl_binding_eV: float
    bote_salvat_edge_eV: float
    overvoltage: np.ndarray
    same_energy_ratio: np.ndarray
    eedl_edge_ratio: np.ndarray
    native_node_ratio: np.ndarray
    neighboring_node_ratio: np.ndarray


def compare_shells(
    parameters: dict[int, BoteSalvatElement],
    element: str,
    atomic_number: int,
    overvoltages: Sequence[float],
) -> list[ShellComparison]:
    """Compare every shell both models tabulate.

    ``overvoltage`` is relative to the Bote--Salvat edge for the same-energy
    ratio and to the EEDL binding energy for the EEDL-edge ratio. EEDL is
    interpolated lin-lin as in production; ``native_node_ratio`` repeats the
    EEDL-edge comparison at EEDL's own nodes with ``1 < U <= max(overvoltages)``,
    separating tabulation from interpolation.

    Validation: eedl-shell-ionization-comparison
    """
    bote = parameters[atomic_number]
    tables = {
        EEDL_SUBSHELL_LABELS[table.shell_designator]: table
        for table in load_eedl_shell_ionization(element)
    }
    grid = np.asarray(overvoltages, dtype=float)
    results = []
    for shell in bote.shells:
        table = tables.get(shell)
        if table is None:
            continue
        edge = float(bote.edge_eV[BOTE_SALVAT_SHELLS.index(shell)])
        binding = table.binding_energy_eV

        def eedl(energy, table=table):
            return float(np.interp(energy, table.projectile_energy_eV, table.cross_section_cm2))

        same = [eedl(u * edge) / bote.cross_section_cm2(shell, u * edge) for u in grid]
        own = [
            eedl(u * binding) / bote.cross_section_cm2(shell, u * binding, binding) for u in grid
        ]
        nodes = table.projectile_energy_eV / binding
        keep = (nodes > 1.0) & (nodes <= grid.max())
        node_ratio = np.array(
            [
                value / bote.cross_section_cm2(shell, energy, binding)
                if energy > binding
                else np.nan
                for energy, value in zip(
                    table.projectile_energy_eV, table.cross_section_cm2, strict=True
                )
            ]
        )
        native = node_ratio[keep]
        upper = np.searchsorted(table.projectile_energy_eV, grid * binding, side="right")
        lower = np.clip(upper - 1, 0, node_ratio.size - 1)
        upper = np.clip(upper, 0, node_ratio.size - 1)
        neighboring = np.column_stack((node_ratio[lower], node_ratio[upper]))
        results.append(
            ShellComparison(
                element,
                shell,
                binding,
                edge,
                grid,
                np.array(same),
                np.array(own),
                native,
                neighboring,
            )
        )
    return results


def _line_family(shell: str) -> str | None:
    if shell in _SHELL_FAMILY:
        return _SHELL_FAMILY[shell]
    return "M" if shell.startswith("M") else None


@dataclass(frozen=True, slots=True)
class ProductionComparison:
    """EEDL over Bote--Salvat characteristic production per line family."""

    element: str
    incident_energy_eV: np.ndarray
    thick_target_energy_eV: np.ndarray
    ratio: dict[str, np.ndarray]
    thick_target_ratio: dict[str, np.ndarray]
    k_edge_eV: float | None


def _bethe_stopping_weight(atomic_number: int, energy_eV: np.ndarray) -> np.ndarray:
    """Relative ``1/S(E)`` from the relativistic Bethe formula, for weighting only.

    ``I = 9.76 Z + 58.8 Z**-0.19`` eV (Berger and Seltzer). The absolute scale
    cancels in a ratio of yields; the logarithm is floored where the formula
    fails far below ``I``.
    """
    mc2 = _ELECTRON_REST_ENERGY_EV
    mean_excitation = 9.76 * atomic_number + 58.8 * atomic_number**-0.19
    tau = energy_eV / mc2
    beta2 = 1.0 - 1.0 / (1.0 + tau) ** 2
    log_term = np.log(tau**2 * (tau + 2.0) / (2.0 * (mean_excitation / mc2) ** 2)) + (
        1.0 - beta2 + (tau**2 / 8.0 - (2.0 * tau + 1.0) * np.log(2.0)) / (tau + 1.0) ** 2
    )
    return beta2 / np.maximum(log_term, 1.0)


def compare_production(
    parameters: dict[int, BoteSalvatElement],
    element: str,
    atomic_number: int,
    incident_energies_eV: Sequence[float],
    thick_target_energies_eV: Iterable[float] = (),
) -> ProductionComparison:
    """Production ``sum_i sigma_i Y_i,line`` summed per line family, EEDL/Bote--Salvat.

    Thick-target ratios integrate each production cross section over
    ``dE/S(E)`` from threshold to the beam energy (continuous slowing down,
    approximate Bethe stopping, no backscatter or absorption).

    Validation: eedl-shell-ionization-comparison
    """
    bote = parameters[atomic_number]
    table = load_characteristic_cross_sections(element)
    shells = table.ionization_shell_labels
    families = [_line_family(shell) for shell in table.line_initial_shell]

    def cross_sections(energy_eV: float) -> tuple[np.ndarray, np.ndarray]:
        eedl = np.array(
            [
                np.interp(energy_eV, grid, values, left=0.0, right=float(values[-1]))
                for grid, values in zip(
                    table.projectile_energy_eV_by_shell,
                    table.ionization_cross_sections_cm2_by_shell,
                    strict=True,
                )
            ]
        )
        bote_values = eedl.copy()
        for index, shell in enumerate(shells):
            if shell in bote.shells:
                bote_values[index] = bote.cross_section_cm2(
                    shell, energy_eV, float(table.shell_binding_energy_eV[index])
                )
        return eedl, bote_values

    def production(energy_eV: float) -> tuple[dict[str, float], dict[str, float]]:
        eedl, bote_values = cross_sections(energy_eV)
        eedl_lines = eedl @ table.line_yield_per_vacancy
        bote_lines = bote_values @ table.line_yield_per_vacancy
        out_eedl, out_bote = {}, {}
        for family in LINE_FAMILIES:
            chosen = np.array([f == family for f in families])
            out_eedl[family] = float(eedl_lines[chosen].sum())
            out_bote[family] = float(bote_lines[chosen].sum())
        return out_eedl, out_bote

    def ratio(e: float, b: float) -> float:
        return e / b if b > 0.0 else np.nan

    energies = np.asarray(incident_energies_eV, dtype=float)
    thick = np.asarray(tuple(thick_target_energies_eV), dtype=float)
    ratios = {family: [] for family in LINE_FAMILIES}
    for energy in energies:
        eedl, bote_values = production(float(energy))
        for family in LINE_FAMILIES:
            ratios[family].append(ratio(eedl[family], bote_values[family]))
    thick_ratios = {family: [] for family in LINE_FAMILIES}
    for beam in thick:
        grid = np.geomspace(10.0, beam, 4000)
        weight = _bethe_stopping_weight(atomic_number, grid)
        samples = [production(float(energy)) for energy in grid]
        for family in LINE_FAMILIES:
            eedl_yield = np.trapezoid([s[0][family] for s in samples] * weight, grid)
            bote_yield = np.trapezoid([s[1][family] for s in samples] * weight, grid)
            thick_ratios[family].append(ratio(eedl_yield, bote_yield))
    return ProductionComparison(
        element,
        energies,
        thick,
        {family: np.array(values) for family, values in ratios.items()},
        {family: np.array(values) for family, values in thick_ratios.items()},
        float(table.shell_binding_energy_eV[shells.index("K")]) if "K" in shells else None,
    )

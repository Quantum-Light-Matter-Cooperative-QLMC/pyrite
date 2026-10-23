"""Bote--Salvat K/L/M electron-impact shell-ionization cross sections.

The analytical formulas of D. Bote et al., At. Data Nucl. Data Tables 95, 871
(2009), fitted to the distorted-wave and plane-wave Born calculations of Bote
and Salvat, Phys. Rev. A 77, 042701 (2008), for the K, L1--L3 and M1--M5 shells
of Z = 1--99. The parameters are NIST's public-domain BoteSalvatICX.jl
(``src/xione.jl``), an opt-in fetched dataset (``pyrite tables fetch
bote-salvat``) pinned by commit and SHA-256.

For overvoltage ``U = E/E_i`` with ``E_i`` the shell edge energy,

    sigma = 4 pi a0**2 (U - 1) (f(U)/U)**2,
    f(U) = A1 + A2 U + (A3 + (A4 + A5/(1+U)**2)/(1+U)**2)/(1+U),      U <= 16,

    sigma = 4 pi a0**2 (A_nlj/beta**2) U/(U + B_e)
            [(ln X**2 - beta**2)(1 + g1/X) + g2 + g3 sqrt(mc**2/(E + mc**2)) + g4/X],
    X = sqrt(E (E + 2 mc**2))/mc**2,                                   U > 16.

Assumptions: free neutral atoms; the fit reproduces its own DWBA/PWBA
calculation to about 1 %. Limiting cases: zero at and below the edge, linear
in ``U - 1`` just above it, and a logarithmic Bethe rise at high energy.

Validation: eedl-shell-ionization-comparison

Validation: bote-salvat-ionization-backend
"""

import re
from dataclasses import dataclass
from functools import cache

import numpy as np

from ..datasets import BOTE_SALVAT, DatasetMismatchError, file_sha256, require_dataset

BOTE_SALVAT_SHA256 = BOTE_SALVAT.sha256
BOTE_SALVAT_SHELLS = ("K", "L1", "L2", "L3", "M1", "M2", "M3", "M4", "M5")
#: Overvoltage where the fit switches from the DWBA to the PWBA branch.
BOTE_SALVAT_BRANCH_OVERVOLTAGE = 16.0
_BOHR_RADIUS_CM = 5.291772108e-9
_ELECTRON_REST_ENERGY_EV = 5.10998918e5  # the value xion.f uses


@dataclass(frozen=True, slots=True)
class BoteSalvatElement:
    """Bote--Salvat parameters for the shells ``K, L1, ...`` of one element."""

    atomic_number: int
    b_e: np.ndarray
    a_nlj: np.ndarray
    g: np.ndarray
    edge_eV: np.ndarray
    a: np.ndarray

    def cross_sections_cm2(self, shell: str, energy_eV, edge_eV: float | None = None) -> np.ndarray:
        """Ionization cross sections of ``shell`` at each of ``energy_eV`` [cm²].

        ``edge_eV`` replaces the fit's own edge in ``U`` only; ``beta`` and
        ``X`` always use the actual incident energy.

        Validation: eedl-shell-ionization-comparison

        Validation: bote-salvat-ionization-backend
        """
        index = BOTE_SALVAT_SHELLS.index(shell)
        edge = float(self.edge_eV[index]) if edge_eV is None else float(edge_eV)
        energy = np.asarray(energy_eV, dtype=float)
        overvoltage = energy / edge
        a = self.a[index]
        g = self.g[index]
        mc2 = _ELECTRON_REST_ENERGY_EV
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = 1.0 / (1.0 + overvoltage)
            fit = (
                a[0]
                + a[1] * overvoltage
                + inverse * (a[2] + inverse**2 * (a[3] + inverse**2 * a[4]))
            )
            low = (overvoltage - 1.0) * (fit / overvoltage) ** 2
            beta2 = energy * (energy + 2.0 * mc2) / (energy + mc2) ** 2
            x = np.sqrt(energy * (energy + 2.0 * mc2)) / mc2
            bethe = (
                (2.0 * np.log(x) - beta2) * (1.0 + g[0] / x)
                + g[1]
                + g[2] * np.sqrt(mc2 / (energy + mc2))
                + g[3] / x
            )
            high = self.a_nlj[index] / beta2 * overvoltage / (overvoltage + self.b_e[index]) * bethe
        reduced = np.where(overvoltage <= BOTE_SALVAT_BRANCH_OVERVOLTAGE, low, high)
        return 4.0 * np.pi * _BOHR_RADIUS_CM**2 * np.where(overvoltage > 1.0, reduced, 0.0)

    def cross_section_cm2(self, shell: str, energy_eV: float, edge_eV: float | None = None):
        """Ionization cross section of ``shell`` at one ``energy_eV`` [cm²].

        Validation: eedl-shell-ionization-comparison
        """
        return float(self.cross_sections_cm2(shell, energy_eV, edge_eV))

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


@cache
def load_bote_salvat_parameters() -> dict[int, BoteSalvatElement]:
    """Parse the installed, checksum-verified Bote--Salvat parameter file.

    Raises
    ------
    pyrite.datasets.DatasetNotFoundError
        If it is not installed; the message names ``pyrite tables fetch bote-salvat``.
    pyrite.datasets.DatasetMismatchError
        If the installed bytes differ from the pinned file.
    """
    path = require_dataset(BOTE_SALVAT.name)
    actual = file_sha256(path)
    if actual != BOTE_SALVAT_SHA256:
        raise DatasetMismatchError(
            f"Bote-Salvat parameter checksum mismatch at {path}: expected "
            f"{BOTE_SALVAT_SHA256}, got {actual}; remove it and run "
            "`pyrite tables fetch bote-salvat`"
        )
    return parse_bote_salvat(path.read_text(encoding="utf-8"))


def bote_salvat_energy_grid(edge_eV: float, max_energy_eV: float, per_decade: int = 64):
    """Incident-energy nodes for lin-lin tabulation of one Bote--Salvat shell.

    Nodes are geometric in ``U - 1`` from ``1e-4`` to 1 and geometric in ``E``
    above, so the threshold onset and the Bethe rise are both resolved. The
    grid starts at the edge (cross section zero), ends exactly at
    ``max_energy_eV``, and carries the branch switch
    ``U = 16`` as the pair ``16`` and ``16 (1 + 1e-12)``, so interpolation never
    blends the two branches.

    Validation: bote-salvat-ionization-backend
    """
    if not edge_eV > 0.0 or not max_energy_eV > 2.0 * edge_eV:
        raise ValueError("Bote-Salvat grid needs 0 < 2 edge < max energy")
    onset = 1.0 + np.geomspace(1.0e-4, 1.0, 4 * per_decade + 1)
    decades = np.log10(max_energy_eV / (2.0 * edge_eV))
    rise = 2.0 * np.geomspace(
        1.0, max_energy_eV / (2.0 * edge_eV), int(np.ceil(decades * per_decade)) + 1
    )
    branch = BOTE_SALVAT_BRANCH_OVERVOLTAGE
    overvoltage = np.unique(
        np.concatenate(([1.0], onset, rise, [branch, branch * (1.0 + 1.0e-12)]))
    )
    energy = edge_eV * overvoltage
    # The last node is exactly ``max_energy_eV``: a rounded rebuild could fall
    # one ulp short and score zero at the end of the EEDL row it replaces.
    energy = energy[energy < max_energy_eV * (1.0 - 1.0e-9)]
    return np.append(energy, max_energy_eV)


__all__ = [
    "BOTE_SALVAT_BRANCH_OVERVOLTAGE",
    "BOTE_SALVAT_SHA256",
    "BOTE_SALVAT_SHELLS",
    "BoteSalvatElement",
    "bote_salvat_energy_grid",
    "load_bote_salvat_parameters",
    "parse_bote_salvat",
]

"""Parse native-grid differential cross sections written by ELSEPA.

Validation: elsepa-vendor-reference
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

_SCALARS = {
    "energy_ev": r"Kinetic energy\s*=\s*([0-9.E+-]+)\s*eV",
    "total_elastic_cm2": r"Total elastic cross section\s*=\s*([0-9.E+-]+)\s*cm\*\*2",
    "transport1_cm2": r"1st transport cross section\s*=\s*([0-9.E+-]+)\s*cm\*\*2",
    "transport2_cm2": r"2nd transport cross section\s*=\s*([0-9.E+-]+)\s*cm\*\*2",
    "absorption_cm2": r"Absorption cross section\s*=\s*([0-9.E+-]+)\s*cm\*\*2",
}


@dataclass(frozen=True)
class ElsepaResult:
    """One energy panel from an ELSEPA ``dcs_*.dat`` output."""

    energy_ev: float
    theta_deg: np.ndarray
    mu: np.ndarray
    dcs_cm2_sr: np.ndarray
    dcs_a0_2_sr: np.ndarray
    sherman: np.ndarray
    relative_error: np.ndarray
    total_elastic_cm2: float
    transport1_cm2: float
    transport2_cm2: float
    absorption_cm2: float


def _scalar(text: str, name: str, *, default: float | None = None) -> float:
    match = re.search(_SCALARS[name], text)
    if match is None:
        if default is not None:
            return default
        raise ValueError(f"ELSEPA output is missing {name}")
    return float(match.group(1).replace("D", "E"))


def parse_dcs(data: str | bytes) -> ElsepaResult:
    """Parse one complete vendor-format ``dcs_*.dat`` file."""
    text = data.decode("ascii") if isinstance(data, bytes) else data
    rows: list[list[float]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"ELSEPA data row has {len(fields)} columns, expected 6")
        try:
            rows.append([float(field.replace("D", "E")) for field in fields])
        except ValueError as exc:
            raise ValueError(f"ELSEPA data row is not numeric: {line!r}") from exc
    if len(rows) < 2:
        raise ValueError("ELSEPA output contains fewer than two angular rows")
    values = np.asarray(rows, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("ELSEPA output contains non-finite values")
    theta, mu, dcs_cm2, dcs_a0, sherman, error = values.T
    if np.any(np.diff(theta) <= 0) or theta[0] < 0 or theta[-1] > 180:
        raise ValueError("ELSEPA theta grid must increase from 0 to 180 degrees")
    if np.any(np.diff(mu) < 0) or mu[0] < 0 or mu[-1] > 1:
        raise ValueError("ELSEPA mu grid must be monotone within [0, 1]")
    if np.any(dcs_cm2 <= 0) or np.any(dcs_a0 <= 0) or np.any(error < 0):
        raise ValueError("ELSEPA cross sections must be positive and errors non-negative")
    return ElsepaResult(
        energy_ev=_scalar(text, "energy_ev"),
        theta_deg=theta,
        mu=mu,
        dcs_cm2_sr=dcs_cm2,
        dcs_a0_2_sr=dcs_a0,
        sherman=sherman,
        relative_error=error,
        total_elastic_cm2=_scalar(text, "total_elastic_cm2"),
        transport1_cm2=_scalar(text, "transport1_cm2"),
        transport2_cm2=_scalar(text, "transport2_cm2"),
        # ``elscata`` writes this line only for MABS > 0. With no absorption
        # potential selected there is no absorption, so its cross section is
        # zero rather than missing.
        absorption_cm2=_scalar(text, "absorption_cm2", default=0.0),
    )


def _normalized_cdf(panel: ElsepaResult) -> np.ndarray:
    """Integrate the DCS over ``mu=(1-cos(theta))/2`` and normalize it.

    For an azimuthally symmetric DCS, ``dOmega = 4*pi*dmu``. The constant
    cancels in ``F(mu)=integral_0^mu DCS(u)du / integral_0^1 DCS(u)du``.
    Trapezoidal integration stays on ELSEPA's native angular grid. The exact
    endpoint limits are ``F(0)=0`` and ``F(1)=1``.

    Validation: elsepa-vendor-reference
    """
    increments = 0.5 * (panel.dcs_cm2_sr[:-1] + panel.dcs_cm2_sr[1:]) * np.diff(panel.mu)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
        raise ValueError("ELSEPA DCS has no positive finite angular integral")
    cdf /= cdf[-1]
    cdf[-1] = 1.0
    return cdf


def table_arrays(
    results: Iterable[ElsepaResult], *, energies_ev: Iterable[float] | None = None
) -> dict[str, np.ndarray]:
    """Stack parsed energy panels into the native-grid table-store schema."""
    panels = sorted(results, key=lambda panel: panel.energy_ev)
    if not panels:
        raise ValueError("cannot build an ELSEPA table without panels")
    reference = panels[0]
    for panel in panels[1:]:
        if not np.array_equal(panel.theta_deg, reference.theta_deg) or not np.array_equal(
            panel.mu, reference.mu
        ):
            raise ValueError("ELSEPA energy panels use different angular grids")
    energy_grid = np.asarray([panel.energy_ev for panel in panels])
    if energies_ev is not None:
        expected = np.asarray(sorted(float(value) for value in energies_ev))
        if expected.shape != energy_grid.shape or not np.allclose(
            expected, energy_grid, rtol=5.1e-6, atol=0.0
        ):
            raise ValueError("ELSEPA outputs do not match the requested energy grid")
        energy_grid = expected
    return {
        "energy_eV": energy_grid,
        "theta_deg": reference.theta_deg,
        "mu": reference.mu,
        "dcs_cm2_sr": np.stack([panel.dcs_cm2_sr for panel in panels]),
        "dcs_a0_2_sr": np.stack([panel.dcs_a0_2_sr for panel in panels]),
        "angular_cdf": np.stack([_normalized_cdf(panel) for panel in panels]),
        "sherman": np.stack([panel.sherman for panel in panels]),
        "relative_error": np.stack([panel.relative_error for panel in panels]),
        "total_elastic_cm2": np.asarray([panel.total_elastic_cm2 for panel in panels]),
        "transport1_cm2": np.asarray([panel.transport1_cm2 for panel in panels]),
        "transport2_cm2": np.asarray([panel.transport2_cm2 for panel in panels]),
        "absorption_cm2": np.asarray([panel.absorption_cm2 for panel in panels]),
    }

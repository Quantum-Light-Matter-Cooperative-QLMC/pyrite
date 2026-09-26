"""Bremsstrahlung sources against the Seltzer--Berger tables.

Validation: brem-source-comparison
"""

import hashlib

import numpy as np
import pytest

from pyrite.validation import brem_sources
from pyrite.validation.brem_sources import (
    SeltzerBergerTable,
    SeltzerBergerUnavailableError,
    beta_squared,
    compare_sources,
    load_seltzer_berger,
    parse_seltzer_berger,
)

# Seltzer's BREME.DAT (NBS, 1984) as distributed in EGSnrc nist_brems.data:
# scaled chi = (beta^2/Z^2) k dsigma/dk in mb on the table's kappa = k/T nodes.
_KAPPA = (
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
    0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9995, 0.9999,
    0.99995, 0.99999, 1.0,
)  # fmt: skip
_SELTZER_BERGER_CHI_MB = {
    (6, 0.1): (
        12.3787, 11.2376, 10.1584, 9.22272, 8.40044, 7.67313, 7.02623, 6.44614, 5.9239,
        5.4616, 5.04543, 4.65945, 4.29275, 3.94472, 3.61129, 3.28784, 2.97132, 2.65007,
        2.31365, 2.14014, 1.96669, 1.83028, 1.70072, 1.66943, 1.64446, 1.64147, 1.63894,
        1.63863, 1.63836, 1.63826,
    ),
    (6, 1.0): (
        16.3892, 13.119, 11.0942, 9.4922, 8.24269, 7.25889, 6.45955, 5.78158, 5.17872,
        4.63288, 4.13455, 3.67551, 3.248, 2.84623, 2.46706, 2.10784, 1.76867, 1.45481,
        1.15327, 0.99309, 0.81444, 0.64953, 0.45335, 0.39948, 0.35592, 0.35043, 0.34593,
        0.34539, 0.34489, 0.34484,
    ),
    (6, 10.0): (
        20.5294, 16.0727, 14.5622, 13.3301, 12.2735, 11.3185, 10.4575, 9.67374, 8.95922,
        8.30502, 7.70321, 7.14507, 6.62055, 6.1194, 5.6312, 5.13963, 4.61512, 3.9981,
        3.20187, 2.69652, 2.08267, 1.49693, 0.80704, 0.58939, 0.37513, 0.33403, 0.28851,
        0.28268, 0.28464, 0.28033,
    ),
    (74, 0.1): (
        8.92442, 8.8157, 8.59365, 8.3051, 7.9943, 7.70177, 7.43315, 7.1873, 6.96312,
        6.75963, 6.57379, 6.40247, 6.24333, 6.09543, 5.95839, 5.83166, 5.71472, 5.60641,
        5.50342, 5.45247, 5.40287, 5.36844, 5.34787, 5.33888, 5.31196, 5.30662, 5.30624,
        5.30717, 5.3058, 5.3068,
    ),
    (74, 1.0): (
        11.9731, 10.8209, 9.84616, 8.98516, 8.22606, 7.55502, 6.95797, 6.4221, 5.9362,
        5.49402, 5.09054, 4.72033, 4.37692, 4.05532, 3.7533, 3.46767, 3.19541, 2.93381,
        2.68008, 2.55611, 2.43377, 2.33719, 2.23862, 2.21397, 2.19514, 2.19299, 2.19104,
        2.19079, 2.19056, 2.18941,
    ),
    (74, 10.0): (
        11.5735, 10.5727, 9.85857, 9.21301, 8.62476, 8.0851, 7.59062, 7.13732, 6.72174,
        6.34018, 5.98847, 5.66128, 5.35133, 5.05168, 4.75689, 4.45722, 4.13277, 3.7393,
        3.2138, 2.87466, 2.45823, 2.05312, 1.56906, 1.43027, 1.35023, 1.35045, 1.35265,
        1.34892, 1.3365, 1.35163,
    ),
}  # fmt: skip
# ESTAR radiative stopping, MeV cm^2/g (NIST SRD 124), at 0.1, 1 and 10 MeV.
_ESTAR = {(6, 0.1): 0.003414, (6, 1.0): 0.01053, (6, 10.0): 0.1513,
          (74, 0.1): 0.04084, (74, 1.0): 0.1159, (74, 10.0): 1.132}  # fmt: skip
_MOLAR_MASS = {6: 12.011, 74: 183.84}
_SYMBOL = {6: "C", 74: "W"}


def _excerpt_table() -> SeltzerBergerTable:
    energies = np.array([0.1, 1.0, 10.0])
    chi = np.ones((100, len(_KAPPA), energies.size))
    for (Z, energy), row in _SELTZER_BERGER_CHI_MB.items():
        chi[Z - 1, :, list(energies).index(energy)] = row
    return SeltzerBergerTable(energies, np.array(_KAPPA), chi)


@pytest.mark.parametrize(("Z", "energy"), sorted(_ESTAR))
def test_seltzer_berger_first_moment_reproduces_estar(Z, energy):
    # Limiting case: ESTAR radiative stopping is the Seltzer-Berger first
    # moment, S/rho = (N_A/A) (Z^2/beta^2) T integral(chi dkappa).
    table = _excerpt_table()
    moment_mb = np.trapezoid(table.chi(Z, energy), table.kappa)
    stopping = 6.02214076e23 / _MOLAR_MASS[Z] * Z**2 / beta_squared(energy) * energy
    np.testing.assert_allclose(stopping * moment_mb * 1e-27, _ESTAR[(Z, energy)], rtol=0.01)


def _synthetic_text(n_energy=3, n_kappa=4):
    energy = np.geomspace(0.01, 1.0, n_energy)
    kappa = np.linspace(0.0, 1.0, n_kappa)
    chi = np.arange(1, 100 * n_energy * n_kappa + 1, dtype=float)
    body = " ".join(f"{v:g}" for v in (*energy, *kappa))
    data = "\n".join(" ".join(f"{v:g}" for v in chi[i : i + 6]) for i in range(0, chi.size, 6))
    return f"BREME.DAT: synthetic\n{n_energy} {n_kappa}\n{body}\nBREMX.DAT\n{data}\n", chi


def test_parser_reads_kappa_major_blocks_per_element():
    text, chi = _synthetic_text()
    table = parse_seltzer_berger(text)
    assert table.chi_mb.shape == (100, 4, 3)
    # Each element holds n_kappa consecutive blocks of n_energy values.
    np.testing.assert_array_equal(table.chi_mb[0, 1], chi[3:6])
    np.testing.assert_array_equal(table.chi_mb[99, 3], chi[-3:])


def test_parser_rejects_truncated_table():
    text, _ = _synthetic_text()
    with pytest.raises(ValueError, match="chi values"):
        parse_seltzer_berger(text.rsplit("\n", 2)[0])


def test_missing_table_is_actionable_and_digest_is_pinned(tmp_path, monkeypatch):
    with pytest.raises(SeltzerBergerUnavailableError, match="--download"):
        load_seltzer_berger(tmp_path / "absent.data")
    text, _ = _synthetic_text()
    path = tmp_path / "nist_brems.data"
    path.write_text(text)
    with pytest.raises(ValueError, match="SHA-256"):
        load_seltzer_berger(path)
    monkeypatch.setattr(
        brem_sources, "SELTZER_BERGER_SHA256", hashlib.sha256(text.encode()).hexdigest()
    )
    assert load_seltzer_berger(path).chi_mb.shape == (100, 4, 3)


def test_installed_table_matches_the_committed_excerpt():
    try:
        table = load_seltzer_berger()
    except SeltzerBergerUnavailableError as exc:
        pytest.skip(str(exc))
    np.testing.assert_array_equal(table.kappa, _KAPPA)
    for (Z, energy), row in _SELTZER_BERGER_CHI_MB.items():
        # The excerpt keeps six significant figures.
        np.testing.assert_allclose(table.chi(Z, energy), row, rtol=5e-6)


@pytest.mark.parametrize("Z", [6, 74])
def test_released_bremslib_tracks_seltzer_berger(Z):
    from pyrite.xsgen._errors import TableNotFoundError
    from pyrite.xsgen.bremslib import tables as bremslib_tables

    element = _SYMBOL[Z]
    try:
        tables = bremslib_tables.load_bremsstrahlung_tables([element])
    except TableNotFoundError as exc:
        pytest.skip(f"released BremsLib tables are not installed: {exc}")
    results = compare_sources(
        _excerpt_table(),
        [(element, Z)],
        (0.1, 1.0, 10.0),
        models=("bremslib",),
        bremslib_tables=tables,
    )
    gated = (np.array(_KAPPA) >= 0.05) & (np.array(_KAPPA) <= 0.95)
    # Seltzer-Berger includes electron-electron bremsstrahlung, up to about
    # 1/Z of the nuclear term; BremsLib is electron-atom only.
    floor = 1.0 / (1.0 + 1.0 / Z)
    for result in results:
        ratio = result.chi_ratio[gated]
        assert ratio.min() >= floor - 0.05, (result.incident_energy_MeV, ratio.min())
        assert ratio.max() <= 1.05, (result.incident_energy_MeV, ratio.max())
        assert floor - 0.03 <= result.first_moment_ratio <= 1.03

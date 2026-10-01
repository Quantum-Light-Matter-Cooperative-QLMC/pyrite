"""EPDL2025 photon cross sections: packaged table, XCOM agreement, generator."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pyrite.materials.photon_cross_sections import (
    EPDL_TABLE_PATH,
    PHOTON_CHANNELS,
    _table,
    photoelectric_edges,
    photon_cross_sections_ang2,
)

REPO = Path(__file__).resolve().parents[2]
XCOM_DIR = REPO / "tests" / "data" / "xcom"
_AVOGADRO = 6.02214076e23

# XCOM column -> EPDL channel.
_XCOM_COLUMNS = {
    "coherent": 1,
    "incoherent": 2,
    "photoelectric": 3,
    "pair_nuclear": 4,
    "pair_electron": 5,
}

# Total mu/rho tolerance away from absorption edges (more than 2% in energy
# from any XCOM or EPDL edge). Low/mid Z agree to 0.35% (Se 0.48%); W and Pb
# differ by up to 2.8% just below their L and K edges, where XCOM and EPDL
# photoionization normalizations differ.
_TOTAL_TOLERANCE = {"C": 5e-3, "N": 5e-3, "Al": 5e-3, "Si": 5e-3, "Se": 6e-3, "W": 3e-2, "Pb": 3e-2}


def _xcom(element):
    rows = [
        line.split()
        for line in (XCOM_DIR / f"xcom_{element}.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    edge = np.array([row[0] != "-" for row in rows])
    data = np.array([[float(value) for value in row[1:]] for row in rows])
    return edge, data


def _mass_attenuation(element, energy_eV):
    import xraydb

    per_atom = photon_cross_sections_ang2(element, energy_eV)
    scale = 1.0e-16 * _AVOGADRO / xraydb.atomic_mass(element)
    return {name: values * scale for name, values in per_atom.items()}


def _far_from_edges(element, energy_eV, xcom_edges_eV, fraction=0.02):
    edges = np.concatenate([xcom_edges_eV, [e for e, _ in photoelectric_edges(element)]])
    if edges.size == 0:
        return np.ones(energy_eV.shape, dtype=bool)
    distance = np.min(np.abs(np.log(energy_eV[:, None] / edges[None, :])), axis=1)
    return distance > fraction


def test_packaged_table_is_pinned_and_compact() -> None:
    tables = _table()
    assert sorted(tables) == list(range(1, 101))
    assert all(sorted(tables[z]) == sorted(PHOTON_CHANNELS.values()) for z in tables)
    # ADR-0014 class (a) neighbourhood; the upstream tape is 86 MB.
    assert EPDL_TABLE_PATH.stat().st_size < 2 * 1024 * 1024
    with np.load(EPDL_TABLE_PATH) as archive:
        assert archive["tolerance"].item() == 5.0e-4


@pytest.mark.parametrize("element", sorted(_TOTAL_TOLERANCE))
def test_total_mass_attenuation_matches_xcom(element) -> None:
    """Total mu/rho against NIST XCOM, 1 keV - 100 GeV, away from edges."""
    edge, data = _xcom(element)
    energy = data[:, 0] * 1.0e6
    keep = ~edge & _far_from_edges(element, energy, energy[edge])

    ours = sum(_mass_attenuation(element, energy[keep]).values())

    np.testing.assert_allclose(ours, data[keep, 6], rtol=_TOTAL_TOLERANCE[element])


@pytest.mark.parametrize("element", ["C", "Si", "W", "Pb"])
@pytest.mark.parametrize("channel", ["pair_nuclear", "pair_electron"])
def test_pair_production_matches_xcom(element, channel) -> None:
    """Pair terms agree to 0.2% wherever XCOM gives them above 1.5 MeV."""
    _edge, data = _xcom(element)
    energy = data[:, 0] * 1.0e6
    reference = data[:, _XCOM_COLUMNS[channel]]
    keep = (reference > 0.0) & (energy >= 1.5e6)
    # Within a few percent of threshold both rise steeply from zero; compare
    # where the electron-field term has left its own threshold too.
    if channel == "pair_electron":
        keep &= energy >= 3.0e6

    ours = _mass_attenuation(element, energy[keep])[channel]

    np.testing.assert_allclose(ours, reference[keep], rtol=2.0e-3)


@pytest.mark.parametrize("element", ["C", "Al", "Pb"])
def test_incoherent_matches_xcom_below_50_MeV(element) -> None:
    _edge, data = _xcom(element)
    energy = data[:, 0] * 1.0e6
    keep = (energy >= 1.0e4) & (energy <= 5.0e7)

    ours = _mass_attenuation(element, energy[keep])["incoherent"]

    np.testing.assert_allclose(ours, data[keep, 2], rtol=2.0e-2)


def test_photoelectric_edges_are_right_continuous() -> None:
    """At an edge energy the post-edge value is returned; just below, the pre-edge."""
    edge_eV, jump = max(photoelectric_edges("Pb"), key=lambda item: item[0])
    assert edge_eV == pytest.approx(88_011.0, rel=1e-4)  # Pb K
    below, at = photon_cross_sections_ang2("Pb", np.array([edge_eV * (1 - 1e-9), edge_eV]))[
        "photoelectric"
    ]
    assert at / below == pytest.approx(jump, rel=1e-6)
    assert jump > 4.0


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "release_epdl_table", REPO / "scripts" / "release_epdl_table.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generator_thinning_keeps_edges_zeros_and_tolerance() -> None:
    generator = _load_generator()
    energy = np.concatenate([[10.0, 10.0], np.linspace(10.0, 100.0, 400)[1:], [100.0]])
    sigma = np.concatenate([[0.0, 5.0], 5.0 * (np.linspace(10.0, 100.0, 400)[1:] / 10.0) ** -2.5])
    sigma = np.concatenate([sigma, [sigma[-1] * 3.0]])  # an edge at 100 eV

    kept_e, kept_s = generator.thin_section(energy, sigma, 1.0e-3)

    assert kept_e.size < energy.size // 4
    assert kept_e[0] == 10.0 and kept_s[0] == 0.0  # zero onset kept verbatim
    assert np.count_nonzero(kept_e == 100.0) == 2  # edge pair kept
    interior = slice(2, -1)
    rebuilt = np.interp(energy[interior], kept_e[1:-1], kept_s[1:-1])
    np.testing.assert_allclose(rebuilt, sigma[interior], rtol=1.0e-3)


def test_generator_npz_is_byte_reproducible(tmp_path) -> None:
    generator = _load_generator()
    arrays = {"b": np.arange(5, dtype=np.int16), "a": np.linspace(0.0, 1.0, 7)}

    first = generator.write_deterministic_npz(tmp_path / "one.npz", arrays)
    second = generator.write_deterministic_npz(tmp_path / "two.npz", arrays)

    assert first == second
    with np.load(tmp_path / "one.npz") as archive:
        np.testing.assert_array_equal(archive["a"], arrays["a"])

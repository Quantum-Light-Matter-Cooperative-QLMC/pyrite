"""BremsLib soft/hard radiative moment and photon checks.

Validation: bremslib-radiative-partition
"""

import hashlib
import io
from pathlib import Path

import numpy as np
import pytest

from pyrite.montecarlo.spectrum.brem_bremslib import (
    bremslib_segment_state,
    evaluate_bremslib,
    prepare_bremslib_table,
    stage_bremslib_table,
)
from pyrite.montecarlo.transport._jit_radiative import (
    radiative_moments_scalar,
    sample_hard_photon_energy_scalar,
)
from pyrite.montecarlo.transport.hard_radiative import (
    build_radiative_partition,
    sample_hard_radiative_photon,
)
from tests.helpers.bremslib import synthetic_bremslib_arrays


@pytest.fixture
def table():
    return prepare_bremslib_table(synthetic_bremslib_arrays(), atomic_number=6)


def test_cutoff_partitions_one_fixed_sdcs_without_double_counting(table):
    partitions = [
        build_radiative_partition(table, 60_000.0, k) for k in (100.0, 3_330.0, 30_000.0, 60_000.0)
    ]
    reference = partitions[0].total_stopping_cs_eV_cm2
    for partition in partitions:
        np.testing.assert_allclose(partition.total_stopping_cs_eV_cm2, reference, rtol=2e-14)
        np.testing.assert_allclose(
            partition.soft_stopping_cs_eV_cm2 + partition.hard_stopping_cs_eV_cm2,
            reference,
            rtol=2e-14,
        )
    assert all(
        a.hard_rate_cs_cm2 > b.hard_rate_cs_cm2
        for a, b in zip(partitions[:-1], partitions[1:], strict=True)
    )
    assert partitions[-1].hard_rate_cs_cm2 == 0.0
    assert partitions[-1].soft_stopping_cs_eV_cm2 == reference

    # Independent dense quadrature of the public BremsLib evaluation checks
    # the units and exact-cell integration against the same physical SDCS.
    k = np.linspace(1.0, 60_000.0, 20_000)
    staged = stage_bremslib_table(table)
    state = bremslib_segment_state(staged, [60.0])
    sdcs = np.asarray(evaluate_bremslib(staged, state, k))[0]
    quadrature = np.trapezoid(k * sdcs, k)
    np.testing.assert_allclose(reference, quadrature, rtol=2e-4)


def test_seeded_hard_energy_samples_match_exact_first_moment(table):
    partition = build_radiative_partition(table, 60_000.0, 2_500.0)
    rng = np.random.default_rng(93)
    energies = np.array([partition.sample_photon_energy(float(u)) for u in rng.random(4000)])
    assert np.all((energies >= partition.cutoff_eV) & (energies <= partition.incident_energy_eV))
    expected = partition.hard_stopping_cs_eV_cm2 / partition.hard_rate_cs_cm2
    assert abs(energies.mean() - expected) < 0.04 * expected


def test_photon_and_recoiled_electron_have_valid_states(table):
    partition = build_radiative_partition(table, 60_000.0, 2_500.0)
    incoming = np.array([0.6, 0.0, 0.8])
    photon = sample_hard_radiative_photon(partition, incoming, 0.4, 0.3, 0.2)
    np.testing.assert_allclose(photon.energy_eV + photon.electron_energy_eV, 60_000.0)
    np.testing.assert_allclose(np.linalg.norm(photon.direction), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(photon.electron_direction), 1.0, atol=1e-12)
    np.testing.assert_array_equal(photon.electron_direction, incoming)
    assert photon.energy_eV > partition.cutoff_eV
    rest = 510_998.95
    p_in = np.sqrt(60_000.0 * (60_000.0 + 2.0 * rest))
    p_out = np.sqrt(photon.electron_energy_eV * (photon.electron_energy_eV + 2.0 * rest))
    np.testing.assert_allclose(
        p_in * incoming,
        photon.energy_eV * photon.direction
        + p_out * photon.electron_direction
        + photon.target_momentum_eV_c,
        atol=1e-10,
    )


def test_invalid_partition_and_empty_hard_range(table):
    with pytest.raises(ValueError, match="positive"):
        build_radiative_partition(table, 60_000.0, 0.0)
    with pytest.raises(ValueError, match="outside"):
        build_radiative_partition(table, 1_000.0, 100.0)
    with pytest.raises(ValueError, match="no hard"):
        build_radiative_partition(table, 60_000.0, 60_000.0).sample_photon_energy(0.5)


@pytest.mark.parametrize("energy", [10_000.0, 13_700.0, 50_000.0, 71_000.0, 200_000.0])
def test_numba_scalar_partition_and_sampler_match_host(table, energy):
    arrays = (
        table.incident_energy_keV,
        table.nominal_reduced_energy,
        table.top_reduced_energy,
        table.scaled_sdcs_mb,
        table.atomic_number,
    )
    for cutoff in (100.0, 3_330.0, 0.5 * energy, energy):
        host = build_radiative_partition(table, energy, cutoff)
        got = radiative_moments_scalar(*arrays, energy, cutoff)
        np.testing.assert_allclose(
            got,
            (host.soft_stopping_cs_eV_cm2, host.hard_rate_cs_cm2, host.hard_stopping_cs_eV_cm2),
            rtol=2e-12,
            atol=1e-30,
        )
        if cutoff == energy:
            continue
        for uniform in (0.0, 0.01, 0.5, 0.99, np.nextafter(1.0, 0.0)):
            sampled = sample_hard_photon_energy_scalar(*arrays, energy, cutoff, uniform)
            np.testing.assert_allclose(sampled, host.sample_photon_energy(uniform), rtol=2e-12)


def test_cutoff_node_rounding_keeps_scalar_sampler_on_the_hard_cells(table):
    # (kc / T) * T rounds below kc at these energies. The moment kernel and
    # sampler must still agree on which cell is the first hard one.
    arrays = (
        table.incident_energy_keV,
        table.nominal_reduced_energy,
        table.top_reduced_energy,
        table.scaled_sdcs_mb,
        table.atomic_number,
    )
    cutoff = 1_000.0
    energies = [e for e in np.linspace(20_000.0, 60_000.0, 401) if (cutoff / e) * e < cutoff]
    assert energies
    for energy in energies[:20]:
        host = build_radiative_partition(table, energy, cutoff)
        assert cutoff in host.photon_grid_eV
        for uniform in (0.0, 0.01, 0.5):
            sampled = sample_hard_photon_energy_scalar(*arrays, energy, cutoff, uniform)
            assert sampled >= cutoff
            np.testing.assert_allclose(sampled, host.sample_photon_energy(uniform), rtol=2e-12)


# ESTAR radiative stopping powers, MeV cm^2/g (NIST SRD 124, Seltzer-Berger;
# retrieved 2026-09-25) at 0.01, 0.03, 0.1, 0.3, 1, 3, 10 and 30 MeV.
_ESTAR_ENERGIES_MEV = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
_ESTAR_RADIATIVE_MEV_CM2_G = {
    "C": (0.00315, 0.003194, 0.003414, 0.004489, 0.01053, 0.03561, 0.1513, 0.5435),
    "Al": (0.006559, 0.007059, 0.007476, 0.009487, 0.02119, 0.06924, 0.2858, 1.003),
    "Ge": (0.01267, 0.01575, 0.01837, 0.02344, 0.04926, 0.1512, 0.5926, 2.029),
    "W": (0.01977, 0.02908, 0.04084, 0.05797, 0.1159, 0.3158, 1.132, 3.735),
    "Bi": (0.02064, 0.03125, 0.04524, 0.06579, 0.1313, 0.3478, 1.222, 4.01),
}


@pytest.mark.parametrize("element", sorted(_ESTAR_RADIATIVE_MEV_CM2_G))
def test_released_bremslib_total_moment_matches_estar(element):
    # Independent benchmark of the integrated first moment. The staged
    # BremsLib electron-nucleus term carries Seltzer-Berger's electron-electron
    # ratio, as ESTAR does. Without it carbon fell 11 % short at 10-30 MeV;
    # with it every element is within ESTAR's 2-50 MeV uncertainty (3-10 %,
    # Seltzer and Berger 1985), and heavy elements within 3 %.
    from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
    from pyrite.xsgen._errors import TableNotFoundError
    from pyrite.xsgen.bremslib import tables as bremslib_tables

    try:
        table = bremslib_tables.load_bremsstrahlung_tables([element])[element]
    except TableNotFoundError as exc:
        pytest.skip(f"released BremsLib tables are not installed: {exc}")
    Z = TRANSPORT_ELEMENTS[element]["Z"]
    grams_per_atom = TRANSPORT_ELEMENTS[element]["A"] / 6.02214076e23
    ratio = np.array(
        [
            build_radiative_partition(table, energy * 1e6, 1.0).total_stopping_cs_eV_cm2
            * 1e-6
            / grams_per_atom
            / estar
            for energy, estar in zip(
                _ESTAR_ENERGIES_MEV, _ESTAR_RADIATIVE_MEV_CM2_G[element], strict=True
            )
        ]
    )
    assert np.all((ratio >= 0.97) & (ratio <= 1.06))
    if Z >= 30:
        np.testing.assert_allclose(ratio, 1.0, atol=0.03)


# Pinned NIST ESTAR downloads (checks/reference/estar/README.md).
_ESTAR_DIR = Path(__file__).resolve().parents[2] / "checks" / "reference" / "estar"
_ESTAR_SHA256 = {
    "graphite": "427849a47e310ba2ced63988965a1f1a4f5409790c7a100f034b91fbfc5b6dca",
    "molybdenum": "d44a6fd7e496fc770679f752bf26545a2c9b94f98f6297d625c3f0d31234867f",
    "silicon": "ac02e55f68576901068169864dc0ff88ad24e4a6c4f26337002afd94f24f0dec",
    "tungsten": "6b50749e1e5ecdc49b84ada9142b64691299e963b4032c83190f0e581e228a3d",
}
_MEV_ENERGIES_MEV = (1.0, 3.0, 10.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0)


def _estar_rows(name):
    data = (_ESTAR_DIR / f"{name}.txt").read_bytes()
    assert hashlib.sha256(data).hexdigest() == _ESTAR_SHA256[name]
    rows = np.loadtxt(io.StringIO(data.decode("ascii")), skiprows=8)
    return {round(float(row[0]), 6): row[1:4] for row in rows}


def _released_table(element):
    from pyrite.xsgen._errors import TableNotFoundError
    from pyrite.xsgen.bremslib import tables as bremslib_tables

    try:
        return bremslib_tables.load_bremsstrahlung_tables([element])[element]
    except TableNotFoundError as exc:
        pytest.skip(f"released BremsLib tables are not installed: {exc}")


@pytest.mark.parametrize(
    ("estar_name", "element", "plasma_key"),
    [
        ("graphite", "C", None),
        ("silicon", "Si", "silicon"),
        ("molybdenum", "Mo", None),
        ("tungsten", "W", "tungsten"),
    ],
)
def test_released_partition_closes_and_matches_estar_to_100_mev(estar_name, element, plasma_key):
    # Issue #439 extension to 100 MeV, with the dielectric suppression of
    # coupled runs where the material has a catalog plasma energy. Soft + hard
    # must equal the uncut first moment exactly; the moment itself must stay
    # inside Seltzer-Berger's 3-10 % (2-50 MeV) uncertainty. Measured
    # 2026-10-22: 1.009-1.053, highest for C and W at 100 MeV.
    from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
    from pyrite.materials.attenuation import plasma_energy_eV

    table = _released_table(element)
    plasma = 0.0 if plasma_key is None else plasma_energy_eV(plasma_key)
    grams_per_atom = TRANSPORT_ELEMENTS[element]["A"] / 6.02214076e23
    estar = _estar_rows(estar_name)
    for energy_MeV in _MEV_ENERGIES_MEV:
        energy = energy_MeV * 1e6
        cut = build_radiative_partition(table, energy, 1.0e3, plasma_energy_eV=plasma)
        uncut = build_radiative_partition(table, energy, energy, plasma_energy_eV=plasma)
        np.testing.assert_allclose(
            cut.soft_stopping_cs_eV_cm2 + cut.hard_stopping_cs_eV_cm2,
            uncut.soft_stopping_cs_eV_cm2,
            rtol=1e-12,
        )
        ratio = cut.total_stopping_cs_eV_cm2 * 1e-6 / grams_per_atom / estar[energy_MeV][1]
        assert 0.97 <= ratio <= 1.06, (energy_MeV, ratio)


@pytest.mark.parametrize("material", ["silicon", "tungsten"])
def test_coupled_total_stopping_matches_estar_to_100_mev(material):
    # Maintainer decision (2026-10-22, #439): split gates. SBETHE collision
    # stopping within 2 % of ESTAR; total within 5 %, because BremsLib + SB85
    # e-e radiative stopping sits 2-5 % above ESTAR (SB's own 3-10 %).
    # Measured: collision 0.998-1.017, total 1.005-1.021 (Si), 1.013-1.044 (W).
    from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
    from pyrite.materials.attenuation import plasma_energy_eV
    from pyrite.montecarlo.transport.shell_rates import adopted_stopping_cs
    from pyrite.xsgen._errors import TableNotFoundError

    element = {"silicon": "Si", "tungsten": "W"}[material]
    table = _released_table(element)
    plasma = plasma_energy_eV(material)
    grams_per_atom = TRANSPORT_ELEMENTS[element]["A"] / 6.02214076e23
    estar = _estar_rows(material)
    for energy_MeV in _MEV_ENERGIES_MEV:
        energy = energy_MeV * 1e6
        try:
            collision = adopted_stopping_cs(material, energy) * 1e-6 / grams_per_atom
        except TableNotFoundError as exc:
            pytest.skip(f"released SBETHE tables are not installed: {exc}")
        radiative = (
            build_radiative_partition(
                table, energy, 1.0e3, plasma_energy_eV=plasma
            ).total_stopping_cs_eV_cm2
            * 1e-6
            / grams_per_atom
        )
        estar_collision, _, estar_total = estar[energy_MeV]
        np.testing.assert_allclose(collision, estar_collision, rtol=0.02)
        np.testing.assert_allclose(collision + radiative, estar_total, rtol=0.05)


@pytest.mark.parametrize("material", ["silicon", "tungsten"])
def test_coupled_csda_range_matches_estar_to_100_mev(material):
    # Issue #439: CSDA range from SBETHE collision plus coupled BremsLib
    # radiative stopping, integrated from 10 keV (ESTAR's own 10 keV range is
    # added for its zero lower limit). Measured 2026-10-22: Si 0.989-0.996,
    # W 0.978-0.992 of ESTAR from 1 to 100 MeV, short by the radiative excess.
    from pyrite.materials._transport_data import TRANSPORT_ELEMENTS
    from pyrite.materials.attenuation import plasma_energy_eV
    from pyrite.montecarlo.transport.shell_rates import adopted_stopping_cs
    from pyrite.xsgen._errors import TableNotFoundError

    element = {"silicon": "Si", "tungsten": "W"}[material]
    table = _released_table(element)
    plasma = plasma_energy_eV(material)
    grams_per_atom = TRANSPORT_ELEMENTS[element]["A"] / 6.02214076e23
    energy = np.geomspace(1.0e4, 1.0e8, 401)
    try:
        collision = np.array([adopted_stopping_cs(material, e) for e in energy])
    except TableNotFoundError as exc:
        pytest.skip(f"released SBETHE tables are not installed: {exc}")
    radiative = np.array(
        [
            build_radiative_partition(table, e, e, plasma_energy_eV=plasma).total_stopping_cs_eV_cm2
            for e in energy
        ]
    )
    inverse = grams_per_atom / (collision + radiative)  # g/cm^2 per eV
    range_g_cm2 = np.r_[0.0, np.cumsum(0.5 * (inverse[1:] + inverse[:-1]) * np.diff(energy))]
    rows = np.loadtxt(_ESTAR_DIR / f"{material}.txt", skiprows=8)
    assert (
        hashlib.sha256((_ESTAR_DIR / f"{material}.txt").read_bytes()).hexdigest()
        == (_ESTAR_SHA256[material])
    )
    estar_range = dict(zip(np.round(rows[:, 0], 6), rows[:, 4], strict=True))
    for energy_MeV in (1.0, 3.0, 10.0, 30.0, 50.0, 100.0):
        ours = np.interp(np.log(energy_MeV * 1e6), np.log(energy), range_g_cm2) + estar_range[0.01]
        assert ours / estar_range[energy_MeV] == pytest.approx(1.0, abs=0.03), energy_MeV

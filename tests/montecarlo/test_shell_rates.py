"""PENELOPE-2024 inner-shell substitution and stopping closure (Eqs. 3.141–3.142)."""

import dataclasses

import numpy as np
import pytest
from scipy.constants import N_A

from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.eedl_ionization import EEDL_SUBSHELL_LABELS, load_eedl_shell_ionization
from pyrite.montecarlo.spectrum import characteristic
from pyrite.montecarlo.transport import shell_gos as sg
from pyrite.montecarlo.transport import shell_oscillators as so
from pyrite.montecarlo.transport import shell_rates as sr
from pyrite.montecarlo.transport.shell_partition import partition_shell_rates
from pyrite.xsgen.sbethe.catalog import resolve_catalog_table

KEYS = ("silicon", "sio2", "mos2")
ENERGIES = (1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 1e6)


def _shell(z, designator, label, orbital, occupation, energy):
    return config.AtomicShell(z, designator, label, orbital, occupation, energy, 0.1, 0.0, 0.0)


def _silicon_shells(merge_l=False):
    l_shells = (
        (_shell(14, 3, "L2", "2p1/2", 6, 104.0),)
        if merge_l
        else (_shell(14, 3, "L2", "2p1/2", 2, 104.0), _shell(14, 4, "L3", "2p3/2", 4, 104.0))
    )
    return {
        14: (
            _shell(14, 1, "K", "1s1/2", 2, 1844.0),
            _shell(14, 2, "L1", "2s1/2", 2, 154.0),
            *l_shells,
            _shell(14, 5, "M1", "3s1/2", 2, 13.46),
            _shell(14, 6, "M2", "3p1/2", 2, 8.151),
        )
    }


def _silicon_fixture(merge_l=False):
    band = so.ConductionBand(4.0, 16.7, "fixture", {14: 1.0})
    return so.build_shell_oscillators({14: 1.0}, 173.0, 31.05, _silicon_shells(merge_l), band)


def _require_pdatconf():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")


def _stp(key, energy):
    arrays = resolve_catalog_table(key).arrays()
    grid, stopping = arrays["stopping_energy_eV"], arrays["stopping_cs_eV_cm2"]
    return float(np.exp(np.interp(np.log(energy), np.log(grid), np.log(stopping))))


def _eedl(symbol, labels, energy):
    total = 0.0
    for shell in load_eedl_shell_ionization(symbol):
        if EEDL_SUBSHELL_LABELS[shell.shell_designator] in labels:
            if energy > shell.binding_energy_eV:
                total += np.interp(energy, shell.projectile_energy_eV, shell.cross_section_cm2)
    return total


def test_default_threshold_matches_characteristic_relaxation_cutoff():
    assert sr.DEFAULT_INNER_SHELL_THRESHOLD_EV == characteristic._MIN_RELAXATION_CUTOFF_EV


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("energy", ENERGIES)
def test_catalog_closure_reproduces_stp_exactly_with_positive_scale(key, energy):
    _require_pdatconf()
    closure = sr.catalog_shell_rate_closure(key, energy)
    assert np.isfinite(closure.outer_scale) and closure.outer_scale > 0.0
    assert closure.moments.total[1] == pytest.approx(_stp(key, energy), rel=1e-12, abs=0.0)
    assert np.all(closure.moments.per_shell >= 0.0)


# Documented note, not a gate (owner decision, 2026-09-24): the closed model's
# total IMFP is 26-35% below independent full Penn calculations. The recorded
# ratios pin that note so a model change cannot silently move it; see
# docs/validation/beam-transport/penelope-shell-rate-closure.md.
@pytest.mark.parametrize(
    ("key", "energy_eV", "penn_imfp_nm", "offset_eV", "recorded_ratio"),
    [
        ("silicon", 1998.2, 4.25, 0.0, 0.745),
        ("silicon", 9897.1, 16.04, 0.0, 0.714),
        ("silicon", 19930.4, 28.77, 0.0, 0.734),
        ("sio2", 1998.2, 5.09, 10.0, 0.674),
        ("sio2", 9897.1, 19.0, 10.0, 0.645),
        ("sio2", 19930.4, 33.9, 10.0, 0.660),
    ],
)
def test_closed_imfp_to_full_penn_ratio_matches_recorded_note(
    key, energy_eV, penn_imfp_nm, offset_eV, recorded_ratio
):
    """Shinotsuka et al. (2015) Table 2 (Si); (2019) Tables 1/5 (SiO2)."""
    _require_pdatconf()
    arrays = resolve_catalog_table(key).arrays()
    formula_density_cm3 = (
        float(arrays["density_g_cm3"]) * N_A / float(arrays["molecular_weight_g_mol"])
    )
    # Si: energy above Fermi level. SiO2: Table 5 gives E = T - Eg - Ev;
    # Eq. 7 uses T-prime = T - Eg and Table 1 gives Ev = 10 eV.
    sigma_cm2 = sr.catalog_shell_rate_closure(key, energy_eV + offset_eV).moments.total[0]
    model_imfp_nm = 1e7 / (formula_density_cm3 * sigma_cm2)
    assert model_imfp_nm / penn_imfp_nm == pytest.approx(recorded_ratio, abs=5e-4)


@pytest.mark.parametrize("key", ("silicon", "sio2", "mos2"))
@pytest.mark.parametrize("energy", (1998.2, 9897.1, 19930.4))
def test_imfp_excess_channel_is_condensed_above_conduction_resonance(key, energy):
    """The IMFP note is benign only for ``W_c > W_cb``: the conduction-band
    distant loss (76-85% of the total rate) is then entirely soft, so hard
    events never sample it and its rate enters transport only through the
    stopping-closed soft first moment."""
    _require_pdatconf()
    closure = sr.catalog_shell_rate_closure(key, energy)
    band = [k for k, o in enumerate(closure.raw.oscillators) if o.atomic_number == 0]
    assert len(band) == 1
    k = band[0]
    w_cb = closure.raw.oscillators[k].resonance_energy_eV
    distant = closure.moments.distant_longitudinal[k, 0] + closure.moments.distant_transverse[k, 0]
    assert 0.75 < distant / closure.moments.total[0] < 0.86
    material = sr.catalog_shell_oscillators(key)
    for cutoff in (np.nextafter(w_cb, np.inf), 50.0, 100.0):
        part = partition_shell_rates(material, closure, cutoff)
        assert part.hard.distant_longitudinal[k, 0] == 0.0
        assert part.hard.distant_transverse[k, 0] == 0.0


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("energy", (2e3, 1e4, 1e5, 1e6))
def test_inner_shell_loss_pdf_unchanged_and_outer_shells_share_one_scale(key, energy):
    _require_pdatconf()
    closure = sr.catalog_shell_rate_closure(key, energy)
    raw, new = closure.raw.per_shell, closure.moments.per_shell
    for k in np.flatnonzero(closure.inner & (raw[:, 0] > 0.0)):
        assert new[k, 1] / new[k, 0] == pytest.approx(raw[k, 1] / raw[k, 0], rel=1e-12, abs=0.0)
        assert new[k, 2] / new[k, 0] == pytest.approx(raw[k, 2] / raw[k, 0], rel=1e-12, abs=0.0)
        assert new[k, 0] == pytest.approx(closure.adopted_inner_cm2[k], rel=1e-12, abs=0.0)
    outer = ~closure.inner
    assert np.allclose(new[outer], closure.outer_scale * raw[outer], rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("energy", (5e3, 1e5, 1e6))
def test_inner_cross_section_is_eedl_times_density_ratio(energy):
    _require_pdatconf()
    closure = sr.catalog_shell_rate_closure("sio2", energy)
    counts = {14: 1.0, 8: 2.0}
    for k, osc in enumerate(closure.raw.oscillators):
        if not closure.inner[k]:
            continue
        symbol = {14: "Si", 8: "O"}[osc.atomic_number]
        eedl = counts[osc.atomic_number] * _eedl(symbol, {osc.label}, energy)
        assert eedl > 0.0
        assert closure.adopted_inner_cm2[k] == pytest.approx(
            eedl * closure.density_ratio[k], rel=1e-12, abs=0.0
        )


def test_catalog_inner_shell_selection():
    _require_pdatconf()
    closure = sr.catalog_shell_rate_closure("mos2", 1e5)
    inner = {
        (o.atomic_number, o.label)
        for o, flag in zip(closure.raw.oscillators, closure.inner, strict=True)
        if flag
    }
    assert (42, "N1") in inner  # U = 68 eV > E_c = 50 eV
    assert (42, "N2") not in inner and (42, "N3") not in inner  # U = 45, 42 eV
    assert (16, "L3") in inner and (42, "K") in inner
    assert not any(
        o.atomic_number == 0
        for o, f in zip(closure.raw.oscillators, closure.inner, strict=True)
        if f
    )


def test_no_inner_shells_limit_gives_stp_over_gos():
    material = _silicon_fixture()
    raw = sg.shell_gos_moments(material, 1e4)
    closure = sr.close_shell_rates(material, 1e4, 0.9 * raw.total[1], {})
    assert not closure.inner.any()
    assert closure.outer_scale == pytest.approx(0.9, rel=1e-13, abs=0.0)
    assert np.allclose(closure.moments.total, 0.9 * raw.total, rtol=1e-13, atol=0.0)


def test_zero_density_effect_gives_unit_ratio():
    _require_pdatconf()
    closure = sr.catalog_shell_rate_closure("mos2", 1e4)
    assert closure.raw.density_effect == 0.0
    assert np.all(closure.density_ratio == 1.0)


def test_density_ratio_matches_transverse_bracket_without_delta():
    material = _silicon_fixture()
    energy = 1e6
    raw = sg.shell_gos_moments(material, energy)
    assert raw.density_effect > 0.0
    gamma = 1.0 + energy / (sg._MC2_EV)
    bracket0 = np.log(gamma * gamma) - (1.0 - 1.0 / gamma**2)
    bracket = bracket0 - raw.density_effect
    assert bracket > 0.0
    bare = (
        raw.distant_longitudinal[:, 0]
        + raw.close[:, 0]
        + raw.distant_transverse[:, 0] * bracket0 / bracket
    )
    keys = {(14, "K"): 1e-22, (14, "L1"): 1e-21}
    closure = sr.close_shell_rates(material, energy, raw.total[1], keys)
    for k in np.flatnonzero(closure.inner):
        expected = raw.per_shell[k, 0] / bare[k]
        assert expected < 1.0
        assert closure.density_ratio[k] == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_spin_orbit_partner_rates_are_merged():
    material = _silicon_fixture(merge_l=True)
    energy = 5e3
    sigma = sr.eedl_inner_cross_sections(
        {14: 1.0}, _silicon_shells(merge_l=True), material, energy, 50.0
    )
    assert set(sigma) == {(14, "K"), (14, "L1"), (14, "L2")}
    merged = _eedl("Si", {"L2", "L3"}, energy)
    assert merged > _eedl("Si", {"L2"}, energy) > 0.0
    assert sigma[(14, "L2")] == pytest.approx(merged, rel=1e-12, abs=0.0)


def test_formula_count_multiplies_eedl_cross_section():
    shells = {
        8: (
            _shell(8, 1, "K", "1s1/2", 2, 538.0),
            _shell(8, 2, "L1", "2s1/2", 2, 28.48),
            _shell(8, 3, "L2", "2p1/2", 1, 13.62),
            _shell(8, 4, "L3", "2p3/2", 3, 13.62),
        )
    }
    material = so.build_shell_oscillators({8: 3.0}, 95.0, 30.0, shells, default_threshold_eV=15.0)
    sigma = sr.eedl_inner_cross_sections({8: 3.0}, shells, material, 2e3, 50.0)
    assert set(sigma) == {(8, "K")}
    assert sigma[(8, "K")] == pytest.approx(3.0 * _eedl("O", {"K"}, 2e3), rel=1e-12, abs=0.0)


def test_inner_shell_cutoff_follows_heaviest_outer_shell():
    shells = _silicon_shells()
    assert sr.inner_shell_cutoff_eV({14: 1.0}, shells) == 50.0
    assert sr.inner_shell_cutoff_eV({14: 1.0}, shells, 200.0) == 200.0
    heavy = {
        **shells,
        79: (_shell(79, 1, "O1", "5s1/2", 2, 114.0), _shell(79, 2, "P1", "6s1/2", 1, 9.2)),
    }
    assert sr.inner_shell_cutoff_eV({14: 1.0, 79: 1.0}, heavy) == 114.0
    for bad in (0.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="threshold"):
            sr.inner_shell_cutoff_eV({14: 1.0}, shells, bad)
    oscillator = so.ShellOscillator(14, "L1", 2.0, 154.0, 200.0)
    assert sr.is_inner_shell(oscillator, 50.0) and not sr.is_inner_shell(oscillator, 154.0)
    assert not sr.is_inner_shell(so.ShellOscillator(79, "O1", 2.0, 114.0, 150.0), 50.0)
    assert not sr.is_inner_shell(so.ShellOscillator(0, "cb", 4.0, 0.0, 16.7), 0.0)


def test_inner_stopping_reaching_adopted_stopping_raises():
    material = _silicon_fixture()
    raw = sg.shell_gos_moments(material, 1e4)
    k = next(i for i, o in enumerate(material.oscillators) if o.label == "K")
    big = 10.0 * raw.total[1] / raw.per_shell[k, 1] * raw.per_shell[k, 0]
    with pytest.raises(ValueError, match="N\\(E\\)"):
        sr.close_shell_rates(material, 1e4, raw.total[1], {(14, "K"): big})


@pytest.mark.parametrize("stopping", (0.0, -1.0, np.nan, np.inf))
def test_invalid_adopted_stopping_raises(stopping):
    with pytest.raises(ValueError, match="stopping"):
        sr.close_shell_rates(_silicon_fixture(), 1e4, stopping, {})


def test_invalid_inner_inputs_raise():
    material = _silicon_fixture()
    with pytest.raises(ValueError, match="not a bound oscillator"):
        sr.close_shell_rates(material, 1e4, 1e-13, {(14, "M1"): 1e-20})
    with pytest.raises(ValueError, match="finite"):
        sr.close_shell_rates(material, 1e4, 1e-13, {(14, "K"): -1e-20})
    # Below the K threshold GOS has no loss PDF to carry a positive rate.
    with pytest.raises(ValueError, match="no loss PDF"):
        sr.close_shell_rates(material, 1e3, 1e-13, {(14, "K"): 1e-22})
    closed = sr.close_shell_rates(material, 1e3, 1e-13, {(14, "K"): 0.0})
    assert closed.scale[closed.inner][0] == 0.0


def test_energy_outside_stp_or_eedl_grid_raises():
    _require_pdatconf()
    with pytest.raises(ValueError, match="stopping table"):
        sr.catalog_shell_rate_closure("silicon", 500.0)
    with pytest.raises(ValueError, match="stopping table"):
        sr.catalog_shell_rate_closure("silicon", 2e9)
    material = _silicon_fixture()
    with pytest.raises(ValueError, match="EEDL projectile-energy grid"):
        sr.eedl_inner_cross_sections({14: 1.0}, _silicon_shells(), material, 2e11, 50.0)


def test_gos_consistent_inputs_return_raw_moments():
    material = _silicon_fixture()
    energy = 1e6
    raw = sg.shell_gos_moments(material, energy)
    bare = sg.shell_gos_moments(dataclasses.replace(material, plasma_energy_eV=0.0), energy)
    keys = {
        (o.atomic_number, o.label): float(bare.per_shell[k, 0])
        for k, o in enumerate(material.oscillators)
        if o.atomic_number > 0
    }
    closure = sr.close_shell_rates(material, energy, float(raw.total[1]), keys)
    assert closure.outer_scale == pytest.approx(1.0, rel=1e-12, abs=0.0)
    assert np.allclose(closure.scale, 1.0, rtol=1e-12, atol=0.0)
    assert np.allclose(closure.moments.per_shell, raw.per_shell, rtol=1e-12, atol=0.0)

"""Positron shell soft/hard tables and Bhabha hard-collision kernels (#276).

Validation: bhabha-close
"""

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad

from pyrite.montecarlo import shell_configuration as config
from pyrite.montecarlo.transport import shell_gos as sg
from pyrite.montecarlo.transport.hard_inelastic import (
    POSITRON_BRANCH_OFFSET,
    _hard_loss_bounds,
    _hard_primary_cosine,
    _hard_secondary_cosine,
    _sample_hard_transfer_eV,
)
from pyrite.montecarlo.transport.shell_partition import catalog_shell_partition
from pyrite.montecarlo.transport.shell_rates import catalog_shell_oscillators
from pyrite.montecarlo.transport.shell_sampling import BRANCHES, sample_shell_hard_collision
from pyrite.montecarlo.transport.shell_transport import build_shell_inelastic_tables
from pyrite.montecarlo.transport.stopping import prepare_sbethe_stopping_table
from pyrite.xsgen.sbethe.catalog import resolve_catalog_table


def _positron_table(key):
    try:
        return resolve_catalog_table(key, projectile="positron").arrays()
    except Exception as error:  # pragma: no cover - depends on fetched tables
        pytest.skip(f"positron SBETHE tables are not installed: {error}")


@pytest.fixture(autouse=True)
def _require_pdatconf():
    if not config._default_path().is_file():
        pytest.skip("pinned SBETHE reference data have not been fetched")


@pytest.mark.parametrize("key", ("silicon", "mos2"))
@pytest.mark.parametrize("energy_eV", (6.0e3, 3.0e4, 1.0e6))
def test_numba_positron_transfer_and_recoil_match_host_sampler(key, energy_eV):
    _positron_table(key)
    material = catalog_shell_oscillators(key)
    part = catalog_shell_partition(key, energy_eV, 50.0, projectile="positron")
    assert part.closure.projectile == "positron"
    probabilities = part.hard_channel_probabilities.ravel()
    cumulative = np.cumsum(probabilities)
    checked = 0
    for flat in np.flatnonzero(probabilities > 0.0):
        index, branch = divmod(int(flat), 3)
        osc = material.oscillators[index]
        tagged = branch + POSITRON_BRANCH_OFFSET
        u_channel = (cumulative[flat] - 0.5 * probabilities[flat]) / cumulative[-1]
        for u_loss, u_recoil in ((0.0, 0.0), (0.37, 0.61), (0.93, 0.08), (0.999, 0.5)):
            host = sample_shell_hard_collision(material, part, u_channel, u_loss, u_recoil, 0.25)
            assert host.loss.oscillator_index == index and host.loss.branch == BRANCHES[branch]
            args = (energy_eV, osc.ionization_energy_eV, osc.resonance_energy_eV, tagged)
            w = _sample_hard_transfer_eV(*args, 50.0, u_loss)
            assert w == pytest.approx(host.loss.transfer_eV, rel=1e-10, abs=1e-9)
            assert _hard_primary_cosine(*args, w, u_recoil) == pytest.approx(
                host.cos_primary, rel=1e-9, abs=1e-12
            )
            if host.cos_secondary is not None:
                assert _hard_secondary_cosine(*args, w, u_recoil) == pytest.approx(
                    host.cos_secondary, rel=1e-9, abs=1e-12
                )
            checked += 1
    assert checked >= 8


@pytest.mark.parametrize("energy_eV", (1.0e4, 1.0e6))
def test_positron_close_loss_reaches_the_full_kinetic_energy(energy_eV):
    """W_max = E for positrons (Eq. 3.92), (E+U)/2 for electrons (Eq. 3.88)."""
    u, w = 100.0, 120.0
    _, upper_e, _ = _hard_loss_bounds(energy_eV, u, w, 2, 50.0)
    _, upper_p, _ = _hard_loss_bounds(energy_eV, u, w, 2 + POSITRON_BRANCH_OFFSET, 50.0)
    assert upper_e == pytest.approx(0.5 * (energy_eV + u))
    assert upper_p == energy_eV
    top = _sample_hard_transfer_eV(energy_eV, u, w, 2 + POSITRON_BRANCH_OFFSET, 50.0, 1.0 - 1e-15)
    assert 0.5 * (energy_eV + u) < top <= energy_eV


@pytest.mark.parametrize("energy_eV", (2.0e4, 1.0e6))
def test_sampled_close_transfers_follow_the_bhabha_dcs(energy_eV):
    """KS test of kernel draws against the normalized Eq. 3.92 CDF on [W_c, E]."""
    cutoff, u_shell = 200.0, 0.0
    branch = 2 + POSITRON_BRANCH_OFFSET
    rng = np.random.default_rng(276)
    draws = np.array(
        [
            _sample_hard_transfer_eV(energy_eV, u_shell, 15.0, branch, cutoff, u)
            for u in rng.random(20000)
        ]
    )
    assert draws.min() >= cutoff and draws.max() <= energy_eV
    b1, b2, b3, b4 = sg.bhabha_coefficients(energy_eV)

    def density(w):
        x = w / energy_eV
        return (1.0 - b1 * x + b2 * x * x - b3 * x**3 + b4 * x**4) / (w * w)

    norm, _ = quad(density, cutoff, energy_eV, epsrel=1e-12, limit=400)
    grid = np.geomspace(cutoff, energy_eV, 4001)
    pieces = [
        quad(density, a, b, epsrel=1e-12)[0] for a, b in zip(grid[:-1], grid[1:], strict=True)
    ]
    cdf_grid = np.concatenate([[0.0], np.cumsum(pieces)]) / norm

    def cdf(w):
        return np.interp(w, grid, cdf_grid)

    assert stats.kstest(draws, cdf).pvalue > 1e-3


def test_positron_tables_close_to_the_positron_stopping_at_every_node():
    arrays = _positron_table("silicon")
    prepared = prepare_sbethe_stopping_table(arrays)
    tables = build_shell_inelastic_tables(
        ["silicon"], 50.0, [prepared], 5.0, 30.0, projectile="positron"
    )
    assert tables.metadata()["projectile"] == "positron"
    n = tables.n_channels[0]
    assert np.all(tables.channel_branch[0, :n] >= POSITRON_BRANCH_OFFSET)
    assert np.all(tables.channel_code[0, :n] % 3 == tables.channel_branch[0, :n] - 3)
    log_e, soft_log_s = tables.soft_stopping_tables[0]
    lo = int(np.flatnonzero(np.isclose(prepared[0], log_e[0], rtol=0.0, atol=1e-15))[0])
    full = np.exp(prepared[1][lo : lo + log_e.size]) * 1e3
    soft = np.exp(soft_log_s) * 1e3
    for j, energy in enumerate(np.exp(log_e) * 1e3):
        part = catalog_shell_partition("silicon", float(energy), 50.0, projectile="positron")
        total = part.soft.total[1] + part.hard.total[1]
        assert soft[j] + part.hard.total[1] / total * full[j] == pytest.approx(full[j], rel=1e-12)


def test_electron_tables_are_unchanged_by_the_species_argument():
    arrays = resolve_catalog_table("silicon").arrays()
    prepared = prepare_sbethe_stopping_table(arrays)
    default = build_shell_inelastic_tables(["silicon"], 50.0, [prepared], 5.0, 30.0)
    explicit = build_shell_inelastic_tables(
        ["silicon"], 50.0, [prepared], 5.0, 30.0, projectile="electron"
    )
    assert "projectile" not in default.metadata()
    for name in ("hard_rate_per_ang", "channel_rate_per_ang", "channel_branch", "channel_code"):
        np.testing.assert_array_equal(getattr(default, name), getattr(explicit, name))
    assert np.all(default.channel_branch < POSITRON_BRANCH_OFFSET)


def test_positron_inner_shells_keep_their_bhabha_gos():
    """No EEDL substitution for positrons: inner-shell scale is exactly 1."""
    _positron_table("mos2")
    part = catalog_shell_partition("mos2", 3.0e4, 50.0, projectile="positron")
    closure = part.closure
    assert np.any(closure.inner)
    np.testing.assert_allclose(closure.scale[closure.inner], 1.0, rtol=1e-12)

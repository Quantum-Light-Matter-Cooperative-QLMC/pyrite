"""Positron ELSEPA elastic tables through the shared angular sampler (#276).

Validation: elsepa-positron-elastic-sampling
"""

import numpy as np
import pytest

from pyrite.montecarlo.transport.scattering import (
    _sample_cos_theta_elsepa,
    elsepa_angular_pdf,
    pack_elsepa_tables,
)


def _arrays(element, projectile):
    from pyrite.xsgen.elsepa.catalog import resolve_layer_tables

    try:
        return resolve_layer_tables([(element, 1.0)], projectile=projectile)[0].arrays
    except Exception as error:  # pragma: no cover - depends on fetched tables
        pytest.skip(f"{projectile} ELSEPA tables are not installed: {error}")


def _node(arrays, energy_eV):
    index = int(np.argmin(np.abs(np.log(arrays["energy_eV"] / energy_eV))))
    return index, float(arrays["energy_eV"][index])


def _sampled_mu(arrays, energy_eV, xi):
    _, start, length, logE, _, cdf, pdf, mu = pack_elsepa_tables([[arrays]], [np.array([1.0])])
    return np.array(
        [
            0.5
            * (
                1.0
                - _sample_cos_theta_elsepa(
                    energy_eV * 1e-3, x, logE, cdf, pdf, mu, start[0, 0], length[0, 0]
                )
            )
            for x in xi
        ]
    )


@pytest.mark.parametrize("element", ["Si", "W"])
@pytest.mark.parametrize("energy_eV", [1.0e3, 1.0e5, 1.0e7])
def test_sampled_positron_angles_follow_the_elsepa_positron_dcs(element, energy_eV):
    """KS distance at a node below the 99% critical value.

    The exact first moment of the sampled piecewise-linear density reproduces
    ELSEPA's positron ``sigma_1/sigma`` within 1% (the electron tolerance of
    ``elsepa-elastic-sampling``; the 10 MeV forward peak sets it).
    """
    arrays = _arrays(element, "positron")
    index, node_eV = _node(arrays, energy_eV)
    pdf, cdf = elsepa_angular_pdf(arrays["mu"], arrays["dcs_cm2_sr"][index : index + 1])
    draws = _sampled_mu(arrays, node_eV, np.random.default_rng(276).random(50_000))
    empirical = np.searchsorted(np.sort(draws), arrays["mu"], side="right") / draws.size
    assert np.max(np.abs(empirical - cdf[0])) < 1.63 / np.sqrt(draws.size)
    moment = np.trapezoid(2.0 * arrays["mu"] * pdf[0], arrays["mu"])
    expected = arrays["transport1_cm2"][index] / arrays["total_elastic_cm2"][index]
    assert moment == pytest.approx(expected, rel=1e-2)


def test_positrons_scatter_less_than_electrons_off_heavy_nuclei_at_low_energy():
    """Nuclear repulsion: positron sigma_1 < electron sigma_1 for W at 1 keV."""
    positron, electron = _arrays("W", "positron"), _arrays("W", "electron")
    index, _ = _node(positron, 1.0e3)
    assert positron["energy_eV"][index] == electron["energy_eV"][index]
    assert positron["transport1_cm2"][index] < 0.8 * electron["transport1_cm2"][index]

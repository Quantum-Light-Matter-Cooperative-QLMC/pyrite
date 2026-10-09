"""Independent captured-field sample uncertainties for issue #350."""

from dataclasses import replace

import numpy as np
import pytest

from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.spectrum.coherent_dispersion import CoherentDispersionLaw
from pyrite.montecarlo.spectrum.coherent_normalization import _sample_row_fields
from pyrite.montecarlo.spectrum.coherent_sample_certificates import (
    sample_dispersion_bounds,
    sample_row_norm_certificate,
)
from pyrite.montecarlo.spectrum.coherent_windows import CoherentRowField


def _row():
    return CoherentRowField(
        label="sample certificate",
        energy_eV=np.array([1000.0, 1000.0]),
        amplitude=np.array([[1.0, -0.999], [0.3j, -0.2j]]),
        duration_ang=np.array([100.0, 120.0]),
        centre_ang=np.array([50.0, 160.0]),
        electron=np.array([0, 1]),
        start_transmission=np.ones(2),
        end_transmission=np.ones(2),
        mean_transmission=np.ones(2),
        attenuation_slope_ang=np.zeros(2),
        escape_mid_ang=np.array([50.0, 300.0]),
        escape_change_ang=np.array([20.0, -40.0]),
        phase_rad=np.array([0.1, 0.2]),
    )


def _reference_norms(row, energy, omega):
    # Independent midpoint formula at high precision, rather than the
    # certificate's bright-endpoint formula or production formation reducer.
    from mpmath import mp

    with mp.workdps(90):
        pieces = []
        for j in range(row.energy_eV.size):
            dd = mp.mpf(float(row.duration_ang[j]))
            q = mp.mpf(float(row.slope_ang[j])) * dd / 2
            v = (
                dd
                * (mp.mpf(float(energy)) - mp.mpf(float(row.energy_eV[j])))
                / (2 * mp.mpf(HBARC_EV_ANG))
            )
            v -= mp.mpf(float(row.escape_change_ang[j])) * mp.mpf(float(omega)) / 2
            bright = mp.mpf(float(max(row.start_transmission[j], row.end_transmission[j])))
            z = -q + 1j * v
            formation = bright * mp.exp(-abs(q)) * (mp.sinh(z) / z if z != 0 else 1)
            phase = mp.mpf(float(energy)) * mp.mpf(float(row.centre_ang[j])) / mp.mpf(HBARC_EV_ANG)
            phase -= mp.mpf(float(row.phase_rad[j])) + mp.mpf(
                float(row.escape_mid_ang[j])
            ) * mp.mpf(float(omega))
            pieces.append(
                [
                    mp.mpc(complex(c)) * dd * formation * mp.exp(1j * phase)
                    for c in row.amplitude[:, j]
                ]
            )
        fields = [
            [
                sum(pieces[j][p] for j in range(len(pieces)) if row.electron[j] == e)
                for e in np.unique(row.electron)
            ]
            for p in range(row.amplitude.shape[0])
        ]
        return (
            mp.sqrt(sum(abs(z) ** 2 for polarization in fields for z in polarization)),
            mp.sqrt(sum(abs(sum(polarization)) ** 2 for polarization in fields)),
        )


@pytest.mark.parametrize("damping", [0.0, 0.01, -0.01, 20.0, -20.0])
def test_certificate_encloses_signed_absorption_and_nonlinear_phase(damping):
    row = _row()
    q = damping * row.duration_ang / 2
    row = replace(
        row,
        attenuation_slope_ang=np.full(2, damping),
        start_transmission=np.exp(-np.maximum(-2 * q, 0)),
        end_transmission=np.exp(-np.maximum(2 * q, 0)),
    )
    energies = np.array([999.9, 1000.0, 1000.1])
    phase = 1e-9 * energies**2
    samples = _sample_row_fields(row, lambda E: 1e-9 * E**2, energies, remove_global_phase=True)
    norms, errors = sample_row_norm_certificate(
        row, samples, energies, np.column_stack((phase, phase))
    )
    for k, energy in enumerate(energies):
        for component, exact in enumerate(_reference_norms(row, energy, phase[k])):
            assert abs(exact - float(norms[k, component])) <= float(errors[k, component])
    assert np.all(errors < 1e-9 * np.maximum(1, norms))


def test_certificate_charges_an_inaccurate_nominal_sample():
    row = _row()
    energies = np.array([1000.0])
    samples = _sample_row_fields(
        row, lambda E: np.zeros_like(E), energies, remove_global_phase=True
    )
    samples[0, 0, 0] += 0.25
    norms, errors = sample_row_norm_certificate(row, samples, energies, np.zeros((1, 2)))
    for component, exact in enumerate(_reference_norms(row, 1000.0, 0.0)):
        assert abs(exact - float(norms[0, component])) <= float(errors[0, component])
    assert errors.max() > 1e-3


def test_certificate_propagates_material_phase_uncertainty():
    row = _row()
    energies = np.array([1000.0])
    bounds = np.array([[-1e-5, 1e-5]])
    samples = _sample_row_fields(
        row, lambda E: np.zeros_like(E), energies, remove_global_phase=True
    )
    norms, errors = sample_row_norm_certificate(row, samples, energies, bounds)
    for omega in np.linspace(*bounds[0], 11):
        for component, exact in enumerate(_reference_norms(row, 1000.0, omega)):
            assert abs(exact - float(norms[0, component])) <= float(errors[0, component])


def test_certificate_rounds_subnormal_norm_errors_outward():
    from mpmath import mp

    row = replace(
        _row(),
        energy_eV=np.full(3, 1000.0),
        amplitude=np.full((2, 3), 1e-320 + 0j),
        duration_ang=np.full(3, 100.0),
        centre_ang=np.zeros(3),
        electron=np.array([7, 2, 7]),
        start_transmission=np.ones(3),
        end_transmission=np.ones(3),
        mean_transmission=np.ones(3),
        attenuation_slope_ang=np.zeros(3),
        escape_mid_ang=np.zeros(3),
        escape_change_ang=np.zeros(3),
        phase_rad=np.zeros(3),
    )
    norms, errors = sample_row_norm_certificate(
        row, np.zeros((2, 2, 1), complex), np.array([1000.0]), np.zeros((1, 2))
    )
    with mp.workdps(90):
        amplitude = mp.mpf(float(row.amplitude[0, 0].real))
        for component, exact in enumerate(
            (mp.sqrt(10) * 100 * amplitude, mp.sqrt(2) * 300 * amplitude)
        ):
            assert norms[0, component] == 0.0
            assert mp.mpf(float(errors[0, component])) >= exact


@pytest.mark.parametrize("bounds", [[[0.1, -0.1]], [[np.nan, 0.1]], [[0.0, np.inf]]])
def test_certificate_refuses_uncertified_material_phase_inputs(bounds):
    with pytest.raises(ValueError, match="dispersion"):
        sample_row_norm_certificate(
            _row(), np.zeros((2, 2, 1), complex), np.array([1000.0]), np.array(bounds)
        )


@pytest.mark.parametrize("crystal,use_henke", [("hopg", True), ("wse2", True), ("hopg", False)])
def test_material_point_enclosures_cover_an_independent_complex_square_root(crystal, use_henke):
    from mpmath import mp

    law = CoherentDispersionLaw(crystal, 200.0, 5000.0, use_henke=use_henke)
    energies = np.array([201.0, 283.0, 1000.0, 4999.0])
    bounds = sample_dispersion_bounds(law, energies)
    with mp.workdps(90):
        for k, energy in enumerate(energies):
            E = mp.mpf(float(energy))
            forward = mp.mpc(law.forward_constant)
            for atom in law.atoms:
                poly = atom.f1[0]
                i = int(np.searchsorted(poly.x, energy, side="right") - 1)
                # Explicit power sum, distinct from interval Horner evaluation.
                f1 = sum(
                    mp.mpf(float(c)) * (E - mp.mpf(float(poly.x[i]))) ** (len(poly.c) - 1 - j)
                    for j, c in enumerate(poly.c[:, i])
                )
                i = int(np.searchsorted(atom.energies, energy, side="right") - 1)
                fraction = mp.log(E / mp.mpf(float(atom.energies[i]))) / mp.log(
                    mp.mpf(float(atom.energies[i + 1])) / mp.mpf(float(atom.energies[i]))
                )
                f2 = (
                    mp.mpf(float(atom.f2[i]))
                    * (mp.mpf(float(atom.f2[i + 1])) / mp.mpf(float(atom.f2[i]))) ** fraction
                )
                forward += atom.count * (f1 + 1j * f2)
            chi = -mp.mpf(law.prefactor) * forward / E**2
            exact = (1 - mp.sqrt(1 + chi).real) * E / mp.mpf(HBARC_EV_ANG)
            assert mp.mpf(float(bounds[k, 0])) <= exact <= mp.mpf(float(bounds[k, 1]))


def test_certificates_do_not_change_global_interval_or_scalar_precision():
    from mpmath import iv, mp

    before = mp.dps, iv.dps
    row = _row()
    sample_row_norm_certificate(
        row, np.zeros((2, 2, 1), complex), np.array([1000.0]), np.zeros((1, 2)), precision_digits=30
    )
    sample_dispersion_bounds(
        CoherentDispersionLaw("hopg", 900.0, 1100.0), np.array([1000.0]), precision_digits=70
    )
    assert (mp.dps, iv.dps) == before


def test_certificate_refuses_finite_fields_whose_nominal_norm_overflows():
    with pytest.raises(ValueError, match="norms overflowed"):
        sample_row_norm_certificate(
            _row(), np.full((2, 2, 1), 1e308, complex), np.array([1000.0]), np.zeros((1, 2))
        )


def test_composed_band_bounds_need_no_guessed_norm_errors():
    from mpmath import mp
    from scipy.integrate import quad

    from pyrite.montecarlo.spectrum.coherent_normalization import (
        _certified_physical_row_power_bounds,
    )

    row = _row()
    law = CoherentDispersionLaw("hopg", 990.0, 1010.0, use_henke=False)
    nodes = np.linspace(990.1, 1009.9, 31)
    lower, upper = _certified_physical_row_power_bounds(
        row, law, 990.0, 1010.0, nodes, form_factor_bounds=(0.4, 0.4)
    )

    def intensity(energy):
        with mp.workdps(90):
            E = mp.mpf(float(energy))
            chi = -mp.mpf(law.prefactor) * mp.mpf(law.forward_constant) / E**2
            omega = (1 - mp.sqrt(1 + chi)) * E / mp.mpf(HBARC_EV_ANG)
            # Float phase conversion is much smaller than this band envelope;
            # the independent point-phase test above checks exact enclosures.
            grouped, flat = _reference_norms(row, energy, float(omega))
            return float(0.6 * grouped**2 + 0.4 * flat**2)

    actual = quad(intensity, 990.0, 1010.0, epsabs=1e-6)[0]
    assert 0 < lower <= actual <= upper
    with pytest.raises(ValueError, match="strictly interior"):
        _certified_physical_row_power_bounds(row, law, 990.0, 1010.0, np.array([990.0]))

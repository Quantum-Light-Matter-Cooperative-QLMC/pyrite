"""Directional observation of axial fibre cases under the admission opt-in (#1148).

Validation: cr-axial-directional-observation
"""

import warnings

import numpy as np
import pytest

import pyrite as pr
from pyrite.campaign.lowering import build_case
from pyrite.config import FixedSampling
from pyrite.detectors import EnergyBins, PointDetector
from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo import runner
from pyrite.montecarlo.case import AxialAdmissionError
from pyrite.montecarlo.channeling.axial_fiber_angular import (
    linear_depth_weights,
    polarization_spectrum,
)
from pyrite.montecarlo.channeling.bloch import longitudinal_kinematics

GRID = np.linspace(800.0, 10000.0, 47)
KINETIC = 3.5e6


def _scene(detector, admission="accept-inaccurate", lines=4):
    beam = pr.Beam(
        energy_keV=3500.0, transverse=pr.beams.Spot(), longitudinal=pr.beams.PointBunch()
    )
    target = pr.geometry.Slab(
        "silicon",
        thickness_ang=3000.0,
        tilt_deg=0.0,
        width=None,
        height=None,
        cut=pr.geometry.CrystalCut(surface_hkl=(0, 0, 1)),
        allow_normal_incidence=True,
    )
    numerics = pr.Numerics(
        sampling=FixedSampling(cxr_histories=lines, brem_histories=lines),
        energy_bins=EnergyBins(cxr=GRID, brem=GRID),
    )
    channeling = pr.Channeling(
        "axial", axial={"version": 2, "axis_uvw": [0, 0, 1]}, admission=admission
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the unadmitted warning
        return build_case(
            pr.Scene(beam, target, detector),
            numerics,
            physics=pr.Physics(emission="incoherent", radiative_model="uncoupled"),
            channeling=channeling,
        )


def _planar_detector():
    return pr.detectors.PlanarDetector(
        pr.detectors.PlanarPose((0.0, 0.0, 1000.0)),
        pixels=pr.detectors.PixelGrid((2, 2), (1.0, 1.0)),
        response=pr.detectors.IdealPhotonCounter(),
        scorer=pr.config.PixelScorer((2, 2)),
    )


# ---- quadrature ---------------------------------------------------------------


def test_linear_weights_reduce_to_the_trapezoid_rule_without_phase():
    z = np.array([0.0, 2.0, 5.0, 9.0, 10.0])
    weights = linear_depth_weights(z, [0.0])[0]
    spacing = np.diff(z)
    trapezoid = np.r_[spacing[0], spacing[:-1] + spacing[1:], spacing[-1]] / 2
    np.testing.assert_allclose(weights, trapezoid, rtol=1e-15)


@pytest.mark.parametrize("omega", [1e-5, 0.02, 0.7, 6.0, 90.0])
def test_linear_weights_integrate_the_interpolant_exactly(omega):
    rng = np.random.default_rng(5)
    z = np.sort(np.r_[0.0, rng.uniform(0.0, 50.0, 17), 50.0])
    f = rng.normal(size=z.size) + 1j * rng.normal(size=z.size)
    # Exact: sum over intervals of int (a + b t) e^{i omega (z_k + h t)} h dt.
    exact = 0.0
    for k in range(z.size - 1):
        h = z[k + 1] - z[k]
        t = np.linspace(0.0, 1.0, 400001)
        values = (f[k] + (f[k + 1] - f[k]) * t) * np.exp(1j * omega * (z[k] + h * t))
        exact += h * np.trapezoid(values, t)
    np.testing.assert_allclose(linear_depth_weights(z, [omega])[0] @ f, exact, rtol=1e-6)


def test_linear_quadrature_has_no_line_replicas_at_large_angle():
    """One line, far off axis: the trapezoid sum aliases replicas, linear does not."""
    gamma, beta = longitudinal_kinematics(KINETIC)
    hbar_v = HBARC_EV_ANG * beta
    z = np.linspace(0.0, 3000.0, 201)
    delta = 30.0  # eV transition
    phase = np.exp(-1j * delta * z / hbar_v)
    g = np.zeros((z.size, z.size, 2, 2), dtype=complex)
    g[:, :, 0, 0] = phase[:, None] * phase.conj()[None, :]
    theta = np.deg2rad(30.0)
    doppler = (1.0 - beta) + 2.0 * beta * np.sin(theta / 2) ** 2
    line = delta / doppler
    replica = (delta + 2 * np.pi * hbar_v / (z[1] - z[0])) / doppler
    energies = np.array([line, replica, 0.5 * (line + replica)])
    linear = polarization_spectrum(g, z, energies, KINETIC, [theta], [0.0], quadrature="linear")[
        0, 0, 0, 0
    ].real
    trapezoid = polarization_spectrum(
        g, z, energies, KINETIC, [theta], [0.0], max_phase_step=np.inf
    )[0, 0, 0, 0].real
    assert trapezoid[1] > 0.5 * trapezoid[0]  # the aliased replica
    # Linear: only the finite-slab sinc^2 sidelobe, (2 h / 2 pi L)^2 ~ 2.5e-6 of the peak.
    assert linear[1] < 1e-4 * linear[0]
    # The line itself agrees with the continuous integral |L|^2 to O((delta h / hbar v)^2).
    np.testing.assert_allclose(linear[0], trapezoid[0], rtol=2e-2)


def test_polarization_spectrum_refuses_an_unknown_quadrature():
    z = np.linspace(0.0, 10.0, 3)
    with pytest.raises(ValueError, match="quadrature"):
        polarization_spectrum(
            np.zeros((3, 3, 2, 2)), z, [1000.0], KINETIC, [0.0], [0.0], quadrature="simpson"
        )


# ---- admission ----------------------------------------------------------------


@pytest.mark.parametrize(
    "detector",
    [PointDetector(observation_angle_deg=1.0), pr.Detector(observation_angle_deg=0.0), None],
    ids=["off-axis", "aperture", "planar"],
)
def test_directional_observation_is_unadmitted_and_needs_the_opt_in(detector):
    detector = _planar_detector() if detector is None else detector
    with pytest.raises(AxialAdmissionError, match="zero-aperture.*accept-inaccurate"):
        _scene(detector, admission="admitted")
    case = _scene(detector)
    assert case["channeling"]["case_key"] == "unadmitted"
    assert case["channeling"]["observable"] == "directional"


def test_directional_observation_refuses_fewer_than_two_line_electrons():
    with pytest.raises(AxialAdmissionError, match="at least 2 line electrons"):
        _scene(PointDetector(observation_angle_deg=1.0), lines=1)


def test_on_axis_point_detector_stays_admitted():
    case = _scene(PointDetector())
    assert case["channeling"]["case_key"] == "si-fibre/B"
    assert case["channeling"]["observable"] == "on-axis"


def test_admitted_on_axis_cases_refuse_observation_directions_before_work(monkeypatch):
    case = _scene(PointDetector())
    monkeypatch.setattr(runner, "_transport_case", lambda *a, **k: pytest.fail("transported"))
    with pytest.raises(AxialAdmissionError, match="directional observable"):
        runner.run_case_directions(case, np.array([[0.0, 0.0, 1.0]]))
    with pytest.raises(AxialAdmissionError, match="directional observable"):
        runner.run_case(case, observation_directions=np.array([[0.0, 0.0, 1.0]]))


# ---- end to end ---------------------------------------------------------------


@pytest.fixture(scope="module")
def directional_run():
    case = _scene(_planar_detector())
    directions = np.array(
        [
            [0.0, 0.0, 1.0],
            [np.sin(0.1), 0.0, np.cos(0.1)],
            [0.0, np.sin(0.1), np.cos(0.1)],
            [np.sin(np.pi / 6), 0.0, np.cos(np.pi / 6)],
            [np.sin(2.0), 0.0, np.cos(2.0)],
        ]
    )
    transport = runner._transport_case(case)
    from pyrite.montecarlo.runner.directions import directional_outputs

    result = directional_outputs(case, transport, directions, runner._spectrum_case)
    return case, transport, directions, result


@pytest.mark.slow
def test_directional_source_matches_the_on_axis_fibre_and_falls_off_axis(directional_run):
    from pyrite.montecarlo.runner.axial_directional import directional_spectrum

    case, transport, directions, result = directional_run
    info = transport["segs"]["channeling"]
    record = info["directional"]
    energies = info["source"]["E_eV"]
    band = lambda s, e=energies: np.trapezoid(s, e)  # noqa: E731
    on_axis, sem = directional_spectrum(case, record, directions[0], energies)
    # Same histories and grid; linear vs trapezoid depth quadrature on axis.
    assert band(on_axis) == pytest.approx(band(info["source"]["spectrum"]), rel=0.02)
    assert np.all(sem >= 0.0) and band(sem) > 0.0
    pairs = [directional_spectrum(case, record, d, energies) for d in directions]
    at = [band(spectrum) for spectrum, _ in pairs]
    # C4 symmetry of <100>: +x and +y agree within the batch error (4 electrons).
    assert abs(at[1] - at[2]) < 4.0 * np.hypot(band(pairs[1][1]), band(pairs[2][1]))
    assert at[1] < at[0]
    assert 0.0 < at[3] < 1e-2 * at[0]
    # The runner's directional outputs are these spectra on the joint observation grid.
    grid = np.asarray(result["E_grid"])
    spectra = np.asarray(result["spec_channeling_by_direction"])
    for direction, spectrum in zip(directions, spectra, strict=True):
        expected = directional_spectrum(case, record, direction, grid)[0]
        np.testing.assert_allclose(spectrum, expected, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(spectra[4], 0.0)  # backward hemisphere: forward-only


@pytest.mark.slow
def test_directional_fates_and_on_axis_source_are_the_admitted_draws(directional_run):
    """The directional opt-in adds a record; it changes no draw of the same geometry."""
    _, transport, _, _ = directional_run
    admitted = _scene(PointDetector())
    plain = runner._transport_case(admitted)
    a, b = transport["segs"]["channeling"], plain["segs"]["channeling"]
    assert a["fate_fractions"] == b["fate_fractions"]
    np.testing.assert_array_equal(a["source"]["spectrum"], b["source"]["spectrum"])
    assert "directional" in a and "directional" not in b

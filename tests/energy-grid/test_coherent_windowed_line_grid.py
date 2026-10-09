"""Coherent windowed automatic line-grid resolution (#350).

A windowed policy resolves ``coherent_emission`` inside per-row coherent
windows; a policy without windows still refuses it (#117). The synthetic
tests pin the derived rules and their limiting cases; the tiny real transport
anchors windowed-auto against a fine explicit grid on identical trajectories.
"""

from dataclasses import replace

import numpy as np
import pytest

from pyrite import _line_grid_policy
from pyrite._line_grid_policy import (
    LineGridToleranceError,
    LineShapePrecisionWarning,
    resolve_line_grid_policy,
)
from pyrite.energy_grid import convergence_case as cc
from pyrite.energy_grid.convergence import spectrum_observables
from pyrite.materials.crystal import HBARC_EV_ANG
from pyrite.montecarlo.runner.line_grid import resolve_line_grid
from pyrite.montecarlo.spectrum import coherent_windows as cw
from pyrite.montecarlo.spectrum.line_seeds import sincsq_upper_tail_bound

_ENV_NAMES = (
    "PYRITE_ENERGY_GRID_RTOL",
    "PYRITE_ENERGY_GRID_RTOL_INTRINSIC_SOURCE",
    "PYRITE_ENERGY_GRID_RTOL_DETECTED_COUNTS",
    "PYRITE_ENERGY_GRID_MAX_SPACING_EV",
    "PYRITE_ENERGY_GRID_ULPS",
    "PYRITE_ENERGY_GRID_MAX_POINTS",
)
#: The intrinsic-source tolerance the issue asks the windowed axis to meet.
INTRINSIC_RTOL = 1.0e-3


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(_line_grid_policy, "cache_dir", lambda: tmp_path)


# ---- synthetic rows -----------------------------------------------------------
def _field(durations, energies, amplitudes, electrons=None, gaps=None, attenuation=None):
    """Pieces contiguous in ``tau`` per electron (or separated by ``gaps``)."""
    durations = np.asarray(durations, dtype=float)
    n = durations.size
    electrons = np.zeros(n, dtype=int) if electrons is None else np.asarray(electrons)
    gaps = np.zeros(n) if gaps is None else np.asarray(gaps, dtype=float)
    lo = np.cumsum(np.concatenate(([0.0], durations[:-1] + gaps[1:])))
    ones = np.ones(n) if attenuation is None else np.asarray(attenuation, dtype=float)
    return cw.CoherentRowField(
        label="synthetic",
        energy_eV=np.asarray(energies, dtype=float) * np.ones(n),
        amplitude=(np.asarray(amplitudes, dtype=complex) * np.ones(n)).reshape(1, n),
        duration_ang=durations,
        centre_ang=lo + 0.5 * durations,
        electron=electrons,
        start_transmission=ones,
        end_transmission=ones,
        mean_transmission=ones,
    )


def _seeds(rows, decoherence=None, count=1, start=10.0, stop=5000.0):
    return cw.coherent_window_seeds(
        rows,
        start_eV=start,
        stop_eV=stop,
        electron_count=count,
        decoherence=decoherence,
    )


def _exact_tail(field, edge_eV, *, reach=4000.0):
    """Independent upper-tail power fraction of the time-domain proxy field.

    Each piece is ``a_j exp(-i omega_j tau + i psi_j)`` on its support with
    ``psi`` chosen so the phase is continuous at contiguous joints; its
    transform is integrated numerically beyond ``edge_eV`` and the remainder
    past ``reach`` widths is bounded by ``(sum |a| (2/x))^2``.
    """
    omega0 = edge_eV / HBARC_EV_ANG
    x_max = reach * (edge_eV - field.energy_eV.max()) / HBARC_EV_ANG
    lo = field.centre_ang - 0.5 * field.duration_ang
    hi = field.centre_ang + 0.5 * field.duration_ang
    span = max(
        float(np.ptp(np.concatenate((lo[field.electron == e], hi[field.electron == e]))))
        for e in np.unique(field.electron)
    )
    nodes = max(200_001, int(x_max * span / 0.2) + 1)
    omega = omega0 + np.linspace(0.0, x_max, nodes)
    carrier = field.energy_eV / HBARC_EV_ANG
    psi = np.zeros(lo.size)
    for j in range(1, lo.size):
        if field.electron[j] == field.electron[j - 1] and np.isclose(lo[j], hi[j - 1]):
            psi[j] = psi[j - 1] + (carrier[j] - carrier[j - 1]) * lo[j]
    amplitude = field.amplitude[0]
    total = np.zeros((field.electron.max() + 1, omega.size), dtype=complex)
    for j in range(lo.size):
        x = omega - carrier[j]
        piece = (np.exp(1j * x * hi[j]) - np.exp(1j * x * lo[j])) / (1j * x)
        total[field.electron[j]] += amplitude[j] * np.exp(1j * psi[j]) * piece
    power = np.sum(np.abs(total) ** 2, axis=0)
    integral = np.trapezoid(power, omega)
    remainder = (2.0 * np.abs(amplitude).sum()) ** 2 / (omega[-1] - carrier.max())
    return (integral + remainder) / (2.0 * np.pi * field.power)


def test_single_segment_leak_is_the_incoherent_sinc_tail_bound():
    """Limiting case: within ``M hbar c / dd`` one segment gives ``w / (pi^2 u)``."""
    duration = 800.0
    field = _field([duration], 1000.0, 1.0)
    width = 2.0 * np.pi * HBARC_EV_ANG / duration
    jumps = cw._RowJumps(field)
    for distance in (5.0, 50.0):
        leak = cw.coherent_edge_leak(jumps, field.power, 1000.0 + distance, upper=True)
        assert leak == pytest.approx(float(sincsq_upper_tail_bound(width, distance)), rel=1e-12)
    # Far beyond: the two edges decorrelate; the bound stays above the exact tail.
    far = cw.coherent_edge_leak(jumps, field.power, 1500.0, upper=True)
    assert _exact_tail(field, 1500.0) <= far <= float(sincsq_upper_tail_bound(width, 500.0))

    _seeds_out, summary = _seeds([field])
    row = summary["rows"][0]
    # Step limit: pi hbar c over the segment's own support, i.e. w / 2.
    assert row["nyquist_electron_eV"] == pytest.approx(0.5 * width, rel=1e-12)
    assert row["nyquist_all_eV"] == pytest.approx(0.5 * width, rel=1e-12)
    assert row["step_electron_eV"] == pytest.approx(
        0.5 * width / summary["oversampling"], rel=1e-12
    )


@pytest.mark.parametrize(
    "duration,pieces,distance",
    [(800.0, 2, 50.0), (50.0, 4, 5000.0), (5000.0, 400, 50.0), (5000.0, 400, 500.0)],
)
@pytest.mark.parametrize("upper", [False, True])
def test_a_collinear_split_leaks_exactly_what_the_whole_flight_leaks(
    duration, pieces, distance, upper
):
    """Joints of one electron share their phase: equal amplitudes cancel."""
    whole = _field([duration], 1000.0, 1.0)
    split = _field(np.full(pieces, duration / pieces), 1000.0, 1.0)
    edge = 1000.0 + (distance if upper else -distance)
    whole_leak = cw.coherent_edge_leak(cw._RowJumps(whole), whole.power, edge, upper=upper)
    split_leak = cw.coherent_edge_leak(cw._RowJumps(split), split.power, edge, upper=upper)
    assert split_leak == pytest.approx(whole_leak, rel=1e-12)
    _, whole_summary = _seeds([whole])
    _, split_summary = _seeds([split])
    for key in ("window_eV", "span_all_ang", "span_electron_ang"):
        np.testing.assert_allclose(
            whole_summary["rows"][0][key], split_summary["rows"][0][key], rtol=1e-12
        )
    assert split_summary["rows"][0]["joints"] == pieces - 1


@pytest.mark.parametrize("change", ["amplitude", "carrier", "slope", "subnormal"])
@pytest.mark.parametrize("polarization", [0, 1])
def test_nonzero_captured_joints_are_retained(change, polarization):
    """No tolerance or squared-norm underflow may erase a nonzero jump."""
    field = _field([25.0, 25.0], 1000.0, 1.0)
    field = replace(field, amplitude=np.ones((2, 2), dtype=complex))
    if change == "amplitude":
        field.amplitude[polarization, 1] = np.nextafter(1.0, 2.0)
    elif change == "carrier":
        field.energy_eV[1] = np.nextafter(1000.0, 1001.0)
    elif change == "slope":
        field = replace(field, attenuation_slope_ang=np.array([0.0, 1e-15]))
    else:
        field.amplitude[:] = np.nextafter(0.0, 1.0)
        field.amplitude[polarization, 1] *= 2.0
    jumps = cw._RowJumps(field)
    assert jumps.joints == 1
    assert jumps.jumps == 3


def test_zero_endpoint_fields_leave_no_tail_clusters():
    """Both-zero fields cancel even with distinct carriers and slopes."""
    field = replace(
        _field([25.0, 25.0], [1000.0, 1001.0], 0.0),
        attenuation_slope_ang=np.array([0.0, 0.1]),
    )
    jumps = cw._RowJumps(field)
    assert jumps.jumps == 0
    for upper, edge in ((True, 1500.0), (False, 500.0)):
        assert cw.coherent_edge_leak(jumps, field.power, edge, upper=upper) == 0.0


def test_default_coherent_step_resolves_a_straight_track_half_maximum():
    """An analytic sinc line catches the four-node FWHM interpolation error.

    One constant-energy undamped track has density sinc(E-E0)**2 when
    its retardation duration is 2*pi*hbar*c. Its half-maximum root is
    independent of the grid; the intrinsic tolerance is the issue's 1e-3.
    """
    from scipy.optimize import brentq

    from pyrite._line_windows import build_window_plan

    centre = 1000.0
    field = _field([2 * np.pi * HBARC_EV_ANG], centre, 1.0)
    seeds, _ = _seeds([field], start=980.0, stop=1020.0)
    grid = build_window_plan(980.0, 1020.0, 1.0, seeds).coordinates()
    density = np.sinc(grid - centre) ** 2
    actual = spectrum_observables(grid, density, np.zeros_like(grid), detectors={})
    exact_fwhm = 2 * brentq(lambda x: np.sinc(x) ** 2 - 0.5, 0.0, 1.0, xtol=1e-14)
    assert abs(actual["fwhm_eV"] / exact_fwhm - 1) <= INTRINSIC_RTOL


@pytest.mark.parametrize("origin_ang", [0.0, 1.0e7])
def test_real_gaps_do_not_cancel_under_retardation_translation(origin_ang):
    """Fresh verifier's separated-interval tail, independent of time origin.

    Ten 1 Ang pieces separated by 1 Ang gaps have a known transform. A
    common time translation changes only its phase, never tail power.
    The old position-relative tolerance merged nine real gaps at 1e7 Ang,
    understating the infinite-tail bound by 7.38x versus a finite integral.
    """
    from dataclasses import replace

    count = 10
    field = _field(np.ones(count), 1000.0, 1.0, gaps=np.ones(count))
    field = replace(field, centre_ang=field.centre_ang + origin_ang)
    frequency = np.linspace(100.0, 1000.0, 180001)
    phase_sum = np.zeros(frequency.size, complex)
    for index in range(count):
        phase_sum += np.exp(2j * index * frequency)
    transform = 2 * np.sin(frequency / 2) / frequency * phase_sum
    finite_tail = np.trapezoid(np.abs(transform) ** 2, frequency) / (2 * np.pi * count)
    jumps = cw._RowJumps(field)
    assert finite_tail <= cw.coherent_edge_leak(
        jumps, field.power, 1000.0 + 100 * HBARC_EV_ANG, upper=True
    )
    assert jumps.joints == 0


def test_offset_free_centres_keep_the_transport_roundoff_budget():
    """Subtracting a per-electron offset must not shrink the joint tolerance.

    Abutting pieces whose transport coordinates were ~4e6 Ang carry ~1e-9 Ang
    endpoint residuals. After an offset-free translation to ~1 Ang centres the
    absolute-operand scale still joins them; a real 1 Ang gap stays a gap.
    Validation: coherent-line-grid-windowed-resolution
    """
    from dataclasses import replace

    field = _field(np.ones(4), 1000.0, 1.0)
    residual = np.array([0.0, 1e-9, -1e-9, 1e-9])
    shifted = replace(field, centre_ang=field.centre_ang + residual)
    assert cw._RowJumps(shifted).joints < 3
    scaled = replace(shifted, centre_scale_ang=np.full(4, 4.0e6))
    assert cw._RowJumps(scaled).joints == 3
    gapped = _field(np.ones(4), 1000.0, 1.0, gaps=np.ones(4))
    assert cw._RowJumps(replace(gapped, centre_scale_ang=np.full(4, 4.0e6))).joints == 0


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("copies", [1, 3])
def test_continuous_attenuation_cusp_stays_inside_the_tail_bound(upper, copies):
    """Independent two-piece transform from the 2026-10-06 verifier.

    Equal endpoint fields at the central joint have opposite attenuation
    slopes. The undamped jump formula incorrectly cancels that joint.
    A finite integral alone suffices to falsify an infinite-tail bound.
    """
    length, optical_half_depth, energy = 1000.0, 8.0, 1000.0
    slope = optical_half_depth / length
    epsilon = np.exp(-optical_half_depth)
    centres = np.arange(copies) * 6000.0
    field = cw.CoherentRowField(
        label="continuous attenuation cusp",
        energy_eV=np.full(2 * copies, energy),
        amplitude=np.ones((1, 2 * copies), complex),
        duration_ang=np.full(2 * copies, length),
        centre_ang=(centres[:, None] + np.array([-length / 2, length / 2])).ravel(),
        electron=np.zeros(2 * copies, int),
        start_transmission=np.tile([epsilon, 1.0], copies),
        end_transmission=np.tile([1.0, epsilon], copies),
        mean_transmission=np.full(2 * copies, (1 - epsilon**2) / (2 * optical_half_depth)),
    )
    distance = np.linspace(100.0, 1000.0, 100001)
    transform = 2 * np.real(
        (1 - np.exp((-slope + 1j * distance / HBARC_EV_ANG) * length))
        / (slope - 1j * distance / HBARC_EV_ANG)
    )
    transform = transform * np.exp(1j * centres[:, None] * distance / HBARC_EV_ANG).sum(axis=0)
    finite_tail = np.trapezoid(np.abs(transform) ** 2, distance) / (
        2 * np.pi * HBARC_EV_ANG * field.power
    )
    if copies == 1:
        assert finite_tail == pytest.approx(8.09864494379e-4, rel=2e-6)
    edge = energy + (100.0 if upper else -100.0)
    assert finite_tail <= cw.coherent_edge_leak(cw._RowJumps(field), field.power, edge, upper=upper)


def test_near_cancelling_joint_keeps_a_positive_tail_bound():
    """Verifier's tiny carrier separation once returned a negative integral."""
    u1, u2 = 100.0, 100.0 * (1 + 1e-8)
    field = _field([100.0, 100.0], [1100.0 - u1, 1100.0 - u2], [1.0, u2 / u1])
    # Joint is between the two pieces; finite integral computed directly,
    # using a common numerator to avoid reference cancellation too.
    distance = np.linspace(0.0, 1000.0, 10001)
    joint = ((u2 - u1) * distance / u1) / ((u1 + distance) * (u2 + distance))
    finite_tail = np.trapezoid(joint**2, distance)
    assert finite_tail > 0.0
    assert cw._RowJumps(field).terms(1100.0, True)[1] >= finite_tail


def test_capture_preserves_extreme_attenuation_endpoints_and_slopes():
    """Capture must retain dim endpoints without cancellation or log(0)."""
    from types import SimpleNamespace

    q = np.array([-400.0, -20.0, 0.0, 20.0, 400.0])
    start = np.exp(np.minimum(2 * q, 0.0))
    end = np.exp(np.minimum(-2 * q, 0.0))
    duration = np.full(q.size, 1000.0)
    st = SimpleNamespace(
        d_all_geom=np.arange(q.size) * 1000.0,
        seg_elec_id=np.arange(q.size),
        escape_ends=(np.arange(q.size) * 100.0, np.arange(q.size) * 100.0 + 20.0),
        capture_phase_rad=np.arange(q.size) * 0.25,
    )
    collector = cw.CoherentRowCollector()
    collector(
        st,
        np.arange(q.size),
        [duration.astype(complex)],
        np.ones(q.size, bool),
        (
            np.full(q.size, 1000.0),
            duration / (2 * HBARC_EV_ANG),
            np.full(q.size, 10.0),
            start + end,
            end - start,
            q,
        ),
    )
    (row,) = collector.rows
    np.testing.assert_allclose(row.start_transmission, start, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(row.end_transmission, end, rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(row.slope_ang, 2 * q / duration)
    np.testing.assert_allclose(row.mean_transmission, [1 / 1600, 1 / 80, 1, 1 / 80, 1 / 1600])
    assert np.all(np.isfinite(cw._RowJumps(row).terms(1100.0, True)))
    np.testing.assert_array_equal(row.escape_mid_ang, np.arange(q.size) * 100.0 + 10.0)
    np.testing.assert_array_equal(row.escape_change_ang, np.full(q.size, 20.0))
    np.testing.assert_array_equal(row.phase_rad, st.capture_phase_rad)


@pytest.mark.parametrize("delta", [-1e-5, 0.0, 1e-5])
@pytest.mark.parametrize("intercept", [-1e-4, 0.0, 1e-4])
def test_affine_dispersion_maps_the_complete_formation_field(delta, intercept):
    """Independent direct field pins Jacobian, midpoint phase and signed damping."""
    from pyrite.montecarlo.spectrum.lines._formation import formation_coefficients, formation_factor

    duration, change, midpoint = 100.0, 1e6, 6e5
    tau_start, tau_end = np.array([0.2]), np.array([0.8])
    apb, bma, q = formation_coefficients(tau_start, tau_end)
    row = replace(
        _field([duration], 1000.0, 1.0),
        start_transmission=np.exp(-tau_start / 2),
        end_transmission=np.exp(-tau_end / 2),
        mean_transmission=(np.exp(-tau_start) - np.exp(-tau_end)) / (tau_end - tau_start),
        attenuation_slope_ang=(tau_end - tau_start) / (2 * duration),
        escape_mid_ang=np.array([midpoint]),
        escape_change_ang=np.array([change]),
    )
    mapped = cw._affine_dispersion_row(row, delta / HBARC_EV_ANG, intercept)
    energies = np.linspace(800.0, 1400.0, 2001)
    dw = delta * energies / HBARC_EV_ANG + intercept
    argument = duration * (energies - 1000.0) / (2 * HBARC_EV_ANG) - 0.5 * change * dw
    actual = (
        duration
        * formation_factor(argument, apb, bma, q)
        * np.exp(1j * (energies * row.centre_ang[0] / HBARC_EV_ANG - midpoint * dw))
    )
    # Direct integrate the mapped exponential, independently of formation_factor.
    x = (energies - mapped.energy_eV[0]) / HBARC_EV_ANG
    z = 1j * x - mapped.slope_ang[0]
    half = mapped.duration_ang[0] / 2
    transformed = (
        mapped.amplitude[0, 0]
        * np.exp(-(tau_start[0] + tau_end[0]) / 4)
        * (np.exp(z * half) - np.exp(-z * half))
        / z
        * np.exp(1j * (energies * mapped.centre_ang[0] / HBARC_EV_ANG - intercept * midpoint))
    )
    np.testing.assert_allclose(transformed, actual, rtol=2e-11, atol=2e-11)
    jacobian = 1.0 - delta * change / duration
    assert mapped.power == pytest.approx(row.power / jacobian)


def test_affine_dispersion_moves_the_counterexample_inside_the_resonance_band():
    """The old 1000 eV carrier misses the exact 1111.11 eV resonance."""
    row = replace(
        _field([100.0], 1000.0, 1.0),
        escape_mid_ang=np.array([5e5]),
        escape_change_ang=np.array([1e6]),
    )
    mapped = cw._affine_dispersion_row(row, 1e-5 / HBARC_EV_ANG)
    assert mapped.energy_eV[0] == pytest.approx(1000.0 / 0.9)
    assert mapped.duration_ang[0] == pytest.approx(90.0)
    energies = np.linspace(1100.0, 1200.0, 20001)
    actual = 100.0 * np.sinc(100.0 * (0.9 * energies - 1000.0) / (2 * np.pi * HBARC_EV_ANG))
    finite = np.trapezoid(actual**2, energies) / (2 * np.pi * HBARC_EV_ANG * row.power)
    proxy = cw.coherent_edge_leak(cw._RowJumps(row), row.power, 1100.0, upper=True)
    assert finite > 4.0 * proxy
    _, summary = _seeds([mapped], start=900.0, stop=1400.0)
    record = summary["rows"][0]
    assert record["window_eV"][0] < mapped.energy_eV[0] < record["window_eV"][1]
    assert record["nyquist_electron_eV"] == pytest.approx(np.pi * HBARC_EV_ANG / 90.0)


def test_affine_dispersion_refuses_nonmonotone_and_overlapping_support():
    row = replace(
        _field([100.0], 1000.0, 1.0),
        escape_mid_ang=np.array([5e5]),
        escape_change_ang=np.array([1e6]),
    )
    with pytest.raises(ValueError, match="positive piece durations"):
        cw._affine_dispersion_row(row, 1e-4 / HBARC_EV_ANG)
    row = replace(
        _field([100.0, 100.0], 1000.0, 1.0, gaps=[0.0, 10.0]),
        escape_mid_ang=np.array([0.0, 2e6]),
        escape_change_ang=np.zeros(2),
    )
    with pytest.raises(ValueError, match="overlapping support"):
        cw._affine_dispersion_row(row, 1e-5 / HBARC_EV_ANG)
    row = replace(row, escape_mid_ang=np.array([0.0, 1e6]))
    with pytest.raises(ValueError, match="genuine gap"):
        cw._affine_dispersion_row(row, 1e-5 / HBARC_EV_ANG)


@pytest.mark.parametrize("electrons", [[0, 0], [0, 1]])
def test_dispersion_residual_bound_includes_midpoint_and_formation_slopes(electrons):
    """An independently bounded nonlinear residual charges the full complex field."""
    row = replace(
        _field([100.0, 120.0], [1000.0, 1010.0], [1.0, 0.7j], electrons=electrons),
        escape_mid_ang=np.array([500.0, 700.0]),
        escape_change_ang=np.array([200.0, -300.0]),
    )
    energies = np.linspace(900.0, 1200.0, 20001)
    residual_max = 1e-4
    residual = residual_max * np.sin((energies - 900.0) / 30.0)
    difference = np.zeros((max(electrons) + 1, energies.size), dtype=complex)
    for j, electron in enumerate(electrons):
        v = row.duration_ang[j] * (energies - row.energy_eV[j]) / (2 * HBARC_EV_ANG)
        common = (
            row.amplitude[0, j]
            * row.duration_ang[j]
            * np.exp(1j * energies * row.centre_ang[j] / HBARC_EV_ANG)
        )
        actual = common * np.sinc((v - row.escape_change_ang[j] * residual / 2) / np.pi)
        actual *= np.exp(-1j * row.escape_mid_ang[j] * residual)
        difference[electron] += actual - common * np.sinc(v / np.pi)
    finite = np.trapezoid(np.sum(np.abs(difference) ** 2, axis=0), energies)
    bound = cw._dispersion_residual_power_bound(row, residual_max, 300.0)
    assert 0.0 < finite <= bound
    assert cw._dispersion_residual_power_bound(row, 0.0, 300.0) == 0.0


@pytest.mark.parametrize("tau", [0.0, 1e-9, 4.0, 1600.0])
def test_dispersion_residual_bound_retains_the_attenuated_field_norm(tau):
    """The L1 field norm is an independent exponential integral, including underflow."""
    row = replace(
        _field([100.0], 1000.0, 1.0),
        start_transmission=np.ones(1),
        end_transmission=np.array([np.exp(-tau / 2)]),
        attenuation_slope_ang=np.array([tau / 200.0]),
        escape_mid_ang=np.array([500.0]),
        escape_change_ang=np.array([200.0]),
    )
    norm = 100.0 if tau == 0.0 else 200.0 * (-np.expm1(-tau / 2)) / tau
    # Phase error saturates at 2; the absolute bound is W (2 integral |f|)^2.
    assert cw._dispersion_residual_power_bound(row, 1.0, 300.0) == pytest.approx(
        300.0 * (2 * norm) ** 2
    )


def test_dispersion_helpers_require_complete_geometry_and_a_valid_certificate():
    row = _field([100.0], 1000.0, 1.0)
    with pytest.raises(ValueError, match="escape-path geometry"):
        cw._affine_dispersion_row(row, 0.0)
    with pytest.raises(ValueError, match="escape-path geometry"):
        cw._dispersion_residual_power_bound(row, 0.0, 100.0)
    row = replace(row, escape_mid_ang=np.array([0.0]), escape_change_ang=np.array([0.0]))
    for invalid in (-1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="uniform dispersion residual"):
            cw._dispersion_residual_power_bound(row, invalid, 100.0)
        with pytest.raises(ValueError, match="bandwidth"):
            cw._dispersion_residual_power_bound(row, 0.0, invalid)
    with pytest.raises(ValueError, match="finite escape-path geometry"):
        cw._affine_dispersion_row(replace(row, escape_mid_ang=np.array([np.nan])), 0.0)


@pytest.mark.parametrize("delta,residual", [(1e-5, 0.0), (1e-5, 1e-4), (1.1e-4, 0.0)])
def test_dispersion_window_audit_bounds_the_actual_excluded_field(delta, residual):
    """Direct physical-field integration covers a shifted resonance and L1 fallback."""
    from types import SimpleNamespace

    from pyrite.montecarlo.spectrum.coherent_dispersion import DispersionCertificate

    slope = delta / HBARC_EV_ANG
    law = SimpleNamespace(
        start=900.0,
        stop=1200.0,
        breaks=np.array([900.0, 1200.0]),
        fingerprint="analytic test law",
        certificate=lambda lo, hi: DispersionCertificate(
            lo, hi, slope, 0.0, residual, abs(slope) + residual / 30.0
        ),
    )
    row = replace(
        _field([100.0], 1000.0, 1.0),
        escape_mid_ang=np.array([5e5]),
        escape_change_ang=np.array([1e6]),
    )
    summary = {"rows": [{"window_eV": [950.0, 1050.0]}], "oversampling": 2.0}
    audit = cw._dispersion_window_audit([row], summary, law, None, 1)
    finite = 0.0
    for lo, hi in ((900.0, 950.0), (1050.0, 1200.0)):
        energies = np.linspace(lo, hi, 20001)
        dw = slope * energies + residual * np.sin((energies - 900.0) / 30.0)
        v = 100.0 * (energies - 1000.0) / (2 * HBARC_EV_ANG) - 5e5 * dw
        actual = 100.0 * np.sinc(v / np.pi)
        finite += np.trapezoid(actual**2, energies)
    result = audit["rows"][0]
    assert 0.0 < finite <= result["excluded_power_bound_eV"]
    assert audit["relative_production_bound"] is False
    assert result["phase_slope_step_all_eV"] > 0.0
    if delta > 1e-4:
        assert result["l1_fallback_intervals"] == 2
    else:
        assert result["l1_fallback_intervals"] == 0
    if residual == 0.0:
        assert result["affine_residual_power_bound_eV"] == 0.0


@pytest.mark.parametrize("edge_distance", [20.0, 80.0, 400.0])
def test_clustered_joints_stay_inside_the_bound(edge_distance):
    """Verification case: amplitude ramps whose joints fall within ``hbar c / u``.

    A diagonal jump sum underestimates these 7-11x; the cluster bound must not.
    """
    ramp = np.linspace(1.0, 2.0, 20)
    field = _field(np.full(40, 1.0), 1000.0, np.concatenate((ramp, ramp)))
    bound = cw.coherent_edge_leak(
        cw._RowJumps(field), field.power, 1000.0 + edge_distance, upper=True
    )
    assert _exact_tail(field, 1000.0 + edge_distance) <= bound


@pytest.mark.slow
@pytest.mark.parametrize("gap", [1500.0, 4000.0])
def test_near_cancelling_joints_far_apart_stay_inside_the_bound(gap):
    """Verification counterexample: joints whose ``|J|^2`` vanishes inside the tail.

    101 at 1000 eV joined to 100 at 1003 eV puts a zero of the jump spectrum at
    1303 eV, beyond the 1100 eV edge, so ``|J|^2`` is not monotone there and
    the cross term between distant copies reaches ~3.9 of ``sqrt(T T)/(u dtau)``.
    The cross allowance must bound it through the decreasing-envelope moduli.
    """
    durations = np.full(8, 200.0)
    energies = np.tile([1000.0, 1003.0], 4)
    amplitudes = np.tile([101.0, 100.0], 4)
    gaps = np.tile([gap, 0.0], 4)
    gaps[0] = 0.0
    field = _field(durations, energies, amplitudes, gaps=gaps)
    jumps = cw._RowJumps(field)
    bound = cw.coherent_edge_leak(jumps, field.power, 1100.0, upper=True)
    assert jumps.joints == 4
    assert _exact_tail(field, 1100.0) <= bound


@pytest.mark.slow
@pytest.mark.parametrize("seed", [0, 1])
def test_a_scattered_track_with_moving_carriers_stays_inside_the_bound(seed):
    """Random amplitudes, carriers and piece lengths along two tracks."""
    rng = np.random.default_rng(seed)
    n = 30
    field = _field(
        rng.uniform(2.0, 200.0, n),
        rng.uniform(900.0, 1100.0, n),
        rng.uniform(0.2, 3.0, n) * np.exp(1j * rng.uniform(-0.3, 0.3, n)),
        electrons=np.repeat([0, 1], n // 2),
    )
    jumps = cw._RowJumps(field)
    for distance in (10.0, 60.0, 300.0):
        edge = 1100.0 + distance
        assert _exact_tail(field, edge) <= cw.coherent_edge_leak(
            jumps, field.power, edge, upper=True
        )


def test_vanishing_decoherence_takes_the_per_electron_step_everywhere():
    """Limiting case ``F -> 0``: the grouped floor's span sets every bin."""
    field = _field(
        [600.0, 500.0, 300.0],
        [1000.0, 1010.0, 990.0],
        1.0,
        electrons=[0, 0, 1],
        gaps=[0.0, 0.0, 1.0e6],
    )
    seeds, summary = _seeds([field], decoherence=lambda e: np.zeros(np.shape(e)), count=2)
    row = summary["rows"][0]
    assert row["nyquist_electron_eV"] > 10.0 * row["nyquist_all_eV"]
    assert {seed.spacing_eV for seed in seeds} == {row["step_electron_eV"]}


def test_unbounded_decoherence_takes_the_all_electron_step():
    """No offsets, or an empirical F: no bound below one, all-electron span."""
    field = _field(
        [600.0, 500.0, 300.0],
        [1000.0, 1010.0, 990.0],
        1.0,
        electrons=[0, 0, 1],
        gaps=[0.0, 0.0, 1.0e6],
    )
    seeds, summary = _seeds([field], decoherence=None, count=2)
    assert {seed.spacing_eV for seed in seeds} == {summary["rows"][0]["step_all_eV"]}


def test_the_analytic_bunch_factor_switches_to_the_grouped_step_above_its_energy():
    sigma_fs = 1.0e-3
    segments = {
        "initial_t0_ang": np.array([1.0, -1.0]),
        "initial_r_ang": np.zeros((2, 3)),
        "crystal_width_ang": 1.0e7,
        "crystal_height_ang": 1.0e7,
    }
    bound = cw.decoherence_bound(segments, electron_limit=2, longitudinal_rms_fs=sigma_fs)
    assert bound is not None
    sigma = sigma_fs * 2997.92458
    energy = np.array([100.0, 3000.0])
    np.testing.assert_allclose(bound(energy), np.exp(-((energy / HBARC_EV_ANG * sigma) ** 2)))
    # Infinite slab: the empirical characteristic function has no closed form.
    slab = {key: value for key, value in segments.items() if "crystal" not in key}
    assert cw.decoherence_bound(slab, electron_limit=2, longitudinal_rms_fs=sigma_fs) is None
    # No offsets: the reducer squares the all-electron sum.
    still = {**segments, "initial_t0_ang": np.zeros(2)}
    assert cw.decoherence_bound(still, electron_limit=2, longitudinal_rms_fs=sigma_fs) is None


# ---- runner policy ------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny():
    """Ne=3 through 1 um of HOPG at 30 keV, a 100 fs bunch on a finite footprint."""
    case = cc.build_ladder_case("hopg", 30.0, 5.0, 45.0, thickness_ang=1.0e4, n_electrons=3, seed=0)
    case = dict(case)
    case["bunch_length_fs"] = 100.0
    ladder = cc.CaseLadder(case, transport_core="lockstep")
    return case, ladder


def _policy(case, **per_call):
    bandwidth = case["line_grid_policy"]["bandwidth"]
    return resolve_line_grid_policy(
        start_eV=bandwidth["start_eV"], stop_eV=bandwidth["stop_eV"], per_call=per_call
    ).payload()


def _resolve(case, ladder, payload, coherent=True):
    case = {**case, "line_grid_policy": payload, "coherent_emission": coherent}
    tp = ladder.transport
    return resolve_line_grid(case, ladder.segments, tp["n_hat"], tp["Ne_lines"], None)


def test_captured_midpoint_phases_reproduce_the_production_coherent_source(tiny):
    from pyrite.materials.crystal import refractive_index
    from pyrite.montecarlo.runner import _lines_for_segments
    from pyrite.montecarlo.runner.line_grid import _coherent_rows, longitudinal_rms_fs
    from pyrite.montecarlo.spectrum.coherent_dispersion import CoherentDispersionLaw
    from pyrite.montecarlo.spectrum.coherent_normalization import _sample_row_fields

    case, ladder = tiny
    case = {**case, "mosaic_mc_fwhm_rad": None, "mosaic_mc_nodes": 1}
    tp = ladder.transport
    energies = np.array([10.0, 17.0, 1900.0, 2500.0, 5000.0])
    rows = _coherent_rows(
        case, ladder.segments, tp["n_hat"], tp["Ne_lines"], energies[0], energies[-1], None, None
    )
    law = CoherentDispersionLaw(case["crystal"], energies[0], energies[-1])
    F = cw.decoherence_bound(
        ladder.segments,
        electron_limit=tp["Ne_lines"],
        longitudinal_rms_fs=longitudinal_rms_fs(case),
    )(energies)
    reconstructed = np.zeros(energies.size)
    power_rounding_guard = np.zeros(energies.size)
    public_phase = (1 - refractive_index(case["crystal"], energies).real) * energies / HBARC_EV_ANG
    for row in rows:
        assert row.phase_rad is not None
        samples = _sample_row_fields(row, law, energies, remove_global_phase=True)
        grouped = np.sum(np.abs(samples) ** 2, axis=(0, 1))
        flat = np.sum(np.abs(samples.sum(axis=1)) ** 2, axis=0)
        reconstructed += (1 - F) * grouped + F * flat
        # Condition the comparison on the absolute field, not a fixed
        # relative tolerance at a nearly cancelling spectral node. Include
        # the measured interpolant drift and the float64 operation/argument
        # scales of both reductions; this is a regression guard, not the
        # yet-missing universal production sample certificate.
        escape_max = np.abs(row.escape_mid_ang) + 0.5 * np.abs(row.escape_change_ang)
        phase_scale = (
            np.abs(row.centre_ang[:, None]) * energies / HBARC_EV_ANG
            + np.abs(row.phase_rad[:, None])
            + escape_max[:, None] * np.abs(public_phase)
        )
        formation_scale = (
            row.duration_ang[:, None]
            * (energies + np.abs(row.energy_eV[:, None]))
            / (2 * HBARC_EV_ANG)
            + 0.5 * np.abs(row.escape_change_ang[:, None]) * np.abs(public_phase)
            + 0.5 * np.abs(row.slope_ang[:, None]) * row.duration_ang[:, None]
        )
        uncertainty = (
            1e-12
            + 128 * np.finfo(float).eps * (row.energy_eV.size + phase_scale + formation_scale)
            + escape_max[:, None] * np.abs(law(energies) - public_phase)
        )
        amplitude_guard = np.linalg.norm(
            (cw._piece_l1_amplitudes(row)[..., None] * uncertainty).sum(axis=1), axis=0
        )
        power_rounding_guard += (
            (1 - F) * (2 * np.sqrt(grouped) * amplitude_guard + amplitude_guard**2)
            + F * (2 * np.sqrt(flat) * amplitude_guard + amplitude_guard**2)
            + 64 * np.finfo(float).eps * (grouped + flat)
        )
    actual = _lines_for_segments(
        ladder.segments, energies, case, tp["n_hat"], None, None, coherent=True, Ne=tp["Ne_lines"]
    )
    assert np.all(
        np.abs(reconstructed / tp["Ne_lines"] - actual) <= power_rounding_guard / tp["Ne_lines"]
    )
    assert power_rounding_guard.max() / tp["Ne_lines"] < INTRINSIC_RTOL * actual.max()


def test_captured_mosaic_weights_reconstruct_the_production_spectrum(tiny):
    from pyrite.montecarlo.runner import _lines_for_segments
    from pyrite.montecarlo.runner.line_grid import _coherent_rows
    from pyrite.montecarlo.spectrum.coherent_dispersion import CoherentDispersionLaw
    from pyrite.montecarlo.spectrum.coherent_normalization import _sample_row_fields

    case, ladder = tiny
    case = {**case, "mosaic_mc_fwhm_rad": 1e-3, "mosaic_mc_nodes": 2}
    tp = ladder.transport
    rows = _coherent_rows(
        case, ladder.segments, tp["n_hat"], tp["Ne_lines"], 10.0, 5000.0, None, None
    )
    assert rows and all(0 < row.mosaic_weight < 1 for row in rows)
    carriers = np.concatenate([row.energy_eV for row in rows])
    energies = np.unique(np.r_[10.0, carriers, carriers + 0.1, 5000.0])
    law = CoherentDispersionLaw(case["crystal"], 10.0, 5000.0)
    reconstructed = np.zeros_like(energies)
    # This 100fs anchor's Gaussian factor underflows across its line band.
    for row in rows:
        fields = _sample_row_fields(row, law, energies)
        reconstructed += row.mosaic_weight * np.sum(np.abs(fields) ** 2, axis=(0, 1))
    actual = _lines_for_segments(
        ladder.segments, energies, case, tp["n_hat"], None, None, coherent=True, Ne=tp["Ne_lines"]
    )
    assert actual.max() > 0
    assert np.max(np.abs(reconstructed / tp["Ne_lines"] - actual)) <= (
        INTRINSIC_RTOL * actual.max()
    )


def test_weighted_spectrum_audit_checks_a_same_trajectory_production_band(tiny):
    from functools import partial

    from pyrite.montecarlo.runner import _lines_for_segments
    from pyrite.montecarlo.runner.line_grid import _coherent_rows, longitudinal_rms_fs
    from pyrite.montecarlo.spectrum.coherent_dispersion import CoherentDispersionLaw
    from pyrite.montecarlo.spectrum.coherent_form_factor import gaussian_form_factor_bounds
    from pyrite.montecarlo.spectrum.coherent_spectrum_audit import audit_captured_spectrum_yield
    from pyrite.montecarlo.transport.kinematics import C_ANG_PER_FS

    case, ladder = tiny
    case = {**case, "hkl_list": [(0, 0, 2)], "mosaic_mc_fwhm_rad": None, "mosaic_mc_nodes": 1}
    tp = ladder.transport
    rows = _coherent_rows(
        case, ladder.segments, tp["n_hat"], tp["Ne_lines"], 10.0, 5000.0, None, None
    )
    assert len(rows) == 1 and rows[0].mosaic_weight == 1.0
    carrier = float(np.median(rows[0].energy_eV))
    band = (carrier - 0.0005, carrier + 0.0005)
    fine = np.linspace(*band, 257)
    # Preserve the full production material-table axis while measuring only
    # this narrow band. No second transport is run.
    source = _lines_for_segments(
        ladder.segments,
        np.r_[10.0, fine, 5000.0],
        case,
        tp["n_hat"],
        None,
        None,
        coherent=True,
        Ne=tp["Ne_lines"],
    )
    estimate = float(np.trapezoid(source[1:-1], fine))
    assert estimate > 0
    result = audit_captured_spectrum_yield(
        rows,
        CoherentDispersionLaw(case["crystal"], 10.0, 5000.0),
        electron_count=tp["Ne_lines"],
        numerical_yield=estimate,
        bands=[band],
        form_factor_bounds=partial(
            gaussian_form_factor_bounds,
            sigma_z_ang=longitudinal_rms_fs(case) * C_ANG_PER_FS,
        ),
        max_evaluations=1000,
    )
    assert result.within_tolerance
    assert result.relative_error_upper <= INTRINSIC_RTOL


@pytest.mark.slow
def test_automatic_full_axis_yield_audit_uses_the_production_capture():
    """One short physical track exercises the full automatic axis and runner."""
    from pyrite.montecarlo import runner

    case = dict(
        cc.build_ladder_case("hopg", 30.0, 5.0, 45.0, thickness_ang=1.0, n_electrons=1, seed=0)
    )
    from scipy.constants import elementary_charge

    case.update(
        bunch_charge_pc=elementary_charge * 1e12,
        coherent_emission=True,
        hkl_list=[(0, 0, 2)],
        mosaic_mc_fwhm_rad=None,
        mosaic_mc_nodes=1,
        bunch_length_fs=100.0,
        _coherent_yield_audit={
            "max_evaluations": 16000,
            "centroid_relative_tolerance": INTRINSIC_RTOL,
            "integration_bins": 2048,
        },
    )
    case["line_grid_policy"] = _policy(case, windows=True)
    tp = runner._transport_case(case, transport_core="lockstep")
    out = runner._spectrum_case(case, tp)
    record = out["line_grid_coherent_yield_audit"]
    assert record["scope"] == "stored-input full finite-axis yield and centroid"
    assert record["relative_error_upper"] <= INTRINSIC_RTOL
    assert record["centroid_relative_error_upper"] <= INTRINSIC_RTOL
    assert record["centroid_lower_eV"] > 0
    assert record["centroid_lower_eV"] <= record["centroid_upper_eV"]
    assert record["start_eV"] == out["E_grid"][0]
    assert record["stop_eV"] == out["E_grid"][-1]
    assert record["points"] == out["E_grid"].size
    assert record["rows"] == 1
    assert record["yield_lower"] > 0
    assert record["evaluations"] <= 16000
    # Same trajectory and coordinates, no second transport: capture is neutral.
    unaudited = runner._spectrum_case(
        {k: v for k, v in case.items() if k != "_coherent_yield_audit"}, tp
    )
    np.testing.assert_array_equal(out["spec_coherent"], unaudited["spec_coherent"])
    assert "line_grid_coherent_yield_audit" not in unaudited


@pytest.mark.parametrize("method", ["cone", "curvature", "field"])
def test_directed_band_enclosure_covers_the_same_trajectory_production_source(tiny, method):
    from pyrite.montecarlo.runner import _lines_for_segments
    from pyrite.montecarlo.runner.line_grid import _coherent_rows, longitudinal_rms_fs
    from pyrite.montecarlo.spectrum.coherent_dispersion import CoherentDispersionLaw
    from pyrite.montecarlo.spectrum.coherent_normalization import (
        _certified_interpolated_row_power_bounds,
        _certified_physical_row_power_bounds,
    )

    case, ladder = tiny
    case = {**case, "mosaic_mc_fwhm_rad": None, "mosaic_mc_nodes": 1}
    tp = ladder.transport
    rows = _coherent_rows(
        case, ladder.segments, tp["n_hat"], tp["Ne_lines"], 10.0, 5000.0, None, None
    )
    law = CoherentDispersionLaw(case["crystal"], 10.0, 5000.0)
    carrier = float(np.median(rows[0].energy_eV))
    index = np.searchsorted(law.breaks, carrier, side="right") - 1
    lower = max(carrier - 0.25, float(law.breaks[index]))
    upper = min(carrier + 0.25, float(law.breaks[index + 1]))
    fine = np.linspace(lower, upper, 257)
    F = cw.decoherence_bound(
        ladder.segments,
        electron_limit=tp["Ne_lines"],
        longitudinal_rms_fs=longitudinal_rms_fs(case),
    )(fine)
    # In this numerical 100fs anchor, keV production decoherence underflows
    # to zero. The regression isolates that grouped production branch.
    assert np.all(F == 0.0)
    margin = (upper - lower) / 100
    nodes = np.linspace(lower + margin, upper - margin, 33)
    if method == "field":
        bounds = [
            _certified_interpolated_row_power_bounds(
                row, law, lower, upper, nodes, form_factor_bounds=(0.0, 0.0)
            )
            for row in rows
        ]
    else:
        bounds = [
            _certified_physical_row_power_bounds(
                row,
                law,
                lower,
                upper,
                nodes,
                form_factor_bounds=(0.0, 0.0),
                curvature=method == "curvature",
            )
            for row in rows
        ]
    # Full-axis endpoints keep production's spline selection identical to
    # the captured law. Integrate only the narrow band on the same transport.
    source = _lines_for_segments(
        ladder.segments,
        np.r_[10.0, fine, 5000.0],
        case,
        tp["n_hat"],
        None,
        None,
        coherent=True,
        Ne=tp["Ne_lines"],
    )
    actual = np.trapezoid(source[1:-1], fine)
    total_lower = sum(lo for lo, _ in bounds) / tp["Ne_lines"]
    total_upper = sum(hi for _, hi in bounds) / tp["Ne_lines"]
    assert 0 < total_lower <= actual <= total_upper
    if method != "cone":
        cones = [
            _certified_physical_row_power_bounds(
                row, law, lower, upper, nodes, form_factor_bounds=(0.0, 0.0)
            )
            for row in rows
        ]
        cone_width = sum(hi - lo for lo, hi in cones) / tp["Ne_lines"]
        assert total_upper - total_lower < cone_width


def test_uniform_automatic_policy_still_refuses_coherent_emission(tiny):
    case, ladder = tiny
    with pytest.raises(ValueError, match="without feature windows") as excinfo:
        _resolve(case, ladder, _policy(case))
    assert not isinstance(excinfo.value, LineGridToleranceError)
    assert "windows = true" in str(excinfo.value)


def test_a_windowed_policy_resolves_coherent_emission_and_records_its_windows(tiny):
    case, ladder = tiny
    policy = _policy(case, windows=True)
    with pytest.warns(LineShapePrecisionWarning, match="dispersion remains uncertified"):
        grid, record = _resolve(case, ladder, policy)
    coherent = record["coherent_windows"]
    assert grid.size == record["num"]
    assert coherent["decoherence_bound"] == "analytic F_z"
    rows = [row for row in coherent["rows"] if row.get("points")]
    assert rows
    for row in rows:
        lo, hi = row["window_eV"]
        assert lo <= row["band_eV"][0] and row["band_eV"][1] <= hi
        assert row["leak_bound"]["upper"] <= coherent["leak_limit"] or hi == grid[-1]
        inside = grid[(grid > lo) & (grid < hi)]
        assert float(np.diff(inside).max()) <= row["step_electron_eV"] * (1.0 + 1e-9)
    audit = coherent["dispersion"]
    assert audit["intervals"] > 0 and audit["relative_production_bound"] is False
    assert audit["rows"] and all(row["excluded_power_bound_eV"] >= 0 for row in audit["rows"])
    with pytest.warns(LineShapePrecisionWarning, match="dispersion remains uncertified"):
        warm_grid, warm_record = _resolve(case, ladder, policy)
    np.testing.assert_array_equal(warm_grid, grid)
    assert warm_record["cache"] == "hit"
    assert warm_record["coherent_windows"]["dispersion"] == audit


def test_coherent_and_incoherent_grids_never_share_a_cache_entry(tiny):
    case, ladder = tiny
    payload = _policy(case, windows=True)
    _, coherent = _resolve(case, ladder, payload)
    _, incoherent = _resolve(case, ladder, payload, coherent=False)
    assert coherent["cache_key"] != incoherent["cache_key"]
    assert "coherent_windows" not in incoherent
    _, again = _resolve(case, ladder, payload)
    assert again["cache"] == "hit" and again["cache_key"] == coherent["cache_key"]


def test_coherent_cache_tracks_the_frozen_coupling_weight(tiny):
    case, ladder = tiny
    policy = _policy(case, windows=True)
    with pytest.warns(LineShapePrecisionWarning, match="dispersion remains uncertified"):
        _, first = _resolve({**case, "B_ang2": 0.0}, ladder, policy)
    with pytest.warns(LineShapePrecisionWarning, match="dispersion remains uncertified"):
        _, changed = _resolve({**case, "B_ang2": 1.0}, ladder, policy)
    assert first["cache"] == changed["cache"] == "miss"
    assert first["cache_key"] != changed["cache_key"]


def test_zero_jump_filter_does_not_reuse_the_previous_coherent_cache(tiny, monkeypatch):
    from pyrite.montecarlo.runner import line_grid

    case, ladder = tiny
    policy = _policy(case, windows=True)
    with monkeypatch.context() as previous:
        previous.setattr(line_grid, "COHERENT_WINDOW_REVISION", 8)
        _, old = _resolve(case, ladder, policy)
    grid, current = _resolve(case, ladder, policy)
    assert current["cache"] == "miss"
    assert current["cache_key"] != old["cache_key"]
    warm_grid, warm = _resolve(case, ladder, policy)
    assert warm["cache"] == "hit"
    np.testing.assert_array_equal(warm_grid, grid)


def test_coherent_refinement_does_not_reuse_the_four_node_cache(tiny, monkeypatch):
    from pyrite.montecarlo.runner import line_grid

    case, ladder = tiny
    policy = _policy(case, windows=True)
    with monkeypatch.context() as old_policy:
        old_policy.setattr(line_grid, "COHERENT_NYQUIST_OVERSAMPLING", 4.0)
        old_policy.setattr(line_grid, "COHERENT_WINDOW_REVISION", 5)
        old_policy.setattr(
            cw.coherent_window_seeds,
            "__kwdefaults__",
            {**cw.coherent_window_seeds.__kwdefaults__, "oversampling": 4.0},
        )
        _, old = _resolve(case, ladder, policy)
    grid, refined = _resolve(case, ladder, policy)
    assert refined["cache"] == "miss"
    assert refined["cache_key"] != old["cache_key"]
    assert refined["num"] > old["num"]
    warm_grid, warm = _resolve(case, ladder, policy)
    assert warm["cache"] == "hit"
    np.testing.assert_array_equal(warm_grid, grid)
    # A budget that admitted the previous axis must refuse the refined one.
    with pytest.raises(LineGridToleranceError, match=r"above the .*point budget"):
        _resolve(case, ladder, _policy(case, windows=True, max_points=old["num"]))


def test_a_plan_above_the_budget_is_refused_with_its_count(tiny):
    case, ladder = tiny
    payload = _policy(case, windows=True, max_points=200)
    with pytest.raises(LineGridToleranceError, match=r"coherent windowed .* needs \d+ coordinates"):
        _resolve(case, ladder, payload)


def test_a_windowed_policy_refuses_a_sinc_cutoff(tiny):
    """The cutoff truncates line support after capture; the bound does not cover it."""
    case, ladder = tiny
    payload = _policy(case, windows=True)
    with pytest.raises(ValueError, match="does not support sinc_cutoff"):
        _resolve({**case, "sinc_cutoff": 200.0}, ladder, payload)


def test_a_float32_reducer_warns_that_the_bound_needs_float64_phases(tiny, monkeypatch):
    from pyrite._line_grid_policy import LineShapePrecisionWarning
    from pyrite.montecarlo.runner import line_grid

    case, ladder = tiny
    monkeypatch.setattr(line_grid, "REAL", np.float32)
    with pytest.warns(LineShapePrecisionWarning, match="float64 phases only"):
        _resolve(case, ladder, _policy(case, windows=True))


def _finest_used_step(row):
    """Finest step a row's bins actually take; all-electron bins lie below the switch."""
    switch = row.get("electron_step_from_eV")
    if switch is None or switch > row["window_eV"][0]:
        return min(row["step_electron_eV"], row["step_all_eV"])
    return row["step_electron_eV"]


@pytest.mark.slow
def test_windowed_auto_matches_a_fine_explicit_grid_on_identical_trajectories(tiny):
    """Anchor: yield, centroid and FWHM within the 1e-3 intrinsic-source share.

    The reference is uniform over the whole axis at a quarter of the finest
    window step, so it resolves every row's fringes everywhere; both grids are
    evaluated on the one transport. Peak height is excluded (#110). The
    dominant-line FWHM reads its half-maximum crossing by linear interpolation.
    Eight samples per Nyquist step are the numerical policy, informed by
    the remote 60 keV ladder; this tiny anchor also checks all three gates.
    """
    case, ladder = tiny
    grid, record = _resolve(case, ladder, _policy(case, windows=True))
    coherent_case = {**case, "coherent_emission": True}
    coherent = cc.CaseLadder(coherent_case, transport=ladder.transport, coherent=True)
    finest = min(
        _finest_used_step(row) for row in record["coherent_windows"]["rows"] if row.get("points")
    )
    reference_grid = np.arange(float(grid[0]), float(grid[-1]) + 1e-9, 0.25 * finest)
    zeros = np.zeros_like
    auto = spectrum_observables(grid, coherent.lines(grid), zeros(grid), detectors={})
    reference = spectrum_observables(
        reference_grid,
        coherent.lines(reference_grid),
        zeros(reference_grid),
        detectors={},
    )
    assert reference_grid.size > 2 * grid.size
    for key in ("yield", "centroid_eV"):
        assert abs(auto[key] - reference[key]) <= INTRINSIC_RTOL * abs(reference[key]), key
    assert abs(auto["fwhm_eV"] - reference["fwhm_eV"]) <= INTRINSIC_RTOL * reference["fwhm_eV"]


@pytest.mark.slow
@pytest.mark.parametrize("hkl", [(0, 0, 2), (0, 0, 4)])
def test_the_production_reducer_leaks_less_than_the_window_bound(tiny, hkl):
    """Verification anchor: the reducer's own tail beyond each window edge.

    One reflection, so the case has one row; the reducer is evaluated over
    the whole axis at half the window step and its power beyond each edge
    that lies inside the axis is held to ``leak_limit`` of the row's axis
    power. The 2026-10-05 verification measured the pre-fix proxy envelope
    at ~1e-4 here; the tail beyond the axis stop is not sampled.
    """
    case, ladder = tiny
    row_case = {**case, "hkl_list": [hkl]}
    _, record = _resolve(row_case, ladder, _policy(row_case, windows=True))
    summary = record["coherent_windows"]
    (row,) = [entry for entry in summary["rows"] if entry.get("points")]
    lower, upper = row["window_eV"]
    coherent = cc.CaseLadder(
        {**row_case, "coherent_emission": True}, transport=ladder.transport, coherent=True
    )
    bandwidth = row_case["line_grid_policy"]["bandwidth"]
    energy = np.arange(bandwidth["start_eV"], bandwidth["stop_eV"], 0.5 * row["step_electron_eV"])
    lines = coherent.lines(energy)
    total = np.trapezoid(lines, energy)
    limit = summary["leak_limit"]
    above = energy >= upper
    assert not row["at_axis"]["upper"]
    assert np.trapezoid(lines[above], energy[above]) <= limit * total
    if not row["at_axis"]["lower"]:
        below = energy <= lower
        assert np.trapezoid(lines[below], energy[below]) <= limit * total


def test_physical_charge_widens_windows_and_keeps_incident_misses_in_bound():
    rows = [_field([800000.0, 800000.0], 1000.0, 1.0, electrons=[0, 1], gaps=[0.0, 10000000.0])]
    segments = {"Ne": 4}
    summaries = []
    for population in (4.0, 40.0):
        _, summary = cw.coherent_case_seeds(
            rows,
            segments,
            electron_limit=4,
            start_eV=10.0,
            stop_eV=5000.0,
            longitudinal_rms_fs=None,
            physical_electrons=population,
        )
        summaries.append(summary)
        assert summary["incident_samples"] == 4
        assert summary["emitting_samples"] == 2
        assert summary["physical_electrons"] == population
    low, high = [summary["rows"][0]["window_eV"] for summary in summaries]
    assert high[0] < low[0] and high[1] > low[1]


def test_charge_changes_the_coherent_grid_cache_key(tiny):
    case, ladder = tiny
    policy = _policy(case, windows=True)
    _, first = _resolve({**case, "bunch_charge_pc": 1.0}, ladder, policy)
    _, changed = _resolve({**case, "bunch_charge_pc": 2.0}, ladder, policy)
    assert changed["cache"] == "miss"
    assert first["cache_key"] != changed["cache_key"]
    _, warm = _resolve({**case, "bunch_charge_pc": 2.0}, ladder, policy)
    assert warm["cache"] == "hit"

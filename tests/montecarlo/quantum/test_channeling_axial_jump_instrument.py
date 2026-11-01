"""Independent finite absorbing-model references, not physical escape anchors."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pyrite.montecarlo.channeling.axial_jump_instrument import (
    absorbing_axial_model,
    sample_absorbing_outgoing,
)
from pyrite.montecarlo.channeling.axial_phase_space import phase_space_stream


def _scalar(*, outgoing=0.2, unresolved=0.1, weight=0.6, **extra):
    vector = np.array([1, 1j, -0.4, 0.2j])
    vector /= np.linalg.norm(vector)
    args = dict(
        hamiltonian_eV=np.array([[3.0]]),
        initial_density=np.array([[weight]]),
        outgoing_jumps_per_sqrt_ang=(np.sqrt(outgoing) * vector[:, None],),
        unresolved_effect_inv_ang=np.array([[unresolved]]),
        outgoing_shape=(2, 2),
        hbar_v_eV_ang=2.0,
    )
    args.update(extra)
    return absorbing_axial_model(**args), vector


def test_competing_absorbers_match_exponential_and_incident_trace():
    model, vector = _scalar()
    projector = np.outer(vector, vector.conj())
    for depth in (0.0, 0.5, 3.0, 30.0):
        result = model.at_depth(depth)
        survival = np.exp(-0.3 * depth)
        np.testing.assert_allclose(result.retained_density, [[0.6 * survival]], atol=2e-15)
        np.testing.assert_allclose(
            result.outgoing_density, 0.6 * 0.2 / 0.3 * (1 - survival) * projector, atol=2e-15
        )
        np.testing.assert_allclose(result.outgoing_rate_inv_ang, 0.6 * 0.2 * survival * projector)
        assert result.unresolved_weight == pytest.approx(0.6 * 0.1 / 0.3 * (1 - survival))
        assert (
            np.trace(result.retained_density).real
            + np.trace(result.outgoing_density).real
            + result.unresolved_weight
        ) == pytest.approx(model.initial_weight, abs=3e-15)
        assert np.linalg.eigvalsh(result.outgoing_density).min() >= -1e-15


def test_retained_recycling_is_gain_not_escape():
    decay, escape, weight = 0.7, 0.15, 0.8
    retained_jump = np.array([[0, np.sqrt(decay)], [0, 0]])
    outgoing_jump = np.zeros((4, 2))
    outgoing_jump[:2] = np.sqrt(escape) * np.eye(2)
    model = absorbing_axial_model(
        np.zeros((2, 2)),
        np.diag([0, weight]),
        retained_jumps_per_sqrt_ang=(retained_jump,),
        outgoing_jumps_per_sqrt_ang=(outgoing_jump,),
        outgoing_shape=(2, 2),
        hbar_v_eV_ang=1,
    )
    z = 2.3
    result = model.at_depth(z)
    excited = weight * np.exp(-(decay + escape) * z)
    ground = weight * np.exp(-escape * z) - excited
    np.testing.assert_allclose(result.retained_density, np.diag([ground, excited]), atol=2e-15)
    assert np.trace(result.outgoing_density).real == pytest.approx(weight * -np.expm1(-escape * z))
    assert result.unresolved_weight == 0
    assert result.outgoing_density[1, 1].real == pytest.approx(
        weight * escape / (decay + escape) * -np.expm1(-(decay + escape) * z)
    )


def test_coherent_outgoing_density_and_depth_dependence_match_closed_form():
    energy, hbar_v, escape = np.array([0.0, 1.7]), 2.5, 0.18
    psi = np.array([1, 1j]) * np.sqrt(0.7 / 2)
    rho = np.outer(psi, psi.conj())
    jump = np.zeros((4, 2), complex)
    jump[:2] = np.sqrt(escape) * np.eye(2)
    model = absorbing_axial_model(
        np.diag(energy),
        rho,
        outgoing_jumps_per_sqrt_ang=(jump,),
        outgoing_shape=(2, 2),
        hbar_v_eV_ang=hbar_v,
    )
    for z in (0.25, 1.5, 5):
        exponent = escape + 1j * (energy[:, None] - energy[None, :]) / hbar_v
        expected_retained = rho * np.exp(-exponent * z)
        expected_integrated = escape * rho * -np.expm1(-exponent * z) / exponent
        result = model.at_depth(z)
        np.testing.assert_allclose(result.retained_density, expected_retained, atol=2e-15)
        np.testing.assert_allclose(result.outgoing_rate_inv_ang[:2, :2], escape * expected_retained)
        np.testing.assert_allclose(result.outgoing_density[:2, :2], expected_integrated, atol=2e-15)
        assert abs(result.outgoing_density[0, 1]) > 0.01


def test_augmented_exponential_matches_independent_matrix_ode():
    # Direct matrix products, independent of the implementation's Kronecker
    # generator and vectorization. Tight tolerance is numerical oracle accuracy.
    h = np.array([[0.2, 0.4j], [-0.4j, 1.3]])
    a = np.array([[0.1, 0.2j], [0.3, -0.1j]])
    b = np.array([[0.1, 0.2j], [0.3j, -0.2], [0, 0.1j], [0.2, 0.1]])
    c = np.array([[0.2j, 0], [0.1, 0.15], [0.1j, 0.1], [0, -0.2j]])
    v = np.array([0.1, 0.2j])
    unresolved = np.outer(v, v.conj())
    rho = np.array([[0.5, 0.1j], [-0.1j, 0.3]])
    loss = a.conj().T @ a + b.conj().T @ b + c.conj().T @ c + unresolved
    model = absorbing_axial_model(
        h,
        rho,
        retained_jumps_per_sqrt_ang=(a,),
        outgoing_jumps_per_sqrt_ang=(b, c),
        unresolved_effect_inv_ang=unresolved,
        outgoing_shape=(2, 2),
        hbar_v_eV_ang=1.8,
    )

    def derivative(z, state):
        density = state[:4].reshape(2, 2)
        drho = -1j * (h @ density - density @ h) / 1.8 + a @ density @ a.conj().T
        drho -= (loss @ density + density @ loss) / 2
        gain = b @ density @ b.conj().T + c @ density @ c.conj().T
        return np.r_[drho.ravel(), gain.ravel(), np.trace(unresolved @ density)]

    initial = np.r_[rho.ravel(), np.zeros(17)]
    oracle = solve_ivp(derivative, (0, 4.0), initial, method="DOP853", rtol=1e-12, atol=1e-14)
    assert oracle.success
    expected = oracle.y[:, -1]
    result = model.at_depth(4)
    np.testing.assert_allclose(result.retained_density, expected[:4].reshape(2, 2), atol=2e-13)
    np.testing.assert_allclose(result.outgoing_density, expected[4:-1].reshape(4, 4), atol=2e-13)
    assert result.unresolved_weight == pytest.approx(expected[-1].real, abs=2e-13)
    assert (
        np.trace(result.retained_density).real
        + np.trace(result.outgoing_density).real
        + result.unresolved_weight
    ) == pytest.approx(0.8, abs=2e-14)


class RecordedDraws:
    def __init__(self, uniform=0.37):
        self.uniform = uniform
        self.uniform_calls = 0
        self.probabilities = []

    def random(self):
        self.uniform_calls += 1
        return self.uniform

    def choice(self, size, p):
        self.probabilities.append(p.copy())
        return 0


def test_depth_inverse_cdf_matches_competing_exponential():
    model, _ = _scalar()
    rng = RecordedDraws()
    event = sample_absorbing_outgoing(model, np.ones((2, 2)) / 2, 3.0, rng)
    expected_depth = -np.log1p(-rng.uniform * -np.expm1(-0.3 * 3)) / 0.3
    assert event.depth_ang == pytest.approx(expected_depth, abs=1e-8)
    assert event.outgoing_weight == pytest.approx(0.6 * 0.2 / 0.3 * -np.expm1(-0.9))
    assert event.rate_inv_ang == pytest.approx(0.6 * 0.2 * np.exp(-0.3 * event.depth_ang))
    assert rng.uniform_calls == 1 and len(rng.probabilities) == 2


def test_joint_depth_and_phase_space_uses_local_coherent_gain():
    escape, frequency = 0.2, 1.5
    psi = np.array([1, 0, 0, 1j]) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    energies = np.array([0, 0, 0, frequency])
    model = absorbing_axial_model(
        np.diag(energies),
        rho,
        outgoing_jumps_per_sqrt_ang=(np.sqrt(escape) * np.eye(4),),
        outgoing_shape=(2, 2),
        hbar_v_eV_ang=1,
    )
    window = np.ones((2, 2)) / 2
    conditionals = []
    for uniform in (0.15, 0.8):
        rng = RecordedDraws(uniform)
        outcome = sample_absorbing_outgoing(model, window, 3.0, rng)
        local_psi = psi * np.exp(-1j * energies * outcome.depth_ang)
        # At the scripted position (0,0), construct the four carriers directly.
        x, y = np.indices((2, 2))
        expected = []
        for kx, ky in np.ndindex((2, 2)):
            mode = window * np.exp(1j * np.pi * (kx * x + ky * y))
            expected.append(abs(np.vdot(mode.ravel(), local_psi)) ** 2)
        expected = np.array(expected) / np.sum(expected)
        np.testing.assert_allclose(rng.probabilities[0], 0.25, atol=2e-15)
        np.testing.assert_allclose(rng.probabilities[1], expected, atol=2e-15)
        conditionals.append(rng.probabilities[1])
    assert np.max(np.abs(conditionals[0] - conditionals[1])) > 0.1


def test_incident_streams_are_processing_order_independent():
    model, _ = _scalar()
    window = np.ones((2, 2)) / 2
    forward = {
        i: sample_absorbing_outgoing(model, window, 3, phase_space_stream(623, i)) for i in range(4)
    }
    reverse = {
        i: sample_absorbing_outgoing(model, window, 3, phase_space_stream(623, i))
        for i in (3, 1, 0, 2)
    }
    assert forward == reverse


def test_zero_scattering_and_zero_initial_weight_refuse_without_random_draws():
    for args in (dict(outgoing=0, unresolved=0), dict(weight=0)):
        model, _ = _scalar(**args)
        result = model.at_depth(10)
        assert np.trace(result.outgoing_density) == 0
        assert result.unresolved_weight == 0
        assert np.trace(result.retained_density).real == pytest.approx(model.initial_weight)
        with pytest.raises(ValueError, match="zero-weight"):
            sample_absorbing_outgoing(model, np.ones((2, 2)) / 2, 10, None)
    model, _ = _scalar()
    with pytest.raises(ValueError, match="zero-weight"):
        sample_absorbing_outgoing(model, np.ones((2, 2)) / 2, 0, None)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"hamiltonian_eV": np.array([[1j]])}, "Hermitian"),
        ({"initial_density": np.array([[-0.2]])}, "positive"),
        ({"initial_density": np.array([[1.2]])}, "trace"),
        ({"unresolved_effect_inv_ang": np.array([[-0.1]])}, "positive"),
        ({"unresolved_effect_inv_ang": np.eye(2)}, "dimensions"),
        ({"outgoing_jumps_per_sqrt_ang": (np.zeros((3, 1)),)}, "dimensions"),
        ({"outgoing_jumps_per_sqrt_ang": (np.full((4, 1), np.nan),)}, "finite"),
        ({"hbar_v_eV_ang": 0}, "hbar_v"),
        ({"outgoing_shape": (2, True)}, "positive integer"),
        ({"max_working_bytes": True}, "positive integer"),
    ],
)
def test_invalid_models_refuse(changes, message):
    with pytest.raises(ValueError, match=message):
        _scalar(**changes)


def test_memory_refusal_precedes_superoperators_and_eigensolver(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dense construction before admission")

    monkeypatch.setattr(np, "kron", forbidden)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)
    with pytest.raises(ValueError, match="working-memory"):
        _scalar(max_working_bytes=1)


def test_invalid_sampling_inputs_refuse_before_random_draws():
    model, _ = _scalar()
    for bad in (-1, np.inf, np.nan):
        with pytest.raises(ValueError, match="depth_ang"):
            model.at_depth(bad)
        with pytest.raises(ValueError, match="depth_ang"):
            sample_absorbing_outgoing(model, np.ones((2, 2)) / 2, bad, None)
    for bad in (0, -1, np.nan):
        with pytest.raises(ValueError, match="depth_tolerance"):
            sample_absorbing_outgoing(model, np.ones((2, 2)) / 2, 2, None, depth_tolerance_ang=bad)
    for bad in (np.ones((2, 2)), np.ones((1, 2)), np.full((2, 2), np.nan)):
        with pytest.raises(ValueError, match="window"):
            sample_absorbing_outgoing(model, bad, 2, None)


def test_nonfinite_assembled_model_refuses_finite_but_overflowing_inputs():
    with pytest.raises(FloatingPointError, match="overflow"):
        _scalar(outgoing_jumps_per_sqrt_ang=(np.full((4, 1), 1e308),))
    with pytest.raises(FloatingPointError, match="overflow"):
        absorbing_axial_model(
            np.diag([1e308, -1e308]),
            np.eye(2) / 2,
            outgoing_shape=(2, 2),
            hbar_v_eV_ang=1,
        )


def test_prepared_model_does_not_alias_caller_matrices_or_shape():
    h, rho, jump = np.array([[2.0]]), np.array([[0.4]]), np.full((4, 1), 0.1)
    shape = [2, 2]
    model = absorbing_axial_model(
        h,
        rho,
        outgoing_jumps_per_sqrt_ang=(jump,),
        outgoing_shape=shape,
        hbar_v_eV_ang=1,
    )
    before = model.at_depth(3)
    h[:] = 99
    rho[:] = 0
    jump[:] = 0
    shape[:] = [100, 100]
    after = model.at_depth(3)
    np.testing.assert_array_equal(after.retained_density, before.retained_density)
    np.testing.assert_array_equal(after.outgoing_density, before.outgoing_density)
    assert model.outgoing_shape == (2, 2)


def test_equal_loss_effect_does_not_fix_outgoing_momentum_law():
    window = np.ones((2, 2)) / 2
    alternating = np.array([[1, -1], [1, -1]]) / 2
    events = []
    for vector in (window.ravel(), alternating.ravel()):
        model, _ = _scalar(outgoing_jumps_per_sqrt_ang=(np.sqrt(0.2) * vector[:, None],))
        event = sample_absorbing_outgoing(model, window, 3, phase_space_stream(623, 9))
        events.append(event)
    assert events[0].depth_ang == pytest.approx(events[1].depth_ang, abs=1e-12)
    assert events[0].outgoing_weight == pytest.approx(events[1].outgoing_weight)
    assert events[0].momentum_index == (0, 0)
    assert events[1].momentum_index == (0, 1)


def _large_phase_model(**extra):
    """Physical-scale phase: ||H|| z / hbar v reaches ~1e4 at z = 1 mm."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    jump = 5e-4 * (rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    args = dict(
        initial_density=np.outer(psi, psi.conj()),
        outgoing_jumps_per_sqrt_ang=(jump,),
        unresolved_effect_inv_ang=np.diag([1e-6, 0, 2e-6, 0]).astype(complex),
        outgoing_shape=(2, 2),
        hbar_v_eV_ang=1973.0,
    )
    args.update(extra)
    return absorbing_axial_model(5 * (x + x.conj().T), **args)


@pytest.mark.parametrize("depth", [1e4, 1e5, 1e6])
def test_large_phase_snapshots_stay_hermitian_positive_and_closed(depth):
    snapshot = _large_phase_model().at_depth(depth)
    for density in (snapshot.retained_density, snapshot.outgoing_density):
        np.testing.assert_array_equal(density, density.conj().T)
        assert np.linalg.eigvalsh(density).min() >= -1e-15  # eigvalsh roundoff
    total = (
        np.trace(snapshot.retained_density).real
        + np.trace(snapshot.outgoing_density).real
        + snapshot.unresolved_weight
    )
    assert total == pytest.approx(1.0, abs=1e-12)


def test_large_phase_sampling_does_not_refuse_on_roundoff():
    model = _large_phase_model()
    window = np.array([[1, 0], [0, 0]], dtype=complex)
    for seed in range(5):
        event = sample_absorbing_outgoing(model, window, 1e6, np.random.default_rng(seed))
        assert 0 <= event.depth_ang <= 1e6 and event.rate_inv_ang > 0


def test_vanishing_local_rate_refuses_instead_of_sampling_noise():
    model, _ = _scalar(outgoing=1.0, unresolved=0.0, weight=1.0)
    rng = RecordedDraws(uniform=np.nextafter(1.0, 0.0))
    with pytest.raises(FloatingPointError, match="vanishes"):
        sample_absorbing_outgoing(model, np.ones((2, 2)) / 2, 1e3, rng)
    assert rng.probabilities == []


def test_large_outgoing_grid_costs_only_retained_dimension():
    """Outgoing tallies are formed from X(z); a 24^2 grid fits the 64 MiB default."""
    rng = np.random.default_rng(3)
    jumps = tuple(0.1 * rng.normal(size=(24 * 24, 2)) for _ in range(3))
    model = absorbing_axial_model(
        np.diag([0.0, 1.0]),
        np.eye(2) / 2,
        outgoing_jumps_per_sqrt_ang=jumps,
        outgoing_shape=(24, 24),
        hbar_v_eV_ang=2.0,
    )
    snapshot = model.at_depth(0.5)
    effect = sum(b.T @ b for b in jumps)
    occupation = solve_ivp(
        lambda z, y: (
            -0.5 * (effect @ y.reshape(2, 2) + y.reshape(2, 2) @ effect)
            - 1j * (np.diag([0.0, 0.5]) @ y.reshape(2, 2) - y.reshape(2, 2) @ np.diag([0.0, 0.5]))
        ).ravel(),
        (0, 0.5),
        (np.eye(2) / 2).astype(complex).ravel(),
        dense_output=True,
        rtol=1e-11,
        atol=1e-13,
    )
    depths = np.linspace(0, 0.5, 2001)
    integral = np.trapezoid(occupation.sol(depths), depths, axis=1).reshape(2, 2)
    expected = sum(b @ integral @ b.T for b in jumps)
    np.testing.assert_allclose(snapshot.outgoing_density, expected, atol=1e-7)

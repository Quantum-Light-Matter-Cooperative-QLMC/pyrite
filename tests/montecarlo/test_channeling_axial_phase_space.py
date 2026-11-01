"""Independent finite-Weyl oracles; these do not validate physical escape."""

import numpy as np
import pytest

from pyrite.montecarlo.channeling.axial_phase_space import (
    phase_space_stream,
    phase_space_weights,
    sample_phase_space,
)


def _direct_weights(phi, window):
    """Explicit complex exponentials, independently of FFT conventions."""
    shape = phi.shape
    x, y = np.indices(shape)
    out = np.empty((*shape, *shape))
    for position in np.ndindex(shape):
        shifted = window[(x - position[0]) % shape[0], (y - position[1]) % shape[1]]
        for momentum in np.ndindex(shape):
            carrier = np.exp(2j * np.pi * (momentum[0] * x / shape[0] + momentum[1] * y / shape[1]))
            mode = shifted * carrier
            out[(*position, *momentum)] = abs(np.vdot(mode, phi)) ** 2 / phi.size
    return out


def _case():
    rng = np.random.default_rng(620)
    phi = rng.normal(size=(3, 4)) + 1j * rng.normal(size=(3, 4))
    phi *= np.sqrt(0.7 / np.vdot(phi, phi).real)
    window = rng.normal(size=phi.shape) + 1j * rng.normal(size=phi.shape)
    window /= np.linalg.norm(window)
    return phi, window


def test_weyl_weights_match_direct_modes_and_preserve_outgoing_mass():
    phi, window = _case()
    actual = phase_space_weights(phi, window)
    np.testing.assert_allclose(actual, _direct_weights(phi, window), atol=2e-16)
    assert np.min(actual) >= 0
    assert np.sum(actual) == pytest.approx(0.7, abs=1e-15)


def test_retained_and_outgoing_instrument_closes_for_mixed_density():
    rng = np.random.default_rng(3)
    _, window = _case()
    basis, _ = np.linalg.qr(rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12)))
    p = basis[:, :4] @ basis[:, :4].conj().T
    f = np.eye(12) - p
    retained = 0.0
    outgoing = np.zeros((3, 4, 3, 4))
    for weight, psi in zip((0.2, 0.8), basis.T[:2], strict=True):
        retained += weight * np.linalg.norm(p @ psi) ** 2
        outgoing += weight * phase_space_weights((f @ psi).reshape(3, 4), window)
    assert retained + outgoing.sum() == pytest.approx(1.0, abs=2e-15)
    assert outgoing.min() >= 0


def test_conditional_sampler_preserves_joint_correlations():
    phi, window = _case()
    expected = _direct_weights(phi, window)

    class Choices:
        def __init__(self, position):
            self.position = position
            self.probabilities = []

        def choice(self, size, p):
            self.probabilities.append(p)
            return self.position if len(self.probabilities) == 1 else 0

    for index in np.ndindex(phi.shape):
        choices = Choices(np.ravel_multi_index(index, phi.shape))
        outcome = sample_phase_space(phi, window, choices)
        np.testing.assert_allclose(
            choices.probabilities[0], expected.sum(axis=(2, 3)).ravel() / 0.7, atol=2e-15
        )
        np.testing.assert_allclose(
            choices.probabilities[1], expected[index].ravel() / expected[index].sum(), atol=2e-15
        )
        assert outcome.position_index == index
        assert outcome.momentum_index == (0, 0)
        assert outcome.state_weight == pytest.approx(0.7)


def test_delta_window_and_plane_wave_limits():
    phi, _ = _case()
    delta = np.zeros_like(phi)
    delta[0, 0] = 1
    q = phase_space_weights(phi, delta)
    np.testing.assert_allclose(
        q, np.broadcast_to(np.abs(phi[:, :, None, None]) ** 2 / phi.size, q.shape)
    )
    plane = np.ones_like(phi) / np.sqrt(phi.size)
    q = phase_space_weights(plane, plane)
    assert q[:, :, 0, 0].sum() == pytest.approx(1)
    assert q[:, :, 1:, :].sum() == pytest.approx(0, abs=1e-30)


def test_measurement_stream_is_incident_addressed_and_reproducible():
    phi, window = _case()
    direct = {i: sample_phase_space(phi, window, phase_space_stream(17, i)) for i in range(8)}
    shuffled = {
        i: sample_phase_space(phi, window, phase_space_stream(17, i))
        for i in (7, 2, 1, 5, 4, 6, 3, 0)
    }
    assert direct == shuffled
    assert not np.array_equal(
        phase_space_stream(17, 0).random(8), phase_space_stream(17, 1).random(8)
    )


def test_zero_state_and_recording_budget():
    phi, window = _case()
    assert phase_space_weights(np.zeros_like(phi), window).sum() == 0
    with pytest.raises(ValueError, match="zero-weight"):
        sample_phase_space(np.zeros_like(phi), window, phase_space_stream(0, 0))
    with pytest.raises(ValueError, match="max_outcomes"):
        phase_space_weights(phi, window, max_outcomes=phi.size**2 - 1)


@pytest.mark.parametrize("bad", [True, -1, 0.5])
def test_invalid_measurement_keys(bad):
    with pytest.raises(ValueError, match="nonnegative integer"):
        phase_space_stream(bad, 0)
    with pytest.raises(ValueError, match="nonnegative integer"):
        phase_space_stream(0, bad)


def test_invalid_window_and_nonfinite_state():
    phi, window = _case()
    with pytest.raises(ValueError, match="unit discrete"):
        phase_space_weights(phi, 2 * window)
    with pytest.raises(ValueError, match="finite"):
        phase_space_weights(phi * np.nan, window)
    with pytest.raises(ValueError, match="same nonempty"):
        phase_space_weights(phi, window[:1])


def _direct_density_weights(rho, window):
    """Dense POVM oracle using explicit carrier phases, no FFTs."""
    x, y = np.indices(window.shape)
    result = np.empty((*window.shape, *window.shape))
    for a in np.ndindex(window.shape):
        shifted = window[(x - a[0]) % window.shape[0], (y - a[1]) % window.shape[1]]
        for b in np.ndindex(window.shape):
            mode = (
                shifted
                * np.exp(2j * np.pi * (b[0] * x / window.shape[0] + b[1] * y / window.shape[1]))
            ).ravel()
            result[(*a, *b)] = np.vdot(mode, rho @ mode).real / window.size
    return result


def test_mixed_density_matches_direct_povm_and_is_linear():
    from pyrite.montecarlo.channeling.axial_phase_space import density_phase_space_weights

    phi, window = _case()
    rho = np.outer(phi.ravel(), phi.ravel().conj())
    diagonal = np.diag(np.linspace(0.01, 0.05, phi.size))
    mixed = 0.4 * rho + 0.6 * diagonal
    q = density_phase_space_weights(mixed, window)
    np.testing.assert_allclose(q, _direct_density_weights(mixed, window), atol=2e-16)
    np.testing.assert_allclose(
        q,
        0.4 * phase_space_weights(phi, window)
        + 0.6 * density_phase_space_weights(diagonal, window),
        atol=2e-16,
    )
    assert q.sum() == pytest.approx(np.trace(mixed).real, abs=2e-15)
    assert q.min() >= 0
    assert np.max(np.abs(q - density_phase_space_weights(np.diag(np.diag(mixed)), window))) > 1e-4


def test_mixed_sampler_conditionals_match_direct_joint_at_every_position():
    from pyrite.montecarlo.channeling.axial_phase_space import sample_density_phase_space

    phi, window = _case()
    rho = np.outer(phi.ravel(), phi.ravel().conj()) + np.eye(phi.size) * 0.02
    q = _direct_density_weights(rho, window)

    class Choices:
        def __init__(self, index):
            self.index = index
            self.probabilities = []

        def choice(self, size, p):
            self.probabilities.append(p)
            return self.index if len(self.probabilities) == 1 else size - 1

    for index in np.ndindex(phi.shape):
        choices = Choices(np.ravel_multi_index(index, phi.shape))
        outcome = sample_density_phase_space(rho, window, choices)
        assert len(choices.probabilities) == 2
        np.testing.assert_allclose(choices.probabilities[0], q.sum(axis=(2, 3)).ravel() / q.sum())
        np.testing.assert_allclose(choices.probabilities[1], q[index].ravel() / q[index].sum())
        assert outcome.position_index == index
        assert outcome.momentum_index == (2, 3)
        assert outcome.state_weight == pytest.approx(np.trace(rho).real)


def test_density_delta_identity_zero_limits_and_incident_streams():
    from pyrite.montecarlo.channeling.axial_phase_space import (
        density_phase_space_weights,
        sample_density_phase_space,
    )

    phi, window = _case()
    rho = np.outer(phi.ravel(), phi.ravel().conj())
    delta = np.zeros_like(window)
    delta[0, 0] = 1
    q = density_phase_space_weights(rho, delta)
    np.testing.assert_allclose(q, phase_space_weights(phi, delta), atol=1e-16)
    np.testing.assert_allclose(density_phase_space_weights(np.eye(phi.size), window), 1 / phi.size)
    assert density_phase_space_weights(np.zeros_like(rho), window).sum() == 0
    first = {
        i: sample_density_phase_space(rho, window, phase_space_stream(623, i)) for i in range(5)
    }
    reordered = {
        i: sample_density_phase_space(rho, window, phase_space_stream(623, i))
        for i in (4, 2, 0, 3, 1)
    }
    assert first == reordered


def test_density_refusals_precede_allocations_and_random_draws(monkeypatch):
    from pyrite.montecarlo.channeling.axial_phase_space import (
        density_phase_space_weights,
        sample_density_phase_space,
    )

    phi, window = _case()
    rho = np.outer(phi.ravel(), phi.ravel().conj())
    for bad in (rho[:1], rho * np.nan, rho + 1j * np.eye(phi.size), -np.eye(phi.size)):
        with pytest.raises(ValueError, match="density"):
            sample_density_phase_space(bad, window, None)
    with pytest.raises(ValueError, match="zero-weight"):
        sample_density_phase_space(np.zeros_like(rho), window, None)
    with pytest.raises(ValueError, match="unit discrete"):
        density_phase_space_weights(rho, window * 2)
    with pytest.raises(ValueError, match="max_outcomes"):
        density_phase_space_weights(rho, window, max_outcomes=phi.size**2 - 1)
    for bad in (True, 0, 1.5):
        with pytest.raises(ValueError, match="max_working_bytes"):
            sample_density_phase_space(rho, window, None, max_working_bytes=bad)

    def forbidden(*args, **kwargs):
        raise AssertionError("eigensolver must not run before admission")

    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)
    with pytest.raises(ValueError, match="working-memory"):
        density_phase_space_weights(rho, window, max_working_bytes=1)
    with pytest.raises(ValueError, match="working-memory"):
        sample_density_phase_space(rho, window, None, max_working_bytes=1)

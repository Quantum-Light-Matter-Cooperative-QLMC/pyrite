"""Small deterministic checks for the diagnostic entry-projector bookkeeping."""

from types import SimpleNamespace

import numpy as np
import pytest

from checks import channeling_submev_w001_entry as entry


def _states(coefficients):
    return SimpleNamespace(
        orders=np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
        energies_eV=np.array([-3.0, -2.0, -2.0, 1.0]),
        coefficients=coefficients,
    )


def test_doublet_projector_is_invariant_under_mixing():
    c = np.eye(4, dtype=complex)
    baseline = entry._components_from_state(_states(c), (1, 0), 0.0)
    c[:, 1:3] = c[:, 1:3] @ np.array([[1, 1], [-1, 1]]) / np.sqrt(2)
    mixed = entry._components_from_state(_states(c), (1, 0), 0.0)
    assert baseline["upper_2p"] == pytest.approx(1.0)
    assert mixed == pytest.approx(baseline)
    assert sum(mixed.values()) == pytest.approx(1.0)


def test_outside_basis_is_not_renormalized(monkeypatch):
    profile = SimpleNamespace(g_inv_ang=(2.0, 3.0))
    called = []

    def solve(_profile, _gamma, folded, *, max_order):
        called.append((folded, max_order))
        return _states(np.eye(4))

    monkeypatch.setattr(entry, "axial_states", solve)
    outside = entry._at_tilt(profile, 1.2, (4.1, 0.0), 0.0, (1, 1))
    assert outside["outside_basis"] == 1.0
    assert not called
    inside = entry._at_tilt(profile, 1.2, (2.1, 0.0), 0.0, (1, 1))
    assert inside["upper_2p"] == pytest.approx(1.0)
    assert called[0][0] == pytest.approx((0.1, 0.0))


def test_zone_average_is_normalized_and_resolves_order(monkeypatch):
    profile = SimpleNamespace(g_inv_ang=(2.0, 3.0))
    monkeypatch.setattr(entry, "axial_states", lambda *args, **kwargs: _states(np.eye(4)))
    narrow, wide = 0.15, 2.0
    (tight, broad), diag = entry._zone_entry(profile, 1.2, 0.0, (1, 1), (narrow, wide), 32)
    # A narrow beam stays on the incident order (0, 0), which maps to band 0.
    assert tight["normalization"] == pytest.approx(1.0, abs=1e-12)
    assert tight["ground_1s"] == pytest.approx(1.0, abs=1e-6)
    # A broad beam spills onto orders the 4-wave basis lacks; that weight is
    # reported, not renormalized away.
    assert broad["outside_basis"] > 0.1
    assert sum(broad[key] for key in entry.COMPONENTS) == pytest.approx(broad["normalization"])
    assert diag["n_zone"] == 32


def test_zone_average_rejects_nonpositive_width():
    profile = SimpleNamespace(g_inv_ang=(2.0, 3.0))
    with pytest.raises(ValueError, match="positive"):
        entry._zone_entry(profile, 1.2, 0.0, (1, 1), (0.0,), 4)


def test_screen_refuses_missing_radiating_doublet(monkeypatch):
    profile = SimpleNamespace(extrema=lambda: (-3.0, 0.0))
    monkeypatch.setattr(entry, "longitudinal_kinematics", lambda _e: (1.2, 0.5))
    monkeypatch.setattr(entry, "_orders", lambda *args: (1, 1))
    absent = _states(np.eye(4))
    absent.energies_eV[2] = 0.5  # No longer a bound 2p pair.
    monkeypatch.setattr(entry, "axial_states", lambda *args, **kwargs: absent)
    with pytest.raises(ValueError, match="no longer has the assumed bound"):
        entry.screen_energy(profile, 30.0, (0.0,), 2, 3)

"""``xray_dispersion`` switch: vacuum baseline vs in-medium line kinematics.

The refractive model replaces ``k = omega`` with the Maxwell dispersion relation
``k = n(omega) omega`` in the bulk crystal, which moves the CXR resonance from
``omega_res = v.g / (1 - v.n_hat)`` to ``omega_res = v.g / (1 - Re n (v.n_hat))``.
For ``delta = 1 - Re n << 1`` that is a fractional line shift of
``-delta (v.n_hat) / (1 - v.n_hat)``, which is what the first tests measure.

The same dispersion relation governs the coherent propagation phase: the escape
leg runs at the medium's phase velocity, so segment ``j`` accumulates
``-delta(E) omega(E) L_esc,j`` on top of the vacuum ``omega d_j`` -- the real
partner of the Beer-Lambert amplitude already applied over that same path. Its
observable is the RELATIVE phase between segments at different depths, which the
later tests extract from the two-segment interference term.
"""

import numpy as np
import pytest

from pyrite.materials.crystal import CRYSTALS, refractive_index
from pyrite.montecarlo.spectrum import mc_spectrum
from pyrite.montecarlo.spectrum.lines import _observation_direction
from pyrite.montecarlo.transport import beta_from_keV

CRYSTAL = "hopg"
HKL = (0, 0, 2)
B_ANG2 = 0.4
E_KEV = 100.0
THICKNESS_ANG = 1e5

# Long segment -> narrow sinc envelope, so the spectral peak resolves a shift of
# order delta * E_res (~6e-2 eV here) instead of drowning in a ~100 eV wide line.
SEG_LENGTH_ANG = 8000.0
E_GRID = np.linspace(1599.0, 1601.5, 100_001)


def _segments():
    return {
        "r_mid": np.array([[0.0, 0.0, 5000.0]]),
        "v_hat": np.array([[0.0, 0.0, 1.0]]),
        "L_ang": np.array([SEG_LENGTH_ANG]),
        "E_keV": np.array([E_KEV]),
        "t_ang": np.zeros(1),
        "t0_ang": np.zeros(1),
        "elec_id": np.zeros(1, dtype=int),
        "layer": np.zeros(1, dtype=int),
        "Ne": 1,
        "thickness_ang": THICKNESS_ANG,
        "crystal_width_ang": 1e6,
        "crystal_height_ang": 1e6,
        "n_backscattered": 0,
        "n_missed": 0,
        "n_layers": 1,
    }


def _single_layer():
    """One-layer absorber stack, which routes the run onto the per-hkl path."""
    info = CRYSTALS[CRYSTAL]
    n_atoms = len(info["basis"]) / info["V_cell"]
    return [(0.0, THICKNESS_ANG, [("C", n_atoms)])]


def _spectrum(**kwargs):
    return mc_spectrum(_segments(), E_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2, **kwargs)


def _predicted_relative_shift(E_res_eV):
    v_dot_n = float(beta_from_keV(np.array([E_KEV]))[0]) * float(
        np.array([0.0, 0.0, 1.0]) @ _observation_direction(np.deg2rad(119.0), None)
    )
    delta = 1.0 - float(refractive_index(CRYSTAL, E_res_eV).real)
    return -delta * v_dot_n / (1.0 - v_dot_n)


def test_vacuum_is_the_default_and_leaves_the_spectrum_untouched():
    np.testing.assert_array_equal(_spectrum(), _spectrum(xray_dispersion="vacuum"))


@pytest.mark.parametrize("layers", [None, _single_layer()], ids=["batched", "per_hkl"])
def test_refractive_shifts_the_line_by_the_in_medium_denominator(layers):
    vac = _spectrum(layers=layers)
    ref = _spectrum(layers=layers, xray_dispersion="refractive")
    assert vac.max() > 0.0

    E_vac = E_GRID[vac.argmax()]
    E_ref = E_GRID[ref.argmax()]
    measured = (E_ref - E_vac) / E_vac
    # Positive: the detector looks upstream (v.n_hat < 0), so the in-medium
    # denominator is larger than the vacuum one and the line moves up in energy.
    assert measured > 0.0
    np.testing.assert_allclose(measured, _predicted_relative_shift(E_vac), rtol=2e-3)


def test_unknown_dispersion_model_is_rejected():
    with pytest.raises(ValueError, match="xray_dispersion must be one of"):
        _spectrum(xray_dispersion="in_medium")


def test_refractive_coherent_through_a_layer_stack_is_refused():
    with pytest.raises(NotImplementedError, match="LAYERED absorber"):
        _spectrum(xray_dispersion="refractive", coherent=True, layers=_single_layer())


# --- in-medium coherent propagation phase ---------------------------------
#
# The escape leg accumulates phase at the medium's phase velocity, so segment j
# picks up ``-delta(E) omega(E) L_esc,j`` on top of the vacuum ``omega d_j``.
# Only DIFFERENCES matter, so the observable is the relative phase between two
# segments at different depths -- which is also the design brief's question:
# does a delta ~ 1e-4 accumulate into an order-unity phase over microns?

N_Z = float(np.cos(np.deg2rad(119.0)))  # detector looks upstream, out the entrance face


def _two_segments(z1, z2):
    seg = _segments()
    seg["r_mid"] = np.array([[0.0, 0.0, z1], [0.0, 0.0, z2]])
    seg["v_hat"] = np.tile(seg["v_hat"], (2, 1))
    for key in ("L_ang", "E_keV", "t_ang", "t0_ang"):
        seg[key] = np.tile(seg[key], 2)
    for key in ("elec_id", "layer"):
        seg[key] = np.zeros(2, dtype=int)
    return seg


def _one_segment(z):
    seg = _segments()
    seg["r_mid"] = np.array([[0.0, 0.0, z]])
    return seg


def _relative_phase(z1, z2, i_E, **kwargs):
    """Extract cos(delta phi) between the two segments' fields at grid point i_E.

    Both segments share v, so they share E_res, the sinc envelope, and the
    amplitude A; they differ only in Beer-Lambert escape weight. The coherent
    total is therefore ``I1 + I2 + 2 sqrt(I1 I2) cos(delta phi)`` with the
    single-segment runs supplying I1 and I2.
    """

    def run(seg):
        return mc_spectrum(seg, E_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2, coherent=True, **kwargs)

    i_1 = run(_one_segment(z1))[i_E]
    i_2 = run(_one_segment(z2))[i_E]
    i_both = run(_two_segments(z1, z2))[i_E]
    return (i_both - i_1 - i_2) / (2.0 * np.sqrt(i_1 * i_2))


@pytest.mark.parametrize("sinc_cutoff", [None, 200.0], ids=["batched", "per_hkl"])
def test_vacuum_coherent_is_untouched_by_the_dispersion_switch(sinc_cutoff):
    seg = _two_segments(5_000.0, 15_000.0)
    kw = dict(coherent=True, sinc_cutoff=sinc_cutoff)
    base = mc_spectrum(seg, E_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2, **kw)
    same = mc_spectrum(seg, E_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2, xray_dispersion="vacuum", **kw)
    assert base.max() > 0.0
    np.testing.assert_array_equal(base, same)


@pytest.mark.parametrize("sinc_cutoff", [None, 200.0], ids=["batched", "per_hkl"])
def test_refractive_adds_the_escape_path_phase_between_two_depths(sinc_cutoff):
    z1, z2 = 5_000.0, 10_000.0
    kw = dict(sinc_cutoff=sinc_cutoff)
    i_E = int(_spectrum(coherent=True, **kw).argmax())
    E = float(E_GRID[i_E])

    cos_vac = _relative_phase(z1, z2, i_E, **kw)
    cos_ref = _relative_phase(z1, z2, i_E, xray_dispersion="refractive", **kw)

    # -delta(E) omega(E) (L1 - L2) with L_esc = z / (-n_z) out the entrance face.
    delta = 1.0 - float(refractive_index(CRYSTAL, E).real)
    d_phase = -delta * (E / 1973.269804) * (z1 - z2) / (-N_Z)
    assert 0.3 < abs(d_phase) < np.pi - 0.3  # resolvable, and no arccos branch cut

    # arccos is even, so compare the shift's magnitude on both branches and take
    # the one the phase actually sits on.
    got = np.arccos(np.clip(cos_ref, -1.0, 1.0)) - np.arccos(np.clip(cos_vac, -1.0, 1.0))
    # Exact to rounding: the geometry (L_esc = z / -n_z), the sign, and delta(E)
    # on the output grid are all pinned, not just the order of magnitude.
    np.testing.assert_allclose(min(abs(got - d_phase), abs(got + d_phase)), 0.0, atol=1e-9)


def test_micron_scale_depth_separation_inverts_the_interference():
    """The design brief's headline question, answered by the spectrum itself.

    delta ~ 1e-4 is negligible per Angstrom, but over ~1 micron of depth
    separation it reaches pi: constructive interference between the two segments
    becomes destructive. That is the whole reason this model exists.
    """
    z1, z2 = 5_000.0, 5_000.0 + 9_900.0  # escape paths ~1 micron apart
    i_E = int(_spectrum(coherent=True).argmax())
    E = float(E_GRID[i_E])
    delta = 1.0 - float(refractive_index(CRYSTAL, E).real)
    d_phase = -delta * (E / 1973.269804) * (z1 - z2) / (-N_Z)
    assert abs(abs(d_phase) - np.pi) < 0.05  # this geometry is tuned to half a cycle

    # A half-cycle shift flips the interference term's sign, whatever the vacuum
    # phase happened to be: cos(x +- pi) = -cos(x). No arccos branch to pick.
    cos_vac = _relative_phase(z1, z2, i_E)
    cos_ref = _relative_phase(z1, z2, i_E, xray_dispersion="refractive")
    assert abs(cos_vac) > 0.2  # the vacuum interference term is not already ~0
    np.testing.assert_allclose(cos_ref, -cos_vac, atol=0.05)


def test_single_segment_coherent_is_pure_phase_under_refraction():
    """One segment has no relative phase, so the new factor must cancel in |.|^2."""
    seg = _one_segment(9_000.0)
    kw = dict(xray_dispersion="refractive")
    coh = mc_spectrum(seg, E_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2, coherent=True, **kw)
    inc = mc_spectrum(seg, E_GRID, CRYSTAL, [HKL], B_ang2=B_ANG2, **kw)
    assert coh.max() > 0.0
    np.testing.assert_allclose(coh, inc, rtol=1e-10, atol=1e-14 * inc.max())

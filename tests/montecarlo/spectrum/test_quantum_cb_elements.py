"""Exact second-order PXR/CB line elements (#1134).

Validation: cbs-amplitude
"""

import numpy as np
import pytest

from pyrite._backend import _to_cpu
from pyrite.montecarlo.spectrum.cxr import _recoil


def _quantum_line_factors(*args):
    """Host values of the production helper (CuPy scalars under CUDA)."""
    return tuple(float(_to_cpu(x)) for x in _recoil._quantum_line_factors(*args))


MC2 = 510998.95
H = 1973.269804 / MC2  # hbar c / m c^2 [Angstrom]


def _factors(h, gamma, v, g, k, e):
    return _quantum_line_factors(h, gamma, v @ g, k @ v, k @ g, g @ g, k @ k, v @ e, g @ e, k @ e)


def _vector_elements(gamma, v, g, k, e):
    """Independent momentum-vector form: transfer -hbar g, emit hbar k (units m c^2)."""
    p_i = gamma * v
    p_1, p_2, p_f = p_i - H * g, p_i - H * k, p_i - H * (g + k)
    E = [np.sqrt(1.0 + p @ p) for p in (p_i, p_1, p_2, p_f)]
    e_i, e_1, e_2, e_f = E
    j_fi = (p_i + p_f) / (e_i + e_f)
    j_f1 = (p_1 + p_f) / (e_1 + e_f)
    j_2i = (p_i + p_2) / (e_i + e_2)
    d_1 = (-2 * H * (p_i @ g) + H * H * (g @ g)) / (e_1 + e_i)  # E_1 - E_i
    d_2 = (-2 * H * (p_2 @ g) + H * H * (g @ g)) / (e_f + e_2)  # E_f - E_2
    kg = k + g
    pxr = (j_fi @ kg) * (kg @ e) - (k @ k) * (j_fi @ e)
    return pxr, -(j_f1 @ e) / d_1 + (j_2i @ e) / d_2


def _random_kinematics(rng, T_eV):
    gamma = 1.0 + T_eV / MC2
    v = rng.normal(size=3)
    v *= np.sqrt(1.0 - 1.0 / gamma**2) / np.linalg.norm(v)
    g, n = rng.normal(size=3) * 3.0, rng.normal(size=3)
    n /= np.linalg.norm(n)
    k = (0.05 + rng.random()) * n
    e = np.cross(n, rng.normal(size=3))
    return gamma, v, g, k, e / np.linalg.norm(e)


def test_hbar_zero_is_the_classical_braced_form():
    rng = np.random.default_rng(0)
    for _ in range(200):
        gamma, v, g, k, e = _random_kinematics(rng, 1e5 * rng.random() + 1e3)
        vdg, kv, kg = v @ g, k @ v, k @ g
        pxr0 = (vdg + kv) * (g @ e + k @ e) - (k @ k) * (v @ e)
        cbs0 = -((g @ e - vdg * (v @ e)) + (v @ e) * (kg - kv * vdg) / vdg) / (gamma * vdg)
        np.testing.assert_allclose(_factors(0.0, gamma, v, g, k, e), [pxr0, cbs0], rtol=1e-12)


@pytest.mark.parametrize("T_eV", [1e4, 3e4, 1e5, 1e6])
def test_closed_form_matches_vector_elements(T_eV):
    rng = np.random.default_rng(int(T_eV))
    for _ in range(50):
        args = _random_kinematics(rng, T_eV)
        scale = np.abs(_factors(0.0, *args))
        got = np.asarray(_factors(H, *args))
        np.testing.assert_array_less(np.abs(got - _vector_elements(*args)) / scale, 1e-10)


def test_float32_stays_on_the_classical_scale():
    rng = np.random.default_rng(32)
    for T_eV in (1e4, 5e6, 1e9):
        for _ in range(50):
            gamma, v, g, k, e = _random_kinematics(rng, T_eV)
            scalars = (gamma, v @ g, k @ v, k @ g, g @ g, k @ k, v @ e, g @ e, k @ e)
            ref = _quantum_line_factors(H, *scalars)
            got = _quantum_line_factors(np.float32(H), *(np.float32(x) for x in scalars))
            scale = np.abs(_quantum_line_factors(0.0, *scalars))
            assert np.all(np.abs(np.asarray(got, float) - ref) / scale < 1e-3)


def test_reference_pin_si_222_30kev():
    """Si (222), 30 keV along z, photon at 100 deg in x-z, at the exact vacuum root.

    Reference values: an independent 60-digit mpmath evaluation of the
    second-order elements (Shi 2024 SI S12-S15 with convective currents) from
    explicit momenta, on-shell energies and currents, importing no PyRITE
    physics: ``checks/quantum_cb_elements_1134.py::reference_pin`` (#1144).
    The classical values pin the braced form, including its single
    ``1/gamma``: for sigma (e perpendicular to v and k) ``f_cbs`` is
    ``-(g.e)/(gamma v.g)``.
    """
    gamma = 1.0 + 3e4 / MC2
    v = np.array([0.0, 0.0, np.sqrt(1.0 - 1.0 / gamma**2)])
    g = np.full(3, 4.0 * np.pi / 5.431)
    th = np.radians(100.0)
    n = np.array([np.sin(th), 0.0, np.cos(th)])
    k = REFERENCE_OMEGA * n
    e_s = np.array([0.0, 1.0, 0.0])
    for e, (quantum, classical) in zip((e_s, np.cross(n, e_s)), REFERENCE, strict=True):
        np.testing.assert_allclose(_factors(H, gamma, v, g, k, e), quantum, rtol=1e-12)
        np.testing.assert_allclose(_factors(0.0, gamma, v, g, k, e), classical, rtol=1e-12)
    np.testing.assert_allclose(REFERENCE[0][1][1], -(g @ e_s) / (gamma * (v @ g)), rtol=1e-12)


# 60-digit reference (checks/quantum_cb_elements_1134.py::reference_pin), rounded
# to float64: exact vacuum root omega [1/Angstrom] and, for sigma then pi,
# ((pxr_num, f_cbs) quantum, (pxr_num, f_cbs) classical).
REFERENCE_OMEGA = 0.68665746147068140754
REFERENCE = (
    ((1.5907956780189942, -2.9955100457228503), (1.6674555552543886, -2.876417643261264)),
    ((1.690198839167976, -3.9242147431280814), (1.7791971109401228, -3.724387851171339)),
)
